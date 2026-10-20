# RTX 2060 Performance Results

Test date: 2026-10-20 (earlier sections: 2026-10-19)
Machine: NVIDIA GeForce RTX 2060 (6 GB, driver 581.29, Vulkan queue family 2), 8 host threads
Sample: `test_j2k_rgb12_1080p24_3min.mxf`, 1920x1080 RGB12, reversible 5/3, 1 tile, 64x64 codeblocks (~879 per frame)
FFmpeg: `C:/Users/zhangqingxuan/Desktop/Tools/ffmpeg/bin/ffmpeg.exe`

## Summary

| mode | ms/frame | fps |
| --- | ---: | ---: |
| vk-j2k-dec cpu (10 frames) | 212.3 | 4.71 |
| FFmpeg native (240 frames, multi-thread) | 37.3 | 26.8 |
| gpu-tier1 before batching (main @ 88fd8ee, 30 frames) | 29.2 | 34.2 |
| gpu-tier1, batch 1, no CPU offload | 11.7 | 85.4 |
| gpu-tier1, batch 8, no CPU offload (pure GPU, 6 runs) | 7.8-8.2 | 122.5-128.9 |
| gpu-tier1, fully auto (6 runs) | 7.2-8.5 | 117.8-139.6 |
| nvjpeg2000, 1 worker (reference) | 12.9 | 77.6 |
| nvjpeg2000, 8 workers, decode only (reference) | 6.9 | 146.0 |

gpu-tier1 rows are whole-loop averages over 240 frames with the current defaults: packed-state
Tier-1 shader (`tier1_packed.comp.glsl`), fused 2D IDWT (`idwt53_fused_2d.comp.glsl`) and
readback on the dedicated transfer queue. Every gpu-tier1 configuration produces the same
240-frame hash (`VK_J2K_HASH_FRAMES=1` -> `47d938a71bb308e5`), and the last frame matches
FFmpeg's decode bit-exactly (`diff-rgb48`: 0 differing samples).

Run-to-run spread on this machine is large: two identical shader builds measured back to back
differed by up to ~10 fps in one round, and the same build ranged 88-109 fps at batch 8 over
the day. The GPU idles at ~1050 MHz and boosts to ~1830 MHz during runs, and the desktop has
other GPU clients. Compare changes with interleaved multi-round A/B, not single runs.

## GPU stage breakdown

`VK_J2K_GPU_TIMING=1`, ms per frame:

| stage | batch 1 | batch 8 | auto (batch 8 + CPU offload) |
| --- | ---: | ---: | ---: |
| upload | 0.02 | 0.03 | 0.22 |
| Tier-1 | 10.84 | 6.82 | 5.51 |
| scatter | 0.14 | 0.14 | 0.14 |
| IDWT | 0.56 | 0.51 | 0.51 |
| color + pack | 0.12 | 0.13 | 0.12 |
| readback copy (transfer queue) | 1.17 | 1.17 | 1.22 |

The readback copy runs on the transfer queue and mostly overlaps compute, so the stage sum is
larger than the loop time per frame.

## Optimization history (pure GPU, batch 8, no CPU offload)

| change | effect on this machine |
| --- | --- |
| Packed stripe-column Tier-1 state | 52.8 -> 95.4 fps |
| Per-stage GPU timestamps | no throughput change (A/B vs previous build) |
| Fused 2D IDWT, one dispatch per level for all components | IDWT 1.37 -> 0.52 ms/frame, 90.9 -> 96.5 fps |
| Readback on dedicated transfer queue | 88.8 -> 93.5 fps (5-round mean) |
| Tier-1 micro-optimizations (hoisted reconstruction mode, atomic refinement/mirror writes, computed contexts) | Tier-1 8.25-8.59 -> 8.04-8.15 ms, 105.1-109.0 -> 110.0-111.4 fps (4-round interleaved) |
| Full MQ table entry stored in each context | Tier-1 9.89-10.13 -> 9.54-9.92 ms (4-round interleaved, low-clock session) |
| Skip stripes with no significant sample (refinement) or no significant neighbour (significance) | Tier-1 7.78-7.91 -> 6.84-6.97 ms, 112.0-114.7 -> 126.3-128.3 fps (4-round interleaved) |
| Tried, not adopted: unconditional byte loads (no word cache) | within noise, slower in 2 of 4 rounds |
| Tried, not adopted: 2 or 8 warps per workgroup (vs 4) | 2 slower in all rounds, 8 mixed |

The computed significance/sign contexts were checked exhaustively against the old lookup tables
(all 189 input cases).

## Tier-1 shader comparison

Same 240 frames, `VK_J2K_TIER1_SHADER` selects the shader:

| configuration | per-sample state (`tier1_cleanup_nostats`) | packed column state (`tier1_packed`) |
| --- | ---: | ---: |
| batch 1, no CPU offload | 40.9 fps | 69.0 fps |
| batch 8, no CPU offload | 52.8 fps | 95.4 fps |
| fully auto | 59.0-59.7 fps | 104.4-104.8 fps |

A 16x16-codeblock re-encode of the same clip (40 frames, batch 1, no offload):
thread-per-codeblock 66.2 fps, packed 124.4 fps, both with hash `459b94164d4f7e25`.
The packed shader is therefore the default for both codeblock sizes; thread-per-codeblock is
only a fallback when `tier1_packed.comp.spv` is missing.

## gpu-tier1, fully auto (2026-10-20, before stripe skip)

```
profile-mxf-ffmpeg test_j2k_rgb12_1080p24_3min.mxf 240 0 gpu-tier1 ...
  autotune batch 8 (max 8) probes [8:8.17ms 4:8.59ms] cpu_codeblocks_per_frame 80 (max 239)
  loop_avg_ms_per_frame total 9.239 fps 108.24
```

Whole-loop fps over 7 auto runs: 116.53, 111.55, 115.68, 116.44, 113.80, 109.77, 108.24.
The tuner settled on batch 8 with 71-84 CPU-decoded codeblocks per frame.

## Fixed configurations (2026-10-20, before stripe skip)

```
VK_J2K_BATCH=8 VK_J2K_CPU_CBLKS=0
  loop_avg_ms_per_frame total 10.074 fps 99.26
  (5 more runs: 106.86, 111.55, 110.90, 110.98, 110.98 fps)

VK_J2K_BATCH=1 VK_J2K_CPU_CBLKS=0
  loop_avg_ms_per_frame total 12.964 fps 77.14
```

## Where the gains come from (2026-10-19, older pipeline)

| change | effect on this machine |
| --- | --- |
| Tier-1 shaders built with `-DTIER1_NO_STATS` | ~6% faster Tier-1 dispatch |
| Multi-frame batches (one Tier-1 dispatch over N frames) | 39.4 -> 45.9 fps at batch 8 (old shader) |
| CPU decodes each frame's heaviest codeblocks, overlapped with the GPU | 45.9 -> ~56 fps (old shader) |
| Packed stripe-column Tier-1 state | 52.8 -> 95.4 fps pure GPU at batch 8 |

## nvjpeg2000 reference

Same 240 frames (copied as `.j2c`), `decode_bench` from `D:\temp\nvjpeg2000dec`:

```
decode_bench.exe target/samples/nvj_frames 3 1,2,4,8 all
=== Stage: decode ===
      1     3091.95      77.62
      2     1772.95     135.37
      4     1647.55     145.67
      8     1644.32     145.96
=== Stage: decode+shift+D2H ===
      8     1714.97     139.94
```

CUPTI kernel timing (1 worker): one `tier1decodemultiple_channels_k` launch per frame, 11.0 ms
average; IDWT and inverse quantization together ~0.4 ms per frame. nvjpeg2000 output is planar
16-bit (no RGB interleave) and was not checked for correctness here.

## FFmpeg native decode

```
ffmpeg -hide_banner -benchmark -i test_j2k_rgb12_1080p24_3min.mxf -frames:v 240 -f null -
bench: utime=56.234s stime=2.359s rtime=8.957s
```

## cpu (full CPU path)

```
profile-mxf-ffmpeg: mode cpu, 1920x1080 12-bit, discard 0, frames 10
  avg_skip_first_frame_ms decode_total 212.251 fps 4.71 tier1 56.190 reconstruct 143.518 pack 10.779
```

## Notes

- `readback` is ~0 in most gpu-tier1 rows because the profiler only copies out the last frame;
  frames are read straight from mapped memory. A real consumer pays for its own copy or encode.
  With `VK_J2K_HASH_FRAMES=1` every frame is read, which costs ~6 ms/frame here.
- The CPU path fails on frame 18 (`negative coefficient bitplane`): some codeblocks signal more
  zero bitplanes than magnitude bitplanes. The GPU shaders handle them, so the CPU offload leaves
  those codeblocks on the GPU. Not fixed yet.
- The `gpu` mode (CPU Tier-1 + GPU DWT) was not re-measured.
- Smoke and bench commands (`tier1-gpu-bench` and friends) still use the per-sample state
  shaders and their checksums.
- The 2026-05-22 results in earlier versions of this file (10.4 fps gpu-tier1) predate the
  thread-per-codeblock, local-state and double-buffer commits. The 10-frame FFmpeg run there
  (95.8 ms/frame) was dominated by startup cost.
