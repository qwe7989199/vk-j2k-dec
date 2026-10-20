# RTX 2060 Performance Results

Test date: 2026-10-19
Machine: NVIDIA GeForce RTX 2060 (6 GB, driver 581.29, Vulkan queue family 2), 8 host threads
Sample: `test_j2k_rgb12_1080p24_3min.mxf`, 1920x1080 RGB12, reversible 5/3, 1 tile, 64x64 codeblocks (~879 per frame)
FFmpeg: `C:/Users/zhangqingxuan/Desktop/Tools/ffmpeg/bin/ffmpeg.exe`

## Summary

| mode | ms/frame | fps |
| --- | ---: | ---: |
| vk-j2k-dec cpu (10 frames) | 212.3 | 4.71 |
| FFmpeg native (240 frames, multi-thread) | 37.3 | 26.8 |
| gpu-tier1 before batching (main @ 88fd8ee, 30 frames) | 29.2 | 34.2 |
| gpu-tier1, batch 1, no CPU offload | 14.5 | 69.0 |
| gpu-tier1, batch 8, no CPU offload (pure GPU) | 10.5 | 95.4 |
| gpu-tier1, fully auto (3 runs) | 9.5-9.6 | 104.4-104.8 |
| nvjpeg2000, 1 worker (reference) | 12.9 | 77.6 |
| nvjpeg2000, 8 workers, decode only (reference) | 6.9 | 146.0 |

gpu-tier1 rows use the packed-state Tier-1 shader (`tier1_packed.comp.glsl`) and are whole-loop
averages over 240 frames. Every gpu-tier1 configuration produces the same 240-frame hash
(`VK_J2K_HASH_FRAMES=1` -> `47d938a71bb308e5`), and the last frame matches FFmpeg's decode
bit-exactly (`diff-rgb48`: 0 differing samples).

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

## gpu-tier1, fully auto

```
profile-mxf-ffmpeg test_j2k_rgb12_1080p24_3min.mxf 240 0 gpu-tier1 ...
  tier1_shader target\shaders\tier1_packed.comp.spv (warp-per-codeblock)
  autotune batch 8 (max 8) probes [8:9.07ms 4:9.38ms] cpu_codeblocks_per_frame 100 (max 239)
  loop_avg_ms_per_frame total 9.541 fps 104.81
  steady_ms_per_frame total 9.398 fps 106.41 cpu_tier1 6.422 gpu_wait 1.747 readback 0.001 (iterations 16, frames 124)
```

| run | chosen batch | CPU codeblocks/frame | steady fps | whole-loop fps |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 100 | 106.41 | 104.81 |
| 2 | 8 | 101 | 106.11 | 104.59 |
| 3 | 8 | 99 | 105.95 | 104.37 |

## Fixed configurations

```
VK_J2K_BATCH=8 VK_J2K_CPU_CBLKS=0
  loop_avg_ms_per_frame total 10.482 fps 95.40
  steady_ms_per_frame total 10.483 fps 95.39 cpu_tier1 0.000 gpu_wait 10.352 readback 0.001

VK_J2K_BATCH=1 VK_J2K_CPU_CBLKS=0
  loop_avg_ms_per_frame total 14.504 fps 68.95
  steady_ms_per_frame total 14.496 fps 68.98 cpu_tier1 0.002 gpu_wait 14.353 readback 0.002
```

## Where the gains come from

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
