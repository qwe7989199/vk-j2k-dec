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
| gpu-tier1, batch 1, no CPU offload | 25.4 | 39.4 |
| gpu-tier1, batch 8, no CPU offload | 21.8 | 45.9 |
| gpu-tier1, batch 8, auto CPU offload | 17.9 | 56.0 |
| gpu-tier1, fully auto (3 runs) | 16.9-18.7 | 53.5-59.2 |

gpu-tier1 rows are steady-state averages over 240 frames (after auto-tuning settles) unless noted.
Every gpu-tier1 configuration produces the same 240-frame hash (`VK_J2K_HASH_FRAMES=1` ->
`47d938a71bb308e5`), and the last frame matches FFmpeg's decode bit-exactly (`diff-rgb48`: 0 differing samples).

## gpu-tier1, fully auto

```
profile-mxf-ffmpeg test_j2k_rgb12_1080p24_3min.mxf 240 0 gpu-tier1 ...
  device_local_heap_mib 5955 cpu_threads 8
  pipeline_setup_ms 133.838
  autotune batch 8 (max 8) probes [8:16.38ms 4:18.30ms] cpu_codeblocks_per_frame 186 (max 239)
  loop_avg_ms_per_frame total 17.534 fps 57.03
  steady_ms_per_frame total 16.886 fps 59.22 cpu_tier1 11.490 gpu_wait 3.062 readback 0.001 (iterations 16, frames 124)
```

| run | chosen batch | CPU codeblocks/frame | steady fps | whole-loop fps |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 139 | 53.48 | 52.53 |
| 2 | 8 | 186 | 59.22 | 57.03 |
| 3 | 8 | 184 | 54.21 | 53.52 |

Batch 4 and 8 measure within a few percent of each other here, so the tuner's choice varies
between runs. `VK_J2K_BATCH=8` pins it.

## Fixed configurations

```
VK_J2K_BATCH=8 (auto offload)
  autotune batch 8 (max 8, fixed) probes [] cpu_codeblocks_per_frame 170 (max 239)
  steady_ms_per_frame total 17.865 fps 55.98 cpu_tier1 9.879 gpu_wait 5.942 readback 0.001

VK_J2K_BATCH=8 VK_J2K_CPU_CBLKS=0
  steady_ms_per_frame total 21.776 fps 45.92 cpu_tier1 0.001 gpu_wait 21.613 readback 0.001

VK_J2K_BATCH=1 VK_J2K_CPU_CBLKS=0
  steady_ms_per_frame total 25.364 fps 39.43 cpu_tier1 0.005 gpu_wait 25.087 readback 0.004
```

## Where the gains come from

| change | effect on this machine |
| --- | --- |
| Tier-1 shaders built with `-DTIER1_NO_STATS` | ~6% faster Tier-1 dispatch |
| Multi-frame batches (one Tier-1 dispatch over N frames) | 39.4 -> 45.9 fps at batch 8 |
| CPU decodes each frame's heaviest codeblocks, overlapped with the GPU | 45.9 -> ~56 fps |

A 16x16-codeblock re-encode of the same clip (thread-per-codeblock shader) went from 60.6 fps
(batch 1, no offload) to 115.5 fps (batch 4, 64 CPU codeblocks/frame), also bit-exact.

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

- `readback` is ~0 in the batched rows because the profiler only copies out the last frame;
  frames are read straight from mapped memory. A real consumer pays for its own copy or encode.
- The CPU path fails on frame 18 (`negative coefficient bitplane`): some codeblocks signal more
  zero bitplanes than magnitude bitplanes. The GPU shaders handle them, so the CPU offload leaves
  those codeblocks on the GPU. Not fixed yet.
- The `gpu` mode (CPU Tier-1 + GPU DWT) was not re-measured.
- The 2026-05-22 results in the previous version of this file (10.4 fps gpu-tier1) predate the
  thread-per-codeblock, local-state and double-buffer commits. The 10-frame FFmpeg run there
  (95.8 ms/frame) was dominated by startup cost.
