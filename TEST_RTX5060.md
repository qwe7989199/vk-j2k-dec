> 2026-10-20 correctness update: the original `47d938a71bb308e5` hash below describes the pre-fix decoder. A packet-header alignment bug skipped the stuffed byte after a terminal `0xff`, corrupting frames 18, 42, ... of this clip. After the fix, the 240-frame hash is **`b5f65290205ab325`**, independently reproduced by FFmpeg over all 1,492,992,000 RGB48 samples. Keep the earlier timings as historical measurements; they are not the final corrected decoder benchmark. See `TEST_ARTGLASS_RTX5060.md` for the new path and validation.

# RTX 5060 Performance Results

Test date: 2026-10-20 (Asia/Shanghai).
Base revision: `f65991c`, pulled from `origin/main`, plus the local autotuner crash fix described below.

## Machine and input

- NVIDIA GeForce RTX 5060, 8 GB (8151 MiB reported by NVIDIA-SMI), driver 591.86.
- Vulkan 1.4.325; selected compute queue family 2; device-local heap reported by the decoder: 7895 MiB.
- Intel Core i5-13400F, 10 cores / 16 logical threads.
- Windows desktop session with other GPU clients present; GPU clocks were not locked.
- Input: `test_j2k_rgb12_1080p24_3min.mxf`, 1920x1080 RGB12, reversible 5/3, first 240 frames, discard 0.
- Input size: 436,280,070 bytes; SHA-256: `734a38fa63f72e348163fe10250ab90eec29b73fc20a136ddf2b9682ebe4cbfd`.
- Source copied from `D:\Coding\cu-prores\test_assets` into the benchmark output directory.
- FFmpeg: `D:\Coding\cu-prores\vendor\ffmpeg-8.1-full_build-shared\bin\ffmpeg.exe`.

All 15 shaders were compiled with `scripts\compile-shaders.ps1 -VulkanSdk D:\VulkanSDK`
and independently checked with `spirv-val.exe`. The executable was built with `cargo build --release`.
Every successful run confirmed RTX 5060, `tier1_packed.comp.spv`, `idwt fused 2d`, and
`readback transfer queue`.

## Correctness and autotuner fix

Both pure GPU and fully automatic mode produced `frames_hash 47d938a71bb308e5` over
240 frames with `VK_J2K_HASH_FRAMES=1` on the final release binary.
Hash runs are excluded from throughput statistics: hashing reads every frame and adds
substantial CPU work, which also affects the automatic tuning decisions.

The unmodified upstream automatic hash run crashed at `src/tune.rs:76` with an
out-of-bounds candidate index (index 4, length 4). `BatchSizeTuner::current()` used
`unwrap_or`, which evaluated the candidate lookup even after a final batch had been chosen.
Exhausting all candidates advanced the probe index past the end of the vector.
The local fix uses `unwrap_or_else` so the candidate is accessed only while still probing.
It does not change the tuning policy or shaders.

A regression test that finishes all four candidate probes reproduced the panic before
the fix. After the fix, all 9 `tune::tests` passed. The original failing log is retained as
`target/benchmarks/rtx5060-20261020/hash-auto-before-fix.log`.

## Throughput

Each mode received one unmeasured warm-up run. Six measured pairs followed, alternating
order: GPU/auto, auto/GPU, GPU/auto, auto/GPU, GPU/auto, auto/GPU.
Each run used 240 frames with hashing and GPU timestamps disabled.

| Mode | Median fps | Mean fps | Range fps | Median ms/frame |
| --- | ---: | ---: | ---: | ---: |
| Pure GPU (`VK_J2K_BATCH=8 VK_J2K_CPU_CBLKS=0`) | 248.91 | 247.60 | 239.25–252.22 | 4.018 |
| Fully automatic (both variables absent) | 238.20 | 238.81 | 234.56–244.79 | 4.199 |

These are `loop_avg_ms_per_frame` results, including tuning and pipeline drain, but excluding
FFmpeg demux, frame loading/planning, pipeline creation, and writing the last frame.
The profiler performs GPU-to-host transfers for decoded frames, but with hashing disabled
only copies the last frame into an owned CPU output array. This is not an end-to-end
player or encoder measurement.

| Pair | GPU fps | Auto fps | Auto batch | Auto final CPU codeblocks/frame |
| --- | ---: | ---: | ---: | ---: |
| 1 | 252.22 | 236.91 | 4 | 43 |
| 2 | 248.92 | 238.29 | 8 | 60 |
| 3 | 248.89 | 244.79 | 8 | 58 |
| 4 | 250.52 | 240.19 | 8 | 57 |
| 5 | 245.82 | 238.10 | 8 | 60 |
| 6 | 239.25 | 234.56 | 8 | 59 |

For this 240-frame workload, automatic mode's mean whole-loop throughput was 3.6% below
fixed batch 8. The mean reported steady-state throughput was 248.21 fps for pure GPU and
251.68 fps for auto (+1.4%), using different steady-state frame counts; this small difference
does not establish a sustained throughput advantage. A longer run would be needed to
evaluate how tuning amortizes over long clips. No manual settings were used for auto mode.

For context, `TEST_RTX2060.md` reports 122.5–128.9 fps pure GPU and 117.8–139.6 fps automatic
on the same clip and frame count. The 5060 pure-GPU median is approximately twice that
recorded 2060 throughput, but CPU, driver, desktop load, and test session also differ.

## GPU stage timing

Three additional interleaved pairs used `VK_J2K_GPU_TIMING=1`, with hashing disabled.
The table gives the median of the three reported stage averages, in ms/frame.
These runs are separate from the throughput table above.

| Stage | Pure GPU | Automatic |
| --- | ---: | ---: |
| Upload | 0.014 | 0.077 |
| Tier-1 | 3.515 | 2.973 |
| Scatter | 0.084 | 0.084 |
| IDWT | 0.249 | 0.249 |
| Color + pack | 0.090 | 0.090 |
| Readback copy | 0.873 | 0.887 |

Tier-1 remains the dominant GPU stage. Readback runs on the transfer queue and overlaps
compute, so stage times must not be summed to infer whole-loop latency. Automatic timing
runs settled on batch 8 with 47, 58, and 54 CPU codeblocks/frame.

## Reproduction and artifacts

Run from the repository root after compiling shaders and the release binary:

```powershell
& .\target\benchmarks\rtx5060-20261020\run-bench.ps1
```

The local script checks device/shader paths, validates hashes, warms up both modes, runs
six throughput pairs and three timing pairs, and saves each run's output separately.
The copied input, output frames, demuxed frames, `vk-info.log`, `results.csv`, runner, and
all raw logs are under `target/benchmarks/rtx5060-20261020/` (git-ignored).

The source fix and this report are local working-tree changes; no commit or push was made.
