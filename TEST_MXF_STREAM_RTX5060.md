# ArtGlass: sequential MXF to NUL, Vulkan vs nvJPEG2000

2026-10-20. RTX 5060 8 GB, i5-13400F, NVIDIA driver 591.86. Both logs confirm the same RTX 5060. This test replaces the earlier single-frame nvJ2K timing as the relevant comparison for this file.

Source: `D:\Coding\dv_preproc_wgpu\sample\ArtGlass_Mezz_3840x2160.mxf`, 2,925,273,799 bytes. Each measured run reads the entire MXF in file order and decodes all **1,500 distinct picture frames** exactly once. Picture payloads total 2,924,461,058 bytes.

## Measurement

Both backends use the same bounded MXF KLV reader and the same RGB48LE NUL sink. No extracted J2K files or movie-wide frame/packet preloading are involved. Metadata/audio KLV values are consumed and skipped sequentially. The output is unsigned RGB12 left-aligned in packed RGB48LE; both backends perform device-to-host readback and write **74,649,600,000 bytes** to NUL. There is no H.264 stage.

Both use batch 2 and two reusable slots (four in-flight frame buffers). Vulkan uses the packed Tier-1 shader, 9/7 IDWT and ICT with CPU codeblock offload disabled. nvJPEG2000 uses four reusable decoder states/CUDA streams; a CUDA kernel packs its planar output into the identical interleaved RGB48 format before asynchronous pinned-host readback. CPU packet parsing and API submission time are included for each backend.

Each backend first warms up on 120 consecutive source frames. The three measured rounds alternate order: Vulkan/nvJ2K, nvJ2K/Vulkan, Vulkan/nvJ2K. The OS file cache is warm; this is not a forced cold-disk test. Hash calculation is disabled during performance measurement. Each run is a new process and opens the MXF again.

The principal result uses external whole-process wall time: file opening, reading, decoder setup, per-frame parsing, GPU decode, RGB packing, readback, NUL writes, and teardown are all included. The in-program total has the same scope except process startup/shutdown overhead; stream-loop time excludes setup and teardown but still includes all per-frame reading/parsing/output work.

| Round | Vulkan process fps | nvJ2K process fps |
|---|---:|---:|
| 1 | 53.452 | 54.404 |
| 2 | 53.366 | 53.827 |
| 3 | 53.518 | 54.163 |

| Median | Vulkan | nvJ2K |
|---|---:|---:|
| Whole process, fps | 53.452 | 54.163 |
| In-program end-to-end, fps | 53.512 | 54.210 |
| Stream loop, fps | 54.194 | 54.522 |
| Whole process, seconds | 28.063 | 27.694 |

**nvJPEG2000 throughput is 1.33% higher on the whole-process median.** The two implementations are close for this specific file and host-readback workload. This does not establish equal throughput for other JPEG 2000 profiles, GPUs, GPU-only sinks or encoding pipelines. The older 112 fps result is a different benchmark and is not substituted into this comparison.

## Validation and implementation

The first eight consecutive frames were written to raw files through the same streaming path and compared pixel-by-pixel. Every frame, including the formerly problematic eighth frame, differs from nvJPEG2000 by at most **one native 12-bit level** (PSNR approximately 96.94–97.04 dB). This verifies reuse across both slots. The earlier full-clip validation also sampled the last frame; see `TEST_ARTGLASS_RTX5060.md`.

Streaming exposed that codeblocks absent in frame 1 can appear in later frames. The persistent Vulkan template is now built from the complete subband geometry rather than the nonzero blocks in the first frame. This avoids a whole-movie pre-scan and ensures absent blocks are zero-filled when slots are reused. The benchmark rejects frame geometry/quantization changes instead of silently using incompatible state.

The MXF reader uses KLV lengths, so marker-like bytes inside compressed data do not create extra frames. It supports one frame-wrapped JPEG 2000 picture track, with a 64 MiB per-picture bound; this is not a general MXF container implementation. Source buffering remains bounded as movie duration grows.

Validation: `cargo test --lib` 102 passed, 3 existing ignored tests. The new reader tests cover metadata skipping, marker lookalikes inside payloads and truncated KLV/picture data. The release build and `cargo check --all-targets` pass. Both complete-file backends ran successfully in every measured round.

## Reproduce

From the repository root in PowerShell:

```powershell
.\scripts\compile-shaders.ps1 -VulkanSdk D:\VulkanSDK
cargo build --release
.\scripts\build-nvj2k-bridge.ps1
.\scripts\benchmark-mxf-stream.ps1 `
  -InputMxf D:\Coding\dv_preproc_wgpu\sample\ArtGlass_Mezz_3840x2160.mxf `
  -OutputDir target\benchmarks\artglass-stream-repeat `
  -Frames 0 -Batch 2 -Rounds 3
```

`-Frames 0` means EOF. Choose a fresh result directory. The optional nvJ2K bridge requires CUDA Toolkit/MSVC and `refer/nvjpeg2k.h`, `nvjpeg2k.lib`, `nvjpeg2k_0.dll`; `-NvJpeg2k` can point to another installation. Only the comparison backend needs this NVIDIA library; the Vulkan decoder does not.

Direct commands (each reads and decodes the full MXF once):

```powershell
target\release\vk-j2k-dec.exe bench-mxf-stream D:\Coding\dv_preproc_wgpu\sample\ArtGlass_Mezz_3840x2160.mxf vulkan 0 2 NUL
target\release\vk-j2k-dec.exe bench-mxf-stream D:\Coding\dv_preproc_wgpu\sample\ArtGlass_Mezz_3840x2160.mxf nvj2k 0 2 NUL
```

For correctness checks, substitute a raw output path and limit the frame count, e.g. `vulkan 8 2 output.rgb48le`. Optional `VK_J2K_HASH_FRAMES=1` adds CPU hashing and must be disabled for performance runs. `VK_J2K_CUDA_DEVICE` selects the CUDA device index (default 0); check both logged device names on a multi-GPU system.

Evidence: `target/benchmarks/artglass-stream/full-runs/results.csv`, per-run logs in that directory, `summary.json`, `validation-first8.json`, and `tests.log` in its parent. The identical output byte/frame counts are checked by the benchmark runner.
