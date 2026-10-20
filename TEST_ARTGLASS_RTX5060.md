> Update: the requested full-file sequential MXF -> RGB48LE -> NUL comparison is now complete. See [TEST_MXF_STREAM_RTX5060.md](TEST_MXF_STREAM_RTX5060.md): Vulkan 53.45 fps, nvJ2K 54.16 fps (three-round whole-process medians, all 1,500 frames). This supersedes the single-frame speed comparison below.

# ArtGlass RGB12 JPEG 2000 / Vulkan + NVENC

2026-10-20, RTX 5060 8 GB, i5-13400F, NVIDIA 591.86. Base commit f65991c plus local changes. Source: `D:\Coding\dv_preproc_wgpu\sample\ArtGlass_Mezz_3840x2160.mxf` (2,925,273,799 bytes).

## Implemented

The limitation was not RGB12 itself: the fast path only handled reversible 5/3 and one tile-part. ArtGlass uses unsigned RGB12, 3840x2160, irreversible 9/7, ICT, CPRL, five decomposition levels, 64x64 codeblocks, scalar-expounded QCD, and three tile-parts in one tile.

- Persistent GPU pipeline now concatenates same-tile packet payloads, parses CPRL, and runs 9/7 + ICT. GPU dequantization is fused with scatter; only one scale per codeblock is stored, with QCC override support.
- Irreversible coefficients use signed Q8 midpoint reconstruction. This preserves fractional refinement and removes the erroneous LL integer bias. The CPU decoder and CPU offload use the same representation. The dispatch path rejects bitplanes outside its staging range.
- Packet-header finalization consumes the stuffed byte after a terminal 0xff. This fixes ArtGlass frame 8 and the periodic corruption in the old 1080p test clip.
- `decode-rgb48-full-gpu` provides a full GPU single-codestream entry point, producing MSB-aligned RGB48LE.
- Optional Windows bridge exports Vulkan dedicated allocations as opaque Win32 handles, matches the CUDA device by UUID, imports the same memory, converts RGB48 to BT.709 limited NV12 on the GPU, and passes CUDA hardware frames to FFmpeg h264_nvenc. Performance runs do not read raw pictures back to the host. Vulkan fences, external queue ownership transfers and a CUDA stream wait per batch synchronize reuse.
- `VK_J2K_DEMUX=scan` explicitly recovers complete JPEG 2000 codestreams when FFmpeg rejects this MXF header. It buffers at most one frame plus a chunk (64 MiB frame limit), validates extracted codestreams and writes numbered J2K files. It does not recover audio or container timestamps. This benchmark bridge uses fixed 24 fps.

## Correctness

Sampled zero-based ArtGlass frames 0, 2, 7, 119, 239 and 1499 compared against local nvJPEG2000: maximum error **1 native 12-bit level**, PSNR **96.69–100.27 dB**. CPU frame 0 also has maximum error 1. Batched and single-frame Vulkan results match exactly for frame 2. Initial uncorrected output was only 57.11 dB (maximum 47 levels).

240-frame rolling u16 FNV-1a hash: **ede627114b4b9c95**. Pure GPU, automatic mode and forced CPU offload (8 codeblocks/frame, batch 2) all match, including verification through imported CUDA memory.

The encoded MP4 passes full FFmpeg decoding: 3840x2160, H.264, 240 frames, 24/1 fps, exactly 10 seconds. H.264 output is lossy 8-bit 4:2:0; the RGB12 comparison above is performed before that conversion.

The corrected old 1080p RGB12 240-frame hash is **b5f65290205ab325**. FFmpeg independently produced exactly the same hash across all 1,492,992,000 samples. The formerly corrupt frame 18 now matches FFmpeg sample-for-sample. The previous 47d938a71bb308e5 value must not remain the correctness oracle.

## ArtGlass 240-frame performance

Three alternating rounds after both modes warmed up; hash/readback verification disabled, NVENC p4, QP 23, no B frames or lookahead, GOP 48. Loop time includes RGB conversion, encoding, encoder flush and MP4 finalization. Application total also includes essence scan/write, frame reads, packet parsing, pipeline and encoder setup; whole-process time includes teardown.

| Mode | Loop fps, three runs | Median loop fps | Median application fps | Median whole-process fps |
|---|---|---:|---:|---:|
| GPU (`BATCH=8`, `CPU_CBLKS=0`) | 42.21, 41.39, 41.31 | 41.39 | 23.37 | 22.79 |
| Automatic | 43.61, 43.95, 43.70 | 43.70 | 23.94 | 23.43 |

The GPU memory/storage-buffer limits reduce requested batch 8 to 3 at 4K. Automatic runs choose batch 2–3 and about 6–7 CPU codeblocks/frame. No manual adaptation to this CPU/GPU is required.

Local nvJPEG2000 repeated single-frame decode was about 16–16.6 ms (60–63 fps), without H.264 encoding or the same batching/demux work. This is a valid measured reference but not a like-for-like end-to-end comparison. The earlier 112 fps nvJ2K+H.264 result is also a real historical result; its input and complete settings must match before using it to rank this 4K path.

## Entire input

All **1,500 complete frames** were decoded and encoded successfully using GPU batch 2. The MP4 was fully decoded again by FFmpeg/ffprobe: 1,500 H.264 frames, 3840x2160, 24/1 fps, 62.5 seconds under the benchmark's fixed 24 fps setting. Last-frame Vulkan vs nvJPEG2000 maximum error is 1 native level (100.27 dB). Output: `target/benchmarks/artglass/artglass-full.mp4`.

Full-clip loop: 32,473.850 ms (**46.19 fps**); application total: 64,109.503 ms (**23.40 fps**). This is a single coverage run across different content, not another sample in the alternating 240-frame performance table. Input preparation still happens before the timed GPU loop and is a substantial part of total time; this implementation is a batch benchmark, not yet a streaming demux/parse pipeline.

## Reproduce (PowerShell, repository root)

```powershell
.\scripts\compile-shaders.ps1 -VulkanSdk D:\VulkanSDK
cargo build --release
.\scripts\build-nvenc-bridge.ps1 `
  -FFmpeg D:\Coding\cu-prores\vendor\ffmpeg-8.1-full_build-shared `
  -VulkanSdk D:\VulkanSDK
.\scripts\benchmark-nvenc.ps1 `
  -InputMxf D:\Coding\dv_preproc_wgpu\sample\ArtGlass_Mezz_3840x2160.mxf `
  -FFmpeg D:\Coding\cu-prores\vendor\ffmpeg-8.1-full_build-shared\bin\ffmpeg.exe `
  -OutputDir target\benchmarks\artglass-repeat `
  -Frames 240 -Rounds 3 -Demux scan -ExpectedHash ede627114b4b9c95
```

For single-frame validation:

```powershell
target\release\vk-j2k-dec.exe decode-rgb48-full-gpu `
  target\benchmarks\artglass\nvenc\demux\frame_000008.j2k `
  target\benchmarks\artglass\frame8.rgb48le
```

Use a fresh output directory per benchmark invocation (CSV appends). Required runtime DLLs/PTX are in `target/nvenc`; shaders and source media are not tracked by Git. The bridge build needs CUDA Toolkit, Vulkan SDK, MSVC and an FFmpeg shared development distribution. Standard 1080p verification can use `-ExpectedHash b5f65290205ab325` without `-Demux scan`.

## Scope and evidence

Persistent fast path currently requires one tile, zero origin, three equally precise unsigned unsubsampled components, LRCP/CPRL and supported codeblock styles. This does not claim complete JPEG 2000/HTJ2K/MXF support. The existing diagnostic `tier1-gpu-*-bench` commands are not the new corrected production benchmark entry point; use `profile-mxf-ffmpeg ... gpu-tier1` or the script above. The standalone API has older multi-tile/QCC limitations; validation here targets this single-tile, QCD-only clip.

Raw logs and generated files: `target/benchmarks/artglass/` (`pixel-validation.json`, `ffmpeg1080p-reference-hash.txt`, `cpu8-hash.log`, `encoded-probe.json`, `nvenc/results.csv`). Regression tests cover Q8 fractional significance/refinement and packet-header stuffing. Shader builds include SPIR-V validation.

## Corrected 1080p NVENC retest

Three alternating rounds after warmup, same p4/QP23 encoder settings and 240 frames. Both GPU and automatic CUDA-import verification match the independently established `b5f65290205ab325` hash.

| Mode | Median loop fps | Median application fps | Median whole-process fps |
|---|---:|---:|---:|
| gpu | 193.95 | 118.03 | 108.44 |
| auto | 177.78 | 115.86 | 108.21 |

Raw results: `target/benchmarks/rtx5060-nvenc-corrected-20261020/results.csv`. These replace the earlier NVENC measurements made before the packet-header fix. The hot loop must not be compared directly with the historical 112 fps end-to-end nvJ2K result; total/process columns are the relevant timing scope, and preset/input equivalence still matters.

Final validation: `cargo test --lib` **100 passed, 3 ignored**; `cargo check --all-targets`, release build, all shader compilation/SPIR-V validation, and `git diff --check` passed. Ignored tests were already ignored before these changes.
