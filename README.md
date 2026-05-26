# vk-j2k-dec

[中文版](README_ZH.md)

A JPEG 2000 codestream decoder with Vulkan compute acceleration, written in Rust.

The decoder targets common 3-component codestreams found in DCP, broadcast, and archival workflows. Tier-1 arithmetic decoding, coefficient scatter, inverse DWT, color transform, and final packing all run on GPU via Vulkan compute shaders.

## Performance

Tested on NVIDIA GeForce RTX 5060 (8 GB GDDR7, Blackwell, compute capability 12.0, driver 577.00).

Reversible 5/3, 1920x1080 RGB 12-bit, 64x64 codeblocks, single tile (100 iterations):

| Path | FPS | ms/frame | Notes |
|------|-----|----------|-------|
| GPU Tier-1 real MXF (20 frames) | 69.0 | 14.5 | Real container, warm average |
| GPU Tier-1 pipeline (depth 4) | 66.5 | 15.0 | Async sliding-window |
| GPU Tier-1 sync | 63.0 | 15.9 | Single-frame fused dispatch |
| CPU only | 7.4 | 134.5 | Reference path |

Irreversible 9/7, 1920x1080 RGB 16-bit, 16x16 codeblocks, 40 tiles (20 iterations):

| Path | FPS | ms/frame | Notes |
|------|-----|----------|-------|
| GPU full pipeline (Tier-1 + IDWT 9/7 + color) | 70.5 | 14.2 | Fused dispatch, all tiles |
| GPU Tier-1 adaptive | 86.8 | 11.5 | Thread-per-codeblock shader |
| CPU only | 2.95 | 338.6 | Reference path |

GPU vs CPU speedup: 9.3x (5/3 end-to-end), 24x (9/7 end-to-end), 20-31x (Tier-1 only).

The GPU path is bit-exact against the CPU reference. The 5/3 path is also bit-exact against FFmpeg.

Reference: nvjpeg2000 (CUDA) on the same GPU and content achieves ~112 fps end-to-end (including nvenc H.264 encode). Pure J2K decode throughput is estimated at 130-150+ fps.

### Time budget breakdown

Reversible 5/3, single tile, fused GPU dispatch (15.9 ms total):

```
Tier-1 MQ decode       ~14.8 ms   (93%)  ← primary bottleneck
Scatter                 ~0.2 ms   ( 1%)
Inverse DWT 5/3         ~3.6 ms   ( 5%)  (dispatch only; alloc overhead ~5.2 ms amortized in fused path)
Color + RGB48 pack      ~0.4 ms   ( 1%)
D2H copy (staging)      ~1.2 ms         (overlapped in fused command buffer)
Host readback (map)     ~1.9 ms         (overlapped in fused command buffer)
```

Irreversible 9/7, 40 tiles, fused GPU dispatch (14.2 ms total):

```
Tier-1 MQ decode       ~11.5 ms   (81%)  ← primary bottleneck
Scatter                 ~0.1 ms/tile
Inverse DWT 9/7         ~2.0 ms   (14%)  (summed across 40 tiles)
Color + RGB48 pack      ~0.4 ms   ( 3%)
D2H copy + readback     ~2.0 ms         (overlapped)
```

The Tier-1 MQ arithmetic decoder is the dominant cost in both paths. The MQ state machine is inherently serial per codeblock — each decoded bit depends on the previous state. Scatter, DWT, and color are already fast. The primary optimization lever is MQ decode throughput.

Codeblock work distribution (5/3, 879 codeblocks):

```
byte_len:   min 3, p50 52, p90 293, p99 708, max 2386
pass_count: min 1, p50 31, p90 34, p99 37, max 37
Top 3 codeblocks (LL subbands) have 10-20x more work than median → long-tail effect.
```

### What limits further speedup

1. MQ serial dependency: each bit decode depends on the previous A/C register state.
2. Long-tail codeblocks: the 3 LL codeblocks dominate single-frame dispatch time.
3. Vulkan vs CUDA: SPIR-V→PTX compilation is less optimized than native CUDA for serial-heavy kernels.
4. nvjpeg2000 uses only 512 bytes shared memory (vs our global state approach), achieving higher occupancy through register-cached coefficient state.

## Supported Profile

- Raw `.j2k` / `.j2c` codestreams
- 3 components, 8/10/12/16-bit samples
- Reversible 5/3 and irreversible 9/7 wavelet transforms
- LRCP progression order
- RCT and ICT color transforms
- Single and multi-tile layouts

## Requirements

- Rust 2024 edition
- Vulkan 1.1+ capable GPU with compute queue
- Vulkan SDK (for shader compilation)

## Build

```bash
cargo build --release
```

Compile shaders (requires `glslangValidator` from Vulkan SDK):

```powershell
.\scripts\compile-shaders.ps1
```

## Usage

### Decode to raw RGB48

```bash
vk-j2k-dec decode-rgb48 <input.j2k> <output.rgb48le>
vk-j2k-dec decode-rgb48-lowres <input.j2k> <discard-levels> <output.rgb48le>
vk-j2k-dec decode-rgb48-gpu <input.j2k> <discard-levels> <output.rgb48le>
```

### Inspect codestream

```bash
vk-j2k-dec dump <input.j2k>
vk-j2k-dec scan <container-or-bytestream> [max_hits]
vk-j2k-dec scan-summary <container-or-bytestream>
```

### Extract codestream from container

```bash
vk-j2k-dec extract <container-or-bytestream> <index> <output.j2k>
```

### Benchmark

```bash
# Single-frame GPU Tier-1 bench
vk-j2k-dec tier1-gpu-bench <input.j2k> [iterations]

# Async pipeline bench
vk-j2k-dec tier1-gpu-pipeline-bench <input.j2k> [iterations] [pipeline-depth]

# Batch occupancy bench
vk-j2k-dec tier1-gpu-batch-bench <input.j2k> [iterations] [batch-frames]

# Profile with stage breakdown
vk-j2k-dec profile-rgb48-bench <input.j2k> <discard-levels> <cpu|gpu-dwt|gpu> <iterations> <output.rgb48le>

# Real MXF container profiling (requires ffmpeg)
vk-j2k-dec profile-mxf-ffmpeg <input.mxf> <frames> <discard-levels> <cpu|gpu-dwt|gpu|gpu-tier1> <output.rgb48le> [ffmpeg] [work-dir] [idwt.spv] [color.spv]
```

### Vulkan diagnostics

```bash
vk-j2k-dec vk-info
vk-j2k-dec vk-smoke
vk-j2k-dec vk-color-smoke
vk-j2k-dec vk-idwt53-smoke
vk-j2k-dec vk-idwt97-smoke
vk-j2k-dec vk-irreversible-smoke
```

### Diff / validation tools

```bash
vk-j2k-dec diff-rgb48 <actual.rgb48le> <reference.rgb48le>
vk-j2k-dec diff-component-plane <actual.i32le> <reference.i32le>
```

## Architecture

```
CPU: codestream parse → tile layout → packet headers → codeblock metadata
GPU: Tier-1 MQ decode → coefficient scatter → inverse DWT → color transform → RGB48 pack
```

The fused GPU path records Tier-1, scatter, IDWT, and color packing into a single reusable command buffer. Device-local buffers are kept across stages to avoid unnecessary transfers.

## GPU Pipeline

Compute shaders in `shaders/`:

| Shader | Stage |
|--------|-------|
| `tier1_cleanup_smoke.comp.glsl` | Tier-1 MQ arithmetic decode |
| `tier1_scatter_smoke.comp.glsl` | Codeblock → subband plane scatter |
| `idwt53_offset_1d.comp.glsl` | Inverse 5/3 DWT (reversible) |
| `idwt97_offset_1d.comp.glsl` | Inverse 9/7 DWT (irreversible) |
| `color_convert_i32.comp.glsl` | RCT/ICT + level shift + RGB48 pack |

## Validation

Run the full GPU smoke test suite:

```powershell
.\scripts\gpu-smoke.ps1
```

The decoder is bit-exact against FFmpeg for the local reversible 5/3 test corpus at all resolution levels.

## License

TBD
