# GPU pipeline notes

The first production Vulkan path keeps Tier-1 on CPU and sends decoded
coefficient planes to compute shaders. A separate Tier-1 GPU experiment now
flattens packet/body work into GPU-readable jobs so that the next boundary can
move earlier. The current production boundary is:

1. CPU parses codestream, tile layout, packets, codeblock segments, and Tier-1.
2. CPU writes coefficient planes plus per-sample subband step indices.
3. `dequant.comp.glsl` converts integer coefficients to float working samples.
4. `idwt53_1d.comp.glsl` performs inverse 5/3 lifting over rows/columns.
   `idwt97_1d.comp.glsl` covers the irreversible 9/7 float lifting smoke path.
5. Inverse DWT shaders reconstruct component planes in place.
6. `color_convert.comp.glsl` applies no transform, RCT, or ICT and writes RGB16.

The shader contracts are intentionally simple while the Rust Vulkan host grows
toward a reusable pipeline. The CPU side should precompute values that are
awkward or profile-specific:

- per-subband dequant scale from QCD/QCC; QCD scalar-expounded and
  scalar-derived scales are now available on the CPU staging path, while QCC
  still needs wiring;
- level-shift value from component precision;
- output clamp range;
- per-sample subband-to-step lookup, or a cheaper per-dispatch variant later.

For the current RGB12 1080p sample, codeblock style is `0x00`, transform is
reversible 5/3, MCT is disabled, and QCD has 16 no-quantization step entries. That
means Tier-1 remains all MQ-coded on CPU, dequant scale is effectively `1.0`, and
the first GPU work worth wiring is inverse 5/3 DWT plus final RGB16 packing.

## Real-frame GPU DWT path

`decode-rgb48-gpu-dwt` decodes packets and Tier-1 on CPU, uploads the decoded
subband coefficient planes, runs Vulkan inverse 5/3 over each component, then
packs RGB48 on CPU:

```powershell
cargo run -- decode-rgb48-gpu-dwt target\samples\test_j2k_rgb12_1080p24_3min_frame0.j2k 0 target\samples\ours_frame0_rgb48le_gpu_dwt.raw
```

The command accepts the same discard level used by the low-resolution CPU path:

```powershell
cargo run -- decode-rgb48-gpu-dwt target\samples\test_j2k_rgb12_1080p24_3min_frame0.j2k 5 target\samples\ours_frame0_rgb48le_lowres5_gpu_dwt.raw
```

Current validation on the RGB12 sample:

```text
cargo run -- diff-rgb48 target\samples\ours_frame0_rgb48le_gpu_dwt.raw target\samples\ours_frame0_rgb48le.raw
diff-rgb48: samples 6220800, differing 0, max_abs 0 at sample 0, mean_abs 0.000000000000
```

`decode-rgb48-gpu` extends that path by running final color conversion,
level-shift, clamp, and native sample packing through `color_convert.comp.spv`:

```powershell
cargo run -- decode-rgb48-gpu target\samples\test_j2k_rgb12_1080p24_3min_frame0.j2k 0 target\samples\ours_frame0_rgb48le_gpu.raw
cargo run -- diff-rgb48 target\samples\ours_frame0_rgb48le_gpu.raw target\samples\ours_frame0_rgb48le.raw
```

Current validation on the RGB12 sample:

```text
diff-rgb48: samples 6220800, differing 0, max_abs 0 at sample 0, mean_abs 0.000000000000
```

The reversible RGB48 path now keeps DWT and color output device-local through
the final color shader, then copies the packed RGB48 buffer once into
HOST_CACHED staging for CPU readback. On the local RTX 5060 sample this split is
roughly 0.5 ms shader, 1.3 ms device-to-host copy, and 2.1 ms CPU map/read for
the final 12.4 MB RGB48 buffer after DWT outputs are kept device-local through
color conversion. With the high-level path using parallel Tier-1 codeblock
decode, batched DWT subband uploads, and batched DWT dispatch command buffers,
the same frame currently averages about 86 ms/frame through the persistent
CPU-Tier1/GPU-DWT RGB48 benchmark. The device-local GPU Tier-1 fast path is now
wired into the real-MXF profiler and averages about 36.1 ms/frame on 10 real
frames, compared with about 149.7 ms/frame for the local CPU path on the same
run. The next GPU milestones are:

- add persistent buffer pools for repeated full-frame decode;
- overlap CPU Tier-1, coefficient uploads, and GPU reconstruction;
- reduce the remaining GPU Tier-1 fused dispatch cost;
- add multi-frame in-flight decode to keep more codeblocks resident;
- wire irreversible 9/7 DWT into real-frame decode once a representative sample is available.

## Tier-1 GPU handoff experiment

`tier1-gpu-plan` builds a flat GPU-facing Tier-1 job list from the parsed packet
summary. The CPU still parses Tier-2 and computes segment/pass boundaries, then
emits:

- one `GpuTier1CodeblockRecord` per included codeblock;
- one packed u32 word stream containing all compressed packet-body bytes;
- contiguous coefficient offsets for the eventual GPU output buffer.

`tier1_plan_smoke.comp.glsl` currently validates that Vulkan can read both the
records and the compressed byte stream by checking byte ranges and per-codeblock
byte checksums on GPU. On the local RGB12 frame this covers 879 records, 97,023
compressed bytes, and 25,206 MQ passes. This is not a Tier-1 decoder yet, but it
locks down the SSBO shape needed for the real MQ/codeblock shader.

`tier1_mq_smoke.comp.glsl` ports the MQ arithmetic decoder core to GLSL and runs
it against real codeblock bytes. The CLI compares decoded bits, byte position,
`A/C/CT` registers, and context state against the CPU `MqDecoder`.

`tier1_cleanup_smoke.comp.glsl` now runs every MQ-coded pass for every real
codeblock on GPU. It maintains per-coefficient significance/sign/refinement
state plus cached neighbor counts in workgroup memory, handles cleanup
run-length coding, significance propagation, magnitude refinement, context
selection, and exact reversible coefficient reconstruction. The current smoke
writes a flat GPU coefficient buffer and compares all 3,177,152 coefficients,
MQ byte positions, and pass summary counters against CPU Tier-1.

The hot Tier-1 bench command builds the Vulkan pipeline and buffers once, repeats
the same frame in one command buffer, and reads back once for bit-exact
validation:

```powershell
cargo run --release -- tier1-gpu-bench target\samples\test_j2k_rgb12_1080p24_3min_frame0.j2k 10
```

Current local RTX 5060 result after device-local buffers, cached neighbor state,
packing neighbor counts into the coefficient-state word, reading packed state
once per sample in the hot scan loops, direct interior-sample neighbor updates,
division-free Tier-1 pass scheduling, visit-generation tracking, skipping
cleanup-only visit writes, one fused pre-recorded Tier-1+scatter+IDWT+color
command buffer, heavy-codeblock-first scheduling by descending `byte_len *
pass_count`, and the device-local RGB48 fast-path bench:

```text
vulkan full RGB48 hot bench: device NVIDIA GeForce RTX 5060, queue 2, setup_once 8.939 ms, 1580.877 ms total / 31.618 ms avg, 31.63 fps
  avg breakdown: fused_gpu_dispatch 27.823 ms, tier1_cleanup_pass 0.000 ms, scatter 0.000 ms, idwt_dispatch 0.000 ms, idwt_alloc 0.000 ms, color_shader 0.000 ms, color_copy 1.265 ms, color_readback 2.094 ms, summary_readback_once 0.518 ms
cpu full RGB48 reference: 6955.209 ms total / 139.104 ms avg, 7.19 fps; GPU speedup 4.40x
vulkan tier1 bench: device NVIDIA GeForce RTX 5060, queue 2, setup 1.771 ms, hot_dispatch 1556.169 ms total / 31.123 ms avg, readback 4.484 ms
gpu cleanup+scatter vs cpu reference: 8.72x hot speedup
cpu tier1 reference: 13754.919 ms total / 275.098 ms avg, hot GPU speedup 8.84x, first nonzero 2040, checksum 0xfffe2412
```

In these logs, `tier1_cleanup_pass` is the JPEG 2000 Tier-1 cleanup coding pass,
not resource cleanup/destruction. The fused RGB48 bench records cleanup,
scatter, all IDWT dispatches, and color packing into one reusable command
buffer, so `fused_gpu_dispatch` is the measured GPU compute section and the
individual stage slots are left at zero in that mode.

A CUDA ProRes-style multi-lane-per-block variant was tested against this shader
shape. It remained bit-exact, but the larger per-workgroup state reduced
occupancy and lost to the packed single-codeblock workgroup, so the current
fast path keeps one codeblock per workgroup while squeezing shared memory first.
Packing MQ context state and MPS into a single packed state array was successfully implemented, reducing thread-local memory footprint and lookup count. Combined with inner loop unrolling and zero-neighbor bypass, it yielded a combined speedup.

`tier1-gpu-batch-bench` replicates the same frame into a larger Tier-1 GPU
dispatch to test whether the single-frame path is occupancy-starved. On the
local frame, the codeblock distribution is uneven enough to create a tail:

```text
cblk byte_len stats: min 3, p50 52, p90 293, p99 708, max 2386, top1 2.5% bytes, top10 12.6% bytes
cblk pass_count stats: min 1, p50 31, p90 34, p99 37, max 37, total 25206
```

Batching confirms that more in-flight codeblocks help, but only up to a point:

| batch frames | cblk/dispatch | Tier-1 ms/frame | aggregate fps |
| ---: | ---: | ---: | ---: |
| 1 | 879 | 30.912 | 32.35 |
| 4 | 3,516 | 29.405 | 34.01 |
| 8 | 7,032 | 28.151 | 35.52 |
| 16 | 14,064 | 28.034 | 35.67 |

This points to a mixed bottleneck: single-frame work does not fully occupy the
GPU, while the remaining flat spot is the serial MQ path inside each codeblock.
The next practical scheduling step is multi-frame in-flight decode; the next
hard shader step is a lower-state/cooperative codeblock kernel rather than more
simple codeblocks per workgroup.
The current 32-lane workgroup is also the best local zero-initialization shape
so far: 16-lane and 64-lane variants were both bit-exact but slower on the
batch=1 Tier-1 bench.

To address the CPU-GPU transfer bottlenecks and hide the long-tail synchronization delays, we implemented a sliding-window asynchronous pipeline (`tier1-gpu-pipeline-bench`) using multiple fences to process frames concurrently.

For the RTX 5060, using 50 iterations and a pipeline depth of 4:
- **Pipeline Average Frame Time**: `27.463 ms` (`36.41 fps`)
- **Throughput Increase**: **~15.1%** compared to the synchronous baseline of `31.63 fps`.

This demonstrates that concurrent command recording and submission, coupled with sliding-window fence checks, effectively overlaps host-to-device transfers, shader compute execution, and device-to-host readbacks.


`tier1_scatter_smoke.comp.glsl` is the first handoff shader after GPU Tier-1. It
maps the flat per-codeblock coefficient buffer into concatenated component
subband planes using precomputed codeblock rectangles. On the local frame it
scatters 879 codeblocks into 6,220,800 plane coefficients with sub-millisecond
shader dispatch; the current smoke still uploads/zeros/readbacks for validation, but
the dispatch itself is small enough that the next useful step is keeping those
plane buffers device-local and feeding them directly into DWT.

The cleanup+scatter bench keeps the flat Tier-1 coefficient buffer device-local
between the two shaders and records both dispatches into one command buffer per
iteration. The scatter step is effectively hidden beside MQ decode time on the
local frame, while avoiding the separate flat-coefficient upload that the
standalone scatter smoke uses for validation.

`idwt53_offset_1d.comp.glsl` lets the DWT read LL/HL/LH/HH bands from a
concatenated subband-plane buffer via element offsets. The
cleanup+scatter+IDWT/RGB48 smoke now keeps Tier-1 output and scattered subband
planes device-local through inverse 5/3 and integer color packing, then reads
back the packed RGB48 frame for validation. The hot full-chain bench avoids CPU
coefficient placement and intermediate coefficient-plane transfers; the
remaining dominant cost is the one-invocation-per-codeblock Tier-1 cleanup pass.
The bench now preallocates scattered-IDWT scratch/output buffers, pre-records
one fused cleanup/scatter/IDWT/color command buffer, and reuses the RGB48
device/staging buffers across iterations.

This fused path is now exposed by `profile-mxf-ffmpeg ... gpu-tier1` for real
container-backed frames. The profiler builds a union codeblock template across
the measured frame set, pads absent codeblocks with zero-pass records, keeps the
scatter/DWT/color command buffer reusable, writes dynamic Tier-1 records plus
compressed words into host-visible storage buffers each frame, and schedules
heavier codeblocks first within each frame.

## Real MXF profiling

`profile-mxf-ffmpeg` uses the pinned FFmpeg build as the demuxer, copies JPEG
2000 frames from MXF into a temporary `.j2k` sequence, then reuses the existing
CPU/GPU RGB48 profile path. This keeps the measurement tied to real container
input without introducing libavformat FFI yet.

On `D:\Coding\cu-prores\test_assets\test_j2k_rgb12_1080p24_3min.mxf`, the local
10-frame run reports:

```text
profile-mxf-ffmpeg gpu-tier1: ffmpeg_demux 46.472 ms total, avg_skip_first_frame total 33.142 ms / 30.17 fps
  avg_skip_first breakdown: upload 0.014 ms, fused_dispatch 31.145 ms, color_copy 0.000 ms, color_readback 1.982 ms
profile-mxf-ffmpeg gpu: ffmpeg_demux 46.601 ms total, avg_skip_first_frame decode_total 84.241 ms / 11.87 fps
profile-mxf-ffmpeg cpu: ffmpeg_demux 44.405 ms total, avg_skip_first_frame decode_total 149.672 ms / 6.68 fps
gpu-tier1 vs cpu diff-rgb48: differing 0, max_abs 0
ffmpeg native decode-to-null: rtime 0.708s for 10 frames, about 70.8 ms/frame wall time
```

The demux cost is small compared with decode. The fused GPU Tier-1 path is now
about 4.5x faster than the local CPU path on this 10-frame real-MXF run and
bit-exact against the CPU output. The RGB48 device-to-staging copy is recorded
at the end of the fused command buffer, so `color_copy` is included in
`fused_dispatch`; shader-side RGB48 high-bit expansion and uninitialized
readback allocation keep CPU readback near 1.9 ms/frame. The remaining wall
time is dominated by the Tier-1 fused dispatch.

## Shader build

Compile and validate the current compute shaders with:

```powershell
.\scripts\compile-shaders.ps1
```

The script uses `$env:VULKAN_SDK` when it is set, otherwise it falls back to
`D:\VulkanSDK`.

## Local Vulkan device

The current Windows development machine reports:

```text
device 0: NVIDIA GeForce RTX 5060 (discrete GPU, Vulkan 1.4.303)
queue family 2: compute true, graphics false, transfer true
```

`vk-j2k-dec vk-info` selects queue family `2` for compute work.

`vk-j2k-dec vk-smoke` currently creates a logical device, host-visible storage
buffers, descriptor set, compute pipeline, command pool, command buffer, and fence,
then dispatches `dequant.comp.spv`. The smoke test validates GPU output against a
CPU-computed reference.

`vk-j2k-dec vk-color-smoke` runs the same host path against
`color_convert.comp.spv` and validates RGB16 packing for no-transform RGB, RCT,
ICT, and float component-plane paths.

`vk-j2k-dec vk-idwt53-smoke` runs `idwt53_1d.comp.spv` over multiple independent
lines and validates the output against the CPU `inverse_5_3_1d` reference. The
same smoke also exercises strided column access, so vertical IDWT can write
directly into row-major component planes without a transpose-only staging pass.
It also chains three dispatches into a complete one-level 2D inverse 5/3 smoke:
vertical low columns, vertical high columns, then horizontal rows, validated
against the CPU `inverse_5_3_level` reference.

`vk-j2k-dec vk-idwt97-smoke` runs the same row, column, and one-level 2D smoke
shape against `idwt97_1d.comp.spv`, using float buffers and validating against
the CPU `inverse_9_7_1d` and `inverse_9_7_level` references.

The Vulkan backend now also exposes a component-level 9/7 reconstruction helper
that mirrors the 5/3 backend shape and returns float component planes.
`ComponentFloatSubbandPlanes` models dequantized float subbands before 9/7
lifting. CPU and GPU color packing both have float component-plane helpers for
the no-transform and ICT cases. The high-level CPU decode path can now route
irreversible 9/7 codestreams through QCD scalar-expounded or scalar-derived
dequantization, CPU 9/7 reconstruction, and float RGB48 packing. Real
irreversible-frame validation still needs a representative sample and QCC
overrides.

`vk-j2k-dec vk-irreversible-smoke` chains the synthetic irreversible pieces:
synthetically scaled float subbands feed three component-level 9/7
reconstructions, then GPU ICT/color packing, and the result is compared against
CPU 9/7 plus CPU float RGB48 packing.

`vk-j2k-dec dump <input.j2k>` also prints a DWT dispatch plan for supported
5/3 and 9/7 codestreams. The current RGB12 5/3 sample produces 45 planned IDWT stages:
vertical low columns, vertical high columns, then horizontal rows for each
component and reconstruction level. This follows the CPU reference path: inverse
vertical reconstruction of `LL/LH` and `HL/HH`, followed by inverse horizontal
row reconstruction.

Run the whole local GPU check suite with:

```powershell
.\scripts\gpu-smoke.ps1
```
