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
the same frame currently averages about 86 ms/frame through the
persistent GPU RGB48 benchmark, compared with about 119 ms/frame for the local
FFmpeg native JPEG2000 RGB48 loop. The next GPU milestones are:

- add persistent buffer pools for repeated full-frame decode;
- overlap CPU Tier-1, coefficient uploads, and GPU reconstruction;
- replace CPU Tier-1 with a real GPU Tier-1 decoder;
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
visit-generation tracking, and skipping cleanup-only visit writes:

```text
vulkan tier1 cleanup+scatter bench: setup 2.766 ms, hot_dispatch 611.329 ms total / 61.133 ms avg, readback 7.229 ms
vulkan tier1 bench: setup 1.969 ms, hot_dispatch 615.803 ms total / 61.580 ms avg, readback 4.198 ms
cpu tier1 reference: 2645.729 ms total / 264.573 ms avg, hot GPU speedup 4.30x
```

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

The Tier-1 decoder is still a correctness-first kernel and reads coefficients
back for verification in the smoke path. The next implementation step is
connecting the scatter output to downstream GPU DWT without readback and then
replacing the one-invocation-per-codeblock Tier-1 shape with a more cooperative
workgroup layout.

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
