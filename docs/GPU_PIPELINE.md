# GPU pipeline notes

The first Vulkan path keeps Tier-1 on CPU and sends decoded coefficient planes to
compute shaders. The first useful GPU boundary is:

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

- per-subband dequant scale from QCD/QCC;
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

The implementation currently submits one shader dispatch at a time and uses
host-visible buffers for simple staging. That is correct but not the final
performance shape. The next GPU milestones are:

- keep intermediate DWT buffers device-local and avoid CPU readback between levels;
- reuse descriptor layouts, pipelines, command pools, and command buffers;
- keep DWT output on device for color conversion instead of re-uploading planes;
- wire irreversible 9/7 DWT into real-frame decode once a representative sample is available.

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
that mirrors the 5/3 backend shape and returns float component planes. CPU and
GPU color packing both have float component-plane helpers for the no-transform
and ICT cases. Real irreversible-frame decode still needs dequantized float
subband planes and a validated sample before it should be treated as an
end-to-end path.

`vk-j2k-dec vk-irreversible-smoke` chains the synthetic irreversible pieces:
three component-level 9/7 reconstructions feed GPU ICT/color packing, and the
result is compared against CPU 9/7 plus CPU float RGB48 packing.

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
