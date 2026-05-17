# vk-j2k-dec roadmap

`vk-j2k-dec` aims to become a Rust JPEG 2000 decoder with Vulkan compute acceleration for the stages that are naturally GPU-friendly. The first useful target is not a complete ISO 15444 implementation. It is a correct, measurable, open decoder for common 3-component codestreams, especially the kind of material found in DCP, broadcast, scanning, and archival workflows.

## Project shape

- CPU handles container, codestream, tile, packet, and codeblock metadata parsing.
- CPU keeps Tier-1 codeblock decoding in the first production path.
- Vulkan compute handles dequantization, inverse DWT, inverse color transform, level shift, and final packing.
- The decoder should remain usable without Vulkan through a CPU fallback path.
- The public API should be designed so a C ABI wrapper can be added later.

## First supported profile

- Raw `.j2k` / `.j2c` codestream first, `.jp2` later.
- 3 components.
- 8, 10, 12, and 16-bit samples.
- Reversible 5/3 and irreversible 9/7 wavelet transforms.
- Common progression orders, with LRCP as the first target.
- RCT and ICT color transforms.
- No exotic ROI in the first implementation.
- No JPIP, JPWL, or error resilience features in the first implementation.

## Crate layout target

- `j2k-codestream`: marker, tile, packet, precinct, and codeblock parsing.
- `j2k-tier1`: CPU EBCOT Tier-1 codeblock decoder.
- `j2k-dwt`: CPU reference inverse DWT implementation.
- `j2k-vk`: Vulkan compute backend.
- `vk-j2k-dec`: high-level decoder API and CLI tools.

The repository can start as one crate while the data model is still changing. Split crates only once module boundaries become stable.

## Milestones

### v0.1: codestream inspection

- Parse `SOC`, `SIZ`, `COD`, `QCD`, `SOT`, `SOD`, and `EOC`.
- Dump image size, tile grid, component precision, progression order, layer count, transform type, and tile-part metadata.
- Add structured parse errors with byte offsets.
- Add fixtures for minimal valid and invalid codestreams.

### v0.2: CPU reference decode path

- Parse packet headers for the initial supported profile.
- Decode codeblock bitstreams with a CPU Tier-1 path.
- Build coefficient planes per component and subband.
- Add CPU inverse DWT 5/3 and 9/7 reference implementations. Done for one-level references. The reversible 5/3 CPU path now matches FFmpeg's horizontal-first integer reconstruction order.
- Split Tier-1 coefficient reconstruction into exact reversible output and midpoint irreversible output. Done; the local reversible 5/3 frame matches FFmpeg exactly from lowres-5 through full resolution.
- Compare decoded output against OpenJPEG.

### v0.3: Vulkan setup and buffer model

- Add Vulkan device/context selection. Done for the local dynamic-loader path.
- Define GPU buffer layouts for coefficient planes and output planes.
- Implement dequantization compute shaders. Initial storage-buffer smoke is working.
- Implement color conversion compute shaders. No-transform, RCT, and ICT smoke paths are working.
- Add float component-plane packing for irreversible paths. CPU and GPU helpers are covered by local tests/smoke.
- Add dequantized float subband staging for irreversible paths. Synthetic scaling plus QCD scalar-expounded/scalar-derived scales and QCC component overrides are modeled.
- Add a synthetic irreversible GPU pipeline check. Done for scaled float subbands, 9/7 component reconstruction, and ICT/RGB48 packing.
- Keep readback simple and explicit.

### v0.4: Vulkan inverse DWT

- Implement inverse 5/3 lifting in compute. Initial 1D, strided column, one-level 2D smoke, component-backend, and full-frame paths are working with the FFmpeg-compatible horizontal-first reconstruction order.
- Implement inverse 9/7 lifting in compute. Initial 1D, strided column, one-level 2D smoke, and component-backend paths are working.
- Validate against the CPU reference path. Done for 5/3 and 9/7 smoke paths.
- Benchmark tile sizes, workgroup shapes, and memory layouts.

### v0.5: DCP-focused fast path

- Optimize 3-component 12-bit XYZ output.
- Add stage-level profiling around packet parse, Tier-1, reconstruction, color/packing, and file I/O. Done for the CLI path.
- Avoid decoding unused high-resolution codeblocks for reduced-resolution output. Done for the high-level decode path.
- Add DCP profile validation helpers.
- Measure 2K and 4K frame throughput.
- Overlap CPU Tier-1 work, GPU transfer, and GPU reconstruction.

### v0.6: integration prototypes

- Add a stable C ABI wrapper.
- Build a simple FFmpeg integration experiment outside the FFmpeg tree.
- Build a DCP-o-matic integration experiment or replacement decode path.
- Document integration constraints and licensing decisions.

## Longer-term work

- GPU Tier-1 codeblock decoding experiments.
- Multi-tile scheduling and asynchronous transfer queues.
- Direct Vulkan image output to avoid CPU readback.
- SIMD CPU fallback for Tier-1 and DWT.
- JP2 container support.
- 1-component, 4-component, alpha, and unusual component sampling support.
- More progression orders and more codestream edge cases.
- Fuzzing for marker, tile-part, packet, and codeblock parsers.
- Conformance corpus and differential tests against OpenJPEG.
- Optional backends such as Metal, CUDA, or OpenCL if there is a real downstream need.

## Upstream strategy

FFmpeg integration is easier to evaluate after the decoder has a stable C ABI and a proven benchmark story. FFmpeg's core codebase is C and has strict development policy, style, memory allocation, and API compatibility requirements, so an external Rust library is likely the best first step.

DCP-o-matic is a promising downstream target because JPEG 2000 decode performance matters directly for DCP workflows. A DCP-focused fast path and clear comparison against OpenJPEG/Grok-style workflows should make the project easier to evaluate.

## Immediate tasks

- Continue CPU Tier-1: significance propagation, cleanup, magnitude refinement, sign contexts, and exact reversible coefficient reconstruction are in place; the local reversible 5/3 frame is bit-exact against FFmpeg at all reduction levels and full resolution. Remaining differential work is broader corpus/OpenJPEG coverage and irreversible 9/7 samples.
- Build coefficient planes from decoded codeblocks and feed the CPU/GPU reconstruction path.
- Add irreversible 9/7 CPU reference and Vulkan smoke shader. Done.
- Wire irreversible 9/7 into the high-level CPU decode path. Done for QCD scalar-expounded/scalar-derived codestreams and QCC component overrides; corpus validation still needs implementation.
- Turn the Vulkan smoke helpers into a reusable reconstruction backend with persistent pipelines and buffers. Component-level 5/3 and 9/7 helpers exist; persistent pipeline reuse is still pending.
- Use the profiling CLI for every performance run. Current full-resolution CPU decode is ~0.420s on the local frame, with Tier-1 still the largest CPU stage at ~0.325s. Current GPU full path is ~1.544s and is dominated by transient Vulkan reconstruction, packing, and readback rather than Tier-1 arithmetic.
- Continue performance work without sacrificing correctness: lowres pruning, cached Tier-1 significance-neighbor state, byte-backed state planes, direct pass-state access, one fewer MQ input copy, and a fused reversible 5/3 GPU DWT+color call are done. Next likely wins are persistent Vulkan device/pipeline/buffer reuse, GPU-side component-to-color handoff without readback/reupload, CPU Tier-1 allocation reuse, and eventually overlapping CPU Tier-1 with GPU reconstruction.
- Add a frame-level compare path against FFmpeg/OpenJPEG output once Tier-1 is producing coefficients.
- Build a `dump` CLI for codestream headers. Done.
- Parse and print `SIZ`, `COD`, `QCD`, and `SOT`. Done.
- Scan MXF/container files for embedded JPEG 2000 codestreams for early validation. Done for the local MXF sample.
- Track tile-part payload ranges and LRCP packet iteration order. Done for the first profile path.
- Add packet-header bitstream helpers with JPEG 2000 byte stuffing. Done.
- Add tag-tree, codeblock packet-header state, and first-profile LRCP packet parsing. Done for the local sample.
- Add unit tests with tiny synthetic codestreams. Ongoing.
- Add a reference corpus directory once real sample files are available.
- Decide the license before accepting outside contributions.
