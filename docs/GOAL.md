# Performance goal: half of nvjpeg2000

## Reference benchmark

nvjpeg2000 (CUDA) + nvenc H.264 encode on RTX 5060, same MXF source:

```
1920x1080 RGB12 reversible 5/3, 24 fps source
encoded 1001 frames, 111.79 fps, GPU 88%
```

nvenc H.264 high@auto uses the dedicated video encoder (VE at 18.5%), so the
CUDA cores are mostly doing J2K decode. Accounting for nvenc's share of GPU
time and the cspconv RGB16→NV12 step, the pure nvjpeg2000 decode throughput is
likely 130–150+ fps on this content.

## Target

Reach at least **56 fps** (half of 112 fps end-to-end) for the same
1920x1080 RGB12 reversible 5/3 codestream on the Vulkan compute path. That
gives a per-frame budget of **~17.8 ms** from compressed bytes to packed RGB48
in host memory (or device memory if feeding a downstream consumer).

## Current state (RTX 5060, May 2025)

| path | ms/frame | fps | notes |
| --- | ---: | ---: | --- |
| pipeline bench (depth 4) | 26.6 | 37.6 | best current throughput |
| fused GPU Tier-1 (sync) | 33.3 | 30.0 | single-frame dispatch |
| real MXF gpu-tier1 (20f) | 27.8 | 36.0 | real container warm avg |
| CPU-Tier1 + GPU-DWT | 85.9 | 11.6 | CPU bottlenecked |
| batch=8 Tier-1 only | 28.2 | 35.5 | occupancy plateau |

The gap to 56 fps is **~1.5x** from the current pipeline bench.

## Where the time goes (fused single-frame)

```
fused_gpu_dispatch  ~27.8 ms   (93% of frame time)
  ├── Tier-1 MQ decode   ~25 ms   (serial per codeblock, 879 workgroups)
  ├── scatter             <0.3 ms
  ├── IDWT 5/3            ~3.5 ms
  └── color+pack          ~0.5 ms
color_copy              ~1.3 ms
color_readback          ~2.1 ms
```

Tier-1 MQ is 85%+ of GPU frame time. Scatter, DWT, and color are already fast.

## Why nvjpeg2000 is faster

1. **CUDA warp primitives.** Warp shuffle, warp vote, and cooperative groups
   let CUDA kernels share MQ state across lanes without shared memory round
   trips. Vulkan subgroup operations (shuffle, ballot) are available since
   Vulkan 1.1 but less flexible.

2. **Compiler maturity.** NVCC/PTX has years of register allocation and
   instruction scheduling tuning for serial-heavy kernels. SPIR-V → PTX via
   the Vulkan driver is a less optimized path for the same hardware.

3. **Occupancy tuning.** CUDA can precisely control register count, shared
   memory, and block dimensions per kernel launch. Vulkan exposes less
   fine-grained control.

4. **Zero-copy decode-to-encode.** nvjpeg2000 output stays in CUDA device
   memory and feeds nvenc directly through NV12 conversion. No host readback.
   Our current path pays ~3.4 ms for device→host copy+readback.

5. **Library-level optimizations.** nvjpeg2000 is a closed-source library with
   NVIDIA-internal knowledge of the hardware scheduler, cache hierarchy, and
   memory subsystem. It may use techniques not publicly documented.

## Can Vulkan compute reach 56 fps?

**Yes, it is achievable, but not trivially.** The path requires stacking
several improvements:

### Likely wins (high confidence)

| improvement | estimated gain | notes |
| --- | ---: | --- |
| Eliminate host readback | 3.4 ms | keep output device-local for downstream |
| Multi-frame batching (4–8 in flight) | 2–4 ms | already proven in batch bench |
| Subgroup shuffle for MQ renormalization | 1–3 ms | avoid shared memory for neighbor state |
| Persistent buffer pools | 0.5–1 ms | remove per-frame alloc overhead |

Subtotal: ~7–11 ms savings → **16–20 ms/frame** → **50–62 fps**

### Possible wins (medium confidence)

| improvement | estimated gain | notes |
| --- | ---: | --- |
| Table-driven MQ decode | 2–5 ms | replace branch-heavy state machine with LUT |
| Speculative MQ bit decode | 1–3 ms | decode multiple bits speculatively, discard on mismatch |
| Two-pass Tier-1 (coarse then fine) | 1–2 ms | decode significance map first, then coefficients |
| SPIR-V optimization passes | 1–2 ms | manual register pressure reduction |
| Async compute + transfer overlap | 1–2 ms | use separate transfer queue for readback |

### Hard wins (low confidence, high effort)

| improvement | estimated gain | notes |
| --- | ---: | --- |
| Cooperative multi-lane MQ | 3–8 ms | multiple threads decode one codeblock cooperatively |
| Custom SPIR-V with vendor hints | 1–3 ms | hand-tuned instruction scheduling |
| Hybrid CPU+GPU Tier-1 split | variable | decode large codeblocks on GPU, small on CPU |

### Realistic assessment

Stacking the high-confidence wins alone gets close to 56 fps. The key insight
is that **eliminating host readback** (if the output feeds a Vulkan/nvenc
consumer) immediately saves 3.4 ms, bringing the pipeline bench from 26.6 to
~23.2 ms (43 fps). Adding multi-frame batching and subgroup MQ improvements
should close the remaining gap.

If the output must land in host memory (no downstream GPU consumer), 56 fps is
harder but still plausible with aggressive MQ optimization.

## What won't work

- **More codeblocks per dispatch** alone won't help much. The batch bench
  plateaus at 8–16 frames because the bottleneck is serial MQ inside each
  codeblock, not occupancy.

- **Larger workgroups** were tested (64-lane) and lost due to shared memory
  pressure. The 32-lane shape is already near-optimal for this shader.

- **Multi-lane cooperative decode** (CUDA ProRes style) was tested and lost on
  this workload because JPEG 2000 MQ has more serial state than ProRes VLC.

## Milestones toward 56 fps

1. **Remove readback from the critical path.** Keep RGB48 device-local, add an
   optional host-readback path for validation only. Measure fused dispatch
   without readback.

2. **Persistent Vulkan session.** Pool all buffers across frames, eliminate
   per-frame allocation. Measure pipeline bench improvement.

3. **Subgroup MQ experiments.** Use `subgroupShuffle` for MQ renormalization
   and context state sharing. Measure per-codeblock MQ time reduction.

4. **Table-driven MQ.** Replace the branch-heavy `mq_decode` with a
   precomputed transition table indexed by (context, A-range-class, bit).
   Measure branch miss reduction.

5. **Multi-frame pipeline depth 8.** Increase in-flight frames to better hide
   any remaining transfer latency and improve GPU occupancy.

6. **Benchmark against nvjpeg2000 directly.** Add a CLI that decodes the same
   MXF frames with both paths and compares throughput and correctness.

## Bottom line

The 56 fps target is realistic for Vulkan compute on RTX 5060 with this
content. The primary lever is MQ decode speed — everything else (DWT, scatter,
color, packing) is already fast. The secondary lever is eliminating unnecessary
host transfers. CUDA will likely retain a ~1.5–2x advantage on pure MQ
throughput due to compiler and warp-level advantages, but the Vulkan path can
compensate with pipeline depth and transfer elimination.
