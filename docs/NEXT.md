# Next Optimization Steps

## Current State (May 2025, RTX 5060)

| mode | fps | ms/frame | bottleneck |
|------|-----|----------|------------|
| Pipeline depth 4 | 65.9 | 15.2 | H2D/D2H overlap |
| Sync full RGB48 | 54.6 | 18.3 | Tier-1 dispatch + readback |
| Tier-1 only dispatch | - | 14.0 | MQ serial decode |
| Batch=8 Tier-1 | 72.8 | 13.7 | Long-tail codeblocks |

Target: push sync mode toward 60 fps, pipeline toward 75+ fps.

## Time Budget Breakdown (sync full RGB48, 18.3 ms)

```
Tier-1 MQ dispatch     ~14.0 ms  (76%)
Scatter                 ~0.3 ms
IDWT 5/3               ~1.5 ms
Color + pack            ~0.5 ms
color_copy (D2H)        ~1.3 ms
color_readback (map)    ~2.0 ms
```

## Priority 1: Eliminate Host Readback (saves ~3.3 ms)

If the output feeds a GPU consumer (nvenc, display, or another shader), skip
the D2H copy entirely. Keep the RGB48 buffer device-local.

**Implementation:**
- Add a `decode_to_device` path that returns a `VkBuffer` handle instead of host bytes
- The existing `color_convert_i32.comp.spv` already writes to a device-local buffer
- Remove the `color_copy` and `color_readback` stages from the hot path
- Add an optional readback for validation/file-output only

**Expected gain:** 18.3 ms → ~15.0 ms (sync), matching pipeline throughput.

## Priority 2: Fix Pipeline Depth >4 Race Condition

The pipeline bench fails at depth 5+ with checksum mismatches. This limits
async throughput. The race is likely in the shared `state_buffer` — multiple
in-flight frames write to the same global state region.

**Root cause hypothesis:** All pipeline slots share one `state_buffer`. When
depth > 4, a new frame's zero-init overwrites state that a previous frame's
MQ decode is still reading.

**Fix:** Allocate one `state_buffer` per pipeline slot (same as coefficient
buffer is already per-slot). Each in-flight frame gets its own state region.

**Expected gain:** Pipeline depth 6-8 could push from 65.9 to 70+ fps.

## Priority 3: Reduce Long-Tail Codeblock Impact

The batch bench shows 72.8 fps at batch=8 vs 54.6 fps single-frame. The gap
is caused by the codeblock work distribution:

```
cblk byte_len stats: min 3, p50 52, p90 293, p99 708, max 2386
cblk pass_count stats: min 1, p50 31, p90 34, p99 37, max 37
top work: C0 R0 LL 81124, C2 R0 LL 73202, C1 R0 LL 70108
```

The top 3 codeblocks (LL subbands) have 10-20x more work than the median.
When the GPU finishes the 876 small codeblocks, it waits for the 3 large ones.

**Options:**
1. **Multi-frame dispatch** — submit 2-4 frames in one dispatch (like batch bench).
   Requires the host to parse multiple frames ahead and build a combined record list.
2. **Split large codeblocks** — decode the LL codeblocks on CPU while GPU handles
   the rest. The 3 LL codeblocks take ~3 ms on CPU (single-threaded) vs ~14 ms
   GPU tail. Hybrid CPU+GPU could reduce effective dispatch to ~10 ms.
3. **Cooperative codeblock decode** — use multiple warps for one large codeblock.
   Previously tested and failed (shared memory pressure), but with global state
   this might work now. Very complex to implement correctly.

## Priority 4: Table-Driven MQ Decode

Replace the branch-heavy MQ state machine with a precomputed lookup table.
Current `decode_bit` does:
1. Load context (shared mem read)
2. Unpack state + mps
3. Table lookup (constant array)
4. Subtract, compare, conditional assign
5. Conditional context update + renormalize

A table-driven approach would precompute the full transition for each
(state, A-range-class, C-high-bit) tuple, reducing branches.

**Risk:** Previous attempts to restructure `decode_bit` caused massive
regressions. The NVIDIA compiler strongly prefers the current flat structure.
Any table-driven approach must maintain the same control flow shape.

**Approach:** Instead of changing the decode_bit structure, precompute a
larger packed table that encodes both MPS and LPS outcomes:
```
MQ_FULL_TABLE[state * 4 + {mps_no_renorm, mps_renorm, lps_mps, lps_lps}]
```
This eliminates the `choose_mps` ternary and the separate next_mps/next_lps
selection. But it requires careful benchmarking — the current packed table
with 47 entries fits in L1; a 188-entry table might not.

## Priority 5: MQ Context in Registers (Revisited)

The first attempt to move MQ context to private registers failed due to
register spill. But now with 4 warps/block and global state, the register
pressure profile is different. The 19 context entries (76 bytes) might fit
if other register usage is reduced.

**Approach:** Try `uint context_packed[19]` as private variables again, but
only for the MQ decode thread (lane 0). The other 31 lanes don't use context
at all, so their registers are available for the compiler to use for lane 0's
spill.

**Risk:** Medium. The previous failure was with 16 KB shared state; now that
state is global, the register budget is different.

## Priority 6: Vulkan Subgroup Operations

Replace `barrier()` with `subgroupBarrier()` for intra-warp synchronization.
With 4 warps per block, the current `barrier()` synchronizes all 128 threads
even though each warp is independent. Using subgroup operations could reduce
synchronization overhead.

**Note:** On NVIDIA with 32-thread warps, `barrier()` within a single warp is
already a no-op. But the compiler might not know that warps are independent.
Explicit subgroup ops could help the compiler generate better code.

## Priority 7: Persistent Buffer Pools

The real MXF profiler still allocates buffers per-frame. Persistent pools
would eliminate allocation overhead (~0.5-1 ms per frame in cold paths).

**Implementation:**
- Pre-allocate records, compressed, output, coefficient, and state buffers
  sized for the maximum expected frame
- Reuse across frames without reallocation
- Only reallocate if a frame exceeds the pool size

## Architecture Notes for Future Sessions

### What the NVIDIA compiler likes
- Flat conditional assignments (`x = cond ? a : b`)
- Single return point per function
- Constant array lookups (fits in constant cache)
- Sequential memory access patterns
- `findMSB` for batch shift computation

### What the NVIDIA compiler hates
- Early returns / multiple exit points
- Deep nesting with divergent branches
- Large shared memory (reduces occupancy)
- Register-heavy private arrays (spills to local memory)
- Non-uniform memory access patterns

### Key files
- Shader: `shaders/tier1_cleanup_smoke.comp.glsl`
- Host dispatch: `src/vulkan/api.rs` (search `tier1_cleanup_groups`)
- GPU plan: `src/gpu.rs` (GpuTier1Plan, compressed_u32_words)
- CLI bench: `src/cli/tier1_gpu.rs`
- nvjpeg2000 analysis: `docs/NVJPEG2K_REVERSE.md`
- Performance goal: `docs/GOAL.md`

### Benchmark commands
```bash
# Sync full RGB48 (primary metric)
target\release\vk-j2k-dec.exe tier1-gpu-bench target\samples\test_j2k_rgb12_1080p24_3min_frame0.j2k 20

# Pipeline (async overlap)
target\release\vk-j2k-dec.exe tier1-gpu-pipeline-bench target\samples\test_j2k_rgb12_1080p24_3min_frame0.j2k 20 4

# Batch (occupancy ceiling)
target\release\vk-j2k-dec.exe tier1-gpu-batch-bench target\samples\test_j2k_rgb12_1080p24_3min_frame0.j2k 10 8

# Real MXF container
target\release\vk-j2k-dec.exe profile-mxf-ffmpeg D:\Coding\cu-prores\test_assets\test_j2k_rgb12_1080p24_3min.mxf 20 0 gpu-tier1 target\samples\mxf_out.rgb48le D:\Coding\cu-prores\vendor\ffmpeg-8.1-full_build-shared\bin\ffmpeg.exe target\samples\ffmpeg_demux target\shaders\idwt53_offset_1d.comp.spv target\shaders\color_convert_i32.comp.spv
```

### Correctness check
All benchmarks must report `checksum 0xfffe2412`. Any mismatch means the
optimization broke bit-exactness and must be rolled back.

### Decision criteria
- >2% improvement on the sync full RGB48 bench (20 iterations, 3 runs median)
- Bit-exact checksum
- If within noise (±2%), roll back

## Optimization Attempt Log (May 2025)

### Confirmed working
| Optimization | Tier-1 gain | Notes |
|---|---|---|
| Global coefficient_state (from shared) | +31% | Occupancy 8→32+ workgroups/SM |
| 4 codeblocks/workgroup (128 threads) | +14% | 4 independent warps hide latency |
| Per-codeblock 0xFF padding + bounds removal | +2.7% | Eliminates branch in byte_at |
| Column activity bitmap (significance pass) | +10.9% | 64-bit register bitmap, skip inactive columns |
| Subgroup barriers | ~0% alone | Enables pipeline depth >4 correctness |
| Fused D2H copy in command buffer | ~0.2% | Eliminates one submit+wait round-trip |

### Confirmed NOT working (do not retry)
| Optimization | Result | Why |
|---|---|---|
| Private context registers (19 uints) | -74% (21.7ms) | Register spill to local memory |
| Restructured decode_bit (early return) | -70% (16 fps) | NVIDIA compiler hates divergent branches |
| Table-driven MQ (eliminate ternaries) | ±0% | Compiler already generates optimal selp |
| 8 codeblocks/workgroup (256 threads) | ±0% | No benefit over 4 |
| 2-word byte cache | ±0% | Noise |
| Shared memory byte preload | -40% | Occupancy loss |
| Removing byte cache | -2% | Slight regression |
| Stride-based indexing | ±0% | Neutral |
| Multi-lane cooperative MQ decode | failed | MQ serial state too strong (GOAL.md) |
| Larger workgroups (64-lane) | failed | Shared memory pressure (GOAL.md) |
| Refinement pass column-skip | -0.8% | All refined samples are in active columns |

### Remaining opportunities (untried)
| Optimization | Expected gain | Effort | Risk |
|---|---|---|---|
| Packed column state (nvjpeg2000 style) | 5-15% | 3-5 days | High (full shader rewrite) |
| Sign context inline (encode in state bits) | <1% | 1 day | Low |
| Fused scatter+IDWT | 0.5-1ms | 2 days | Medium |
| Async transfer queue for readback | 1-2ms | 1 day | Low |
| IDWT shared memory tiling | 0.5-1ms | 2 days | Medium |
