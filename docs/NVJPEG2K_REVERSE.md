# nvjpeg2000 Tier-1 GPU Kernel 逆向分析

从 `nvjpeg2k_0.dll` 提取的 PTX (sm_120 / Blackwell) 分析。

## Kernel 列表

| Kernel | 用途 |
|--------|------|
| `tier1decodemultiple_channels_k<0>` | 传统 J2K MQ Tier-1 解码 |
| `init_tier1_contexts` | Tier-1 上下文初始化 |
| `htTier1MagSgnDecodeMultiple_channels_k` | HTJ2K 幅值/符号解码 |
| `htTier1MelAndVlcDecodeMultiple_channels_k` | HTJ2K MEL+VLC 解码 |
| `idwt` / `idwt_0d` / `idwt_1d` | 逆向 DWT（多种模板特化） |
| `csc_levelshift` | 色彩空间转换 + level shift |
| `lossy_mct_levelshift` / `lossless_mct_levelshift` | MCT |
| `inv_quantize_partial` | 反量化 |

## Tier-1 解码 Kernel 架构分析

### 基本参数

```
签名: tier1decodemultiple_channels_k<0>(const uint8_t* data1, const uint8_t* data2, tier1BatchedInput* input)
PTX 行数: ~2440 行
基本块数: 236
寄存器: 303 pred, 177 x 16-bit, 1800 x 32-bit, 293 x 64-bit
共享内存: 仅 512 字节 (cblk_mem_shared)
```

### 关键架构差异（vs 我们的实现）

| 特性 | nvjpeg2000 | 我们 |
|------|-----------|------|
| 共享内存 | **512 字节** | 16,460 字节 |
| coefficient_state 存储 | **全局内存** (通过寄存器缓存) | 共享内存 4096×u32 |
| MQ context 存储 | 共享内存 19×u32 (76字节) | 共享内存 19×u32 |
| 每 block 处理 codeblock 数 | **4 个** (blockIdx.x << 2) | 1 个 |
| 线程组织 | 每 warp 1 个 codeblock, 4 warps/block | 32 线程/block, 仅 thread 0 工作 |
| Warp 同步 | `bar.warp.sync` (1次) | `barrier()` (多次) |
| Warp shuffle | 未使用 | 未使用 |
| 全局内存访问 | 116 次 | 通过 byte cache 减少 |
| 共享内存访问 | 173 次 | 大量 (coefficient_state) |

### 核心设计思路

1. **Coefficient state 不在共享内存！** nvjpeg2000 把 coefficient state 放在全局内存（通过 L1/L2 缓存），只用 512 字节共享内存存 MQ 上下文 + 少量元数据。这意味着：
   - 共享内存占用极低 → 占用率极高（理论上可达 128KB/512B = 256 workgroups/SM，受限于寄存器）
   - coefficient state 的读写走 L1 cache，对于 64×64 = 4096 × 4B = 16KB 的 codeblock，完全在 L1 内

2. **每个 block 处理 4 个 codeblock**（4 warps，每 warp 1 个 codeblock）。这比我们的 1 codeblock/block 更高效：
   - 更多 warps 可以互相隐藏延迟
   - 共享内存按 warp 分区（每 warp 128 字节: `shl.b32 %r783, %r2, 7`）

3. **只有 lane 0 做 MQ 解码**（`setp.ne.s32 %p5, %r10, 0; @%p5 bra exit`）— 和我们一样，但因为有 4 个 warp 同时工作，SM 利用率更高。

4. **大量寄存器使用**（1800 个 r32）— 他们把 coefficient state 的热数据缓存在寄存器中，而不是反复读共享内存。PTX 编译器会把这些映射到实际硬件寄存器 + register spill to local memory。

5. **没有 warp shuffle** — MQ 解码本质上是单线程的，他们也没有尝试多线程协作解码。

### 对我们的启示

#### 最大的可借鉴点：把 coefficient_state 从共享内存移到全局内存

当前我们用 16 KB 共享内存存 coefficient_state，限制占用率为 8 workgroups/SM。如果改为：
- coefficient_state 放在全局内存（device-local buffer）
- 依赖 L1 cache（RTX 5060 每 SM 128 KB L1）
- 共享内存只保留 MQ context (76 bytes) + 少量元数据

则：
- 共享内存降到 ~512 字节
- 占用率可能提升到 16-32 workgroups/SM（受限于寄存器）
- 更多 active warps 隐藏 MQ 的指令延迟

#### 次要借鉴点

1. **每 block 多 codeblock** — 把 workgroup 改为 128 线程 (4 warps)，每 warp 处理 1 个 codeblock。4 个独立 MQ 解码器共享一个 block，互相隐藏延迟。

2. **寄存器缓存 coefficient state** — 对于 stripe 内的 4 行样本，把当前 stripe 的 state 缓存在寄存器中，减少全局内存读写。

3. **Batched input 结构** — 他们用 `tier1BatchedInput` 结构传递所有 codeblock 的元数据，而不是逐个 codeblock 的 record 数组。

### 风险评估

把 coefficient_state 移到全局内存是一个大改动：
- 之前我们尝试"共享内存预加载压缩字节"时，增加共享内存反而降低了性能
- 但 nvjpeg2000 的做法是**减少**共享内存（从我们的 16 KB 到 512 字节）
- 这与我们之前的失败方向相反——他们是通过减少共享内存来提升占用率

### 实施优先级

1. **[高优先级]** 把 coefficient_state 移到全局内存 buffer，只保留 MQ context 在共享内存
2. **[中优先级]** 每 block 4 warps × 1 codeblock/warp 的组织方式
3. **[低优先级]** 寄存器缓存 stripe-local state

## DWT Kernel 架构

nvjpeg2000 的 DWT 有大量模板特化：
- `dwt_0d` / `dwt_1d` — 0D (单点) 和 1D lifting
- `idwt_0d` / `idwt_1d` — 逆向
- 模板参数: `<行优先?, MCT?, lifting步数, 步长, 输入类型, 输出类型>`
- 输入/输出类型: `h`=uint8, `s`=int16, `t`=uint16, `i`=int32, `f`=float

这说明他们为每种精度组合生成了专门的 kernel，避免运行时类型分支。

## 文件位置

- 完整 PTX: `refer/ptx/full_ptx.txt` (176306 行)
- Tier-1 kernel: 行 85420-87860
- 源 DLL: `refer/nvjpeg2k_0.dll`
