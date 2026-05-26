# vk-j2k-dec

[English](README.md)

基于 Vulkan 计算着色器加速的 JPEG 2000 码流解码器，使用 Rust 编写。

面向 DCP、广播和存档工作流中常见的 3 分量码流。Tier-1 算术解码、系数散布、逆 DWT、色彩变换和最终打包全部在 GPU 上通过 Vulkan 计算着色器完成。

## 性能

测试环境：NVIDIA GeForce RTX 5060（8 GB GDDR7，Blackwell 架构，计算能力 12.0，驱动 577.00）。

可逆 5/3，1920x1080 RGB 12-bit，64x64 codeblock，单 tile（100 次迭代）：

| 路径 | FPS | ms/帧 | 说明 |
|------|-----|--------|------|
| GPU Tier-1 真实 MXF（20 帧） | 69.0 | 14.5 | 真实容器，热均值 |
| GPU Tier-1 流水线（深度 4） | 66.5 | 15.0 | 异步滑动窗口 |
| GPU Tier-1 同步 | 63.0 | 15.9 | 单帧融合调度 |
| 纯 CPU | 7.4 | 134.5 | 参考路径 |

不可逆 9/7，1920x1080 RGB 16-bit，16x16 codeblock，40 tiles（20 次迭代）：

| 路径 | FPS | ms/帧 | 说明 |
|------|-----|--------|------|
| GPU 全流水线（Tier-1 + IDWT 9/7 + 色彩） | 70.5 | 14.2 | 融合调度，全部 tile |
| GPU Tier-1 自适应 | 86.8 | 11.5 | Thread-per-codeblock 着色器 |
| 纯 CPU | 2.95 | 338.6 | 参考路径 |

GPU 对比 CPU 加速比：9.3x（5/3 端到端），24x（9/7 端到端），20-31x（仅 Tier-1）。

GPU 路径与 CPU 参考逐比特一致。5/3 路径同时与 FFmpeg 逐比特一致。

参考：nvjpeg2000（CUDA）在相同 GPU 和内容上端到端达到约 112 fps（含 nvenc H.264 编码）。纯 J2K 解码吞吐量估计为 130-150+ fps。

### 时间分布

可逆 5/3，单 tile，融合 GPU 调度（总计 15.9 ms）：

```
Tier-1 MQ 解码         ~14.8 ms   (93%)  ← 主要瓶颈
Scatter（系数散布）     ~0.2 ms   ( 1%)
逆 DWT 5/3             ~3.6 ms   ( 5%)  (仅 dispatch；alloc 开销 ~5.2 ms 在融合路径中摊销)
色彩变换 + RGB48 打包   ~0.4 ms   ( 1%)
D2H 拷贝（staging）    ~1.2 ms         (在融合命令缓冲区中重叠)
Host 回读（map）       ~1.9 ms         (在融合命令缓冲区中重叠)
```

不可逆 9/7，40 tiles，融合 GPU 调度（总计 14.2 ms）：

```
Tier-1 MQ 解码         ~11.5 ms   (81%)  ← 主要瓶颈
Scatter（系数散布）     ~0.1 ms/tile
逆 DWT 9/7             ~2.0 ms   (14%)  (40 个 tile 合计)
色彩变换 + RGB48 打包   ~0.4 ms   ( 3%)
D2H 拷贝 + 回读        ~2.0 ms         (重叠)
```

Tier-1 MQ 算术解码器是两条路径的主要开销。MQ 状态机在每个 codeblock 内本质上是串行的——每个解码比特依赖前一个状态。Scatter、DWT 和色彩变换已经很快。主要优化杠杆是 MQ 解码吞吐量。

Codeblock 工作量分布（5/3，879 个 codeblock）：

```
byte_len:   min 3, p50 52, p90 293, p99 708, max 2386
pass_count: min 1, p50 31, p90 34, p99 37, max 37
前 3 个 codeblock（LL 子带）工作量是中位数的 10-20 倍 → 长尾效应。
```

### 进一步加速的限制因素

1. MQ 串行依赖：每个比特解码依赖前一个 A/C 寄存器状态。
2. 长尾 codeblock：3 个 LL codeblock 主导单帧调度时间。
3. Vulkan vs CUDA：SPIR-V→PTX 编译对串行密集型内核的优化不如原生 CUDA。
4. nvjpeg2000 仅使用 512 字节共享内存（对比我们的全局状态方案），通过寄存器缓存系数状态实现更高占用率。

## 支持的 Profile

- 原始 `.j2k` / `.j2c` 码流
- 3 分量，8/10/12/16 位采样
- 可逆 5/3 和不可逆 9/7 小波变换
- LRCP 渐进顺序
- RCT 和 ICT 色彩变换
- 单 tile 和多 tile 布局

## 依赖

- Rust 2024 edition
- 支持 Vulkan 1.1+ 计算队列的 GPU
- Vulkan SDK（用于编译着色器）

## 构建

```bash
cargo build --release
```

编译着色器（需要 Vulkan SDK 中的 `glslangValidator`）：

```powershell
.\scripts\compile-shaders.ps1
```

## 用法

### 解码为 RGB48 原始数据

```bash
vk-j2k-dec decode-rgb48 <input.j2k> <output.rgb48le>
vk-j2k-dec decode-rgb48-lowres <input.j2k> <discard-levels> <output.rgb48le>
vk-j2k-dec decode-rgb48-gpu <input.j2k> <discard-levels> <output.rgb48le>
```

### 检查码流

```bash
vk-j2k-dec dump <input.j2k>
vk-j2k-dec scan <container-or-bytestream> [max_hits]
vk-j2k-dec scan-summary <container-or-bytestream>
```

### 从容器中提取码流

```bash
vk-j2k-dec extract <container-or-bytestream> <index> <output.j2k>
```

### 性能测试

```bash
# 单帧 GPU Tier-1 基准测试
vk-j2k-dec tier1-gpu-bench <input.j2k> [iterations]

# 异步流水线基准测试
vk-j2k-dec tier1-gpu-pipeline-bench <input.j2k> [iterations] [pipeline-depth]

# 批量占用率基准测试
vk-j2k-dec tier1-gpu-batch-bench <input.j2k> [iterations] [batch-frames]

# 分阶段性能分析
vk-j2k-dec profile-rgb48-bench <input.j2k> <discard-levels> <cpu|gpu-dwt|gpu> <iterations> <output.rgb48le>

# 真实 MXF 容器性能分析（需要 ffmpeg）
vk-j2k-dec profile-mxf-ffmpeg <input.mxf> <frames> <discard-levels> <cpu|gpu-dwt|gpu|gpu-tier1> <output.rgb48le> [ffmpeg] [work-dir] [idwt.spv] [color.spv]
```

### Vulkan 诊断

```bash
vk-j2k-dec vk-info
vk-j2k-dec vk-smoke
vk-j2k-dec vk-color-smoke
vk-j2k-dec vk-idwt53-smoke
vk-j2k-dec vk-idwt97-smoke
vk-j2k-dec vk-irreversible-smoke
```

### 对比 / 验证工具

```bash
vk-j2k-dec diff-rgb48 <actual.rgb48le> <reference.rgb48le>
vk-j2k-dec diff-component-plane <actual.i32le> <reference.i32le>
```

## 架构

```
CPU: 码流解析 → tile 布局 → 包头 → codeblock 元数据
GPU: Tier-1 MQ 解码 → 系数散布 → 逆 DWT → 色彩变换 → RGB48 打包
```

融合 GPU 路径将 Tier-1、散布、IDWT 和色彩打包录制到单个可复用的命令缓冲区中。各阶段之间保持 device-local 缓冲区，避免不必要的传输。

## GPU 流水线

计算着色器位于 `shaders/` 目录：

| 着色器 | 阶段 |
|--------|------|
| `tier1_cleanup_smoke.comp.glsl` | Tier-1 MQ 算术解码 |
| `tier1_scatter_smoke.comp.glsl` | Codeblock → 子带平面散布 |
| `idwt53_offset_1d.comp.glsl` | 逆 5/3 DWT（可逆） |
| `idwt97_offset_1d.comp.glsl` | 逆 9/7 DWT（不可逆） |
| `color_convert_i32.comp.glsl` | RCT/ICT + level shift + RGB48 打包 |

## 验证

运行完整 GPU 冒烟测试套件：

```powershell
.\scripts\gpu-smoke.ps1
```

解码器在所有分辨率级别上与 FFmpeg 逐比特一致。

## 许可证

待定
