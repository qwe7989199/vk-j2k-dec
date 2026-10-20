# iDWT 9/7 Gain Bug — 待修复

## 现状

PSNR=46.14 dB vs ffmpeg (应该 80+ dB)。DC 值完美匹配，误差全部来自 detail subbands 的 K 归一化不正确。

## 根因

`src/dwt.rs` 中 `inverse_9_7_1d` 的 **1D DC gain 是 1/K² 而不是 1/K**。

验证方法：constant low=-2047, zero high → 输出 -1352.6 = -2047/K²（应该是 -2047/K = -1664）。

这导致 6 级 2D iDWT 的总 DC gain = (1/K²)^12 = 1/K^24，而标准实现（ffmpeg/OpenJPEG）是 1/K^12。

## 当前 workaround

在 `src/reconstruct.rs` 的 `QcdScaleMap::scale_for` 和 `src/gpu.rs` 的 `compute_subband_scale` 中，对 LL band 乘以 K^(4*NL) = K^24 来补偿。这让 DC 正确了，但 detail bands 的 K power 公式无法正确推导（因为 lifting 本身的增益特性不对）。

当前 detail band 公式：
```rust
SubbandKind::Ll => 4 * nl,
SubbandKind::Hl | SubbandKind::Lh => (4 * (nl - r) - 2).max(0),
SubbandKind::Hh => (4 * (nl - r) - 4).max(0),
```

## 修复结果

修正 `inverse_9_7_1d`（`src/dwt.rs:72`）使其 1D DC gain = 1/K（而不是 1/K²）。修好后：
- LL 的 K power 变为 `2*NL`（标准值）
- HL/LH 的 K power 变为 `2*(NL-r) - 1`
- HH 的 K power 变为 `2*(NL-r) - 4`
- 这和 ffmpeg 的 `init_band_stepsize` 公式一致
- PSNR 应该到 80+ dB

注意：本项目的 `quantization_step_scale` 已经包含 JPEG 2000 的 `gain_bits`
（HL/LH 为 `2^1`，HH 为 `2^2`）。因此 HH 在额外 K 归一化里还要抵消
ffmpeg `F_LFTG_X²` 对应的 `1/K²`，最终是 `2*(NL-r)-4`。

## 问题定位

lifting 步骤本身是正确的（round-trip forward→inverse = identity）。问题在于 **我们的 forward DWT 对 constant signal 产生 K^24 的增益**（实测 constant 1000 → LL=144277 = 1000*K^24），而 Kakadu/ffmpeg 的 forward 产生 K^12 的增益。

这意味着我们的 forward DWT 每个 1D pass 的 DC gain 是 K²（而不是 K）。forward 和 inverse 配对时互相抵消所以 round-trip 正确，但和外部编码器不兼容。

## 对照 ffmpeg/OpenJPEG 的关键差异

ffmpeg 的 `init_band_stepsize`（`libavcodec/jpeg2000.c`）：
```c
// 基础 step: 2^(cbps - expn) * (1 + mant/2048)  [注意：没有 gain_bits]
// 然后对 9/7:
//   HL/LH: *= F_LFTG_X * 2    (F_LFTG_X = 1/K = 0.812893)
//   HH:    *= F_LFTG_X² * 4
//   所有:  *= K^(2*(NL-r) + lband - 2)  [lband=1 for HL/LH, 0 for LL/HH]
```

我们的公式：`2^(cbps + gain_bits - expn) * (1 + mant/2048)`，其中 gain_bits = 0/1/2。
- 我们的 `2^gain_bits` 对应 ffmpeg 的 `F_LFTG_X^n * 2^n`（近似但不完全相等，差 1/K^n）

## 如何验证修复

### 前置准备

```bash
# 从 MXF 提取第一帧 J2K codestream
cargo run -- extract 1.mxf 0 C:/Users/domo/AppData/Local/Temp/cprl_test.j2k

# 用 ffmpeg 解码参考帧（rgb48le raw）
ffmpeg -y -i 1.mxf -frames:v 1 -pix_fmt rgb48le -f rawvideo /tmp/ref_frame.raw
```

### 验证 1D DC gain（单元测试）

```bash
cargo test --release inverse_97_dc_gain -- --nocapture
```

修好后应该看到：
- `6-level 2D iDWT DC gain (center pixel): ≈ 0.0834`（= 1/K^12，不是当前的 1/K^24）
- 或者如果完全去掉 K 修正后 DC gain = 1.0

### CPU 解码 vs ffmpeg 对比（主测试）

```bash
CPRL_TEST_J2K="C:/Users/domo/AppData/Local/Temp/cprl_test.j2k" \
CPRL_REF_RAW="/tmp/ref_frame.raw" \
cargo test --release cprl_4k_dcp_full_gpu -- --ignored --nocapture
```

当前输出：
```
top-left (0,0): ours=(1,1,1) ref=(1,1,1)        ← DC 正确
center (1920,1080): ours=(705,1142,646) ref=(702,1139,642)  ← 差 3-7
vs ffmpeg: PSNR=46.14 dB, max_diff=667
```

目标输出：
```
PSNR > 80 dB, max_diff ≤ 2
```

### GPU pipeline 对比

测试代码在 `src/decode.rs` 的 `cprl_4k_dcp_full_gpu_decode_produces_nonzero_samples` 中。
切换 CPU/GPU 路径：修改该测试中的 `decode_rgb48_with_discard` 为 `decode_rgb48_full_gpu_97`。

当前测试用的是 CPU 路径（`decode_rgb48_with_discard`）。GPU 路径需要同步修正 `shaders/idwt97_offset_1d.comp.glsl`。

### GPU shader 编译

```bash
glslc -fshader-stage=compute shaders/idwt97_offset_1d.comp.glsl -o shaders/idwt97_offset_1d.comp.spv
```

### Player 端到端测试

```bash
cargo run --release -p vk-j2k-player -- 1.mxf --fps 24
```

### dump 解码结果对比

```bash
# 设置 CPRL_OUT_RAW 环境变量会 dump 解码结果
CPRL_TEST_J2K="C:/Users/domo/AppData/Local/Temp/cprl_test.j2k" \
CPRL_REF_RAW="/tmp/ref_frame.raw" \
CPRL_OUT_RAW="/tmp/ours_frame.raw" \
cargo test --release cprl_4k_dcp_full_gpu -- --ignored --nocapture

# 用 ffplay 目视对比
ffplay -f rawvideo -pixel_format rgb48le -video_size 3840x2160 /tmp/ours_frame.raw
ffplay -f rawvideo -pixel_format rgb48le -video_size 3840x2160 /tmp/ref_frame.raw
```

### 全量测试（确保没有回归）

```bash
cargo test --release
```

当前 87 个 lib tests + 4 个 bin tests 全部通过。

## 相关文件

| 文件 | 作用 |
|------|------|
| `src/dwt.rs:72` | `inverse_9_7_1d` — 需要修正 DC gain |
| `src/dwt.rs:284` | `inverse_9_7_level` — 2D wrapper |
| `src/reconstruct.rs:441` | `QcdScaleMap::scale_for` — CPU dequant K 修正 |
| `src/gpu.rs:42` | `compute_subband_scale` — GPU dequant K 修正 |
| `shaders/idwt97_offset_1d.comp.glsl` | GPU iDWT shader（需要同步修正） |
| `src/dwt.rs:624` | `forward_9_7_1d`（test-only）— 用于 round-trip 验证 |

## 排查建议

1. 对照 OpenJPEG `src/lib/openjp2/dwt.c` 中的 `opj_v4dwt_decode_step1/step2` 实现
2. 关键区别可能在：K 是应用在 interleaved signal 上（even/odd）还是 separated bands 上（low/high）
3. 或者 lifting 步骤中 low/high 的邻居索引偏移有误（`high[index]` vs `high[index-1]` 的选择）
4. 用小信号（8 samples）做 forward→quantize→dequant→inverse 全流程对比 ffmpeg 的中间值
