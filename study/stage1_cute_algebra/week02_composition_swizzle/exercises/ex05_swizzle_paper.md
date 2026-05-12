# Ex05 — `Swizzle<B, M, S>` bit math 手算

锚点：`include/cute/swizzle.hpp:42-130`

---

## 速记

源码（`include/cute/swizzle.hpp:67-78`）：

```cpp
yyy_msk = ((1 << B) - 1) << (M + S);                  // 源 bit 的位置（假设 S >= 0）
apply(offset) = offset XOR ((offset & yyy_msk) >> S); // 把源段位 shift 下来 XOR 到目标段位
```

等价的"位段描述"：
- **target bits**（被 XOR 修改的位）：`[M, M+B)`
- **source bits**（参与 XOR 的位）：`[M+S, M+S+B)`
- 两段距离 = `S` bit

**直觉**：
- `M` = 一个"原子单位"占多少 bit（例如 fp32 = 4B = 2 bit、128B = 7 bit）。下面 `M` bit 是不可分割的访问颗粒，绝不动。
- `B` = 多少 bit 参与 swizzle（XOR 几位）
- `S` = 源段距目标段的位移

效果是把"上层" `B` 个 bit XOR 到 `[M, M+B)` 这段——让行号低位"打乱"列号低位，让同一个 bank 不再被"同 col 不同 row"全部命中。

参数三元组到 byte 的速记表（fp32 = 4B/elem）：

| Swizzle | 含义 |
|---------|------|
| `<3, 2, 3>` | 8x8 fp32 tile（每行 32B）：row 低 3 bit XOR 到 col 低 3 bit，消转置访问的 2-way bank conflict（8 行 col=0 命中 banks {0,8,16,24} 各 2 次）|
| `<1, 4, 3>` | SW32：8 行 × 2 chunks of 16B（每行 32B），row[0] XOR chunk[0]，1 bit swizzle |
| `<2, 4, 3>` | SW64：8 行 × 4 chunks of 16B（每行 64B），row[1:0] XOR chunk[1:0]，2 bit swizzle |
| `<3, 4, 3>` | SW128：8 行 × 8 chunks of 16B（每行 128B），row[2:0] XOR chunk[2:0]，最常见的 WGMMA smem swizzle |

注：SW32/64/128 系列共用 M=4、S=3——是因为 Hopper 约定 row 字段对齐到 offset 的 bit 7+（哪怕每行 chunks 不满 8 个，由外层 stride 隔开），让 swizzle 公式跨 SW 系列统一。B = log2(chunks/row)。

---

## 题目

### Q1. 解 `Swizzle<3, 2, 3>` 在 fp32 smem 上的位置

smem 8x8 fp32（即 8x32B = 8 行 × 8 fp32/行）。元素 `[row, col]` 的线性 byte offset = `row * 32 + col * 4`。

bit 布局（offset 共 8 bit）：
```
bit 7..5 = row[2:0]      // 3 bit
bit 4..2 = col[2:0]      // 3 bit
bit 1..0 = byte_in_elem  // M=2
```

`Swizzle<3, 2, 3>` 的 source = `[M+S, M+S+B)` = bits **5..7**，target = `[M, M+B)` = bits **2..4**。
- source = `bit[7:5]` = **row 的低 3 bit**
- target = `bit[4:2]` = **col 的低 3 bit**

**你的答案**：被 XOR 的目标 bit 是 offset 的第 **2** 到第 **4** 位（即 col[2:0]），对应物理意义是：**row 的低 3 bit 与 col 的低 3 bit 异或**——新地址的 bank 编号 ≈ `(row ^ col) % 32`。

---

### Q2. 列出 `Swizzle<3, 2, 3>(addr)` 的具体值

XOR mask = `((offset >> 5) & 0b111) << 2` = `row[2:0] << 2`。

填表（row, col 都是 0..7 演示，足够覆盖 swizzle 一个周期；row stride = 32B）：

| row | col | raw offset (bytes) | XOR mask | swizzled offset |
|-----|-----|---------------------|----------|-----------------|
| 0 | 0 | 0   | 0  | 0   |
| 0 | 1 | 4   | 0  | 4   |
| 0 | 2 | 8   | 0  | 8   |
| 1 | 0 | 32  | 4  | 36  |
| 1 | 1 | 36  | 4  | 32  |
| 2 | 0 | 64  | 8  | 72  |
| 7 | 7 | 252 | 28 | 224 |

**关键观察**：在原始 layout 下，`[row=0..7, col=0]` 命中 banks 0, 8, 16, 24, 0, 8, 16, 24（4 个 bank 各 2 次 → **2-way conflict**）。swizzle 之后 col=0 的 bank id 序列（bank id = `(swizzled_offset >> 2) & 0x1f`）：

| row | raw | swizzled | bank |
|-----|-----|----------|------|
| 0 | 0   | 0   | 0  |
| 1 | 32  | 36  | 9  |
| 2 | 64  | 72  | 18 |
| 3 | 96  | 108 | 27 |
| 4 | 128 | 144 | 4  |
| 5 | 160 | 180 | 13 |
| 6 | 192 | 216 | 22 |
| 7 | 224 | 252 | 31 |

8 行 → 8 个不同 bank → **0 conflict** ✓

---

### Q3. WGMMA 用的 `Swizzle<3, 4, 3>`

WGMMA 操作数从 smem 装载到 register 时，每个 warp lane 装 16B。所以 `M = 4`（16B 原子）。
对 64x64 fp16 tile（128B-line × 64 行 = 8KB），fp16 stride 1 的连续 layout：
- `bit[3:0]` = byte_in_16B
- `bit[6:4]` = "16B chunk in line" 索引
- `bit[12:7]` = row[5:0]（64 行）

`Swizzle<3, 4, 3>` 取 `(offset >> 7) & 0x7` = row 低 3 bit，XOR 到 bit[6:4] = "16B chunk index"。

**问**：WGMMA 64x16 加载时，128 个线程同步从同一 row 读 8 个 16B chunk。在这种 swizzle 下，第 `tid` 个线程读 row `tid/8`、chunk `tid%8`。
- 没 swizzle：所有线程读 row 0 chunk 0..7，命中 8 个 bank pair → 0 conflict？不，8 chunks × 16B = 128B 分布在所有 32 banks，0 conflict。
- 有 swizzle：row r、chunk c 的实际地址是 `c XOR (r&7)`。当所有 lane 读同一 row 不同 chunk，一行内还是 0..7 的排列（被 row 重排），还是 0 conflict ✓

那 swizzle 是为了**什么场景**消 conflict？答：**MMA store 后再 load** 的转置访问，即 `r XOR c` 模式（不同 row 不同 chunk）。具体写出 lane 0..31 转置访问时的 chunk id 序列，证明它仍然是 0..31 的全排列 → 0 conflict。

**你的回答**：用一两句话解释为什么 WGMMA 必须配 swizzle smem。

---

## 验证（运行 ex05_swizzle_smem）

```bash
make study_stage1_w02_ex05_swizzle_smem -j
./study_stage1_w02_ex05_swizzle_smem            # 打印两种 layout 的访问结果

# bank conflict 计数（需要 ncu）
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
  ./study_stage1_w02_ex05_swizzle_smem
```

期望：
- 不带 swizzle 的转置 load → 32-way conflict（每次 load 多 31 个 cycle）
- 带 swizzle 的转置 load → 0 conflict
