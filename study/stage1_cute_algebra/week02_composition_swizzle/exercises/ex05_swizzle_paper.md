# Ex05 — `Swizzle<B, M, S>` bit math 手算

锚点：`include/cute/swizzle.hpp:42-130`

---

## 速记

```cpp
Swizzle<B, M, S>::operator()(offset)
  = offset XOR ((offset >> S) & ((1<<B) - 1)) << M
```

**直觉**：
- `M` = 一个"原子单位"占多少 bit（例如 fp32 = 4B = 2 bit、128B = 7 bit）
- `B` = 多少 bit 参与 swizzle（XOR 几位）
- `S` = "外圈" bit 距 "内圈"（被 XOR 的）有多少位 shift

效果是把高位的 `B` 个 bit XOR 到 `[M, M+B)` 这段位置上 —— 让"列号"打乱"行号低位"，使同一个 bank 不会被同 row 不同 col 命中。

参数三元组到 byte 的速记表（fp32 = 4B/elem）：

| Swizzle | 含义 |
|---------|------|
| `<3, 2, 3>` | 8x8 fp32 tile：3 bit 列 XOR 到 3 bit 行（消 32-way smem 转置 conflict） |
| `<2, 4, 3>` | 16-byte 原子（如 ldmatrix.x4 = 16B），3 bit 行 swizzle |
| `<3, 4, 3>` | 16-byte 原子，128B-line / fp16 用，最常见的 WGMMA smem swizzle |

---

## 题目

### Q1. 解 `Swizzle<3, 2, 3>` 在 fp32 smem 上的位置

smem 32x32 fp32（即 32x128B = 32 行 × 32 bank）。元素 `[row, col]` 的线性 byte offset = `row * 128 + col * 4`。

bit 布局（offset 共 12 bit）：
```
bit 11..7 = row[4:0]      // 5 bit
bit  6..2 = col[4:0]      // 5 bit
bit  1..0 = byte_in_elem  // M=2
```

`Swizzle<3, 2, 3>` 取 `(offset >> 3) & 0b111` = `bit[5:3]` = **col 的低 3 bit**，XOR 到 `bit[4:2]` = ?

**你的答案**：被 XOR 的目标 bit 是 offset 的第 ___ 到第 ___ 位，对应物理意义是：col 的低 3 bit 与 ___ 的低 3 bit 异或。

---

### Q2. 列出 `Swizzle<3, 2, 3>(addr)` 的具体值

填表（row, col 都是 0..7 演示，足够覆盖 swizzle 一个周期）：

| row | col | raw offset (bytes) | XOR mask | swizzled offset |
|-----|-----|---------------------|----------|-----------------|
| 0 | 0 | 0 | 0 | 0 |
| 0 | 1 | 4 | ? | ? |
| 0 | 2 | 8 | ? | ? |
| 1 | 0 | 128 | ? | ? |
| 1 | 1 | 132 | ? | ? |
| 2 | 0 | 256 | ? | ? |
| 7 | 7 | 924 | ? | ? |

**关键观察**：在原始 layout 下，`[row=tid, col=0]` 都映射到 bank 0（32-way conflict）。swizzle 之后呢？写出 row 0..7 下 col=0 的 bank id（bank id = `(swizzled_offset >> 2) & 0x1f`）。

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
