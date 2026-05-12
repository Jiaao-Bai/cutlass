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

### Q3. 真实 WGMMA 场景的 swizzle

WGMMA 用 `Swizzle<3, 4, 3>`（SW128）：M=4（16B 颗粒，`LDS.128`），B=3（8 chunks/row），S=3（row 字段对齐 bit 7）。bit 公式跟 Q1/Q2 同理，把"row[2:0] XOR chunk[2:0]"放到 16B-chunk 颗粒上做。

**为什么必须配 swizzle**：WGMMA mainloop 里 SMEM 既要被"行访问"（warp 沿 K 方向连续读 A operand）又要被"列访问 / 转置访问"（epilogue 把 D 矩阵存回 SMEM 后转置着 load 给下一阶段）。行访问天然 0 conflict，但**转置访问不 swizzle 就 32-way bank conflict**——swizzle 把列方向访问的地址打散到不同 bank 才让两种模式都 0 conflict。

> **本题不展开手算**——swizzle 的真正发力场景跟具体 kernel 强绑定：
> - **W9** 写 Hopper WarpSpec GEMM 时，会配合 `cute::GMMA::Layout_K_SW128_Atom` 实际跑 ncu，看带/不带 swizzle 的 `l1tex__data_bank_conflicts` 差几个数量级
> - **W12** 写 FlashAttention 时再处理 Q/K/V 的 SMEM swizzle 与 transpose 路径
>
> 这里只需要建立直觉：**M 决定颗粒、B 决定打散维度、S 决定 source 字段位置**。具体 bank 表手算放到能跑 kernel 对照的时候再做。

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
