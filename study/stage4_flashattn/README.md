# Stage 4 — FlashAttention（SM90 → SM100）

预计 5 周（W12–W15 Hopper + W21 Blackwell），约 75h。

> **硬件**：🟢 5060 Ti 主战（用 SM120 mainloop 跑 FA 框架；FP4 量化 FA 实验也可在本地）
> 🟡 H20（W13-W15 WGMMA FA 实测 + 88_hopper_fmha 基线对照）｜ 🔴 B200（W21 UMMA + TMEM FA 实测）

## 阶段目标

- 推得动 online softmax 的数学（Dao et al. 那张表）
- 自写 FA forward kernel，跑过正确性 + 达到 88_hopper_fmha 的 ≥ 80%
- 能写 FA backward 的 dQ/dK/dV 流程
- 能把 SM90 FA 迁移到 SM100（UMMA atom + TMEM accumulator）
- 加分项：在 5060 Ti 上做 fp4/fp6 量化 FA 实验

> 课程顺序：**SM90 FA 优化做透（W12-W15），再 SM100 迁移（W21）**。FA mental model 比 GEMM 更细（softmax / causal / pipeline），迁移时只换 atom + accumulator 位置。

## 周次

| 周 | 标题 | 主战硬件 | 输出 |
|----|------|---------|------|
| W12 | [FA 算法](week12_fa_algorithm/) | 🟢 5060 Ti / CPU | numpy/Python 写一份 online softmax 参考 |
| W13 | [FA fwd writeup](week13_fa_fwd_writeup/) | 🟢 5060 Ti + 🟡 H20 | `ex_fa_fwd_v1.cu` 正确性通过 |
| W14 | [FA fwd optimize](week14_fa_fwd_optimize/) | 🟢 5060 Ti + 🟡 H20 | v2 ≥ 80% 88_hopper_fmha；FP4 实验可选 |
| W15 | [FA bwd](week15_fa_bwd/) | 🟢 5060 Ti + 🟡 H20 | dQ/dK/dV 正确性 |
| W21 | [SM100 FA](week21_sm100_fa/) | 🟢 5060 Ti 读 + 🔴 B200 实测 | UMMA + TMEM 迁移版 |

## CHECKPOINT — 进入 Stage 5 前必过

### 综合练习
- `ex_fa_fwd_v2.cu` 在 H20 上：
  - shape `(B=4, H=32, S=4096, d=128)`，causal=True，FP16
  - 性能 ≥ 80% `examples/88_hopper_fmha`
  - 通过 PyTorch SDPA 正确性比对（rtol=1e-2）

### 口答 7 题
1. Online softmax 比朴素 softmax 多出来的 rescale 步骤数学上怎么推？
2. FA 的两次 GEMM（QK^T 和 PV）能不能用同一个 `TiledMma`？为什么？
3. 你的 K/V tile 加载顺序：先 K 再 V，还是 K/V 同时 prefetch？为什么？
4. Causal mask 在 tile 粒度怎么处理？对角线 tile 和上三角 tile 的差异？
5. d=128 vs d=96 在 H20 上的性能差距来源？
6. Persistent kernel 在 FA 里的收益主要来自哪？
7. FA bwd 的 dQ 和 dK/dV 为什么常拆成两个 kernel 分别算？
