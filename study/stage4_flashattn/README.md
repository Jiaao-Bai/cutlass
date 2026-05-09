# Stage 4 — FlashAttention

预计 4 周（W12–W15），约 60h。

## 阶段目标

- 推得动 online softmax 的数学（Dao et al. 那张表）
- 自写 FA forward kernel，跑过正确性 + 达到 88_hopper_fmha 的 ≥ 80%
- 能写 FA backward 的 dQ/dK/dV 流程
- 能解释 FA 在 H20 vs B200 上的差异（B200 部分留到 Stage 6）

## 周次

| 周 | 标题 | 输出 |
|----|------|------|
| W12 | [FA 算法](week12_fa_algorithm/) | numpy/Python 写一份 online softmax 参考 |
| W13 | [FA fwd writeup](week13_fa_fwd_writeup/) | `ex_fa_fwd_v1.cu` 正确性通过 |
| W14 | [FA fwd optimize](week14_fa_fwd_optimize/) | v2 ≥ 80% 88_hopper_fmha |
| W15 | [FA bwd](week15_fa_bwd/) | dQ/dK/dV 正确性 |

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
