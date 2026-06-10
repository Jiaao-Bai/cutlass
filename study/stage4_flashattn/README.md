# Stage 4 — FlashAttention（SM100）

预计 5 周（W14–W17 FA 主线 + W18 GQA/decode 变体），约 75h。硬件：B200（SM100）。

## 阶段目标

- 推得动 online softmax 的数学（Dao et al. 那张表）
- 自写 FA forward kernel（SM100 UMMA + TMEM），跑过正确性 + 达到 `77_blackwell_fmha` 的 ≥ 80%
- 能写 FA backward 的 dQ/dK/dV 流程
- 能写 GQA / decode 低延迟变体——LLM 推理的真实 attention 形态

> 本阶段产出（FA fwd/bwd + GQA/decode）直接构成开源算子库的 attention 家族。

## 周次

| 周 | 标题 | 输出 |
|----|------|------|
| W14 | [FA 算法](week14_fa_algorithm/) | numpy/Python 写一份 online softmax 参考 |
| W15 | [FA fwd writeup](week15_fa_fwd_writeup/) | `ex_fa_fwd_v1.cu` 正确性通过 |
| W16 | [FA fwd optimize](week16_fa_fwd_optimize/) | v2 ≥ 80% 77_blackwell_fmha |
| W17 | [FA bwd](week17_fa_bwd/) | dQ/dK/dV 正确性 |
| W18 | [GQA / decode 变体](week18_sm100_fa/) | GQA + decode 低延迟变体，B200 实测 |

## CHECKPOINT — 进入 Stage 5 前必过

### 综合练习
- `ex_fa_fwd_v2.cu` 在 B200 上：
  - shape `(B=4, H=32, S=4096, d=128)`，causal=True，FP16
  - 性能 ≥ 80% `examples/77_blackwell_fmha`（≈ 70% tensor core 利用率口径）
  - 通过 PyTorch SDPA 正确性比对（rtol=1e-2）

### 口答 7 题
1. Online softmax 比朴素 softmax 多出来的 rescale 步骤数学上怎么推？
2. FA 的两次 GEMM（QK^T 和 PV）能不能用同一个 `TiledMma`？为什么？
3. 你的 K/V tile 加载顺序：先 K 再 V，还是 K/V 同时 prefetch？为什么？
4. Causal mask 在 tile 粒度怎么处理？对角线 tile 和上三角 tile 的差异？
5. d=128 vs d=96 在 B200 上的性能差距来源？
6. Persistent kernel 在 FA 里的收益主要来自哪？
7. FA bwd 的 dQ 和 dK/dV 为什么常拆成两个 kernel 分别算？
