# Stage 6 — B200（SM100）增量

预计 3 周（W19–W21），约 45h。

> 这不是附录，而是正式阶段。学完 SM90 之后，B200 只需"增量"理解几个新概念，但每个都需要亲手跑一次。

## 阶段目标

- 看懂 TMEM（Tensor Memory，第 4 种内存层次）
- 看懂 UMMA（C 矩阵写到 TMEM 而非寄存器）
- 把 Stage 3 的 GEMM 移植到 SM100，跑通
- 把 Stage 4 的 FA 移植到 SM100（B200 才支持的 FA 变体）

## 周次

| 周 | 标题 | 输出 |
|----|------|------|
| W19 | [TMEM + UMMA](week19_tmem_umma/) | 跑通 SM100 minimal example，理解 tmem alloc |
| W20 | [SM100 GEMM](week20_sm100_gemm/) | 把 Stage 3 v3 移植到 SM100 |
| W21 | [SM100 FA](week21_sm100_fa/) | 跑通 fmha_bwd.py（已是 B200），自写 SM100 FA fwd |

## 与 SM90 的对照表（速查）

| 维度 | SM90 (Hopper) | SM100 (Blackwell) | 文件 |
|------|---------------|-------------------|------|
| MMA | WGMMA（C 在 RMEM） | UMMA（C 在 TMEM） | `mma_sm90.hpp` vs `mma_sm100_umma.hpp` |
| MMA 单位 | warpgroup (128 线程) | 1 thread issuing TCGen5 | — |
| Pipeline | producer/consumer 同 CTA | TMEM consumer 语义变化 | `sm90_pipeline.hpp` vs `sm100_pipeline.hpp` |
| TMA | 已有 | 同+ 2-CTA | `copy_sm90_tma.hpp` vs `copy_sm100_tma_*.hpp` |
| 调度 | persistent + cluster 2x1 | persistent + 2-CTA cluster | — |

底层 Layout 代数和 TiledMma/TiledCopy 抽象**完全一样**，只是 atom 换。

## CHECKPOINT — Stage 6 完成

### 综合练习
- `ex_sm100_gemm.cu`：在 B200 上 ≥ 70% cuBLAS（M=N=K=8192 FP16）
- `ex_sm100_fa_fwd.cu`：在 B200 上正确性通过（性能可以慢）
- 写 `B200_VS_H20.md`：同一份算法在两种硬件上的差异、迁移注意点

### 口答 6 题
1. TMEM 跟 SMEM 在地址空间、容量、访问粒度上各有什么差别？
2. UMMA 的 C 矩阵在 TMEM，怎么把累加结果搬到 epilogue 的寄存器？
3. SM100 的"2-CTA cluster"跟 SM90 的 cluster 在 MMA 协同上有什么不同？
4. SM100 pipeline 引入 TMEM consumer，`consumer_release` 的时机比 SM90 晚还是早？为什么？
5. 同样的 algorithm，B200 比 H20 提速主要来自哪？
6. SM100 上 sparse MMA（结构化稀疏）怎么用？什么场景划算？
