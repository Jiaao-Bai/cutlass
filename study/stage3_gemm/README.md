# Stage 3 — 手写 GEMM（SM90 → SM100）

预计 5 周（W9–W12 Hopper + W13 Blackwell），约 75h。

> **硬件**：🟢 5060 Ti 主战（用 `MainloopSm120TmaWarpSpecialized` 在本地跑完整 TMA + WarpSpec + Cluster 框架）
> 🟡 H20（W10-W12 WGMMA 实测 + cuBLAS 基线）｜ 🔴 B200（W13 UMMA + TMEM 实测）

## 阶段目标

- 看懂 CUTLASS 3.x 的 5 层 API（device → kernel → collective → tiled atom → atom），能在每层定位文件
- 能从零写一个 warp-specialized GEMM kernel（不靠 `CollectiveBuilder`），跑出正确结果
- 能解释 Pingpong 与 Cooperative 的差异，知道 H20 上各自的适用场景
- 能把 SM90 WGMMA mainloop 迁移到 SM100 UMMA + TMEM（atom + accumulator 改动，框架 90% 不变）
- **CHECKPOINT**：自写 GEMM 在 H20 上 ≥ 70% cuBLAS；SM100 版本在 B200 上跑通 + 性能对照

> 课程顺序：**SM90 GEMM 优化到底（W9-W12）后再做 SM100 迁移（W13）**。WGMMA mental model 牢固后，SM100 的核心改动（UMMA atom + TMEM accumulator）只是 atom 替换。

## 周次

| 周 | 标题 | 主战硬件 | 输出 |
|----|------|---------|------|
| W9 | [3.x 分层设计](week09_3x_design/) | 🟢 5060 Ti | 跑通 example 48/49，画一份分层架构图 |
| W10 | [WarpSpec writeup](week10_warpspec_writeup/) | 🟢 5060 Ti SM120 mainloop + 🟡 H20 WGMMA | 自写 WarpSpec GEMM v1（正确即可）|
| W11 | [WarpSpec optimize](week11_warpspec_optimize/) | 同上 | v2：smem swizzle + pipeline depth 调参 |
| W12 | [Pingpong vs Cooperative](week12_pingpong_vs_coop/) | 同上 | v3 双版本，benchmark 对比 |
| W13 | [SM100 GEMM](week13_sm100_gemm/) | 🟢 5060 Ti 读 + 🔴 B200 实测 | UMMA + TMEM 迁移版，B200 性能数字 |

## CHECKPOINT — 进入 Stage 4 前必过

### 综合练习
你的 `ex_warpspec_gemm_v3.cu` 在 H20 上：
- M=N=K=4096（FP16），≥ 70% cuBLAS（约 ≥ 600 TFLOPS）
- M=N=K=8192，≥ 75% cuBLAS（约 ≥ 700 TFLOPS）
- 写 PERFORMANCE.md：你比 cuBLAS 慢的 30% 主要损失在哪（用 ncu Roofline 论证）

### 口答 8 题
1. CUTLASS 3.x 的 `Mainloop` 和 `CollectiveOp` 是同一个东西吗？
2. `CollectiveBuilder::CollectiveOp` 决定了哪些东西？哪些是用户必须自己定的？
3. 你的 v1（无优化）相比 cuBLAS 慢了多少倍？瓶颈是 WGMMA 发射还是 TMA 拉取？
4. v2 加了 swizzle 后 bank conflict 应该是 0，但你的 ncu 数据里如果不是 0，可能是什么原因？
5. pipeline depth 选 4 vs 8 vs 12，对 smem 占用、register pressure、性能各有什么影响？
6. Pingpong 在 M=8192/N=8192 vs M=128/N=8192 上哪个更快？为什么？
7. Cooperative 模式下两个 warpgroup 协同算同一个 tile，accumulator 是怎么 reduce 的？
8. 你的 epilogue 是直接寄存器→GMEM 还是经过 SMEM 二次组装？为什么？
