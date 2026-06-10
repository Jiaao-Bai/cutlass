# Stage 3 — 手写 GEMM（SM100）

预计 5 周（W9 3.x 设计 + W10–W12 dense GEMM + W13 量化 GEMM），约 75h。硬件：B200（SM100）。

## 阶段目标

- 看懂 CUTLASS 3.x 的 5 层 API（device → kernel → collective → tiled atom → atom），能在每层定位文件
- 能从零写一个 warp-specialized SM100 GEMM kernel（不靠 `CollectiveBuilder`），用 UMMA + TMEM 累加器，跑出正确结果
- 能解释 1-SM UMMA vs 2-SM UMMA（`cta_group::1` vs `::2`）的差异，知道各自的 TMEM/cluster/吞吐权衡
- 能写 blockscaled 量化 GEMM（NVFP4/FP8 + microscaling）——LLM 推理的主力 GEMM 形态
- **CHECKPOINT**：自写 SM100 GEMM 在 B200 上 ≥ 80% cuBLAS（≈ 70% tensor core 利用率，口径见 `study/README.md`）

> 本阶段产出的 dense GEMM + 量化 GEMM 是未来开源 CuTe 算子库的第一批 kernel——写成扁平、可独立搬走的形态（见 THINKING O33）。

## 周次

| 周 | 标题 | 输出 |
|----|------|------|
| W9 | [3.x 分层设计](week09_3x_design/) | 跑通 example 70/71，画一份分层架构图 |
| W10 | [WarpSpec writeup](week10_warpspec_writeup/) | 自写 WarpSpec GEMM v1（正确即可）|
| W11 | [WarpSpec optimize](week11_warpspec_optimize/) | v2：smem swizzle + pipeline depth 调参 |
| W12 | [1-SM vs 2-SM UMMA](week12_pingpong_vs_coop/) | v3 双版本（1SM / 2SM），benchmark 对比 |
| W13 | [Blockscaled 量化 GEMM](week13_sm100_gemm/) | NVFP4/FP8 blockscaled GEMM，对比 dense v3 与 cuBLAS FP8 |

## CHECKPOINT — 进入 Stage 4 前必过

### 综合练习
你的 `ex_warpspec_gemm_v3.cu`（SM100 版）在 B200 上：
- M=N=K=4096（FP16），≥ 80% cuBLAS
- M=N=K=8192，≥ 85% cuBLAS
- ncu 确认 `sm__pipe_tensor_cycles_active` ≥ 70% of peak
- 写 PERFORMANCE.md：你比 cuBLAS 慢的部分主要损失在哪（用 ncu Roofline 论证）

### 口答 8 题
1. CUTLASS 3.x 的 `Mainloop` 和 `CollectiveOp` 是同一个东西吗？
2. `CollectiveBuilder::CollectiveOp` 决定了哪些东西？哪些是用户必须自己定的？
3. 你的 v1（无优化）相比 cuBLAS 慢了多少倍？瓶颈是 UMMA 发射还是 TMA 拉取？
4. v2 加了 swizzle 后 bank conflict 应该是 0，但你的 ncu 数据里如果不是 0，可能是什么原因？
5. pipeline depth 选 4 vs 8 vs 12，对 smem 占用、TMEM 占用、性能各有什么影响？
6. 1-SM UMMA vs 2-SM UMMA 在 M=8192/N=8192 vs M=128/N=8192 上哪个更快？为什么？
7. 2-SM UMMA（`cta_group::2`）模式下两个 CTA 协同算同一个 tile，TMEM 累加器是怎么分配/合并的？
8. NVFP4 blockscaled GEMM 的 scale factor 是怎么进 UMMA 的（`tcgen05.mma.kind::nvf4` 的 SF 操作数路径）？
