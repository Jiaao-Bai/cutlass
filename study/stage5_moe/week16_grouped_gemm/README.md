# Week 16 — Grouped GEMM 基础

预计 ~15h
> **硬件**：🟢 5060 Ti（用 `MainloopSm120ArrayTmaWarpSpecialized` 跑 grouped GEMM 框架）｜ 🟡 H20（WGMMA 实测 grouped 性能）

## 目标
- 看懂 grouped GEMM 与普通 GEMM 的调度差异
- 跑通 `examples/57_hopper_grouped_gemm`
- 自写一个 minimal grouped GEMM（5 个不等长 problem）

## 读

> Tile scheduler 是 cutlass/ 必读区的最后一块；详见 [cutlass_reading_strategy.md §4](../../cutlass_reading_strategy.md#4-tile-scheduler--includecutlassgemmkernel)。

- `examples/57_hopper_grouped_gemm/57_hopper_grouped_gemm.cu`
- `include/cutlass/gemm/kernel/sm90_tile_scheduler_group.hpp` — Group tile scheduler
- `include/cutlass/gemm/kernel/sm90_gemm_array_tma_warpspecialized*.hpp` — array variants

## Grouped GEMM 核心

每个 problem 有自己的 (M, N, K)，tile scheduler 需要：
1. 把所有 problem 的 tile 拍平到一个全局 tile id 空间
2. 每个 SM 拿到 tile id 后查表知道属于哪个 problem 以及 problem 内坐标
3. 切换 problem 时需要换 ptr / stride

## 写
- `exercises/ex22_run_57.cu` — 跑通 example 57 with M=[128,256,128,512,64], 同 N=K=4096
- `exercises/ex23_grouped_gemm_minimal.cu` — 自己写一个最小 grouped GEMM，复用 Stage 3 的 WarpSpec 骨架，加一层 problem dispatch

## 跑
```bash
make study_stage5_w16_ex22_run_57 -j && ./study_stage5_w16_ex22_run_57
make study_stage5_w16_ex23_grouped_gemm_minimal -j && ./study_stage5_w16_ex23_grouped_gemm_minimal
```

## 自检
1. Grouped GEMM 的 tile id → (problem_idx, tile_in_problem) 怎么算？需要 binary search 吗？
2. 每个 problem 的 ptr / stride 是 host array 还是从 launch param 传？
3. M 列表 [128, 256, 128, 512, 64] 总共多少 tile？SM 利用率怎么估计？
4. 为什么 example 57 用 array variant 的 KernelSchedule？跟 Pingpong 区别？
5. 如果某个 problem 的 M 极小（比如 16），tile scheduler 怎么避免发空 tile？
