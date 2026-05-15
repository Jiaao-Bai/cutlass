# Week 8 — CUTLASS 3.x 分层设计

预计 ~15h
> **硬件**：🟢 5060 Ti（读源码 + 静态编译，主要是理解 3.x 分层，跨 SM90/SM100/SM120 共通）

## 目标
- 能画出 device → kernel → collective → tiled atom → atom 的分层图，每层指向具体文件
- 跑通 example 48 和 49，能解释每个模板参数
- 能说出 `CollectiveBuilder` 帮你省掉了哪些手写代码

## 读

> 本周开始接触 `include/cutlass/gemm/`；先看 [cutlass_reading_strategy.md](../../cutlass_reading_strategy.md) 知道哪些必读、哪些跳过（约 38 万行 2.x 遗产可以直接跳）。

- `media/docs/cpp/gemm_api_3x.md` — **必读**，5 层 API 设计文档
- `media/docs/cpp/cutlass_3x_design.md` — 设计哲学
- `include/cutlass/gemm/kernel/gemm_universal.hpp` — 3.x kernel 骨架
- `include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp:1-200` — mainloop 入口
- `examples/48_hopper_warp_specialized_gemm/48_hopper_warp_specialized_gemm.cu` — 手动组装版
- `examples/49_hopper_gemm_with_collective_builder/49_hopper_gemm_with_collective_builder.cu` — Builder 简化版

## 写
- `exercises/ex17_layered_diagram.md` — 自己画一张分层图（mermaid 或 ASCII），标出每层关键类与文件
- `exercises/ex18_run_48.cu` — 抄并跑通 example 48，把每个模板参数（TileShape、ClusterShape、KernelSchedule、EpilogueSchedule）的含义注释出来
- `exercises/ex19_run_49.cu` — 跑通 example 49，对比 48，列出 Builder 推断了哪些类型

## 跑
```bash
make study_stage3_w08_ex18_run_48 -j && ./study_stage3_w08_ex18_run_48
make study_stage3_w08_ex19_run_49 -j && ./study_stage3_w08_ex19_run_49
```

## 自检
1. `GemmUniversalAdapter` 在哪一层？为什么需要它？
2. `KernelSchedule::Pingpong / Cooperative / Auto` 是在编译期还是运行期切换？
3. `TileShape_MNK = Shape<_128,_128,_64>` 表示什么？为什么 N 维度多用 128 而不是 256？
4. `ClusterShape_MNK = Shape<_2,_1,_1>` 启用了什么？要求架构是？
5. Builder 选不好时的 fallback 是什么？怎么强制指定？
