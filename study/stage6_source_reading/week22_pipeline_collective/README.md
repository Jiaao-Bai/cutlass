# Week 22 — Pipeline + Collective 精读

预计 ~15h
> **硬件**：🟢 5060 Ti（纯读源码 + 写笔记，不需要跑 kernel）

## 目标
- 逐行读完 `sm90_pipeline.hpp`（1388 行），理解 phase bit / arrive / wait 的完整状态机
- 读 `sm100_pipeline.hpp`（1328 行）增量，标注跟 SM90 的差异
- 读 collective mainloop（`sm90_mma_tma_gmma_ss_warpspecialized.hpp` 584 行），理解 load/mma 分工

## 读

1. `include/cutlass/pipeline/sm90_pipeline.hpp` — 全文精读
   - 重点：`PipelineTmaAsync` 的 `producer_acquire` / `producer_commit` / `consumer_wait` / `consumer_release`
   - 画出 phase bit 状态转移图
2. `include/cutlass/pipeline/sm100_pipeline.hpp` — 对照 SM90 读增量
   - 标注：哪些 API 签名变了、哪些语义变了、哪些是新增
3. `include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp` — 全文
   - 重点：`load()` 和 `mma()` 方法的循环结构、pipeline state 怎么传递
4. `include/cutlass/arch/barrier.h` — mbarrier 底层 wrapper（辅助理解 pipeline）

## 写

- `exercises/notes_pipeline_sm90.md` — SM90 pipeline 精读笔记：
  - phase bit 状态机图
  - producer/consumer 时序图（标注 mbarrier 操作）
  - 关键设计决策（为什么用 phase bit 不用 counter？为什么 depth 一般选 4/8？）
- `exercises/notes_pipeline_sm100_diff.md` — SM100 vs SM90 差异清单
- `exercises/notes_collective_mainloop.md` — collective mainloop 架构笔记：
  - load/mma 分工图
  - pipeline state 在 load 和 mma 之间怎么传递
  - warp specialization 的 barrier 同步点

## 自检
1. `PipelineTmaAsync` 的 `producer_acquire` 内部做了什么？为什么需要等 consumer release？
2. phase bit 翻转的时机是什么？如果 depth=4，phase 序列是什么？
3. collective mainloop 的 `load()` 方法里，TMA 发射和 mbarrier arrive 的顺序能不能反？
4. SM100 pipeline 相比 SM90 多了什么？少了什么？
5. 如果 pipeline depth 从 4 改成 8，smem 占用翻倍，但性能不一定翻倍——为什么？
