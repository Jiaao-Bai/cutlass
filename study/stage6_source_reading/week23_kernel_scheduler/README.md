# Week 23 — Kernel + Scheduler 精读

预计 ~15h
> **硬件**：🟢 5060 Ti（纯读源码 + 写笔记）

## 目标
- 逐行读完 pingpong kernel（947 行），画出两个 warpgroup 的时序图
- 读 cooperative kernel，理解 accumulator reduce 机制
- 读 tile_scheduler 系列，理解 persistent / stream-K / grouped 三种调度策略

## 读

1. `include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp`（947 行）— 全文
   - 重点：两个 warpgroup 的 barrier 交替、accumulator 不被覆盖的机制
2. `include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp` — 全文
   - 重点：两个 warpgroup 协作算同一 tile，accumulator 怎么 reduce
3. `include/cutlass/gemm/kernel/sm90_tile_scheduler.hpp` — persistent scheduler
   - 重点：`get_current_work()` 的动态分配逻辑
4. `include/cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp` — stream-K
   - 重点：K 维度怎么拆、partial tile 怎么 reduce
5. `include/cutlass/gemm/kernel/sm90_tile_scheduler_group.hpp` — grouped GEMM
   - 重点：多 problem 的 tile id 拍平 + 查表

## 写

- `exercises/notes_pingpong_kernel.md` — Pingpong 精读笔记：
  - 两个 warpgroup 的完整时序图（标注 barrier wait/arrive）
  - accumulator 寄存器生命周期
  - 跟你 W12 手写版的对比：官方多做了什么？
- `exercises/notes_cooperative_kernel.md` — Cooperative 精读笔记：
  - accumulator reduce 的具体实现（SMEM 中转？还是 shuffle？）
  - 跟 pingpong 的适用场景对比
- `exercises/notes_tile_scheduler.md` — Scheduler 精读笔记：
  - persistent / stream-K / grouped 三种策略的决策树
  - `get_current_work()` 的伪代码
  - stream-K 的 partial tile reduce 机制

## 自检
1. Pingpong 里两个 warpgroup 切换时，前一个的 accumulator 在哪？为什么不会被覆盖？
2. Cooperative 模式下 accumulator reduce 用的是什么同步原语？
3. Persistent scheduler 的 `get_current_work()` 是 atomic 还是 deterministic？
4. Stream-K 把 K 拆给多个 SM，partial result 怎么汇总？用 atomic add 还是 reduction kernel？
5. Grouped GEMM scheduler 的 tile id → (problem_idx, tile_in_problem) 是 O(1) 还是 O(log n)？
