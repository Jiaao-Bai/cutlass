# Week 9 — 手写 WarpSpec GEMM v1

预计 ~15h ｜ 目标硬件：H20

## 目标
- 自己组装 WarpSpec GEMM kernel，**不用** `CollectiveBuilder`
- 跑出正确结果（跟 CPU ref 对齐）
- 性能不重要，正确性优先

## 读
- `include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized.hpp` — 基础 WarpSpec 完整骨架（必读）
- `include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp` — `load()` / `mma()` 分离 mainloop
- `include/cutlass/pipeline/sm90_pipeline.hpp` — pipeline state 流转（W7 已读，复习）
- `examples/48_hopper_warp_specialized_gemm/` — 参照实现

## WarpSpec 模式

```
┌─────────────────────────────────────┐
│  CTA                                │
│  ┌──────────┐   ┌────────────────┐  │
│  │ TMA warp │   │  MMA warpgroup │  │
│  │(producer)│──▶│   (consumer)   │  │
│  └──────────┘   └────────────────┘  │
│       mbarrier 同步                 │
└─────────────────────────────────────┘
```

## 写
- `exercises/ex_warpspec_gemm_v1.cu` — v1 版本：
  - TileShape `<128,128,64>`、ClusterShape `<1,1,1>`
  - pipeline depth = 2（最小可工作）
  - 简单 epilogue：寄存器直接写回 GMEM（不带 alpha/beta）
  - 通过 CPU ref 检查（M=N=K=512）

需要自己实现的关键细节：
- smem 的 swizzle layout（先用 `Layout_K_SW128_Atom`，能跑就行）
- producer warp：循环发起 TMA，`producer_acquire/commit`
- consumer warpgroup：`consumer_wait`、发 WGMMA、`consumer_release`
- 累加器在 register 里，最终写回 GMEM

## 跑
```bash
make study_stage3_w09_ex_warpspec_gemm_v1 -j
./study_stage3_w09_ex_warpspec_gemm_v1 512 512 512   # 期望 PASSED
./study_stage3_w09_ex_warpspec_gemm_v1 4096 4096 4096
```

## 自检
1. 你的 producer warp 用了多少线程？为什么？
2. consumer warpgroup 一共多少线程？跟 WGMMA 的 128 线程怎么对应？
3. pipeline depth = 2 时，TMA 和 WGMMA 能重叠吗？什么时候重叠不上？
4. `consumer_wait` 之后才能发 WGMMA，但 WGMMA 是异步的——`consumer_release` 应该放在 `wgmma.commit_group` 之后还是 `wgmma.wait_group<0>` 之后？为什么？
5. 你的 v1 vs example 48 性能差多少？主要差距在哪？
