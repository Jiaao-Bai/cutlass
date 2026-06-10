# Week 24 — EVT + kernel pattern 总结

预计 ~15h
> **硬件**：B200（SM100）

## 目标
- 读懂 EVT（Epilogue Visitor Tree）的编译期组合机制
- 产出 SM100 cheat sheet（MMA / Pipeline / Cluster / Epilogue 关键 API 与约束一览）
- 回顾 Stage 3-5 自写的全部 kernel，抽出**开源算子库的 kernel pattern 清单**——哪些骨架可共享、哪些必须每算子独立

## 读

1. `include/cutlass/epilogue/collective/sm100_epilogue_tma_warpspecialized.hpp` — epilogue 主体（含 TMEM→RMEM）
2. `include/cutlass/epilogue/fusion/sm100_callbacks_tma_warpspecialized.hpp` — EVT callback
3. `include/cutlass/epilogue/fusion/operations.hpp` — `LinearCombination` / `Bias` / `Activation` 等基础 op
4. `include/cutlass/epilogue/fusion/sm100_visitor_compute_tma_warpspecialized.hpp` / `sm100_visitor_store_tma_warpspecialized.hpp` — TreeVisitor 组合逻辑
5. 抽象机制源码：`include/cutlass/gemm/dispatch_policy.hpp` + `include/cute/atom/mma_traits_sm100.hpp` — tag dispatch / traits 怎么让上层代码不感知 atom 差异

## 写

- `exercises/notes_evt.md` — EVT 精读笔记：
  - TreeVisitor 的编译期组合机制（怎么把 Bias + ReLU + LinearCombination 拼成一棵树）
  - 一个具体例子：`alpha * C + beta * D + bias + relu` 的 visitor 树长什么样
  - EVT vs 手写 epilogue 的 trade-off——开源库里 epilogue fusion 走哪条路
- `exercises/sm100_cheat_sheet.md` — **SM100 cheat sheet**：MMA atom 选型（dense / blockscaled 各 kind）、pipeline 类型与 depth 约束（TMEM/smem 容量）、cluster 配置（1-SM/2-SM）、epilogue 路径（TMEM→RMEM→smem→TMA）、各项的文件位置与关键 API
- `exercises/kernel_patterns.md` — **kernel pattern 清单**（开源算子库的骨架设计，本 stage 核心产出）：
  - 对照自写的 dense GEMM / 量化 GEMM / FA fwd/bwd / GQA / grouped GEMM，列出公共零件：warp 角色分配、pipeline 协议、TMA 封装、TMEM 管理、epilogue
  - 哪些可以做成库内共享代码、哪些每算子必须特化、接口长什么样

## 自检
1. EVT 的 `Sm100TreeVisitor` 是编译期还是运行期组合？怎么保证零开销？
2. 如果你要加一个自定义 fusion op（比如 GELU），需要改哪些文件？
3. 同一套 collective 框架换 MMA atom（WGMMA / UMMA）上层代码不变——是哪一层（traits / dispatch policy）做的解耦？
4. TMEM→RMEM 这一步在哪个文件实现？它对 epilogue 的并行度（4 个 epilogue warp）有什么约束？
5. 你的 GEMM 和 FA kernel 的 pipeline 协议有多少代码真正可共享？差异的根源是什么（双 GEMM 依赖链 / softmax 插入点）？
6. 开源算子库如果只依赖 `include/cute/` 不依赖 `include/cutlass/`，会失去什么（pipeline 类 / EVT / tile scheduler）？哪些值得自己重写？
