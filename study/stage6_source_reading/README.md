# Stage 6 — 源码精读

预计 3 周（W22–W24），约 45h。

> 你已经手写过 GEMM / FA / MoE，现在回头系统精读 CUTLASS 3.x SM100 源码，补全心智模型。
> 目标不是"再写一遍"，而是**读懂官方 SM100 实现的设计决策**，为 Stage 7 调优和开源算子库的工程化打基础。

## 阶段目标

- 能脱稿画出 CUTLASS 3.x 从 device → kernel → collective → atom 的完整数据流
- 能解释每层在哪个文件、关键 API 是什么、为什么这么分层
- 产出一份 SM100 cheat sheet（MMA / Pipeline / Cluster / Epilogue 关键 API 与约束一览）
- 产出开源算子库的 **kernel pattern 清单**：从已写的 GEMM/FA/MoE 中抽出可复用骨架（warp 角色分配 / pipeline 协议 / epilogue 路径），确定库里哪些代码可以共享、哪些必须每算子独立

## 周次

| 周 | 标题 | 精读内容 | 输出 |
|----|------|---------|------|
| W22 | [Pipeline + Collective](week22_pipeline_collective/) | `sm100_pipeline.hpp` 全文 + `sm100_mma_warpspecialized.hpp` collective mainloop | 带注释的架构图 + 设计决策笔记 |
| W23 | [Kernel + Scheduler](week23_kernel_scheduler/) | `sm100_gemm_tma_warpspecialized.hpp` 全文 + sm100 tile_scheduler 系列 + stream-K | annotated notes + 自己的 scheduler 变体 |
| W24 | [EVT + kernel pattern 总结](week24_evt_arch_compare/) | sm100 epilogue fusion 全套 + 回顾自写 kernel 抽公共骨架 | SM100 cheat sheet + kernel pattern 清单 |

## CHECKPOINT — 进入 Stage 7 前必过

### 综合产出

1. **架构图**（手画或 mermaid）：CUTLASS 3.x 五层 API 的数据流 + 每层对应文件
2. **SM100 cheat sheet**：MMA / Pipeline / Cluster / Epilogue 的关键 API、约束、选型表
3. **设计决策笔记**：至少 5 个"为什么官方这么做"的 insight（对比你自己手写时的选择）
4. **kernel pattern 清单**：开源算子库的骨架设计——GEMM/FA/MoE 三类 kernel 的公共零件（pipeline 协议 / warp 角色 / TMA 封装 / epilogue）与差异点

### 口答 6 题

1. `PipelineTmaUmmaAsync` 的 phase bit 翻转时机是什么？为什么不用 counter？
2. Collective mainloop 的 `load` 和 `mma` 方法是怎么被 kernel 层调度的？
3. SM100 kernel 里 1-SM / 2-SM UMMA 的 barrier 同步点在哪？画出时序图。
4. `TileSchedulerPersistent` 的 `get_current_work()` 返回什么？stream-K 模式下怎么拆 K？
5. EVT 的 `Sm100TreeVisitor` 怎么把多个 fusion op 组合成一棵树？编译期还是运行期？
6. 同一套 collective/kernel 框架怎么做到换 MMA atom（WGMMA / UMMA / mma.sync）上层代码不变？traits 机制在哪一层完成解耦？
