# Week 24 — EVT + 架构对比

预计 ~15h
> **硬件**：🟢 5060 Ti（读源码 + 写 cheat sheet）

## 目标
- 读懂 EVT（Epilogue Visitor Tree）的编译期组合机制
- 产出 SM90 / SM100 / SM120 三架构 cheat sheet
- 能解释框架怎么做到"上层代码不变，底层 atom 换架构"

## 读

1. `include/cutlass/epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp` — epilogue 主体
2. `include/cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp` — EVT callback
3. `include/cutlass/epilogue/fusion/operations.hpp` — `LinearCombination` / `Bias` / `Activation` 等基础 op
4. `include/cutlass/epilogue/fusion/sm90_visitor_*.hpp` — TreeVisitor 组合逻辑
5. 架构对比源码：
   - `include/cute/arch/mma_sm90.hpp` vs `mma_sm100_umma.hpp` vs `mma_sm120.hpp`
   - `include/cutlass/gemm/dispatch_policy.hpp` — 看 SM90/SM100/SM120 的 mainloop policy 差异
   - `include/cute/atom/mma_traits_sm90.hpp` vs `mma_traits_sm100.hpp`

## 写

- `exercises/notes_evt.md` — EVT 精读笔记：
  - TreeVisitor 的编译期组合机制（怎么把 Bias + ReLU + LinearCombination 拼成一棵树）
  - 一个具体例子：`alpha * C + beta * D + bias + relu` 的 visitor 树长什么样
  - EVT vs 手写 epilogue 的 trade-off
- `exercises/arch_cheat_sheet.md` — **三架构 cheat sheet**（本 stage 核心产出）：

  | 维度 | SM90 (H20) | SM100 (B200) | SM120 (5060 Ti) |
  |------|-----------|-------------|----------------|
  | MMA 指令 | WGMMA | UMMA (tcgen05.mma) | mma.sync (fp4/fp6/fp8) |
  | 发射粒度 | warpgroup (128 thr) | 1 thread | warp (32 thr) |
  | Accumulator | RMEM | TMEM | RMEM |
  | TMA | ✅ | ✅ | ✅ |
  | Cluster | ✅ | ✅ (含 2-SM 配对) | ✅ (SM90 风格) |
  | WarpSpec mainloop | ✅ | ✅ | ✅ |
  | Pipeline | PipelineTmaAsync | PipelineTmaAsyncSm100 | PipelineTmaAsync (SM90 复用) |
  | Epilogue | TMA writeback | TMA writeback + TMEM→RMEM | TMA writeback |
  | 独有能力 | — | TMEM, 2-SM paired cluster | fp4/fp6 块缩放 mma.sync |

- `exercises/notes_arch_abstraction.md` — 框架抽象笔记：
  - CUTLASS 怎么做到 collective mainloop 代码不变、底层 atom 换架构
  - dispatch_policy 的 tag dispatch 机制
  - Builder 怎么根据 arch tag 选 atom + pipeline + scheduler

## 自检
1. EVT 的 `Sm90TreeVisitor` 是编译期还是运行期组合？怎么保证零开销？
2. 如果你要加一个自定义 fusion op（比如 GELU），需要改哪些文件？
3. SM120 复用 SM90 的 collective mainloop，但 MMA atom 不同——框架哪一层做的 dispatch？
4. SM100 的 epilogue 比 SM90 多一步 TMEM→RMEM，这一步在哪个文件实现？
5. 三架构中，哪个的 pipeline depth 选择空间最大？为什么？
6. `dispatch_policy.hpp` 里 SM90/SM100/SM120 的 mainloop policy 有什么共同基类？
