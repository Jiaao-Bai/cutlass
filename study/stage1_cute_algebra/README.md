# Stage 1 — CuTe 张量代数

预计 4 周（W1–W4），约 60h。

## 阶段目标

- 看到 `Layout<Shape, Stride>` 能用纸笔算出任意 `crd2idx`
- 能解释 `coalesce`、`composition`、`complement`、`swizzle` 的几何含义
- 能从 `MMA_Atom` / `Copy_Atom` 出发，推导出 `TiledMma` / `TiledCopy` 给每个线程的 partition
- 能完全脱离 `cutlass::gemm`，只用 CuTe 原语手写正确的 FP16 GEMM

## 周次

| 周 | 标题 | 输出 |
|----|------|------|
| W1 | [Layout basics](week01_layout_basics/) | `crd2idx` 手算练习；layout `print` 工具 |
| W2 | [Composition + Swizzle](week02_composition_swizzle/) | `ex06_hgemm_naive`（已完成）+ swizzle 验证 |
| W3 | [TiledMMA](week03_tiledmma/) | 能解释 `sgemm_sm80.cu` 里每个 shape |
| W4 | [TiledCopy](week04_tiledcopy/) | 自写 async copy + bank conflict 实测 |

## 推荐阅读顺序

```
include/cute/stride.hpp:47        crd2idx 4 条规则
  ↓
include/cute/layout.hpp:95         Layout struct 与 EBO
  ↓
include/cute/layout.hpp:768        coalesce
  ↓
include/cute/layout.hpp:1021       composition
  ↓
include/cute/layout.hpp:1164       complement
  ↓
include/cute/swizzle.hpp:42        Swizzle<B,M,S>
  ↓
include/cute/atom/mma_atom.hpp:42      MMA_Atom 结构
  ↓
include/cute/atom/mma_atom.hpp:250     TiledMMA::thrfrg_C/A/B
  ↓
include/cute/atom/mma_atom.hpp:460     ThrMMA::partition_C/A/B
  ↓
include/cute/algorithm/gemm.hpp:100    5 层 dispatch
  ↓
include/cute/algorithm/gemm.hpp:260    serpentine 寄存器复用
```

## CHECKPOINT — 进入 Stage 2 前必过

### 综合练习
读懂 `examples/cute/tutorial/sgemm_sm80.cu`，写一个变体：把 thread tile 从 `2x2` 改成 `4x2`，跑出正确结果，并解释每个 `TiledMma` / `TiledCopy` shape 的来源。

放置位置：`stage1_cute_algebra/week04_tiledcopy/exercises/ex_sgemm_sm80_variant.cu`。

### 口答 5 题（自己说出来才算）
1. `crd2idx((2,3), (4,5), (1,4))` 等于多少？过程？
2. 为什么 `Layout` 继承 `cute::tuple`（而不是组合）？EBO 解决了什么？
3. `coalesce` 的"merge"规则在什么条件下成立？为什么必须从右到左做？
4. `composition(L1, L2)` 的代数含义？为什么需要 `divmod`？
5. `Swizzle<3,4,3>` 在 SMEM 上的物理作用？为什么能消 bank conflict？
