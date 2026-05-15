# 思考 / 踩坑 / 方法论

> 跨 stage 的概念笔记、硬件约束、方法论。下一个接手仓库的 agent **务必读完再开始**——避免重复同样的错误。

---

## 一、硬件 / 编译

- **拥有的硬件**：**5060 Ti 16GB (SM120, Blackwell consumer)** —— 日常学习主战
- **租赁硬件**：H20 (SM90) ~$1-3/hr、B200 (SM100) ~$5-15/hr —— 只在需要 WGMMA / UMMA 真实跑通 + ncu 性能数字时短 sprint 用
- **目标**：手写 + 极致优化 Hopper/Blackwell GEMM、FlashAttention、Sparse MoE

**5060 Ti (SM120) 真实能力**（详见 O20）：
- **能跑**：W1-W4（SM80 atom）、W6 TMA、W7 Pipeline+Cluster、W9-W12 + W14-W21 的 WarpSpec 框架（用 `MainloopSm120TmaWarpSpecialized` 代替 WGMMA mainloop）、fp4/fp6 块缩放 mma.sync 量化实验
- **跑不了**：W5 WGMMA atom（要 H20）、W8/W13/W18 UMMA + TMEM（要 B200）、2-SM 配对 cluster（要 B200）

**Stage 内部 SM 串行**：每个 stage 先 SM90 优化做透、再 SM100 迁移，不并行。Stage 2 是例外——一次性消化 SM90 + SM100 prim 比拆两次学好（UMMA 文档大量"differs from WGMMA in X"）。

**编译命令**：`cmake -DCUTLASS_ENABLE_STUDY=ON -DCUTLASS_NVCC_ARCHS=120 ..`（5060 Ti）/ `90` (H20) / `100a` (B200) / `120;90;100a` (跨架构通用)

---

## 二、核心概念笔记

### O1. coalesce 的 merge 条件方向

`bw_coalesce` (`layout.hpp:780-814`) 的实际 merge 规则是 **`d_next == s_prev × d_prev`**（accumulator's stride 等于 incoming dim 的 size × stride），从右往左扫。
- 实战效果：row-major 永远不能 coalesce 成 1D（除非某维 size=1），col-major 才友好。
- `ex03_coalesce.cu` 文件头注释里写的 "`s0:s1*d1   s1:d1  =>  merge`" 方向**反了**，以源码为准。

### O2. bank conflict "N-way" 的定义

"N-way conflict" = **同一 cycle 撞同一 bank 的最大线程数**。
- 8x8 fp32 col=0 访问：8 lanes 命中 banks {0, 8, 16, 24} 各 2 次 → **2-way conflict**（不是 8-way）。
- 32x32 fp32 col=0 访问：32 lanes 全打 bank 0 → 32-way。

### O3. SW32/SW64/SW128 不是方 tile

CUTLASS 的标准 swizzle 都是 **8 行 × N chunks of 16B** 的非方 tile：
- SW128 = `Swizzle<3,4,3>` = 8 行 × 8 chunks（128B/row）
- SW64 = `Swizzle<2,4,3>` = 8 行 × **4** chunks（64B/row）—— **不是 4×4 tile**
- SW32 = `Swizzle<1,4,3>` = 8 行 × 2 chunks（32B/row）

S=3 是因为 Hopper 约定 **row 字段对齐到 offset 的 bit 7+**（外层 stride 撑出来），公式跨 SW 系列统一。

**B+M 必须等于 `log2(单条指令实际触及字节数)`** —— Swizzle<3,2,3> 配 32×32 fp32 (128B/row) 只能从 32-way 降到 16-way，要消干净得 Swizzle<5,2,5>（B+M=7 覆盖整条 bank-row）。

### O4. CuTe 的 B 写作 `(N, K)`，不是 BLAS 的 `(K, N)`

权威出处：`media/docs/cpp/cute/0x_gemm_tutorial.md:80-81`：
> "CuTe follows the convention that the semantics of matrix modes is **`(M,K)` for `A`, `(N,K)` for `B`, and `(M,N)` for `C`**."

理由：让 A、B 都把 K 放在最后一维，K-reduce 循环写起来对称。**讲解时永远用 CuTe 约定**——除非明确在讨论 BLAS API。

### O5. CuTe 的 "X-major" 比 BLAS T/N 更准

`0x_gemm_tutorial.md:103`：
> "Instead of row-major or column-major (or Transposed and Not-Transposed), we have found it much more convenient to say that a matrix is **'M-major'**, **'N-major'**, or **'K-major'** based on which mode has stride-1."

BLAS↔CuTe 对照表（原文 line 107-112）：

| BLAS | A Majorness | A Layout | B Majorness | B Layout |
|------|-------------|----------|-------------|----------|
| NT   | M-major     | `(M,K):(1,ldA)` | N-major | `(N,K):(1,ldB)` |
| **TN** | **K-major** | `(M,K):(ldA,1)` | **K-major** | `(N,K):(ldB,1)` |
| NN   | M-major     | `(M,K):(1,ldA)` | K-major | `(N,K):(ldB,1)` |
| TT   | K-major     | `(M,K):(ldA,1)` | N-major | `(N,K):(1,ldB)` |

**TN = A、B 都 K-major** = 跟 MMA-TN 硬件天然一致。

### O6. Shape 写法顺序不重要，stride 才是真理

**Layout 物理顺序由 stride 决定，shape mode 写在前还是后只是命名**。`(M,K):(K,1)` 和 `(K,M):(1,K)` 指向同一段内存。CuTe 钉死 `(M,K), (N,K)` 是为了**库代码用统一标签**，物理上你随便换都行。

### O7. MMA-TN vs GEMM-TN 是两个层级

- **MMA-TN**（硬件层）：`mma.sync.aligned.m16n8k16.row.col` 强制 A、B K-contig。硬件只支持 TN，**没得选**。
- **GEMM-TN**（用户层）：用户传入的 A、B 在 GMEM 的 stride 配置。NN/NT/TN/TT 都可选，TN 最常用因为跟 MMA-TN 天然一致、无需 SMEM 转置。

### O8. 自下而上读源码

源码该 **PTX → Traits → Atom → Tiled → 算法** 这个方向读。先看 wrapper 再看里面 wrap 啥本末倒置——读者看到 `ALayout`/`ValLayoutSrc` 这些字段时还不知道是啥的容器。

### O9. 反对"空对空"练习

脱离实战场景的纯数学 / bit 推导练习无价值。**bank conflict / ncu 实测**之类的东西**放到能跑真实 kernel 的阶段做**（W10 WGMMA mainloop / W14 FA），不在 W2 单独搞 demo。

### O10. Divide 家族 + local_tile / local_partition 大图

**核心心智模型**：所有 divide / partition 函数都站在 `zipped_divide` 这一个节点之上，只是切刀方向或形状嵌套不同。

```
       composition + complement     (两个底层原语)
                  │
                  ▼
           logical_divide            按"原维度"嵌套
        ((BLK_M, m), (BLK_N, n))
                  │
                  │  tile_unzip 重组：按"角色"分组
                  ▼
           zipped_divide             ◄═══ 核心节点
        ((BLK_M, BLK_N), (m, n))
                  │
       ┌──────────┼──────────────────────┐
       │          │                      │
   形状变体     切 tile 维             切 rest 维
       │          ▼                      ▼
       │   outer_partition       inner_partition
       │   保留 rest             保留 tile
       │          │                      │
       │          ▼                      ▼
       │   local_partition       local_tile
       │   (thr_layout, thr_idx) (tile_shape, coord)
       │
       ├── tiled_divide  ((BLK), m, n)
       └── flat_divide   (BLK_M, BLK_N, m, n)
```

**5 个 divide** 指向同一坨地址，只是 shape tuple 嵌套方式不同：

| 函数 | shape | 怎么取第 (i, j) 块 |
|------|------|------|
| `logical_divide` | `((BLK_M, m), (BLK_N, n))` | `result((_,i), (_,j))` |
| `zipped_divide` | `((BLK_M, BLK_N), (m, n))` | `result(_, (i, j))` |
| `tiled_divide` | `((BLK_M, BLK_N), m, n)` | `result(_, i, j)` ← 最常用 |
| `flat_divide` | `(BLK_M, BLK_N, m, n)` | `result(_, _, i, j)` |

**2 个 partition** 在 zipped_divide 上切一刀：

| 函数 | 切哪边 | 保留哪边 | 语义 |
|------|--------|----------|------|
| `inner_partition` ≈ `local_tile` | 切 rest | 保留 tile | "我是第 (i,j) 个 CTA，给我那一块 tile" |
| `outer_partition` ≈ `local_partition` | 切 tile | 保留 rest | "我是第 t 个 thread，给我散在各个 tile 里的同一位置" |

**决策树**：
```
我要干啥？
├─ 算法层推导 layout                      → tiled_divide / flat_divide
├─ kernel 入口，CTA 取自己那块            → local_tile
├─ thread 协作搬一片 tile（GMEM→SMEM）    → local_partition
├─ 用 TiledCopy 抽象的协作搬运            → TiledCopy::partition_S/D
└─ 用 TiledMMA 给每 thread 取 MMA fragment → TiledMMA::partition_A/B/C
```

`TiledCopy::partition_S/D` 内部本质就是 `outer_partition` + 自动算的 thr_layout。`TiledMMA::partition_A/B/C` 在此基础上还套 (T, V) → element 的 ALayout 硬件接线。

### O11. Per-mode complement vs whole-layout complement——不该直接对比

`zipped_divide` 用 **per-mode complement**：tuple tiler 拆开，每个 mode 在**坐标空间**做 1D complement（cotarget = 该 mode size）。
直接调 `complement(layout, M)` 是 **whole-layout complement**：单个多维 layout 在**地址空间**做 complement（cotarget = 总尺寸）。

**两边输入根本不是同一类对象**——前者吃 tuple of 1D，后者吃单个多维 layout。强行对比要做空间转换，即使能凑出"image 集合相同"（紧凑 L 时），也不等于它们可互换。

它们是**不同 API 解决不同问题**：
- per-mode：tuple tiler API 配合 logical_divide 的 `((tile),(rest))` 嵌套结构，**为 tile-and-grid 切分服务**
- whole-layout：服务于代数原语场景（`logical_product` 复制块、`right_inverse` 求逆、填补非紧凑 layout 的洞、扩展 codomain）

### O12. MMA atom 内的 ALayout/BLayout/CLayout 是硬件接线图

**不是** GMEM→SMEM 加载映射，而是 MMA 指令执行那一刻、(lane_id, val_id) → 该元素在原子 tile 中的位置。

```
A 已经在 RMEM 里
    ↓
ALayout: (lane_id, val_id) → A 矩阵中的元素索引
    ↓
mma.sync 指令（硬件接线钉死）
```

**SM80 m16n8k16 fp16 ALayout 拆解**：
```
ALayout = ((_4, _8), (_2, _2, _2)) : ((_32, _1), (_16, _8, _128))
            └──T=32──┘└────V=8─────┘
```
- Thread mode (4, 8): t0=lane%4=threadID_in_group, t1=lane/4=groupID
- Value mode (2, 2, 2): (reg-internal pair, low/high-M, low/high-K) — 对应 4 个 .b32 register {a0,a1,a2,a3}

设计哲学：硬件接线包进 Layout 函数 → 编译期零开销 → 跨架构统一，上层 cute::gemm 不需要知道架构。

### O13. make_fragment_A/B/C 不只做检查

`mma_atom.hpp:120-195` 的三件事：

1. **检查**（`CUTE_STATIC_ASSERT_V`）：rank≥3、V 维 size 匹配 atom 期望、元素类型匹配
2. **分配寄存器存储**：
   - `make_fragment_C` 永远新建 RMEM tensor（C 是累加器）
   - `make_fragment_A/B`：若 `FrgTypeA` 是 view 类型，返回输入的 reinterpret view；否则新建
3. **跨架构选 view vs copy**：
   - SM80：A/B 必须在 RMEM → 新建寄存器存储
   - SM90 (WGMMA)：A/B 在 SMEM 由硬件 descriptor 读 → 返回 SMEM view，不复制
   - SM100 (UMMA)：类似 SMEM/TMEM view

让上层 `cute::gemm` 跨架构代码不变——同样调 `make_fragment_A`，SM80 分配 RMEM、SM90 给 SMEM view。

### O14. mma_unpack 是 Tensor → PTX 的胶水

`mma_traits.hpp:113` 的 `mma_unpack` 把 4 个 Tensor 翻译成位置参数喂给 `MMA_Op::fma`（裸 inline asm）：

1. 静态断言 4 个 tensor 都在 RMEM
2. `recast<RegTypeA>(A)`：把 `Tensor<half_t, ...>` shape (V=8) 重解释为 `Tensor<uint32_t, ...>` shape (V=4)
3. size 校验：recast 后的元素数必须等于 PTX 接口要的 register 数
4. `detail::explode`：用 integer pack expansion 展开成 N 个 `uint32_t&` 位置参数，调用 `MMA_Op::fma(d0, d1, a0, ..., a3, b0, b1, c0, c1)`

设计哲学：架构特化的 raw `fma` 函数手写 inline asm（每架构一份）；通用 `mma_unpack` 通过 recast + explode 适配任意 register 数。

### O15. partition_X 输出 VMK/VNK/VMN

`partition_C(tile_tensor)` 返回 **rank-3 tensor**：

| 函数 | 输出 shape | 含义 |
|------|-----------|------|
| `partition_A(gA)` | `(V, RestM, RestK)` | V = atom 内 per-thread 持值数（A 是 8）；RestM/K = 该 thread 沿 M/K 覆盖几个 atom |
| `partition_B(gB)` | `(V, RestN, RestK)` | 类似 |
| `partition_C(gC)` | `(V, RestM, RestN)` | C 没有 K |

V 维 size 是 atom 钉死的常量；RestM/RestN/RestK 跟着用户传进来的 tile size scale。`partition_X` 输出仍在原 memory（GMEM/SMEM），**只是 layout 换成 thread 视角**。配 `make_fragment_X` 才得到 register tensor。

### O16. cute::gemm 只占完整 GEMM kernel 的 ~1 行

`cute::gemm(mma, rA, rB, rC)` 只是**最内层的 mma 累加 + serpentine 优化**。完整 sgemm_sm80 ~150 行里，cute::gemm 占 1 行，其余是 local_tile + TiledCopy + cp.async + K-loop pipeline + epilogue。

### O17. GEMM-TN 实战占主流 80%+

GEMM-TN（A、B 都 K-major）在 GMEM 的 stride 配置**天然跟 MMA-TN 的硬件需求一致**——`cp.async` 灌进 SMEM 后 `ldmatrix` 直送 register，**无 SMEM 转置中转**。

其它三种 (NN/NT/TT) 都需要在 SMEM 转置部分操作数。业界 fp16/bf16 tensor core 生产 kernel ~80% 走 TN。PyTorch `torch.matmul(A, B)` 两个 row-major 实际是 TT 配置，后端通常先 transpose B 转成 TN。

### O18. local_partition 实现：outer_partition + thr_layout.get_flat_coord

源码 `tensor_impl.hpp:1079`：
```cpp
local_partition(tensor, thr_layout, thr_idx) {
  return outer_partition(tensor,
                         product_each(shape(thr_layout)),     // ① 每 thread tile size
                         thr_layout.get_flat_coord(thr_idx)); // ② thread idx → 多维 coord
}
```

`outer_partition` 跟 `inner_partition`（即 `local_tile`）相反：切 tile 维保留 rest 维。每个 thread 拿到散在各 tile 中、跟它角色一致的元素，而不是一整块 tile。典型场景：GMEM→SMEM 协作搬运，32 thread 各拿 4 fp16 形成合并访问。

### O19. complement 要求输入是单射 layout

源码 `layout.hpp:1203`：
```cpp
static_assert(not is_constant<0, decltype(new_shape)>::value,
              "Non-injective Layout detected in complement.");
```

非单射输入（如 `(2,2):(1,1)` 有两个坐标命中同一地址）会让 `new_shape` 算出 0，**编译失败**。源码先 `filter(layout)` 去 stride-0 和 size-1 modes，但剩下的必须单射。语义：complement 是"填补 L 没 hit 的地址"，L 内部 overlap 时"hit 哪些"和"hit 多少次"是两个概念。

### O20. SM120（5060 Ti）能力 — 不是 SM100 子集，也不是 SM80 加一点

源码 cross-check 结论：**SM120 ≈ "SM90 软件栈 + fp4/fp6 块缩放 mma.sync"**，跟 SM100 是**不同分支**（不是父子）。

| Cap | TMA | Cluster | WarpSpec 主循环 | MMA path | fp4/fp6 + 块缩放 | UMMA + TMEM |
|-----|-----|---------|----------------|----------|------------------|--------------|
| SM80 | ❌ | ❌ | ❌ | mma.sync (RMEM) | ❌ | ❌ |
| SM90 | ✅ | ✅ | ✅ | **WGMMA** (warpgroup async, RMEM) | ❌ | ❌ |
| **SM120 (5060 Ti)** | ✅ | ✅ (SM90 风格) | ✅ | **mma.sync** (warp issue, RMEM, 新 fp4/fp6/fp8 类型) | ✅ | ❌ |
| SM100 (B200) | ✅ | ✅ (含 2-SM 配对) | ✅ | **UMMA** (`tcgen05.mma`, 单线程 issue, TMEM) | ✅ | ✅ |

源码证据：
- `config.hpp:155-156` 启用 `CUTE_ARCH_MMA_SM120_ENABLED` + `CUTE_ARCH_TMA_SM120_ENABLED`
- `dispatch_policy.hpp:1430` 有 `MainloopSm120TmaWarpSpecialized`（带 ClusterShape + PipelineAsyncMmaStages）—— 一等公民
- `mma_sm120.hpp` 全是 `mma.sync` 不是 `tcgen05` —— SM120 没有 UMMA
- `mma_sm100_umma.hpp` 才是 UMMA，包含 `2x1SM_*`（2-CTA paired cluster）专属变体

**对 5060 Ti 用户的实际意义**：
- W6 TMA / W7 Pipeline+Cluster / W9-W12 WarpSpec GEMM / W14-W21 FA & MoE 的**整个软件框架可以在 5060 Ti 完整跑通**（用 `MainloopSm120TmaWarpSpecialized` 代替 SM90 WGMMA mainloop）
- 5060 Ti 跑不了：WGMMA atom（要 H20）、UMMA atom + TMEM（要 B200）、2-SM 配对 cluster（要 B200）
- 5060 Ti 上独有的：fp4 / fp6 块缩放 mma.sync（量化 FA、量化 MoE 实验的现成硬件）

### O21. TiledCopy 和 TiledMMA 本质是同一个问题：线程→元素映射

不管是 sgemm_1 的裸 ThreadLayout、还是 sgemm_2 的 TiledCopy/TiledMMA，解决的核心问题都是 **"把 0-255 这些线程 ID 映射到 (128, 8) 这块 tile 上"**。TiledCopy 和 TiledMMA 概念上是同一种东西——把 M 个 thread 映射到 N 个元素，区别只是每个 thread 执行 copy 还是 mma。

- 代码层面**没有共享基类**——`TiledCopy` 继承 `Copy_Atom`，`TiledMMA` 继承 `MMA_Atom`，各自独立
- 但底层都用同一套 layout 代数原语（`zipped_divide`、`logical_divide`、`compose`、`right_inverse`）
- 没合并的原因：MMA 多一个 **permutation** 步骤（硬件 lane→element 接线图）+ 用 4D 坐标 `(V,M,N,K)` 而非扁平 thr_idx
- 唯一的代码依赖方向：`copy_atom.hpp` include `mma_atom.hpp`（`make_tiled_copy_A/B/C` 从 TiledMMA 反推兼容的 TiledCopy）

裸 Layout 只解决**映射**；TiledCopy 同时解决**映射 + 指令选择**（Copy_Atom 编码了向量宽度和硬件指令类型）。

### O22. Val Layout + 数据 layout 配合保证向量化连续性

`UniversalCopy<uint128_t>` 怎么保证每线程搬的 4 个 float 在显存连续？**这是程序员的责任，不是框架自动推导的**。

```
Val Layout 的方向  ×  数据 layout 在那个方向的 stride  ==  连续
```

NT 情况：gmem A 是 m-major (stride=1 in M)，Val Layout `(4,1)` 沿 M 排 → 4 个连续地址 → 可发 128-bit load。
TN 情况：gmem A 是 k-major (stride=ldA in M)，M 方向不连续 → 退化为 `UniversalCopy<TA>` (32-bit) + Val `(1,1)`。

越宽的 Copy Atom 对连续性/对齐要求越严格。选错了不会编译报错，运行时 crash 或读错数据。

---

## 三、方法论

1. **回答简短到位，不写小作文**。表格、清单优先于段落。
2. **不能编内容**。API 名、行号、shape、stride 这些精确的事情，**必须先查源码再答**。
3. **优先级**：源码 (`include/cute/`) > 官方文档 (`media/docs/cpp/cute/`) > 脑内记忆。脑内的东西永远要验证。
4. **大题目用 Explore agent 审计**：跨多文件的 cross-check / 路径审计交给 Explore，效率高。
5. **行话纠正要给 receipt**：用户问"X 是啥"时，引用源码 / 文档 line 号，让他能自查。
6. **实战 > 纯理论**：ncu / bank conflict / swizzle 这些放到能跑真 kernel 的阶段做（W10 GEMM、W14 FA），不在抽象阶段空对空 demo。
7. **架构特性查询逐项查源码** (`config.hpp` + `dispatch_policy.hpp` + 各 arch `mma_*.hpp` + `copy_*.hpp`)，不要凭"应该有/应该没有"猜。

---

## 四、给下个 agent 的接手指南

1. 先读 `study/README.md` + `study/PROGRESS.md` 看到了哪一周
2. 再读这个 THINKING.md（概念笔记 O1-O22 是参考材料，碰到对应主题直接引用 line 号）
3. **任何涉及 Layout / Swizzle / MMA atom / API 名 / shape / stride / 架构能力的问题先查源码**，引用 `include/cute/` 和 `media/docs/cpp/cute/`
4. **W3/W4 reading 顺序已经是自下而上**（PTX → Traits → Atom → Tiled → 算法），别退回 wrapper-first
5. **scaffold 必须 SM80-compatible**（5060 Ti 是主战硬件）
6. **不要在 W1-W4 阶段强加 ncu / bank conflict 实测**，挪到 W10+

PR / merge 流程：
- main 受保护，必须 PR 后 merge（直接 push 会 403）
- 用户的开发分支：`claude/review-progress-ex02-7MUIB`
- 用 `mcp__github__create_pull_request` + `mcp__github__merge_pull_request`
