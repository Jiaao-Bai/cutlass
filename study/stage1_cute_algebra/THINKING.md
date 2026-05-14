# Stage 1 — 思考 / 踩坑 / 方法论

> 这个文件记录 Stage 1 学习过程中的用户洞察、agent 犯的错、方法论偏好。下一个接手仓库的 agent **务必读完再开始**——避免重复同样的错误。

---

## 一、用户硬件 / 学习背景

- **硬件**：日常在 **5060 Ti (SM120, Blackwell consumer)** 上学习；偶尔也有 H20 (SM90) 和 B200 (SM100/120)。
- **目标**：手写 + 极致优化 Hopper/Blackwell GEMM、FlashAttention、Sparse MoE。
- **W1-W4 必须在 5060 Ti 上能跑** —— scaffold 只能用 SM80 PTX (`mma_sm80` + `cp.async`)，不能用 SM90+ 独有指令（WGMMA / TMA / cluster）。
- **W5-W18 主要在 H20**；**W19-W21 (Blackwell SM100/SM120) 可以回 5060 Ti 跑 SM120 增量**。
- 编译命令：`cmake -DCUTLASS_ENABLE_STUDY=ON -DCUTLASS_NVCC_ARCHS=120 ..`（5060 Ti）/ `90` (H20) / `100a` (B200)

---

## 二、用户的核心洞察（这次他抓到的、且对的）

按出现顺序，每条都是用户的**纠错或追问**揭示的本质——后面解释相关概念时请直接引用这些洞察。

### O1. coalesce 的 merge 条件方向

`bw_coalesce` (`layout.hpp:780-814`) 的实际 merge 规则是 **`d_next == s_prev × d_prev`**（accumulator's stride 等于 incoming dim 的 size × stride），从右往左扫。
- ex03_coalesce.cu 文件头注释里写的 "`s0:s1*d1   s1:d1  =>  merge`" 方向**反了**，应以源码为准。
- 实战效果：row-major 永远不能 coalesce 成 1D（除非某维 size=1），col-major 才友好。

### O2. bank conflict "N-way" 的定义

"N-way conflict" = **同一 cycle 撞同一 bank 的最大线程数**。
- 8x8 fp32 col=0 访问：8 lanes 命中 banks {0, 8, 16, 24} 各 2 次 → **2-way conflict**（不是 8-way！）。
- 32x32 fp32 col=0 访问：32 lanes 全打 bank 0 → 32-way。

### O3. SW32/SW64/SW128 不是方 tile

CUTLASS 的标准 swizzle 都是 **8 行 × N chunks of 16B** 的非方 tile：
- SW128 = `Swizzle<3,4,3>` = 8 行 × 8 chunks（128B/row）
- SW64 = `Swizzle<2,4,3>` = 8 行 × **4** chunks（64B/row）—— **不是 4×4 tile**
- SW32 = `Swizzle<1,4,3>` = 8 行 × 2 chunks（32B/row）

S=3 是因为 Hopper 约定 **row 字段对齐到 offset 的 bit 7+**（外层 stride 撑出来），让公式跨 SW 系列统一。

### O4. CuTe 的 B 写作 `(N, K)`，不是 BLAS 的 `(K, N)`

权威出处：`media/docs/cpp/cute/0x_gemm_tutorial.md:80-81`：
> "CuTe follows the convention that the semantics of matrix modes is **`(M,K)` for `A`, `(N,K)` for `B`, and `(M,N)` for `C`**."

理由：让 A、B 都把 K 放在最后一维，K-reduce 循环写起来对称。**讲解时永远用 CuTe 约定，不要用 BLAS 的 (K,N)**——除非明确在讨论 BLAS API。

### O5. CuTe 的 "X-major" 比 BLAS T/N 更准

`0x_gemm_tutorial.md:103`：
> "Instead of row-major or column-major (or Transposed and Not-Transposed), we have found it much more convenient to say that a matrix is **'M-major'**, **'N-major'**, or **'K-major'** based on which mode has stride-1."

BLAS↔CuTe 对照表（原文 line 107-112，背下来）：

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

### O8. 反对 wrapper-first 阅读顺序

源码该**自下而上**读：PTX → Traits → Atom → Tiled... → 算法。先看 wrapper 再看里面 wrap 啥本末倒置——读者看到 `ALayout`/`ValLayoutSrc` 这些字段时还不知道是啥的容器。

### 9. 反对"空对空"练习

用户对脱离实战场景的纯数学 / bit 推导练习无耐心。**bank conflict / ncu 实测** 之类的东西**放到能跑真实 kernel 的阶段做**（W9 WGMMA mainloop / W12 FA），不在 W2 单独搞 demo。已经因此删掉 `ex05_swizzle_smem.cu`。

---

## 二.A、W3 学习中沉淀的概念笔记（O10-O18）

这些是 W3 阶段深挖出的概念整理，供下个 agent / 用户回顾时直接引用。

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

**5 个 divide** 指向同一坨地址，只是 shape tuple 嵌套方式不同（同 content 不同 view）：

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

`TiledCopy::partition_S/D` 内部本质就是 `outer_partition` + 自动算的 thr_layout（来自 Copy_Atom 硬件要求）。`TiledMMA::partition_A/B/C` 在此基础上还套 (T, V) → element 的 ALayout 硬件接线。

### O11. Per-mode complement vs whole-layout complement——**不该直接对比**

`zipped_divide` 用 **per-mode complement**：tuple tiler 拆开，每个 mode 在**坐标空间**做 1D complement（cotarget = 该 mode size）。
直接调 `complement(layout, M)` 是 **whole-layout complement**：单个多维 layout 在**地址空间**做 complement（cotarget = 总尺寸）。

**两边输入根本不是同一类对象**——前者吃 tuple of 1D，后者吃单个多维 layout。强行对比要做空间转换，即使能凑出"image 集合相同"（紧凑 L 时），也不等于它们可互换。

**正解**：它们是**不同 API 解决不同问题**：
- per-mode：tuple tiler API 配合 logical_divide 的 `((tile),(rest))` 嵌套结构，**为 tile-and-grid 切分服务**
- whole-layout：服务于代数原语场景（`logical_product` 复制块、`right_inverse` 求逆、填补非紧凑 layout 的洞、扩展 codomain）

不要试图把它们当成"同一操作的两种输入格式"互相替换。

### O19. complement 要求输入是单射 layout

源码 `layout.hpp:1203`：
```cpp
auto new_shape = min_stride / get<i>(result_stride);
static_assert(not is_constant<0, decltype(new_shape)>::value,
              "Non-injective Layout detected in complement.");
```

iteration 中 `new_shape = min_stride / 前一累积 stride`。如果输入 layout 多个 mode stride 撞车（不同坐标命中同一地址，即非单射），new_shape 算出 0，**编译失败**。

例：`(2,2):(1,1)` —— `(0,0)→0, (0,1)→1, (1,0)→1, (1,1)→2`，两个坐标命中 1，非单射 → assert fire。

源码先 `filter(layout)` 去 stride-0 和 size-1 modes，但**剩下的部分必须单射**。语义合理性：complement 是"填补 L 没 hit 的地址"，L 内部 overlap 时"hit 哪些"和"hit 多少次"是两个概念，"剩下"无法良定义。

### O12. MMA atom 内的 ALayout/BLayout/CLayout 是硬件接线图

**误解**：以为是 GMEM→SMEM 加载的映射。  
**实际**：描述 MMA 指令执行那一刻、(lane_id, val_id) → 该元素在原子 tile 中的位置。

```
A 已经在 RMEM 里
    ↓
ALayout: (lane_id, val_id) → A 矩阵中的元素索引
    ↓
mma.sync 指令（硬件接线钉死）
```

硬件 mma 单元的 lane→element 接线是物理连线，软件不能选。CuTe 用一个 Layout 函数把它包起来。

**SM80 m16n8k16 fp16 ALayout 拆解**：
```
ALayout = ((_4, _8), (_2, _2, _2)) : ((_32, _1), (_16, _8, _128))
            └──T=32──┘└────V=8─────┘
```
- Thread mode (4, 8): t0=lane%4=threadID_in_group, t1=lane/4=groupID
- Value mode (2, 2, 2): (reg-internal pair, low/high-M, low/high-K) — 对应 4 个 .b32 register {a0,a1,a2,a3} 的组合

设计哲学：硬件接线包进 Layout 函数 → 编译期零开销 → 跟 logical_divide/composition 兼容 → 统一所有架构（SM75/80/90/100），上层 cute::gemm 完全不需要知道架构。

### O13. make_fragment_A/B/C 不只做检查

`mma_atom.hpp:120-195` 的三件事：

1. **检查**（`CUTE_STATIC_ASSERT_V`）：rank≥3、V 维 size 匹配 atom 期望、元素类型匹配
2. **分配寄存器存储**（关键）：
   - `make_fragment_C` 永远新建一个 RMEM tensor（C 是累加器，必须独立 register）
   - `make_fragment_A/B`：若 `FrgTypeA` 是 view 类型（有 dereference），返回输入的 reinterpret view；否则新建
3. **跨架构选 view vs copy**：
   - SM80：A/B 必须在 RMEM → 新建寄存器存储
   - SM90 (WGMMA)：A/B 在 SMEM 由硬件 descriptor 读 → 返回 SMEM view，不复制
   - SM100 (UMMA)：类似 SMEM/TMEM view

设计哲学：让上层 `cute::gemm` 跨架构代码不变——同样调 `make_fragment_A`，SM80 分配 RMEM、SM90 给 SMEM view，使用者透明。

### O14. mma_unpack 是 Tensor → PTX 的胶水

`mma_traits.hpp:113` 的 `mma_unpack` 把 4 个 Tensor 翻译成位置参数喂给 `MMA_Op::fma`（裸 inline asm）：

1. 静态断言 4 个 tensor 都在 RMEM
2. `recast<RegTypeA>(A)`：把 `Tensor<half_t, ...>` shape (V=8) 重新解释为 `Tensor<uint32_t, ...>` shape (V=4)（2 fp16 packed/uint32）
3. size 校验：recast 后的元素数必须等于 PTX 接口要的 register 数
4. `detail::explode`：用 integer pack expansion 把 4 个 Tensor 展开成 N 个 `uint32_t&` 位置参数，调用 `MMA_Op::fma(d0, d1, a0, ..., a3, b0, b1, c0, c1)`

设计哲学：架构特化的 raw `fma` 函数手写 inline asm（每架构一份）；通用 `mma_unpack` 通过 recast + explode 适配任意 register 数。加新架构时 mma_unpack 不变。

### O15. partition_X 输出 VMK/VNK/VMN

`partition_C(tile_tensor)` 返回 **rank-3 tensor**，含义：

| 函数 | 输出 shape | 含义 |
|------|-----------|------|
| `partition_A(gA)` | `(V, RestM, RestK)` | V = atom 内 per-thread 持值数（A 是 8）；RestM/K = 该 thread 沿 M/K 覆盖几个 atom |
| `partition_B(gB)` | `(V, RestN, RestK)` | 类似 |
| `partition_C(gC)` | `(V, RestM, RestN)` | C 没有 K |

V 维 size 等于 `ALayout/BLayout/CLayout` 的第二个 mode（per-thread 值数，是 atom 钉死的常量）。RestM/RestN/RestK 跟着用户传进来的 tile size scale。

`partition_X` 输出仍在原 memory（GMEM/SMEM），**只是 layout 换成 thread 视角**。配 `make_fragment_X` 才得到 register tensor。

### O16. cute::gemm 只占完整 GEMM kernel 的 ~1 行

`cute::gemm(mma, rA, rB, rC)` 只是**最内层的 mma 累加 + serpentine 优化**，不是完整 GEMM。完整 sgemm_sm80 ~150 行里，cute::gemm 占 1 行，其余是：

```
local_tile (CTA 切块)
TiledCopy partition + cp.async (GMEM→SMEM)
TiledCopy partition_S/D (SMEM→RMEM via ldmatrix)
K-loop: cp_async_wait → ldmatrix → cute::gemm → next cp.async
epilogue: copy 结果回 GMEM
grid/tile 调度
```

不要被 "cute::gemm" 的名字误导以为它能独立实现 GEMM。

### O17. GEMM-TN 实战占主流 80%+

**原因**：GEMM-TN（A、B 都 K-major）在 GMEM 的 stride 配置**天然跟 MMA-TN 的硬件需求一致**——`cp.async` 灌进 SMEM 后 `ldmatrix` 直送 register，**无 SMEM 转置中转**。

其它三种：
- **NT**：A、B 都不 K-major → 需要 SMEM 同时转置 A、B
- **NN**：B K-major 但 A 不 → 只转 A
- **TT**：A K-major 但 B 不 → 只转 B

业界 fp16 / bf16 tensor core 生产 kernel ~80% 走 TN。CUTLASS 的 benchmark / Hopper WGMMA / Blackwell UMMA 例子也几乎全 TN。PyTorch `torch.matmul(A, B)` 两个 row-major 实际是 TT 配置，后端通常先 transpose B 转成 TN。

### O18. local_partition 实现：outer_partition + thr_layout.get_flat_coord

源码 `tensor_impl.hpp:1079`：
```cpp
local_partition(tensor, thr_layout, thr_idx) {
  return outer_partition(tensor,
                         product_each(shape(thr_layout)),     // ① 每 thread tile size
                         thr_layout.get_flat_coord(thr_idx)); // ② thread idx → 多维 coord
}
```

`outer_partition` 跟 `inner_partition`（即 `local_tile`）相反：切第一维（tile）保留第二维（rest）。结果是**每个 thread 拿到散在各 tile 中、跟它角色一致的元素**，而不是一整块 tile。

例子：data (16, 64)，thr_layout `Layout<Shape<_2,_16>>` (32 threads 排成 2×16)：
- "thread tile" size = (2, 16) → 把 data 切成 (8, 4) 个 (2, 16) 块
- thr 0 在每个块里负责位置 (0, 0)，总共拿 8×4=32 个元素
- 返回 shape `(8, 4)`，**散布**在整个 (16, 64) 中

典型场景：GMEM→SMEM 协作搬运，32 thread 各拿 4 fp16 形成合并访问。

---

## 三、Agent 这次犯的错 + 修正记录

每条记录：**错在哪 → 实际正确答案 → 教训**。下次类似话题先查源码再答。

### E1. coalesce merge 方向反了
- **错**：第一版讲 ex03 时说 merge 条件是 `d_prev == s_next × d_next`（左项依赖右项）
- **正**：源码 `layout.hpp:800-805` 是 `d_next == s_prev × d_prev`（右项依赖左项）—— 见 O1
- **教训**：layout 代数的方向性靠记忆容易错，必须先打开源码看

### E2. Swizzle<3,2,3> 配 32×32 fp32 layout
- **错**：在 ex05_swizzle_paper.md 和 ex05_swizzle_smem.cu 里都说 `Swizzle<3,2,3>` 对 32×32 fp32 (128B/row) 能消 32-way conflict
- **正**：`Swizzle<3,2,3>` 只覆盖 32B 子区段（B+M=5），在 32×32 layout 上只能从 32-way 降到 16-way。要消 32×32 的转置 conflict 需 `Swizzle<5,2,5>`（B+M=7 覆盖整条 bank-row）
- **修复**：paper 改成 8×8 fp32 tile（32B/row）让 `Swizzle<3,2,3>` 的预期生效；.cu 改成 `Swizzle<5,2,5>` 保留 32×32 layout
- **教训**：B+M 必须等于 `log2(单条指令实际触及字节数的 log)`，否则 swizzle 不工作

### E3. "8-way conflict" 实际是 2-way
- **错**：ex05 Q2 关键观察里写 "[row=0..7, col=0] 全部映射到 bank 0（8-way conflict）"
- **正**：8 行 col=0 命中 banks 0,8,16,24,0,8,16,24 → 4 banks 各 2 次 → **2-way conflict**
- **教训**：N-way 的 N 是"同 cycle 撞同 bank 的线程数"，不是"撞了几个 bank"

### E4. Swizzle<2,4,3> 描述成 "4×4 tile"
- **错**：速记表说 SW64 是 "4x4 tile 内 2 bit swizzle"
- **正**：源码 `mma_traits_sm100.hpp:173` 写 `((T,4,m),(8,k))` —— SW64 是 **8 行 × 4 chunks of 16B**，不是方 tile。S=3 反映 row 字段对齐 bit 7+ 的硬件约定，跨 SW 系列共用 S=3,M=4，只有 B 变（= log2(chunks/row)）
- **教训**：CUTLASS 标准 swizzle 都是 8 行的非方 tile，由硬件 WGMMA core matrix 8x8 决定

### E5. ex05 Q3 数字自相矛盾
- **错**：原题面 "128 个线程从同一 row 读 8 chunk" 又写 "tid/8 跨 16 row"，数字对不上
- **修复**：删掉 Q3，改成 pointer 指向 W9/W12 实战
- **教训**：自检题要么严格 ground-truth，要么干脆不出。模糊题面比没题更坏

### E6. W3 README 行号 250-380 实际 252-353
- **错**：W3 README 写 "`mma_atom.hpp:250-380` — TiledMMA::thrfrg_C/A/B"
- **正**：实际 252-353（agent 审计抓到）
- **教训**：贴源码行号必须打开文件核对

### E7. W3 提到的 "LayoutA_TV" 实际是 "ALayout"
- **错**：W3 README 把 MMA_Traits 字段叫 `LayoutA_TV / LayoutB_TV / LayoutC_TV`
- **正**：`mma_traits_sm80.hpp:87-91` 实际字段是 `ALayout / BLayout / CLayout`
- **教训**：API 名一字不差，凭印象写 = 制造混乱

### E8. W4 README "Copy_Atom::LayoutSrc_TV" 完全是编造
- **错**：W4 README 自检 Q3 引用 `Copy_Atom::LayoutSrc_TV`
- **正**：`copy_atom.hpp:65-67` 实际字段是 `ValLayoutSrc / ValLayoutDst / ValLayoutRef`（外加 `BitLayoutSrc/Dst/Ref` 原始版）
- **教训**：API 名永远要从源码查再写

### E9. W4 CHECKPOINT "改成 4x2" 含糊
- **错**：CHECKPOINT 让人 "把 sgemm_sm80 thread tile 改成 4x2"，不知道 4x2 指什么
- **正**：实际是把 `make_tiled_mma(SM80_..., Layout<Shape<_2,_2>>{})` 的 atom layout `(2,2)` 改成 `(4,2)` / `(2,4)` / `(4,4)`
- **教训**：CHECKPOINT 描述必须能直接照做，含糊不能放过

### E10. W19 文件名 `copy_traits_sm100_tmem.hpp` 不存在
- **错**：W19 README 引用 `include/cute/atom/copy_traits_sm100_tmem.hpp`
- **正**：实际是 `copy_traits_sm100.hpp`（无 `_tmem`）+ `arch/copy_sm100.hpp:618+` 里的 `SM100_TMEM_LOAD/STORE` PTX
- **教训**：所有"前缀_后缀"组合的文件名先 `ls` 一下

### E11. W3/W4 reading 顺序原本是 wrapper-first
- **错**：W3 让人先读 `mma_atom.hpp:42-200` (MMA_Atom wrapper)，再读 traits 和 PTX
- **正**：改成自下而上 —— PTX (`mma_sm80.hpp:92`) → Traits (`mma_traits_sm80.hpp:77`) → Atom (`mma_atom.hpp:42`) → TiledMMA → ThrMMA → gemm。W4 同理
- **教训**：阅读顺序的反模式是 wrapper-first，正模式是底层原语先行

### E12. B 的 shape 表述用 BLAS 习惯
- **错**：早期讲解时说 "A (M, K), B (K, N), C (M, N)"
- **正**：CuTe 用 **A (M, K), B (N, K), C (M, N)** —— 见 O4 的官方原文
- **教训**：CuTe 跟 BLAS 不一样，永远用 CuTe 约定

### E13. Per-mode vs whole-tile complement 第一次给的对比例子无意义
- **错**：用户问"per-mode complement 跟直接调 complement 结果差别"，agent 第一次给的例子用 `complement(L, cosize(L))` —— L 紧凑时这是 trivial `_1:_0`，根本对比不出来
- **正**：用户指出后改用 `complement(tile, cosize(L))` —— 把 tile 当成 L 地址空间里的一坨 layout 调 complement，得到 `(2,3):(4,16)`
- **教训**：对比两个 API 时，参数选择要让**两边都返回有意义的结果**

### E14. 然后又把 tuple tiler 偷换成 physical tile
- **错**：上一轮修正后，agent 在 O11 里把 "tuple tiler `((2:1),(4:1))`" 跟 "physical tile `(2,4):(8,1)`" 当成同一个对象做"per-mode vs whole-tile complement"对比。两者其实是**不同空间不同对象**——前者在坐标空间是 tuple of 1D，后者在 L 地址空间是单个 rank-2 layout，需要 composition with L 才能桥接
- **正**：per-mode 跟 whole-layout complement **本来就不应该做对比**——输入对象类别不同、服务的问题不同。O11 已重写承认这点
- **教训**：用户最终说"按 mode 和不按 mode 就没法比较"是对的。不是所有"看起来类似的 API"都能转换成对比——有时它们就是不同 API 解决不同问题

---

## 四、用户偏好的方法论

1. **回答简短到位，不写小作文**。表格、清单优先于段落。
2. **不能编内容**。涉及 API 名、行号、shape、stride 这些精确的事情，**必须先查源码再答**——重复犯 E1-E12 这种错会被骂"打脸"。
3. **优先级**：先源码 (`include/cute/`)，次官方文档 (`media/docs/cpp/cute/`)，最后才是脑内记忆。脑内的东西**永远要验证**。
4. **大题目用 Explore agent 审计**。W3+W4 reading 顺序 + W5-W21 cross-check 都是 agent 干的，效率比手翻高。
5. **行话纠正要给 receipt**：用户问"X 是啥"时，引用源码 / 文档 line 号，让他能自查。
6. **实战 > 纯理论**：ncu / bank conflict / swizzle 这些放到能跑真 kernel 的阶段做（W9 GEMM、W12 FA），不在抽象阶段空对空 demo。
7. **每个 Stage 该用哪个硬件**：W1-W4 + W19-21 可 5060 Ti；W5-W18 需 H20。

---

## 五、本次 PR 历史（main 上的提交链）

按时间顺序，每个 PR 修的什么：

| PR | 主题 |
|----|------|
| #6 | study/w02 ex05: fix swizzle paper bit-math & Q1/Q2 layout (E2 修复) |
| #7 | study/w02 ex05: fix Swizzle params for 32x32 fp32 demo (E2 .cu 修复) |
| #8 | study/w02 ex05 paper: fix conflict count 8-way → 2-way (E3) |
| #9 | study/w02 ex05 paper: SW32/64/128 are 8-row tiles, not square (E4) |
| #10 | study/w02 ex05 paper: slim Q3, defer WGMMA drill to W9/W12 (E5) |
| #11 | study/w02: drop ex05_swizzle_smem.cu, defer ncu drill to W9 (用户决定) |
| #12 | study/w02 README: rephrase self-check Q3 |
| #13 | study/PROGRESS: mark W1+W2 done, start W3 |
| #14 | study/w03+w04: scaffold exercises + fix README errors + SM120 ready (E6-E9) |
| #15 | study/w03+w04: rewrite reading sections bottom-up (E11) |
| #16 | study/w19: rewrite reading bottom-up, fix file names (E10) |
| #17 | study/w03+w04: link tutorial sgemm_* examples by abstraction level |

---

## 六、给下个 agent 的建议

接手时务必做：
1. **读 `study/README.md` 和 `study/PROGRESS.md`** 看到了哪一周
2. **读这个 THINKING.md** 避免重复 E1-E12 的错
3. **看 W3/W4 的 reading 顺序**——已经改成自下而上，**别再退回 wrapper-first**
4. **任何涉及 Swizzle / Layout / MMA atom / API 名 / shape / stride 的问题先查源码**，引用 `include/cute/` 和 `media/docs/cpp/cute/`
5. **不要在 W2/W3/W4 阶段强加 ncu / bank conflict 实测**——挪到 W9+
6. **scaffold 必须 SM80-compatible**（5060 Ti 用户）

W3 起步的标准动线：
```
sgemm_1.cu 跑通  →  W3 reading（PTX → Traits → Atom → TiledMMA → gemm）
              →  ex07 partition shape 打印
              →  ex08 serpentine 寄存器对比
              →  W3 自检 5 道
              →  打勾收 W3 进 W4
```

PR / merge 流程：
- main 受保护，只能 PR 后 merge
- 用户的开发分支：`claude/review-progress-ex02-7MUIB`
- 用 `mcp__github__create_pull_request` + `mcp__github__merge_pull_request`，不要直接 `git push origin main`（403）

W3 后还有自检题和 CHECKPOINT，建议跟用户一起做完一关进下一关，避免堆积。
