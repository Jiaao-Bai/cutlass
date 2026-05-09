# CuTe + Hopper/Blackwell 学习计划

目标：能在 H20（SM90）和 B200（SM100）上手写并极致优化 GEMM / FlashAttention / Sparse MoE。

全程只学 CuTe + CUTLASS 3.x，跳过 CUTLASS 2.x 所有内容（`gemm/threadblock/`、`gemm/warp/`、`transform/threadblock/` 等目录）。

---

## 总体路线图

```
第一阶段：CuTe 张量代数    第二阶段：SM90 硬件原语    第三阶段：GEMM
  Layout / Tensor 代数  →  WGMMA + TMA + Barrier  →  手写 WarpSpec GEMM
       (3-4 周)                  (2-3 周)                  (3-4 周)
            ↓
第四阶段：FlashAttention    第五阶段：Sparse MoE      第六阶段：极致调优
   FA Forward/Backward   →  GroupedGEMM + 路由调度  →  Roofline + ncu
        (4 周)                    (3 周)                   (持续)
```

---

## 第一阶段：CuTe 张量代数（3-4 周）

### Week 1-2：Layout 代数

#### 必读代码

**1. `crd2idx` — Layout 的本质（`include/cute/stride.hpp:47-170`）**

整个 CuTe 的核心，4 条递归规则：
```
op(c, s, d)             => c * d                    // 整数：直接乘 stride
op(c, (s,S), (d,D))     => op(c%prod(s),s,d)         // 整数坐标 + tuple shape：divmod
                         + op(c/prod(s), S, D)
op((c,C),(s,S),(d,D))   => op(c,s,d) + op(C,S,D)    // tuple 坐标：逐 mode 求和
```

**2. `Layout` struct（`include/cute/layout.hpp:95-220`）**

两个设计亮点：
- 继承自 `cute::tuple<Shape,Stride>`，用 EBO 使静态 layout 零运行时开销
- `operator()(Coord)` 用 `if constexpr (has_underscore<Coord>)` 同时支持坐标映射（返回整数）和切片（返回子 Layout）

**3. `coalesce`（`include/cute/layout.hpp:768-880`）**

`bw_coalesce` 是从右到左的栈式合并，注释里直接写出了 4 条规则：
```
front(NewLayout)  get<I>(Layout)
s0:d0              _1:d1     =>  continue         // size=1 跳过
_1:d0              s1:d1     =>  replace_front     // 替换 size-1
s0:s1*d1           s1:d1     =>  merge             // 连续内存，合并
s0:d0              s1:d1     =>  prepend           // 不连续，新加一维
```

**4. `composition`（`include/cute/layout.hpp:1021-1165`）**

`lhs o rhs`，含义是 `result(c) = lhs(rhs(c))`。核心是 fold 循环里的 `divmod` 把 rhs 的 stride 在 lhs 的 shape 序列里找位置。

**5. `complement`（`include/cute/layout.hpp:1164-1260`）**

构造 layout 的补集，本质是排序 + 折叠找到所有未被当前 layout 覆盖的地址。`logical_divide` 和 `tiled_divide` 都依赖它。

**6. `Swizzle`（`include/cute/swizzle.hpp:42-130`）**

```cpp
struct Swizzle<B, M, S> {
    operator()(offset) => offset ^ XOR_of_specific_bits
}
```
`make_swizzle` 和 `composition(Swizzle, Swizzle)` 展示了两个 swizzle 的复合可以静态化简为零。

#### 推荐读的顺序

```
stride.hpp:47  →  layout.hpp:95  →  layout.hpp:768  →  layout.hpp:1021  →  layout.hpp:1164  →  swizzle.hpp:42
```

#### 练习

用纯 CuTe 手写一个 naive FP16 GEMM（不用任何 `cutlass::gemm` 组件），能跑出正确结果。

参考：`examples/cute/tutorial/sgemm_1.cu`、`sgemm_2.cu`

---

### Week 3-4：TiledMma / TiledCopy / Atom

#### 必读代码

**1. `MMA_Atom` struct（`include/cute/atom/mma_atom.hpp:42-200`）**

两层包装设计：
```
MMAOperation（PTX 裸指令）
    ↓ MMA_Traits<Op>    — 配上 ThrID / LayoutA_TV / LayoutB_TV / LayoutC_TV
    ↓ MMA_Atom<Traits>  — 提供 call() / make_fragment_A/B/C()
```

`LayoutA_TV`（TV = Thread × Value）描述"这条指令的 A 矩阵，哪些线程负责哪些元素"，是从硬件指令到 CuTe 抽象的关键桥梁。

**2. `TiledMMA::thrfrg_C/A/B`（`include/cute/atom/mma_atom.hpp:250-380`）**

把全局 `(M,N)` tensor 变换成 `((ThrV,(ThrM,ThrN)),(FrgV,(RestM,RestN)))` 的 4 步 Layout 变换：

```cpp
logical_divide(ctensor, t_tile)      // (PermM,PermN)                   — 按 permutation 重排
zipped_divide(t_tensor, c_tile)      // ((AtomM,AtomN),(RestM,RestN))   — 按 Atom 大小切块
c_tensor.compose(AtomLayoutC_TV{})   // ((ThrV,FrgV),(RestM,RestN))     — TV 变换
zipped_divide(tv_tensor, thr_tile)   // ((ThrV,(ThrM,ThrN)),(FrgV,...)) — 线程分组
```

**3. `ThrMMA::partition_C/A/B`（`include/cute/atom/mma_atom.hpp:460-523`）**

用 `thrfrg_C` 的结果按当前线程坐标切片，得到这个线程负责的寄存器分片。Layout 代数里 `composition + slice` 的实战应用。

**4. `cute::gemm` 的 5 层 dispatch（`include/cute/algorithm/gemm.hpp:100-500`）**

```
Dispatch 1: (V) × (V) → (V)            直接调用 mma.call()
Dispatch 2: (M) × (N) → (M,N)          外积，升维到 Dispatch 3
Dispatch 3: (M,K) × (N,K) → (M,N)      标准矩阵乘，升维到 Dispatch 5
Dispatch 4: (V,M) × (V,N) → (V,M,N)   带寄存器复用的批量外积
Dispatch 5: (V,M,K) × (V,N,K) → (V,M,N)  最终展开，对每个 K 调 Dispatch 4
```

重点看 Dispatch 4（`gemm.hpp:260-390`）里的 **serpentine（蛇形）遍历**：
```cpp
int ns = (m & 1) ? N-1-n : n;  // 奇数行反向
```
这是为了最大化寄存器复用，实测影响几十 TFLOPS。

**5. SM90 GMMA smem layout（`include/cute/atom/mma_traits_sm90_gmma.hpp:71-130`）**

```cpp
using Layout_MN_SW128_Atom_Bits = ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag,
                                   Layout<Shape<_1024,_8>, Stride<_1,_1024>>>;
```

WGMMA 对共享内存 layout 的完整要求。`upcast<sizeof_bits<Type>>` 把 bit 单位转换到元素单位。自己写 GEMM 时 smem layout 的直接参考。

#### 推荐读的顺序

```
mma_atom.hpp:42   →  mma_atom.hpp:250   →  mma_atom.hpp:460
(Atom 结构)           (thrfrg：Layout变换)   (partition：线程切片)
      ↓
gemm.hpp:100   →  gemm.hpp:260
(5层dispatch)     (serpentine 寄存器复用)
      ↓
mma_traits_sm90_gmma.hpp:71
(WGMMA smem layout 实战)
```

#### 练习

读懂 `examples/cute/tutorial/sgemm_sm80.cu`，能解释每个 `TiledMma`/`TiledCopy` 的 shape 是怎么来的。

---

## 第二阶段：SM90 硬件原语（2-3 周）

### Week 5-6：WGMMA + TMA

#### WGMMA（Warpgroup MMA，4 个 warp 协同 128 线程做矩阵乘）

| 文件 | 内容 |
|------|------|
| `include/cute/arch/mma_sm90.hpp` | PTX wgmma 封装 |
| `include/cute/atom/mma_traits_sm90_gmma.hpp` | 各种 WGMMA atom 的 layout 定义（非 ext/sparse） |
| `examples/cute/tutorial/hopper/wgmma_sm90.cu` | 最小 WGMMA 示例 |
| `examples/cute/tutorial/hopper/wgmma_tma_sm90.cu` | WGMMA + TMA 组合 |

跳过 `mma_sm90_gmma_ext.hpp`（60k 行）和 `mma_sm90_gmma_sparse_ext.hpp`（56k 行），它们是自动生成的枚举表，没有设计可读。

**关键理解点：**
- WGMMA 的 A 矩阵必须在**寄存器**（RS 模式）或**共享内存**（SS 模式）
- smem 需要特定 swizzle layout，参见 `mma_traits_sm90_gmma.hpp:71-130`

#### TMA（Tensor Memory Accelerator，硬件异步搬运引擎）

| 文件 | 内容 |
|------|------|
| `include/cute/arch/copy_sm90_tma.hpp` | cp.async.bulk / TMA PTX |
| `include/cute/atom/copy_traits_sm90_tma.hpp` | TMA descriptor 构造 |
| `media/docs/cpp/cute/0z_tma_tensors.md` | TMA Tensor 概念文档 |

**关键理解点：**
- TMA 绕过 L1/L2 直接 global→smem，需要 128B 对齐
- TMA descriptor 在 host 上构造，kernel 里只传指针

### Week 7：Hardware Barrier + Thread Block Cluster

| 文件 | 内容 |
|------|------|
| `include/cutlass/pipeline/sm90_pipeline.hpp` | PipelineAsync / PipelineTmaAsync |
| `media/docs/cpp/pipeline.md` | barrier 设计思想文档 |
| `include/cute/arch/cluster_sm90.hpp` | cluster 同步 API |

**关键理解点：**
- `mbarrier` 比 `__syncthreads` 更细粒度，支持 arrive/wait 分离
- `PipelineTmaAsync`：TMA 线程是 producer，MMA warpgroup 是 consumer
- Thread Block Cluster：多个 CTA 共享 smem（DSMEM），用于跨 block 通信

`sm90_pipeline.hpp` 的 10 个核心接口：`producer_acquire` / `producer_commit` / `producer_tail` / `consumer_wait` / `consumer_release`。

---

## 第三阶段：从零手写 Hopper GEMM（3-4 周）

### Week 8：理解 CUTLASS 3.x 分层设计

| 文件 | 内容 |
|------|------|
| `media/docs/cpp/gemm_api_3x.md` | 5 层 API 设计文档（必读） |
| `media/docs/cpp/cutlass_3x_design.md` | 设计哲学 |
| `include/cutlass/gemm/kernel/gemm_universal.hpp` | 3.x kernel 骨架 |
| `include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp` | mainloop 实现 |

**跑通并读懂：**
- `examples/48_hopper_warp_specialized_gemm/` — 手动组装的 WarpSpec GEMM
- `examples/49_hopper_gemm_with_collective_builder/` — 用 Builder 简化

### Week 9-10：手写 WarpSpec GEMM

核心模式是 **Warp Specialization**：

```
┌─────────────────────────────────────┐
│  CTA                                │
│  ┌──────────┐   ┌────────────────┐  │
│  │ TMA warp │   │  MMA warpgroup │  │
│  │(producer)│──▶│  (consumer)   │  │
│  └──────────┘   └────────────────┘  │
│       mbarrier 同步                 │
└─────────────────────────────────────┘
```

参考实现（按复杂度递增）：

| 文件 | 说明 |
|------|------|
| `include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized.hpp` | 基础 WarpSpec |
| `include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp` | `load()` / `mma()` 分离的 mainloop |
| `include/cutlass/pipeline/sm90_pipeline.hpp` | pipeline state 流转 |

需要自己处理的关键细节：
- smem 的 swizzle layout（消除 bank conflict）
- pipeline 深度（一般 depth=4~8）
- epilogue：从寄存器写回 global，支持 alpha/beta

### Week 11：Pingpong vs Cooperative

| 文件 | 说明 |
|------|------|
| `include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp` | Pingpong kernel 骨架 |
| `include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp` | Cooperative kernel 骨架 |

- **Pingpong**：两个 MMA warpgroup 交替计算，隐藏 WGMMA 延迟，适合大矩阵
- **Cooperative**：两个 warpgroup 合作算同一个 tile，适合 batch 小/K 大的场景

H20 上经验：大 batch 推理优先 Pingpong，decode 阶段 M 小时考虑 Cooperative。

---

## 第四阶段：FlashAttention（4 周）

### Week 12：理解 FA 算法

FA 与 GEMM 的核心差异：
```
GEMM：  A × B → C  （单次）
FA：    softmax(Q × Kᵀ / √d) × V  （两次 GEMM + online softmax，必须 fuse）
```

| 文件 | 内容 |
|------|------|
| `examples/88_hopper_fmha/` | 完整 FMHA 实现 |
| `examples/88_hopper_fmha/collective/` | mainloop（online softmax 在这里） |
| `examples/python/CuTeDSL/hopper/fmha.py` | Python DSL 版本，逻辑更清晰，适合先理解算法 |

### Week 13-14：手写 FA Forward

| 技术 | 作用 |
|------|------|
| Online softmax (Dao et al.) | 流式计算 max/sum，无需存储完整 attention score |
| Q/K/V 的 TMA 切分 | Q 按 row tile，K/V 按 col tile 流式加载 |
| WarpSpec 在 FA 中的角色 | TMA warp 预加载 K/V，MMA warpgroup 做 QK^T 和 PV |
| 因果 mask 处理 | 对角线以上整体跳过，半遮挡 tile 用 predication |

H20 上 FA 优化重点：
- Head dim 选择：d=64/128 最优，d=96 需要 padding
- Seq len 分块：tile size 根据 smem 容量决定（SM90 smem 最大 228KB/CTA）
- Persistent kernel：用 tile scheduler 减少 kernel launch overhead

### Week 15：FA Backward

参考 `examples/python/CuTeDSL/blackwell/fmha_bwd.py`

---

## 第五阶段：Sparse MoE（3 周）

### Week 16：Grouped GEMM 基础

MoE 的核心是不同 expert 处理不同数量的 token，天然不等长 → Grouped GEMM。

| 文件 | 内容 |
|------|------|
| `examples/57_hopper_grouped_gemm/` | Hopper grouped gemm |
| `include/cutlass/gemm/kernel/sm90_tile_scheduler_group.hpp` | Group tile scheduler |

### Week 17：Token Routing + Expert 并行

MoE 的四个核心操作：
1. **Gate/Router**：softmax + topk，决定 token 去哪个 expert
2. **Permute**：按 expert id 重排 token（scatter）
3. **Expert GEMM**：每个 expert 做独立 GEMM
4. **Unpermute**：结果按原始顺序归位（gather）

Tile Scheduler 设计参考：`include/cutlass/gemm/kernel/sm90_tile_scheduler.hpp`

`get_current_work()` 的动态调度设计可以直接复用到自定义 MoE kernel。

### Week 18：Fused MoE 优化

H20 上的优化手段：
- Expert 间负载均衡：动态 tile scheduler，让忙的 expert 多占 SM
- Gate + GEMM fusion：把 routing 和第一层 GEMM 合并
- FP8 量化：SM90 原生 FP8 WGMMA，expert weight 用 FP8 存储

参考：`examples/python/CuTeDSL/blackwell/grouped_gemm.py`

---

## 第六阶段：H20 极致调优（持续进行）

### H20 硬件参数

| 参数 | H20 值 | 影响 |
|------|--------|------|
| SM 数量 | 132 | 最大并发 CTA 数 |
| WGMMA 峰值 (FP16) | ~990 TFLOPS | 理论上界 |
| HBM3 带宽 | ~4 TB/s | memory bound 上界 |
| L2 Cache | 60 MB | 注意复用 |
| smem/SM | 228 KB | tile size 上界 |

### Profiling 方法

```bash
ncu --set full -o profile ./my_kernel

# 关键指标
# sm__throughput              SM 利用率
# l1tex__t_bytes              L1 命中率
# dram__bytes_read            实际 HBM 带宽
# smsp__sass_thread_inst_executed_op_hmma  WGMMA 指令数
```

### Roofline 分析

```
算术强度 = FLOP / Bytes

GEMM(M,N,K):   FLOPs = 2MNK,   Bytes = 2(MK+NK+MN)
FA(B,H,S,d):   算术强度 ≈ S/8（d=128 时约 16），decode 阶段 memory bound
```

### 常见优化 checklist

- [ ] smem layout 加 swizzle（消除 bank conflict）
- [ ] TMA descriptor 用 128B 对齐
- [ ] pipeline depth 调参（通常 4 或 8）
- [ ] 用 `ClusterShape` 开启 2×1 或 2×2 cluster
- [ ] epilogue 用 EVT（Epilogue Visitor Tree）fuse 后处理
- [ ] FP8 + blockwise scaling 用于推理

---

## B200（SM100）增量知识

学完 SM90 那层后，B200 只需额外了解：

| 新概念 | 文件 |
|--------|------|
| TMEM（Tensor Memory，第四种内存层次） | `include/cute/arch/tmem_allocator_sm100.hpp` |
| UMMA（C 矩阵写到 TMEM 而非寄存器） | `include/cute/arch/mma_sm100_umma.hpp` |
| SM100 Pipeline 差异 | `include/cutlass/pipeline/sm100_pipeline.hpp` |
| SM100 GEMM kernel | `include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp` |

底层 Layout 代数和 TiledMma/TiledCopy 抽象完全一样，增量学习即可。

---

## `include/cutlass/` 目录阅读策略

`cutlass/` 目录共 67 万行、671 个文件，大部分是 2.x 遗产，对我们的目标没有价值。
以下是精确的取舍判断。

### 必读：直接有用（约 5000 行）

**1. Pipeline 抽象（`include/cutlass/pipeline/`）**

| 文件 | 行数 | 核心内容 |
|------|------|----------|
| `sm90_pipeline.hpp` | 1388 | `PipelineTmaAsync`：TMA producer / MMA consumer 同步模型 |
| `sm100_pipeline.hpp` | 1328 | B200 的 pipeline 差异（TMEM 引入后 consumer 语义变化） |

`sm90_pipeline.hpp` 的设计重点：
- `PipelineState`：循环 buffer 的 index + phase，避免 ABA 问题
- `producer_acquire` → `producer_commit` → `consumer_wait` → `consumer_release` 的 4 步协议
- `spread_arrivals_to_warpgroup`：把 barrier arrive 均摊到 warpgroup 内所有线程

自己写 FlashAttention / MoE kernel 时，producer/consumer 同步直接参照这个模式。

**2. WarpSpec Kernel 骨架（`include/cutlass/gemm/kernel/`）**

| 文件 | 行数 | 核心内容 |
|------|------|----------|
| `sm90_gemm_tma_warpspecialized_pingpong.hpp` | 947 | TMA warp 和 MMA warpgroup 分工、pipeline state 流转的完整实现 |
| `sm90_gemm_tma_warpspecialized_cooperative.hpp` | — | 两个 warpgroup 协同算同一 tile 的变体 |

Pingpong kernel 的结构（`sm90_gemm_tma_warpspecialized_pingpong.hpp`）：
```
kernel()
├── if (is_producer_warp)
│     load()  ← TMA 异步搬运 A/B，配合 producer_acquire/commit
└── else (MMA warpgroup 0 or 1)
      mma()   ← WGMMA + consumer_wait/release，两个 warpgroup 交替
```

写自己的 FA kernel 骨架时直接参考这个分工结构。

**3. Collective Mainloop（`include/cutlass/gemm/collective/`）**

| 文件 | 行数 | 核心内容 |
|------|------|----------|
| `sm90_mma_tma_gmma_ss_warpspecialized.hpp` | 584 | `load()` / `mma()` 分离，pipeline depth 控制，smem tile 管理 |

设计亮点：`load()` 和 `mma()` 是两个独立函数，分别由不同 warpgroup 调用，通过 pipeline state 对齐。这种分离是写 FA 时 QK^T 和 PV 两阶段 mainloop 的直接参考。

**4. Tile Scheduler（`include/cutlass/gemm/kernel/`）**

| 文件 | 核心内容 |
|------|----------|
| `sm90_tile_scheduler.hpp` | 基础 persistent scheduler，`get_current_work()` 接口 |
| `sm90_tile_scheduler_group.hpp` | Grouped GEMM 调度，不等长 expert 的负载分配 |
| `tile_scheduler_params.h` | scheduler 参数结构，stream-K / data-parallel 策略选择 |

`get_current_work()` 的动态调度设计可以直接复用到自定义 MoE kernel 的 expert 负载均衡。

### 选读：设计思路可借鉴

**EVT（Epilogue Visitor Tree）**

| 文件 | 内容 |
|------|------|
| `include/cutlass/epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp` | epilogue TMA 写回 |
| `include/cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp` | EVT callback 组合 |

EVT 把 `alpha*C + beta*D`、bias add、activation 等后处理组合成一棵**编译期类型树**，在 kernel 里零开销展开。写 FA 时 softmax rescale + output accumulate 阶段可以借鉴这个组合模式。

### 不用看（2.x 遗产，共约 38 万行）

| 目录 | 原因 |
|------|------|
| `gemm/threadblock/` | 2.x 的 `DefaultMmaCore`、tile iterator，全部被 collective 替代 |
| `gemm/warp/` | 2.x 的 `WarpMma`，被 TiledMma 替代 |
| `transform/threadblock/` | 2.x 的数据搬运，被 TMA 替代 |
| `conv/` | 卷积，当前目标不涉及 |
| `epilogue/threadblock/` | 2.x 的 epilogue，被 collective epilogue 替代 |

---

## 精简版阅读顺序

```
# 第一阶段：CuTe 代数
stride.hpp:47 → layout.hpp:95 → layout.hpp:768 → layout.hpp:1021 → swizzle.hpp:42
→ mma_atom.hpp:42 → mma_atom.hpp:250 → algorithm/gemm.hpp:100
→ sgemm_1.cu → sgemm_2.cu → sgemm_sm80.cu

# 第二阶段：SM90 硬件
→ wgmma_sm90.cu → wgmma_tma_sm90.cu
→ pipeline.md → sm90_pipeline.hpp

# 第三阶段：GEMM
→ gemm_api_3x.md → 48_hopper_warp_specialized_gemm（跑通+读懂）
→ 手写 WarpSpec GEMM

# 第四阶段：FA
→ 88_hopper_fmha（读+改） → 手写 FA Forward

# 第五阶段：MoE
→ 57_hopper_grouped_gemm → 手写 MoE

# 持续：ncu profiling + 迭代优化
```

---

## 一个务实建议

**不要一开始就用 `CollectiveBuilder`**，它把细节全藏起来了。正确路径：

1. 先用 CuTe 原语手写，理解每个 tensor 的 shape 和 layout
2. 能从零跑出正确结果后，再用 Builder 对比，理解它帮你省了什么
3. 最终优化时回到手写，因为 Builder 生成的不一定是最优配置

H20 上 90% 的性能天花板取决于对 **WGMMA 发射节奏**和 **TMA 预取时序**的理解，这两个只有手写才能真正掌握。
