# Week 3 — TiledMMA / MMA_Atom

预计 ~15h
> **硬件**：🟢 5060 Ti（任何 SM80+ 都行；用 `mma_sm80` atom）

## 目标
- 能解释 `MMA_Atom` 三层包装：PTX → `MMA_Traits` → `MMA_Atom`
- 能从 `ALayout`（TV→A 元素的映射）推出"哪个线程负责 A 矩阵的哪些元素"
- 能解释 `cute::gemm` 5 层 dispatch 的每一层意义
- 能解释 serpentine 遍历为什么省寄存器

## 读（**自下而上**：PTX → Traits → Atom → Tiled → Thr → 算法）

1. `include/cute/arch/mma_sm80.hpp:92-120` — **具体 PTX wrapper** `SM80_16x8x16_F16F16F16F16_TN`，就是 `mma.sync.aligned.m16n8k16.row.col.f16` 的 inline asm。从这里建立"硬件指令到底是啥"的直觉。
2. `include/cute/atom/mma_traits_sm80.hpp:77-92` — **`MMA_Traits<SM80_16x8x16_F16F16F16F16_TN>`**：
   ```
   Shape_MNK = (_16, _8, _16)
   ThrID     = _32                              ← 一个 warp
   ALayout   = ((_4,_8),(_2,_2,_2)):((_32,_1),(_16,_8,_128))   ← (T,V) → A 元素索引
   BLayout   = ((_4,_8),(_2,_2)):((_16,_1),(_8,_64))
   CLayout   = SM80_16x8_Row
   ```
   `ALayout` 第二个 mode 的 size = 2×2×2 = **8** → 单 thread 持 8 个 A 元素。
3. `include/cute/atom/mma_atom.hpp:42-200` — **`MMA_Atom<Traits>`** wrapper：用 Traits 字段提供 `call()` + `make_fragment_A/B/C()`。本质就是把 Traits 包成 C++ struct 接口。
4. `include/cute/atom/mma_atom.hpp:252-353` — **`TiledMMA::thrfrg_C/A/B`**：4 步把 atom 平铺成更大 tile：
   ```cpp
   logical_divide(ctensor, t_tile)      // (PermM,PermN)               — permutation
   zipped_divide(t_tensor, c_tile)      // ((AtomM,AtomN),(RestM,RestN)) — atom 切块
   c_tensor.compose(AtomLayoutC_TV{})   // ((ThrV,FrgV),(RestM,RestN))  — TV 变换
   zipped_divide(tv_tensor, thr_tile)   // ((ThrV,(ThrM,ThrN)),(FrgV,..)) — 线程分组
   ```
5. `include/cute/atom/mma_atom.hpp:460-523` — **`ThrMMA::partition_C/A/B`**：从 TiledMMA 按 thread idx 切片，得到当前 thread 的 fragment。
6. `include/cute/algorithm/gemm.hpp:100-500` — **`cute::gemm` 5 层 dispatch**，编排 atom call：
   ```
   Dispatch 1: (V) × (V) → (V)            标量乘加
   Dispatch 2: (M) × (N) → (M,N)            外积
   Dispatch 3: (M,K) × (N,K) → (M,N)        矩阵乘 → 升维到 5
   Dispatch 4: (V,M) × (V,N) → (V,M,N)      批量外积，寄存器复用
   Dispatch 5: (V,M,K) × (V,N,K) → (V,M,N)  最终展开
   ```
7. `include/cute/algorithm/gemm.hpp:260-390` — Dispatch 4 的 **serpentine 遍历**优化：
   ```cpp
   int ns = (m & 1) ? N-1-n : n;  // 奇数行反向，最大化寄存器复用
   ```

**心智模型**：
```
硬件 PTX → Traits 描述硬件 → Atom 包成 struct → TiledMMA 拼大块 →
ThrMMA 切给 thread → cute::gemm 编排 → serpentine 调寄存器
```

## 实战例子（**读源码前后都建议跑一遍**）

| 例子 | 用到的 | 看点 |
|------|--------|------|
| `examples/cute/tutorial/sgemm_1.cu` | `local_partition` + SMEM + 同步 `cp.async` | 最简完整 GEMM。**不**用 TiledMMA，用更原始的 `local_partition` 切块 + scalar `cute::gemm`。先看这个建立"完整 kernel 长啥样"的直觉。 |
| `examples/cute/tutorial/sgemm_2.cu` | **TiledMMA + TiledCopy**（同步 cp.async）| **W3 主例**。用 TiledMMA 替换 sgemm_1 的 local_partition，看 `partition_A/B/C` 怎么用。 |
| `examples/cute/tutorial/sgemm_sm70.cu` | TiledMMA + Volta tensor core atom | 可选。Volta 的 HMMA atom（不是 SM80 mma_sync），看 TiledMMA 跨架构怎么换 atom 就好。 |

```bash
# 跑：
make sgemm_1 -j && ./examples/cute/tutorial/sgemm_1 4096 4096 4096
make sgemm_2 -j && ./examples/cute/tutorial/sgemm_2 4096 4096 4096
```

`sgemm_sm80.cu` 放到 W4——它带 double-buffer pipeline，跟 W4 的 `cp.async` 异步部分配套。

## 写
- `exercises/ex07_tiled_mma_layout.cu` — 给定 `MMA_Atom` 和 `TiledMMA`，让 thread 0 / 32 / 127 各自打印 partition 出来的 fragment shape
- `exercises/ex08_serpentine_count.cu` — 实测序列遍历 vs serpentine 遍历的寄存器使用差异（`-Xptxas=-v`）

## 跑
```bash
# 5060 Ti / SM120
cmake -DCUTLASS_ENABLE_STUDY=ON -DCUTLASS_NVCC_ARCHS=120 ..
# H20 / SM90
cmake -DCUTLASS_ENABLE_STUDY=ON -DCUTLASS_NVCC_ARCHS=90 ..
# 4090 / SM89 也可：CUTLASS_NVCC_ARCHS=89

make study_stage1_w03_ex07_tiled_mma_layout -j && ./study_stage1_w03_ex07_tiled_mma_layout
make study_stage1_w03_ex08_serpentine_count -j
```

## 自检
1. `MMA_Traits` 里的 `ThrID` 和 `ALayout` 是什么关系？为什么需要两个？
2. `thrfrg_C` 的 4 步变换中，哪一步把"硬件 TV"接到了"用户 (M,N)"上？
3. 为什么 Dispatch 5 要把 K 维拆出来单独处理？
4. serpentine 遍历能省下多少寄存器（数量级）？为什么省？
5. `SM80_16x8x16_F16F16F16F16_TN` 一次需要 **32** 个线程（1 warp）。每个线程持有：A = ___ 个 fp16，B = ___ 个 fp16，C = ___ 个 fp16？（提示：查 `mma_traits_sm80.hpp:78` 的 ALayout/BLayout 的 V 维度积）
