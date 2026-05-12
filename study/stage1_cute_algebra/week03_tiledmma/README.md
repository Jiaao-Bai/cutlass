# Week 3 — TiledMMA / MMA_Atom

预计 ~15h ｜ 目标硬件：**任何 SM80+**（H20 / 5060 Ti / 4090 / A100 都行；用 `mma_sm80` atom）

## 目标
- 能解释 `MMA_Atom` 三层包装：PTX → `MMA_Traits` → `MMA_Atom`
- 能从 `ALayout`（TV→A 元素的映射）推出"哪个线程负责 A 矩阵的哪些元素"
- 能解释 `cute::gemm` 5 层 dispatch 的每一层意义
- 能解释 serpentine 遍历为什么省寄存器

## 读
- `include/cute/atom/mma_atom.hpp:42-200` — `MMA_Atom` 三层包装
  ```
  MMAOperation（PTX 裸指令，如 SM80_16x8x16_F16F16F16F16_TN）
      ↓ MMA_Traits<Op>    : Shape_MNK / ThrID / ALayout / BLayout / CLayout
      ↓ MMA_Atom<Traits>  : call() + make_fragment_A/B/C()
  ```
  - `ALayout`（domain = (ThrID, ValIdx)，codomain = A 的元素索引）= "这条指令的 A 矩阵，哪些线程负责哪些元素"，硬件指令到 CuTe 抽象的桥
- `include/cute/atom/mma_atom.hpp:252-353` — `TiledMMA::thrfrg_C/A/B`
  4 步 Layout 变换把 `(M,N)` tensor 变成 `((ThrV,(ThrM,ThrN)),(FrgV,(RestM,RestN)))`：
  ```cpp
  logical_divide(ctensor, t_tile)      // (PermM,PermN)               — permutation
  zipped_divide(t_tensor, c_tile)      // ((AtomM,AtomN),(RestM,RestN)) — atom 切块
  c_tensor.compose(AtomLayoutC_TV{})   // ((ThrV,FrgV),(RestM,RestN))  — TV 变换
  zipped_divide(tv_tensor, thr_tile)   // ((ThrV,(ThrM,ThrN)),(FrgV,..)) — 线程分组
  ```
- `include/cute/atom/mma_atom.hpp:460-523` — `ThrMMA::partition_C/A/B`
  - 对 `thrfrg_C` 结果按当前线程坐标切片
- `include/cute/algorithm/gemm.hpp:100-500` — 5 层 dispatch
  ```
  Dispatch 1: (V) × (V) → (V)            标量乘加
  Dispatch 2: (M) × (N) → (M,N)            外积
  Dispatch 3: (M,K) × (N,K) → (M,N)        矩阵乘 → 升维到 5
  Dispatch 4: (V,M) × (V,N) → (V,M,N)      批量外积，寄存器复用
  Dispatch 5: (V,M,K) × (V,N,K) → (V,M,N)  最终展开
  ```
- `include/cute/algorithm/gemm.hpp:260-390` — Dispatch 4 的 **serpentine 遍历**
  ```cpp
  int ns = (m & 1) ? N-1-n : n;  // 奇数行反向，最大化寄存器复用
  ```

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
