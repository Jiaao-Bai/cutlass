# Week 3 — TiledMMA / MMA_Atom

预计 ~15h ｜ 目标硬件：H20（SM80 也可，先看 mma_sm80）

## 目标
- 能解释 `MMA_Atom` 三层包装：PTX → `MMA_Traits` → `MMA_Atom`
- 能从 `LayoutA_TV` 推出"哪个线程负责 A 矩阵的哪些元素"
- 能解释 `cute::gemm` 5 层 dispatch 的每一层意义
- 能解释 serpentine 遍历为什么省寄存器

## 读
- `include/cute/atom/mma_atom.hpp:42-200` — `MMA_Atom` 三层包装
  ```
  MMAOperation（PTX 裸指令）
      ↓ MMA_Traits<Op>    ：ThrID + LayoutA_TV + LayoutB_TV + LayoutC_TV
      ↓ MMA_Atom<Traits>  ：call() + make_fragment_A/B/C()
  ```
  - `LayoutA_TV`（TV = Thread × Value）= "这条指令的 A 矩阵，哪些线程负责哪些元素"，硬件指令到 CuTe 抽象的桥
- `include/cute/atom/mma_atom.hpp:250-380` — `TiledMMA::thrfrg_C/A/B`
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
  Dispatch 1: (V) × (V) → (V)
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
make study_stage1_w03_ex07_tiled_mma_layout -j && ./study_stage1_w03_ex07_tiled_mma_layout
make study_stage1_w03_ex08_serpentine_count -j
```

## 自检
1. `MMA_Traits` 里的 `ThrID` 和 `LayoutA_TV` 是什么关系？为什么需要两个？
2. `thrfrg_C` 的 4 步变换中，哪一步把"硬件 TV"接到了"用户 (M,N)"上？
3. 为什么 Dispatch 5 要把 K 维拆出来单独处理？
4. serpentine 遍历能省下多少寄存器（数量级）？为什么省？
5. SM80 的 `mma.m16n8k16` 一次需要多少线程？每个线程持有多少 A / B / C 元素？
