# Week 2 — Composition + Complement + Swizzle

预计 ~15h ｜ 目标硬件：H20

## 目标
- 能用 `composition / complement / divide` 推出任意分块 layout
- 能解释 `Swizzle<B,M,S>` 的 XOR 公式如何消 bank conflict
- 能脱离 cutlass::gemm，**只用 CuTe** 写出正确的 FP16 GEMM

## 读
- `include/cute/layout.hpp:1021-1165` — `composition`
  - `lhs o rhs` 含义 `result(c) = lhs(rhs(c))`
  - fold 循环里 `divmod` 把 rhs 的 stride 在 lhs 的 shape 序列里找位置
- `include/cute/layout.hpp:1164-1260` — `complement`
  - 排序 + 折叠找出未被覆盖的地址
  - `logical_divide` / `tiled_divide` 都依赖它
- `include/cute/swizzle.hpp:42-130` — `Swizzle<B,M,S>`
  ```cpp
  struct Swizzle<B, M, S> {
      operator()(offset) => offset ^ XOR_of_specific_bits
  }
  ```
  - 看 `make_swizzle` 与 `composition(Swizzle, Swizzle)` 的化简
- `media/docs/cpp/cute/02_layout_algebra.md` — 代数文档

## 写
- `exercises/ex04_composition.cu` + `ex04_composition_paper.md` — 预测 `composition` 结果再跑
- `exercises/ex05_swizzle_paper.md` — Swizzle bit math 手算（ncu 对照延迟到 W9 配 WGMMA 实战）
- `exercises/ex06_hgemm_naive.cu` — **已完成**：纯 CuTe FP16 GEMM，单线程一元素，CPU ref check

## 跑
```bash
cmake -DCUTLASS_ENABLE_STUDY=ON -DCUTLASS_NVCC_ARCHS=90a ..
make study_stage1_w02_ex06_hgemm_naive -j
./study_stage1_w02_ex06_hgemm_naive 256 256 128
# 期望：PASSED! All 65536 elements match.
```

参考：`examples/cute/tutorial/sgemm_1.cu`、`sgemm_2.cu`。

## 自检
1. `composition(Layout<_8, _1>, Layout<_4, _2>)` 等于？画出几何形状。
2. `complement(Layout<_4, _3>, _12)` 等于？为什么需要 cosize 参数？
3. `Swizzle<3,4,3>` 在每行 128B 的 smem 上交换了哪些 bit？为什么这个公式跟元素类型（fp32/fp16/fp8）无关？
4. WGMMA 要求的 smem 是 16-byte 对齐还是 128-byte 对齐？跟 swizzle 参数怎么关联？
5. 当前 `ex06_hgemm_naive` 的算术强度大约多少？memory bound 还是 compute bound？为什么慢？
