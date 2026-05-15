# Week 18 — SM100 FA

预计 ~15h
> **硬件**：🟢 5060 Ti（读源码 + 静态编译 `sm_100a`）｜ 🔴 B200（实测 UMMA + TMEM 的 FA 性能数字）  
> 本周承接 W15-W17 的 SM90 FA——核心改动是 MMA atom WGMMA → UMMA，accumulator RMEM → TMEM

## 目标
- 跑通 `examples/python/CuTeDSL/blackwell/fmha.py` 与 `fmha_bwd.py`
- 自写 SM100 FA fwd（基于 Stage 4 v2 + Stage 3 W13 的 SM100 GEMM 经验）
- 写完 Stage 6 CHECKPOINT：`B200_VS_H20.md`

## 读
- `examples/python/CuTeDSL/blackwell/fmha.py` — SM100 FA fwd 参考实现（Python，逻辑清晰）
- `examples/python/CuTeDSL/blackwell/fmha_bwd.py` — bwd 参考
- 如果有 C++ SM100 FA example，对照读

## SM100 FA 的关键差异（vs SM90）

1. P（softmax 结果）落 TMEM 还是 RMEM？落 TMEM 时 P @ V 的 UMMA 怎么发？
2. 累加器 O 在 TMEM 后，softmax rescale `O *= exp(m_old - m_new)` 怎么做（TMEM→RMEM→TMEM）？
3. 2-CTA cluster 在 FA 里能用吗？Q tile 怎么切给 2 个 CTA？

## 写
- `exercises/ex_sm100_fa_fwd.cu` — SM100 FA fwd v1，正确性优先
- `exercises/B200_VS_H20.md` — Stage 6 CHECKPOINT 文档：
  - GEMM 算法层 0 改动，atom 层一对一替换
  - FA 算法层 0 改动，但 P 与 O 的存放位置触发的 layout 变更要列清楚
  - Pipeline 同步多出来的 tmem mbarrier
  - 性能差距：H20 vs B200 在 attention 上的提速主要来自哪些维度

## 跑
```bash
cmake -DCUTLASS_ENABLE_STUDY=ON -DCUTLASS_NVCC_ARCHS=100a ..
python examples/python/CuTeDSL/blackwell/fmha.py
python examples/python/CuTeDSL/blackwell/fmha_bwd.py
make study_stage4_w18_ex_sm100_fa_fwd -j && ./study_stage4_w18_ex_sm100_fa_fwd
```

## 自检
1. P 在 RMEM vs TMEM 的选择对 register pressure 有多大影响？
2. UMMA 的 A 矩阵能从 TMEM 读吗？P @ V 的 P 是 A 还是 B？
3. softmax rescale 涉及 elementwise 运算，TMEM 不能直接做 elementwise，怎么搬运最高效？
4. 2-CTA cluster 在 FA 里收益小于 GEMM，为什么？
5. B200 上 FA 的 ncu Roofline 是 compute-bound 还是 memory-bound？d=128 时？
