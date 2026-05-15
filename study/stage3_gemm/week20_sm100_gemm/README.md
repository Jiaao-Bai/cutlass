# Week 20 — SM100 GEMM

预计 ~15h
> **硬件**：🟢 5060 Ti（读源码 + 静态编译 `sm_100a`，主要是 UMMA atom + TMEM 分配的代码理解）｜ 🔴 B200（实测真 UMMA + TMEM + 2-SM 配对 cluster 的性能数字）  
> 本周承接 W8-W11 的 SM90 GEMM——核心改动是 MMA atom 从 WGMMA → UMMA，累加器从 RMEM → TMEM，其余 mainloop 框架 90% 不变

## 目标
- 把 Stage 3 v3 移植到 SM100，跑出 ≥ 70% cuBLAS（B200 FP16）
- 理解 SM100 pipeline 引入 TMEM consumer 后语义的变化
- 理解 2-CTA cluster

## 读
- `include/cutlass/pipeline/sm100_pipeline.hpp` — pipeline 在 SM100 的修改
- `include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp` — SM100 kernel 骨架
- `include/cutlass/gemm/collective/sm100_mma_*.hpp` — SM100 mainloop
- `examples/`（搜 sm100 / blackwell 的 GEMM example）
- `examples/python/CuTeDSL/blackwell/dense_gemm.py`

## SM90 → SM100 移植清单

| 项 | SM90 写法 | SM100 写法 |
|----|-----------|------------|
| MMA atom | `SM90_64x128x16_F32F16F16_SS` | `SM100_MMA_F16BF16_SS` 系列 |
| C accumulator | rmem 的 `tCrC` | tmem 的 `tCtAcc` |
| Epilogue | rmem → smem → gmem (TMA store) | tmem → rmem → smem → gmem |
| Pipeline | `PipelineTmaAsync` | `PipelineTmaUmmaAsync`（新增 tmem 同步） |
| Cluster | `<2,1,1>`（multicast A）| `<2,1,1>` 但 2-CTA 也参与 MMA |

## 写
- `exercises/ex_sm100_gemm.cu` — 在 Stage 3 v3 基础上做最少改动移植
- `exercises/MIGRATION_DIFF.md` — 记录每行 diff 的语义解释

## 跑
```bash
cmake -DCUTLASS_ENABLE_STUDY=ON -DCUTLASS_NVCC_ARCHS=100a ..
make study_stage6_w20_ex_sm100_gemm -j
./study_stage6_w20_ex_sm100_gemm 8192 8192 8192
ncu --set full -o sm100_gemm ./study_stage6_w20_ex_sm100_gemm 8192 8192 8192
```

## 自检
1. SM100 上 cuBLAS FP16 峰值大约多少 TFLOPS？你的实现到了多少？
2. UMMA 的 K 维度一次累加多少元素？跟 WGMMA 的 16 比是多少？
3. 2-CTA cluster 模式下，谁负责发 UMMA？
4. TMEM 容量 256KB，pipeline depth 受什么限制？
5. SM100 pipeline 多了哪个 mbarrier？做什么用？
