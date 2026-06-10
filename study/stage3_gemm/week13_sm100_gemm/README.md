# Week 13 — Blockscaled 量化 GEMM（NVFP4 / FP8，SM100）

预计 ~15h ｜ 目标硬件：B200（SM100）
> 本周承接 W10-W12 的 dense GEMM——把 v3 改造成 **blockscaled 量化 GEMM**。这是 LLM 推理的主力 GEMM 形态（weight FP4/FP8 + microscaling），也是未来开源算子库的核心 kernel。零件在 W8 已学过（`tcgen05.mma.kind::f8f6f4/mxf4/nvf4`，THINKING O31），本周实跑。

## 目标
- 把 v3 的 FP16 路径改成 FP8（`kind::f8f6f4`，无块缩放），跑出正确结果 + 性能数字
- 再改成 NVFP4 blockscaled（`kind::nvf4`，block 16 + E4M3 scale），理解 scale factor 怎么进 UMMA
- 能解释 mxf4 vs nvf4 vs f8f6f4 三条 PTX 通路的差异和取舍

## 读（由浅入深）
1. `examples/72_blackwell_narrow_precision_gemm/` — SM100 narrow precision GEMM 入门：nvfp4/mxfp 模板参数怎么填、SF tensor 怎么构造
2. `include/cutlass/gemm/collective/sm100_blockscaled_mma_warpspecialized.hpp` — SM100 blockscaled mainloop：SFA/SFB（scale factor）tensor 怎么随 A/B 一起 TMA 进来、怎么喂给 `tcgen05.mma`
3. `include/cute/atom/mma_traits_sm100.hpp` — blockscaled UMMA atom 的 traits：SF 操作数的 layout 约定
4. `media/docs/cpp/blackwell_functionality.md` — f8f6f4 / mxf4 / nvf4 的格式与精度权衡（对照 THINKING O31）

## 写
- `exercises/ex_fp8_gemm.cu` — v3 改 FP8（无块缩放），最少改动：dtype + MMA atom + epilogue 反量化
- `exercises/ex_nvfp4_gemm.cu` — NVFP4 blockscaled 版：A/B 量化 + SFA/SFB 生成在 host 端，kernel 里 SF 随 tile 进 smem
- `exercises/QUANT_NOTES.md` — 记录：SF 的 TMA/smem layout、精度损失实测（vs FP16 reference 的误差分布）、三种 kind 的性能对比

## 跑
```bash
cmake -DCUTLASS_ENABLE_STUDY=ON -DCUTLASS_NVCC_ARCHS=100a ..
make study_stage3_w13_ex_nvfp4_gemm -j
./study_stage3_w13_ex_nvfp4_gemm 8192 8192 8192
ncu --set full -o nvfp4_gemm ./study_stage3_w13_ex_nvfp4_gemm 8192 8192 8192
```
- 期望：FP8 ≥ 80% cuBLAS FP8；NVFP4 吞吐 > FP8（理论 2×，实际看 SF 开销）

## 自检
1. `kind::f8f6f4` / `kind::mxf4` / `kind::nvf4` 三条通路：block size、scale 格式、适用场景各是什么？
2. SFA/SFB 在 smem 里的 layout 跟 A/B 的 swizzle layout 是什么关系？SF 也要 swizzle 吗？
3. NVFP4 GEMM 的 K-loop 里，scale factor 的消耗节奏跟 A/B tile 一致吗（block 16 意味着每 K=16 一个 SF）？
4. 量化 GEMM 的精度验证不能用 rtol=1e-2 的逐元素比对——应该怎么验（误差分布 / 相对 Frobenius 范数）？
5. FP4 理论吞吐是 FP16 的 4×，你实测是多少？没到的损失在哪（SF 加载 / 反量化 epilogue / TMA 粒度）？
