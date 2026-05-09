# H20 Baselines

每跑通一个 kernel 就来这里加一行。命名规则：`stageN_wMM_<exercise>`。

## GEMM 系列

| Kernel | M/N/K | dtype | TFLOPS | % cuBLAS | sm__throughput | DRAM BW | wgmma 指令 | ncu report |
|--------|-------|-------|--------|----------|----------------|---------|------------|-------------|
| ex06_hgemm_naive (W2) | 256/256/128 | FP16 | <待测> | — | — | — | 0 | — |
| ex_sgemm_sm80_variant (W4) | 4096³ | FP32 | | | | | | |
| ex_warpspec_gemm_v1 (W9) | 4096³ | FP16 | | | | | | |
| ex_warpspec_gemm_v2 (W10) | 4096³ | FP16 | | | | | | |
| ex_warpspec_gemm_v3_pingpong (W11) | 4096³ | FP16 | | | | | | |
| ex_warpspec_gemm_v3_pingpong (W11) | 8192³ | FP16 | | | | | | |
| ex_warpspec_gemm_v3_cooperative (W11) | 128/8192/8192 | FP16 | | | | | | |

cuBLAS H20 FP16 4096³ 参考：≈ 850 TFLOPS（hgemm，non-strided）。

## FA 系列

| Kernel | (B,H,S,d) | causal | dtype | TFLOPS | % 88_fmha | DRAM BW | ncu report |
|--------|-----------|--------|-------|--------|-----------|---------|-------------|
| ex_fa_fwd_v1 (W13) | (1,8,1024,128) | F | FP16 | | | | |
| ex_fa_fwd_v2 (W14) | (4,32,4096,128) | T | FP16 | | | | |
| ex_fa_fwd_v2 (W14) | (1,32,16384,128) | T | FP16 | | | | |

## MoE

| Kernel | 配置 | 算子 | runtime (ms) | vs PyTorch baseline | ncu report |
|--------|------|------|---------------|----------------------|-------------|
| ex_moe_forward (W18) | E=8 topk=2 H=2048 I=8192 | end-to-end fwd | | | |

## 调优记录

每次某项优化（W10 / W14 / W18）的前后对比：

| 日期 | kernel | 改动 | TFLOPS before | TFLOPS after | 备注 |
|------|--------|------|---------------|--------------|------|
| | | | | | |
