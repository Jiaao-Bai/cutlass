# B200 Baselines

每跑通一个 kernel 记一行。**TC util% = `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`**（总目标口径，≥ 70% 为过线）。

## GEMM

| Kernel | M/N/K | dtype | TFLOPS | TC util% | % cuBLAS | DRAM BW | TMEM 占用 | ncu report |
|--------|-------|-------|--------|----------|----------|---------|-----------|-------------|
| ex_warpspec_gemm_v3 (W12) | 8192³ | FP16 | | | | | | |
| ex_nvfp4_gemm (W13) | 8192³ | NVFP4 | | | | | | |

cuBLAS B200 FP16 8192³ 参考：≈ 待补。

## FA

| Kernel | (B,H,S,d) | causal | dtype | TFLOPS | TC util% | DRAM BW | ncu report |
|--------|-----------|--------|-------|--------|----------|---------|-------------|
| ex_fa_fwd_v2 (W16) | (4,32,4096,128) | T | FP16 | | | | |
| ex_fa_decode (W18) | (B,8/1,KV=32k,128) | — | FP16 | —(mem bound) | — | | |

## MoE

| Kernel | 配置 | dtype | grouped GEMM TC util% | DRAM BW | ncu report |
|--------|------|-------|------------------------|---------|-------------|
| ex_moe_forward (W21) | 8 expert, topk=2, h=2048, i=8192 | FP16 | | | |
