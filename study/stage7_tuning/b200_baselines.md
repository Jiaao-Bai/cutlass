# B200 Baselines

格式同 [h20_baselines.md](h20_baselines.md)。命名规则：`stage6_wMM_<exercise>`。

## GEMM

| Kernel | M/N/K | dtype | TFLOPS | % cuBLAS | sm__throughput | DRAM BW | TMEM 占用 | ncu report |
|--------|-------|-------|--------|----------|----------------|---------|-----------|-------------|
| ex_sm100_gemm (W20) | 8192³ | FP16 | | | | | | |

cuBLAS B200 FP16 8192³ 参考：≈ 待补。

## FA

| Kernel | (B,H,S,d) | causal | dtype | TFLOPS | DRAM BW | TMEM 占用 | ncu report |
|--------|-----------|--------|-------|--------|---------|-----------|-------------|
| ex_sm100_fa_fwd (W21) | (4,32,4096,128) | T | FP16 | | | | |

## H20 vs B200 对比

填完 W20/W21 后在这里整理：

| Workload | H20 TFLOPS | B200 TFLOPS | 提速 | 主要来源（compute / mem / 算法） |
|----------|------------|-------------|------|-----------------------------------|
| GEMM 8192³ FP16 | | | | |
| FA (4,32,4096,128) | | | | |
