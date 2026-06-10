# Stage 7 — 极致调优（持续进行）

不再按周划分。每完成一个新 kernel，就跑一遍这里的 checklist 并把数据写进 baseline 文档。

**本阶段就是总目标的验收场**：任何 compute-bound kernel 在 B200 上 tensor core 利用率（`sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`）≥ 70%；memory-bound kernel（decode / routing）`dram__throughput` 贴近 roofline。

## 目录

- [profiling_recipes.md](profiling_recipes.md) — ncu / nsys 命令收藏
- [b200_baselines.md](b200_baselines.md) — B200 上跑过的所有 kernel + ncu 数据

## 开放题（总目标终极验收）

任选一个**课程计划之外**的 tensor core 算子（候选：conv、量化 attention、LoRA batched GEMM、attention sink/sliding window 变体），从零写到 ≥ 70% TC 利用率。过了这题，"能写任意 CuTe kernel"才算成立——并把它收进开源算子库。

## 通用 checklist

每个 kernel 都过一遍：

- [ ] smem layout 加 swizzle，bank conflict = 0
- [ ] TMA descriptor 128B 对齐
- [ ] pipeline depth 调到 sweet spot（一般 4 或 8）
- [ ] 开 `ClusterShape<2,1,1>` 或 `<2,2,1>` 看 multicast / 2-SM UMMA 收益
- [ ] epilogue 用 EVT 把 alpha*C+beta*D / bias / activation fuse 掉
- [ ] FP8 / NVFP4 blockscaled 用于推理（如果 dtype 允许）
- [ ] persistent kernel + CLC 减少 launch overhead
- [ ] register usage（`-Xptxas=-v`）没 spill
- [ ] occupancy 至少 1 块/SM

## B200 硬件参数（速查）

| 参数 | 值 | 影响 |
|------|----|------|
| SM 数量 | 148 (per die) | 双 die，需要考虑 cross-die |
| UMMA 峰值 (FP16) | ~2.25 PFLOPS | TC 利用率分母 |
| HBM3e 带宽 | ~8 TB/s | memory bound 上界 |
| TMEM/SM | 256 KB | 累加器容量 / pipeline depth 上界 |
| smem/SM | 228 KB | tile size 上界 |

## Roofline 速算

```
算术强度 = FLOP / Bytes

GEMM(M,N,K):   FLOPs = 2MNK,   Bytes = 2(MK+NK+MN)
FA(B,H,S,d):   算术强度 ≈ S/8（d=128 时约 16），decode 阶段 memory bound
```

B200 ridge point ≈ 2.25e15 / 8e12 ≈ **280 FLOP/Byte**（FP16）。
低于这个数就是 memory bound——此时优化目标切到带宽利用率，TC 利用率指标失效。

## 永远先量后改

- 先 `ncu --set full` 拍下 baseline，记到 baseline 文档
- 改一处，跑一次，对比 baseline，确认改动方向对
- 不要批量优化（一次改一项才能归因）
