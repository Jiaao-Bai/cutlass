# Stage 7 — 极致调优（持续进行）

不再按周划分。每完成一个新 kernel，就跑一遍这里的 checklist 并把数据写进 baseline 文档。

## 目录

- [profiling_recipes.md](profiling_recipes.md) — ncu / nsys 命令收藏
- [h20_baselines.md](h20_baselines.md) — 你在 H20 上跑过的所有 kernel + ncu 数据
- [b200_baselines.md](b200_baselines.md) — 同上，B200

## 通用 checklist

每个 kernel 都过一遍：

- [ ] smem layout 加 swizzle，bank conflict = 0
- [ ] TMA descriptor 128B 对齐
- [ ] pipeline depth 调到 sweet spot（一般 4 或 8）
- [ ] 开 `ClusterShape<2,1,1>` 或 `<2,2,1>` 看 multicast 收益
- [ ] epilogue 用 EVT 把 alpha*C+beta*D / bias / activation fuse 掉
- [ ] FP8 + blockwise scaling 用于推理（如果 dtype 允许）
- [ ] persistent kernel 减少 launch overhead
- [ ] register usage（`-Xptxas=-v`）没 spill
- [ ] occupancy 至少 1 块/SM

## H20 硬件参数（速查）

| 参数 | 值 | 影响 |
|------|----|------|
| SM 数量 | 132 | 最大并发 CTA 数 |
| WGMMA 峰值 (FP16) | ~990 TFLOPS | 理论上界 |
| HBM3 带宽 | ~4 TB/s | memory bound 上界 |
| L2 Cache | 60 MB | 大问题尺寸下要注意复用 |
| smem/SM | 228 KB | tile size 上界 |

## B200 硬件参数（速查）

| 参数 | 值 | 影响 |
|------|----|------|
| SM 数量 | 148 (per die) | 双 die，需要考虑 cross-die |
| UMMA 峰值 (FP16) | ~2.25 PFLOPS | 比 H20 ~2.25× |
| HBM3e 带宽 | ~8 TB/s | 比 H20 2× |
| TMEM/SM | 256 KB | 新增层次 |

## Roofline 速算

```
算术强度 = FLOP / Bytes

GEMM(M,N,K):   FLOPs = 2MNK,   Bytes = 2(MK+NK+MN)
FA(B,H,S,d):   算术强度 ≈ S/8（d=128 时约 16），decode 阶段 memory bound
```

H20 ridge point ≈ 990e12 / 4e12 ≈ **247 FLOP/Byte**。
低于这个数就是 memory bound。

## 永远先量后改

- 先 `ncu --set full` 拍下 baseline，记到 baseline 文档
- 改一处，跑一次，对比 baseline，确认改动方向对
- 不要批量优化（一次改一项才能归因）
