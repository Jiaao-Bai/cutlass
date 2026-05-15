# Week 14 — FA Forward v2：优化

预计 ~15h
> **硬件**：🟢 5060 Ti（继续 SM120 mainloop 上优化 softmax/causal/pipeline）｜ 🟡 H20（真 WGMMA 性能 vs 88_hopper_fmha 基准）  
> 🟢 顺带可在 5060 Ti 试 **FP4/FP6 量化 FA**（`SM120_*_TN<e2m1, e2m1>` atom）

## 目标
- v2 性能 ≥ 80% `examples/88_hopper_fmha`
- 加 causal mask 支持
- 加 persistent kernel

## 读
- `examples/88_hopper_fmha/kernel/sm90_fmha_fwd_kernel_tma_warpspecialized.hpp` — kernel 入口
- `examples/88_hopper_fmha/collective/sm90_fmha_fwd_softmax_*.hpp` — softmax 实现细节
- `include/cutlass/gemm/kernel/sm90_tile_scheduler.hpp` — persistent scheduler，复用到 FA

## H20 上的 FA 优化重点

| 优化 | 预期收益 | 备注 |
|------|---------|------|
| causal 上三角 tile 整体跳过 | causal 场景 +50% | 注意 tile scheduler 也要跳 |
| persistent kernel | +10% | (B,H) 多时收益大 |
| pipeline depth 调到 3 | +5% | smem 紧张，权衡 K/V 同时 prefetch |
| Q stay in smem（不重复加载） | +5% | row tile 内多次复用 |
| Pingpong 两 warpgroup 错开 GEMM2 | +10% | WGMMA 发射满载 |

H20 经验：
- Head dim 选择：d=64/128 最优，d=96 需要 padding
- Seq len 分块：tile size 根据 smem 容量决定（SM90 smem 最大 228KB/CTA）
- Persistent kernel：用 tile scheduler 减少 kernel launch overhead

## 写
- `exercises/ex_fa_fwd_v2.cu` — 在 v1 基础上：
  - causal mask（对角线 tile predication，上三角整体跳）
  - persistent scheduler（复用 GEMM 的）
  - Pingpong 双 warpgroup
- `exercises/PERFORMANCE.md` — 跑 4 组 shape：

| Shape (B,H,S,d) | causal | 类型 |
|-----------------|--------|------|
| (4, 32, 4096, 128) | True | 训练典型 |
| (1, 32, 16384, 128) | True | 长上下文 |
| (32, 32, 1024, 128) | True | 大 batch |
| (1, 16, 8192, 64) | False | 小 head dim |

## 跑
```bash
make study_stage4_w14_ex_fa_fwd_v2 -j
./study_stage4_w14_ex_fa_fwd_v2 4 32 4096 128
ncu --set full -o fa_v2 ./study_stage4_w14_ex_fa_fwd_v2 4 32 4096 128
```

## 自检
1. Causal 下，上三角 tile 在 tile scheduler 里跳过，比在 kernel 里 predication 快多少（数量级）？
2. Pingpong 在 FA 里，两个 warpgroup 各负责什么？是错开 GEMM1 还是 GEMM2？
3. d=128 + BlockM=128 + BlockN=128 + pipeline=3，smem 用量算一下，超 228KB 吗？
4. Q stay in smem 节省的是什么？为什么 S 长时收益反而下降？
5. 你的 v2 在 ncu Roofline 上是 compute-bound 还是 memory-bound？
