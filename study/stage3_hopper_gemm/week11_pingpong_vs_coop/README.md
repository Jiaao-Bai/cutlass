# Week 11 — Pingpong vs Cooperative

预计 ~15h ｜ 目标硬件：H20

## 目标
- 写出 Pingpong 与 Cooperative 两个变体，理解骨架差异
- 在 H20 上做 benchmark，找出各自的最优工作负载
- 完成 Stage 3 CHECKPOINT：v3 ≥ 70% cuBLAS

## 读
- `include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp` — Pingpong 完整骨架
  - 关注 `is_producer_warp` 分支与两个 MMA warpgroup 的 ping-pong 切换
- `include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp` — Cooperative
  - 关注两个 warpgroup 协同算同一 tile 时的 accumulator 处理
- `media/docs/cpp/cutlass_3x_design.md` 里关于 KernelSchedule 的章节

## 两种模式对比

| 维度 | Pingpong | Cooperative |
|------|----------|-------------|
| MMA warpgroup 数 | 2，交替算不同 tile | 2，协同算同一 tile |
| 隐藏的延迟 | WGMMA 发射延迟 | WGMMA 发射 + epilogue 延迟 |
| 适合场景 | 大矩阵（M、N 都大） | M 小 / batch 小 / K 大 |
| H20 经验 | prefill / 大批次推理 | decode / KV cache 长序列 |

## 写
- `exercises/ex_warpspec_gemm_v3_pingpong.cu` — 在 v2 基础上加 ping-pong 切换逻辑
- `exercises/ex_warpspec_gemm_v3_cooperative.cu` — Cooperative 变体
- `exercises/bench.sh` — 跑下面 6 组 shape，记 TFLOPS

| Shape (M,N,K) | 类型 |
|---------------|------|
| 8192,8192,8192 | 大 GEMM |
| 4096,4096,4096 | 中等 |
| 128,8192,8192 | 短 M（decode-like） |
| 8192,128,8192 | 短 N |
| 4096,4096,16384 | 大 K |
| 16,8192,4096 | 极短 M |

- `exercises/PERFORMANCE.md` — 记录 6 组结果 + 对比 cuBLAS

## 跑
```bash
make study_stage3_w11_ex_warpspec_gemm_v3_pingpong -j
make study_stage3_w11_ex_warpspec_gemm_v3_cooperative -j
bash study/stage3_hopper_gemm/week11_pingpong_vs_coop/exercises/bench.sh
```

## 自检
1. Pingpong 切换两个 warpgroup 时，accumulator 寄存器怎么不被覆盖？
2. Cooperative 模式下两个 warpgroup 各算 tile 的一半，最终 accumulator 怎么合并？
3. 在 M=128 的 decode 场景为什么 Pingpong 不划算？
4. Pingpong 的 register 用量比 Cooperative 高还是低？为什么？
5. 你的 v3 在 6 个 shape 中哪几个超过 70% cuBLAS？哪几个不达标？瓶颈分别是什么？
