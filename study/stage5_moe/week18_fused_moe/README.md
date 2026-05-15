# Week 18 — Fused MoE

预计 ~15h
> **硬件**：🟢 5060 Ti（用 SM120 array TMA + WarpSpec 跑 fused MoE 框架；FP4 量化 expert 也能在本地试）｜ 🟡 H20（WGMMA 实测 production 性能数字）

## 目标
- 完成 Stage 5 CHECKPOINT：end-to-end MoE forward
- 实测 H20 上的几个 fusion 优化
- 理解 FP8 expert weight 的 scaling

## 读
- `examples/python/CuTeDSL/blackwell/grouped_gemm.py` — 高层 API 参考
- `include/cutlass/gemm/kernel/sm90_tile_scheduler.hpp` 的 `get_current_work()` 动态调度

## H20 上的 MoE 优化手段

| 优化 | 收益来源 |
|------|----------|
| 动态 tile scheduler | 让忙的 expert 多占 SM，避免长尾 |
| Gate + GEMM fusion | 省一次 X 加载 |
| Permute + grouped GEMM fusion | A 矩阵不落 GMEM，直接从 reordered smem 进 WGMMA |
| FP8 expert weight | weight 减半，mem bw 翻倍可用 |
| SwiGLU fused 在 first GEMM 的 epilogue | 省一遍 hidden buffer 读写 |

## 写
- `exercises/ex_moe_forward.cu` — end-to-end：
  - 8 expert，topk=2，hidden=2048，inter=8192，FP16
  - router + permute + grouped GEMM (`X @ W_up`, `X @ W_gate`) + swiglu + grouped GEMM (`@ W_down`) + unpermute
  - 跟 PyTorch ref 对比 rtol=1e-2

- `exercises/PERFORMANCE.md` — 至少 3 个 baseline：
  - naive 4 个 kernel 串联
  - 加 dynamic scheduler
  - 加 SwiGLU fusion

## 跑
```bash
python ref_torch_moe.py  # 生成参考
make study_stage5_w18_ex_moe_forward -j
./study_stage5_w18_ex_moe_forward
```

## 自检
1. 你的 grouped GEMM 跟 W16 的 minimal 比，多了什么？
2. SwiGLU `silu(x*W_gate) * (x*W_up)` 在哪一层 fuse 最划算（first GEMM epilogue 还是单独 elementwise）？
3. FP8 weight 的 scaling factor 在 GMMA 里怎么应用？accumulator 在哪转 FP32？
4. expert load 不均时，dynamic scheduler 拿到 tile id 后查表 + atomic 分配，开销多大？
5. 你的 end-to-end MoE forward 比 naive 串联快多少？瓶颈在哪？
