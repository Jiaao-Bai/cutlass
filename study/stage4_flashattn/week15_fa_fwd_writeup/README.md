# Week 15 — FA Forward v1（正确性优先）

预计 ~15h
> **硬件**：🟢 5060 Ti（用 SM120 mainloop 在本地跑 FA fwd 正确性）｜ 🟡 H20（WGMMA 实测）

## 目标
- 自写 FA forward kernel，**复用 Stage 3 的 WarpSpec 骨架**
- 跑出正确结果（PyTorch SDPA 对比 rtol=1e-2）
- 性能可以慢，正确优先

## 读
- 复习 `study/stage3_gemm/week10_warpspec_writeup/exercises/ex_warpspec_gemm_v1.cu`（你的 GEMM v1）
- `examples/88_hopper_fmha/collective/sm90_fmha_fwd_*.hpp` — 重点看两次 GEMM 之间 softmax 怎么插入

## 写
- `exercises/ex_fa_fwd_v1.cu`
  - 参数：`(B, H, S_q, S_kv, d) = (1, 8, 1024, 1024, 128)` FP16
  - causal=False（先简单）
  - WarpSpec：1 个 producer warp（TMA Q/K/V），1 个 consumer warpgroup
  - 两次 GEMM：
    1. `S = Q @ K^T`（在寄存器）
    2. softmax → P（仍在寄存器）
    3. `O += P @ V`
  - 不做 persistent，每个 (B,H,row_tile) 一个 CTA
- `exercises/ref_torch.py` — 用 `torch.nn.functional.scaled_dot_product_attention` 算参考结果，写到 binary，C++ 读进来对比

## 跑
```bash
python study/stage4_flashattn/week15_fa_fwd_writeup/exercises/ref_torch.py  # 生成参考
make study_stage4_w15_ex_fa_fwd_v1 -j
./study_stage4_w15_ex_fa_fwd_v1
# 期望：max abs diff < 0.01，PASSED
```

## 自检
1. P（softmax 结果）放寄存器，对 register pressure 多大？d=128 时大概多少 bytes/thread？
2. P @ V 这一步，P 的"layout"和 WGMMA 期望的 A 矩阵 layout 是否对得上？需要 reorder 吗？
3. 你的 K^T 是真的转置（在 smem 里物理转）还是用 layout 玩出来？哪种省事？
4. softmax 的 row reduce 在 warpgroup 内怎么做？用 shfl 还是用 smem？
5. 你的 v1 比 88_hopper_fmha 慢多少？瓶颈是什么（softmax 同步/WGMMA 占比/...）？
