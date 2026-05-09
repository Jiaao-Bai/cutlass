# Week 7 — Hardware Barrier + Thread Block Cluster

预计 ~15h ｜ 目标硬件：H20

## 目标
- 能解释 `mbarrier` 比 `__syncthreads` 强在哪
- 能用 `PipelineTmaAsync` 写 TMA producer / MMA consumer 同步
- 完成 Stage 2 CHECKPOINT：minimal warpspec ping-pong toy

## 读
- `include/cutlass/pipeline/sm90_pipeline.hpp` — `PipelineAsync` / `PipelineTmaAsync`
  - `PipelineState`：循环 buffer 的 index + phase，避免 ABA
  - 4 步协议：`producer_acquire` → `producer_commit` → `consumer_wait` → `consumer_release`
  - `spread_arrivals_to_warpgroup`：barrier arrive 均摊到 warpgroup
- `media/docs/cpp/pipeline.md` — barrier 设计思想（必读）
- `include/cute/arch/cluster_sm90.hpp` — cluster 同步 API
- `include/cutlass/arch/barrier.h` — 底层 mbarrier wrapper

## 写
- `exercises/ex15_mbarrier_basic.cu` — 单个 mbarrier 在 1 个 warp 内 arrive / wait，验证 phase 翻转
- `exercises/ex16_pipeline_state.cu` — 用 `PipelineState` 做循环 buffer，trace 出 (index, phase) 序列
- **CHECKPOINT**：`exercises/ex_warpspec_pingpong_toy.cu`
  - 1 个 producer warp + 2 个 consumer warpgroup 做 reduction（不是 GEMM）
  - 跑 4 轮，CPU ref 验证

## 跑
```bash
make study_stage2_w07_ex15_mbarrier_basic -j && ./study_stage2_w07_ex15_mbarrier_basic
make study_stage2_w07_ex16_pipeline_state -j && ./study_stage2_w07_ex16_pipeline_state
make study_stage2_w07_ex_warpspec_pingpong_toy -j && ./study_stage2_w07_ex_warpspec_pingpong_toy
```

## 自检
1. `mbarrier.arrive` 和 `bar.sync` 在硬件上是同一种实现吗？
2. `PipelineState` 的 phase 为什么不能用 index % depth 替代？
3. `producer_acquire` 阻塞在等待什么？被谁释放？
4. Cluster 内的 `bar.sync` 如何跨 CTA 工作？DSMEM 地址用 `cute::cluster_to_smem_ptr` 转换的语义是什么？
5. 在你的 ping-pong toy 里，最少需要几个 mbarrier？为什么不能用 1 个？
