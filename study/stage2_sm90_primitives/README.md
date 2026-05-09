# Stage 2 — SM90 硬件原语

预计 3 周（W5–W7），约 45h。

## 阶段目标

- 能解释 WGMMA 的 RS 模式 vs SS 模式各自适用何时
- 能在 host 构造 TMA descriptor，在 kernel 里发起 `cp.async.bulk`
- 能用 `mbarrier` 写 producer / consumer 同步，理解 phase / arrive / wait
- 完成一个 minimal warpspec ping-pong toy（不必完整 GEMM，只验证同步正确）

## 周次

| 周 | 标题 | 输出 |
|----|------|------|
| W5 | [WGMMA](week05_wgmma/) | 跑通 wgmma_sm90.cu + 复刻一份最小版 |
| W6 | [TMA](week06_tma/) | 跑通 wgmma_tma_sm90.cu + 自写 TMA G→S 拷贝 |
| W7 | [Pipeline + Cluster](week07_pipeline_cluster/) | minimal mbarrier ping-pong toy |

## CHECKPOINT — 进入 Stage 3 前必过

### 综合练习
写一个最小化的 warp-specialized 程序，**不要求是 GEMM**：
- 1 个 producer warp 用 TMA 把全局 1MB 数据加载到 smem
- 2 个 consumer warpgroup 轮流（ping-pong）从 smem 读出做 reduction
- 用 `mbarrier` + `PipelineState` 做同步
- 跑 4 轮，验证最终 reduction 结果与 CPU 一致

放置位置：`stage2_sm90_primitives/week07_pipeline_cluster/exercises/ex_warpspec_pingpong_toy.cu`。

### 口答 7 题
1. WGMMA 的 RS 模式（A 在寄存器）与 SS 模式（A 在 smem）各自适用什么场景？
2. WGMMA 的 smem layout 为什么必须用 swizzle？哪种 swizzle 配合 BF16 / FP16 / FP8？
3. TMA descriptor 在 host 上构造，传到 kernel 里的是什么？为什么不能在 kernel 里直接 `make_tma_descriptor`？
4. `cp.async.bulk` 与 `cp.async`（SM80）的本质差异是什么？为什么 TMA 需要 128B 对齐？
5. `mbarrier` 的 `phase` 是怎么避免 ABA 问题的？
6. `PipelineTmaAsync::producer_acquire` 内部做了什么？为什么需要 `producer_commit`？
7. Thread Block Cluster 的 DSMEM 跟普通 smem 在地址空间上有什么差别？
