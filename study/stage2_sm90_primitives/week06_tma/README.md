# Week 6 — TMA (Tensor Memory Accelerator)

预计 ~15h ｜ 目标硬件：H20

## 目标
- 能在 host 构造 TMA descriptor，理解 5D tensor 的 box / global stride / element stride
- 能在 kernel 里用 `cp.async.bulk.tensor` 做 G→S 异步搬运
- 跑通 `wgmma_tma_sm90.cu` 并改一改

## 读
- `include/cute/arch/copy_sm90_tma.hpp` — `cp.async.bulk.tensor` PTX wrapper
- `include/cute/atom/copy_traits_sm90_tma.hpp` — TMA descriptor 构造
- `media/docs/cpp/cute/0z_tma_tensors.md` — TMA Tensor 概念文档（必读）
- `examples/cute/tutorial/hopper/wgmma_tma_sm90.cu` — WGMMA + TMA 组合

## 写
- `exercises/ex13_tma_g2s.cu` — 从 host 构造 1D / 2D / 3D 三种 TMA descriptor，每种发一次 G→S 搬运，验证内容
- `exercises/ex14_tma_multicast.cu` — 用 `cp.async.bulk.tensor.multicast` 让 cluster 内多个 CTA 共享一次加载

## 跑
```bash
make study_stage2_w06_ex13_tma_g2s -j && ./study_stage2_w06_ex13_tma_g2s
make study_stage2_w06_ex14_tma_multicast -j && ./study_stage2_w06_ex14_tma_multicast
```

ncu 验证：
- `dram__bytes_read.sum` 应为 `M*N*sizeof(T)`，而不是更多
- `l1tex__t_bytes.sum` 在 multicast 模式下应明显小于 N×（单 CTA 数据量）

## 自检
1. TMA descriptor 为什么要在 host 构造？如果在 kernel 里构造会怎样？
2. TMA 绕过 L1/L2 直接 G→S，对 L2 cache 还有用吗？什么时候依然命中 L2？
3. 128B 对齐要求是对什么对齐？元素地址、box 起点还是 global tensor 起点？
4. `cp.async.bulk.commit_group` 之后等待用什么指令？跟 SM80 的 `cp.async.wait_group` 是同一条吗？
5. multicast 适合什么模式（A 还是 B）？为什么大 K 时 multicast 收益大？
