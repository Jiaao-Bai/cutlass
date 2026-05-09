# Week 4 — TiledCopy / Copy_Atom

预计 ~15h ｜ 目标硬件：H20

## 目标
- 能解释 `Copy_Atom` 与 `MMA_Atom` 设计上的对称性
- 能用 `cp.async`（SM80）写 G→S 拷贝并隐藏 latency
- 能用 `ncu` 实测 bank conflict 数，并通过 swizzle 干掉它
- **CHECKPOINT 通过**：自写 `sgemm_sm80` 变体，跑出正确结果

## 读
- `include/cute/atom/copy_atom.hpp` — `Copy_Atom` 与 `TiledCopy`
- `include/cute/arch/copy_sm80.hpp` — `cp.async.cg.shared.global` PTX 封装
- `include/cute/atom/copy_traits_sm80.hpp` — async copy 的 traits
- `examples/cute/tutorial/sgemm_sm80.cu` — 完整 SM80 GEMM，重点看 `TiledCopy` 的构造
- `media/docs/cpp/cute/0t_mma_atom.md` 的 Copy_Atom 章节

## 写
- `exercises/ex09_async_copy.cu` — 用 `cp.async` 做 G→S 拷贝，对比同步 `__pipeline_*` API 与 `cute::copy`
- `exercises/ex10_bank_conflict.cu` — 写一个会 bank conflict 的 smem layout，再用 swizzle 修掉，记 ncu 数据
- **CHECKPOINT**：`exercises/ex_sgemm_sm80_variant.cu` — 把 `sgemm_sm80.cu` 的 thread tile 改成 `4x2`，跑通

## 跑
```bash
make study_stage1_w04_ex09_async_copy -j
make study_stage1_w04_ex10_bank_conflict -j
make study_stage1_w04_ex_sgemm_sm80_variant -j
./study_stage1_w04_ex_sgemm_sm80_variant 4096 4096 4096
```

ncu 关键指标：
```bash
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
              smsp__sass_thread_inst_executed_op_hmma.sum \
    ./study_stage1_w04_ex_sgemm_sm80_variant
```

## 自检
1. `cp.async.cg` 和 `cp.async.ca` 的区别？什么场景选哪个？
2. `cp.async.commit_group` 之后，`cp.async.wait_group<N>` 的 N 是"等待还剩 N 个"还是"等待第 N 个"？
3. `Copy_Atom` 的 `LayoutSrc_TV` 和 `MMA_Atom` 的 `LayoutA_TV` 在抽象上是同一类东西吗？
4. SM80 上一个 warp 同时发起的 `cp.async` 最大能搬多少字节？
5. 你的 `sgemm_sm80_variant` 比原版的 register 用量多/少了多少？哪里来的差异？
