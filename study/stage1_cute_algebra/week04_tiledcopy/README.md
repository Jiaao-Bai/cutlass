# Week 4 — TiledCopy / Copy_Atom

预计 ~15h ｜ 目标硬件：**任何 SM80+**（H20 / 5060 Ti / 4090 / A100；用 `cp.async` + `mma_sm80`）

## 目标
- 能解释 `Copy_Atom` 与 `MMA_Atom` 设计上的对称性
- 能用 `cp.async`（SM80）写 G→S 拷贝并隐藏 latency
- 能用 `ncu` 实测 bank conflict 数，并通过 swizzle 干掉它
- **CHECKPOINT 通过**：自写 `sgemm_sm80` 变体，跑出正确结果

## 读（**自下而上**：PTX → Traits → Atom → TiledCopy → 实战 → 文档）

1. `include/cute/arch/copy_sm80.hpp:47-157` — **`cp.async` 系列 PTX 封装**：
   ```
   SM80_CP_ASYNC_CACHEALWAYS  (l47-70) : cp.async.ca.shared.global  (L1+L2 缓存)
   SM80_CP_ASYNC_CACHEGLOBAL  (l74-97) : cp.async.cg.shared.global  (仅 L2 缓存)
   *_ZFILL 版本                        : src 越界自动填 0（带 src_size 参数）
   ```
   以及 `copy_sm80.hpp:164-195` 的同步原语：
   ```
   cp_async_fence()    : 把累计的 cp.async 打包成一个 group
   cp_async_wait<N>()  : 等待至多 N 个 group 仍 pending（N=0 等全部完成）
   ```
2. `include/cute/atom/copy_traits_sm80.hpp:42-157` — **`Copy_Traits<SM80_CP_ASYNC_*>` 特化**：
   - `ThrID = Layout<_1>`（cp.async 是单线程指令，"thread group" 就是 1）
   - `SrcLayout / DstLayout / RefLayout` = `(T, V) → bit-offset` 的映射
3. `include/cute/atom/copy_atom.hpp:52-72` — **`Copy_Atom<Traits, ValType>` 字段**：
   ```
   ThrID                     : 来自 Traits
   BitLayoutSrc / Dst / Ref  : (T, V) → bit-offset（原始 PTX 视角）
   ValLayoutSrc / Dst / Ref  : (T, V) → ValType-index（用户视角，重铸后的）
   ```
   **`ValLayoutSrc`** 跟 `MMA_Atom::ALayout` 一样都是 `(T, V) → element_index` 的映射——这就是 Copy 和 MMA 的**对称性**。
4. `include/cute/atom/copy_atom.hpp:180-260` — **`TiledCopy`** + `make_tiled_copy`：把单个 Copy_Atom 用 thread layout / val layout 平铺到一整片数据上。
5. `examples/cute/tutorial/sgemm_sm80.cu` — **完整 SM80 GEMM**，重点看 `TiledCopy` 怎么构造（行 369-376）和 mainloop 怎么 pipeline（cp.async + fence + wait）。
6. `media/docs/cpp/cute/0t_mma_atom.md` 的 Copy_Atom 章节——官方层级化解释。

**心智模型**（跟 W3 MMA 完全对称）：
```
cp.async PTX → Copy_Traits 描述指令 → Copy_Atom 包成 struct →
TiledCopy 拼大块 → ThrCopy 切给 thread → 在 mainloop 里用
```

## 实战例子（**W4 主例 + CHECKPOINT base**）

| 例子 | 用到的 | 看点 |
|------|--------|------|
| `examples/cute/tutorial/sgemm_sm80.cu` | TiledMMA + **TiledCopy（cp.async）+ double-buffer pipeline** | **W4 主例**。重点：行 369-376 `make_tiled_copy` 构造 + mainloop 里 prologue/`cp_async_fence`/`cp_async_wait<PIPE-2>` 的双缓冲编排。**这是 CHECKPOINT `ex_sgemm_sm80_variant` 的直接基础。** |

回顾 W3 已读过的 `sgemm_2.cu` 也可以再扫一遍——它用同步 cp.async，作为"不带 pipeline 的版本"对照看出双缓冲省的是什么。

```bash
make sgemm_sm80 -j && ./examples/cute/tutorial/sgemm_sm80 4096 4096 4096
```

## 写
- `exercises/ex09_async_copy.cu` — 用 `cute::copy` + `SM80_CP_ASYNC_CACHEGLOBAL` 做 G→S 拷贝，对比同步 `LDG.E` 路径
- `exercises/ex10_bank_conflict.cu` — 写一个会 bank conflict 的 smem layout，再用 swizzle 修掉，记 ncu 数据
- **CHECKPOINT**：`exercises/ex_sgemm_sm80_variant.cu` — 把 `sgemm_sm80.cu` 的 MMA atom layout 从 `(2,2,1)`（4 warp）改成 `(4,2,1)`（8 warp）或 `(2,4,1)`，重新跑正确性 + 比 register 用量

## 跑
```bash
# 5060 Ti / SM120
cmake -DCUTLASS_ENABLE_STUDY=ON -DCUTLASS_NVCC_ARCHS=120 ..
# H20 / SM90
cmake -DCUTLASS_ENABLE_STUDY=ON -DCUTLASS_NVCC_ARCHS=90 ..

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
2. `cp.async.commit_group` 之后，`cp.async.wait_group<N>` 的语义是 "等待至多 ___ 个 group 仍 pending"（填空，PTX 标准答案）。
3. `Copy_Atom::ValLayoutSrc` 和 `MMA_Atom::ALayout` 在抽象上都是 `(ThrID, ValIdx) → element_offset`，对吗？
4. 单条 `cp.async` 一次最多搬多少字节？一个 warp 一次发射（32 lane 同时执行）一共多少字节？
5. 你的 `sgemm_sm80_variant` 比原版的 register 用量多/少了多少？哪里来的差异？
