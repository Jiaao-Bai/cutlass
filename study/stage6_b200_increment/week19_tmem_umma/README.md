# Week 19 — TMEM + UMMA

预计 ~15h ｜ 目标硬件：B200（没有可用 H100 退而求其次只能读代码）

## 目标
- 看懂 TMEM 是什么、怎么 alloc、怎么访问
- 看懂 UMMA 与 WGMMA 的核心差异
- 跑通一个 minimal SM100 MMA example

## 读
- `include/cute/arch/tmem_allocator_sm100.hpp` — TMEM 分配
- `include/cute/arch/mma_sm100_umma.hpp` — UMMA PTX 包装
- `include/cute/atom/mma_traits_sm100*.hpp` — UMMA atom traits
- `include/cute/atom/copy_traits_sm100_tmem.hpp` — TMEM ↔ RMEM 拷贝
- `examples/cute/tutorial/blackwell/`（如果有）
- `examples/python/CuTeDSL/blackwell/dense_gemm.py` — 可执行参考

## TMEM vs SMEM vs RMEM

| 内存 | SM90 角色 | SM100 角色 | 容量/SM | 访问 |
|------|----------|------------|---------|------|
| RMEM | A/B/C 都可放 | A/B 仍可放，C 不在这 | 64KB | 一切指令 |
| SMEM | 主要 staging | 同 SM90 | 228KB(SM90) / TBD(SM100) | LDS/STS, TMA |
| TMEM | — | C 矩阵专用 | 256KB | `tcgen05.ld/st`, UMMA 内置 |

## UMMA 与 WGMMA 差异

```
WGMMA (SM90):       UMMA (SM100):
A in SMEM/RMEM      A in SMEM/TMEM
B in SMEM           B in SMEM
C in RMEM (warpgroup spread)   C in TMEM
issuing: warpgroup (128 thread)   issuing: 1 thread (TCGen5)
```

UMMA 单线程发射、C 在 TMEM，意味着 epilogue 必须 TMEM→RMEM→GMEM。

## 写
- `exercises/ex27_tmem_alloc.cu` — alloc 一段 TMEM，写入数据，读回 RMEM 验证
- `exercises/ex28_umma_minimal.cu` — 最小 UMMA：(M,N,K)=(64,128,16) FP16，C 留 TMEM 然后搬出验证

## 跑
```bash
cmake -DCUTLASS_ENABLE_STUDY=ON -DCUTLASS_NVCC_ARCHS=100a ..
make study_stage6_w19_ex27_tmem_alloc -j && ./study_stage6_w19_ex27_tmem_alloc
make study_stage6_w19_ex28_umma_minimal -j && ./study_stage6_w19_ex28_umma_minimal
```

## 自检
1. TMEM 是 SM 私有还是 cluster 共享？
2. `tcgen05.alloc` 与 `__shared__` 声明在编译期 vs 运行期的差别？
3. UMMA 单线程发射，但谁来等 `mma.commit_group` 完成？
4. C 在 TMEM 时，accumulator 的 layout 是什么？跟 WGMMA 的 RMEM C 是同一种 partition 吗？
5. TMEM ↔ RMEM 的 `tcgen05.ld/st` 一次能搬多少 byte/thread？
