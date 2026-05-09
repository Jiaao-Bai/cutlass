# CuTe + Hopper/Blackwell 学习计划

**目标**：在 H20（SM90）和 B200（SM100）上手写并极致优化 GEMM / FlashAttention / Sparse MoE。

**范围**：只学 CuTe + CUTLASS 3.x，跳过 CUTLASS 2.x（`gemm/threadblock/`、`gemm/warp/`、`transform/threadblock/` 等）。

**约定**：本计划及所有练习代码都在 `study/` 目录下，与上游 `include/`、`examples/` 隔离，避免 rebase 冲突。

---

## 路线图

| 阶段 | 内容 | 周 | 预计时长 | 目录 |
|------|------|----|----------|------|
| Stage 1 | CuTe 张量代数 | W1–4 | 60h | [stage1_cute_algebra/](stage1_cute_algebra/) |
| Stage 2 | SM90 硬件原语（WGMMA / TMA / Pipeline） | W5–7 | 45h | [stage2_sm90_primitives/](stage2_sm90_primitives/) |
| Stage 3 | 手写 Hopper GEMM | W8–11 | 60h | [stage3_hopper_gemm/](stage3_hopper_gemm/) |
| Stage 4 | FlashAttention | W12–15 | 60h | [stage4_flashattn/](stage4_flashattn/) |
| Stage 5 | Sparse MoE | W16–18 | 45h | [stage5_moe/](stage5_moe/) |
| Stage 6 | B200（SM100）增量 | W19–21 | 45h | [stage6_b200_increment/](stage6_b200_increment/) |
| Stage 7 | 极致调优 | 持续 | — | [stage7_tuning/](stage7_tuning/) |

进度跟踪：[PROGRESS.md](PROGRESS.md)

---

## 每周统一模板

每个 `weekNN_xxx/README.md` 都按这个 schema 写：

```markdown
# Week N — <标题>

预计 ~15h ｜ 目标硬件：H20 / B200

## 目标
- 能用一句话回答：<核心问题 1>
- 能用一句话回答：<核心问题 2>
- 能手写 <X>，跑出正确结果（或达到 Y TFLOPS / GB/s）

## 读
- `include/cute/...:行号` — 关注什么
- `media/docs/cpp/.../xxx.md`

## 写
- `exercises/exNN_xxx.cu` — 任务列表（TODO / 验证标准）

## 跑
- 编译：`cmake -DCUTLASS_ENABLE_STUDY=ON -DCUTLASS_NVCC_ARCHS=90a ..`
- 运行：`./study/stageX/weekNN/exNN_xxx`
- 期望输出 / 性能基线

## 自检（口头能答上来才算过）
1. <问题 1>
2. <问题 2>
3. <问题 3>
```

每个 stage README 末尾有一个 **CHECKPOINT**：综合性练习 + 5–10 道口答题，过了再进下一阶段。

---

## 目录结构

```
study/
├── README.md                # 本文件
├── PROGRESS.md              # 每周完成情况 + ncu 关键指标
├── CMakeLists.txt           # 一键 build 所有练习
├── common/                  # 共享 util（timing / ref check）
│
├── stage1_cute_algebra/
│   ├── README.md            # 阶段总览 + CHECKPOINT
│   ├── week01_layout_basics/
│   │   ├── README.md
│   │   └── exercises/       # 每个 .cu 一份 TODO + 验证
│   ├── week02_composition_swizzle/
│   ├── week03_tiledmma/
│   └── week04_tiledcopy/
│
├── stage2_sm90_primitives/
│   ├── README.md
│   ├── week05_wgmma/
│   ├── week06_tma/
│   └── week07_pipeline_cluster/
│
├── stage3_hopper_gemm/
│   ├── README.md
│   ├── week08_3x_design/
│   ├── week09_warpspec_writeup/
│   ├── week10_warpspec_optimize/
│   └── week11_pingpong_vs_coop/
│
├── stage4_flashattn/
│   ├── README.md
│   ├── week12_fa_algorithm/
│   ├── week13_fa_fwd_writeup/
│   ├── week14_fa_fwd_optimize/
│   └── week15_fa_bwd/
│
├── stage5_moe/
│   ├── README.md
│   ├── week16_grouped_gemm/
│   ├── week17_routing/
│   └── week18_fused_moe/
│
├── stage6_b200_increment/   # ★ 不是附录，而是正式阶段
│   ├── README.md
│   ├── week19_tmem_umma/
│   ├── week20_sm100_gemm/
│   └── week21_sm100_fa/
│
└── stage7_tuning/
    ├── README.md
    ├── profiling_recipes.md
    ├── h20_baselines.md
    └── b200_baselines.md
```

---

## 如何使用

### 编译练习

```bash
mkdir -p build && cd build
cmake .. -DCUTLASS_ENABLE_STUDY=ON -DCUTLASS_NVCC_ARCHS=90a   # H20
# 或：-DCUTLASS_NVCC_ARCHS=100a 用 B200

# 全部
make study_all -j

# 单个练习
make study_stage1_w02_hgemm_naive -j
```

可执行文件命名：`study_stage<N>_w<MM>_<exercise>`，方便 `ncu` / `nsight-sys` 定位。

### 推进节奏

1. 进入某 stage，先读 `stageN/README.md` 的目标与 checkpoint
2. 进入某 week，读 `weekNN/README.md`
3. 完成 `exercises/` 里的 TODO，跑通，记录到 [PROGRESS.md](PROGRESS.md)
4. 答完自检题再走，遇到答不上来的就回去重读
5. 阶段尾跑 CHECKPOINT 综合练习，过了进下一阶段

---

## `include/cutlass/` 阅读策略

`cutlass/` 共 67 万行，**必读约 5000 行**，**38 万行 2.x 遗产可直接跳过**。完整清单和"哪一周读哪一段"的映射，见单独的：

→ **[cutlass_reading_strategy.md](cutlass_reading_strategy.md)**

简单说：
- `pipeline/sm90_pipeline.hpp`（1388 行）→ Stage 2 W7
- `gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp`（584 行）→ Stage 3 W8/W9
- `gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp`（947 行）→ Stage 3 W11
- `gemm/kernel/sm90_tile_scheduler*.hpp` → Stage 3 W10、Stage 5 W16
- `pipeline/sm100_pipeline.hpp`（1328 行）→ Stage 6 W19
- 选读：EVT（`epilogue/collective/` + `epilogue/fusion/`）→ Stage 3 W10

**不要一上来就把 5000 行刷完**。每周用到再读，否则没上下文。

---

## 务实建议

1. **不要一开始就用 `CollectiveBuilder`**。它把细节全藏起来。先用 CuTe 原语手写，跑出正确结果后，再用 Builder 对比。最终优化时回到手写。
2. H20 上 90% 的性能天花板取决于对 **WGMMA 发射节奏**和 **TMA 预取时序**的理解，这两个只有手写才能掌握。
3. 每个练习先求**正确**（CPU ref check），再求**快**（vs cuBLAS / 上游 example）。
4. 每周都跑一次 `ncu --set full`，把关键指标记到 [PROGRESS.md](PROGRESS.md) 形成自己的 baseline 表。
