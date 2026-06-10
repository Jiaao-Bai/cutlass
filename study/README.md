# CuTe + Blackwell 学习计划

**目标**：能在 SM100（B200）上写出 **tensor core 利用率 ≥ 70%** 的任意 CuTe kernel——GEMM / FlashAttention / Sparse MoE 是学习路径，**不是终点**。

**终点产物**：一个独立开源仓库——把 `include/cute/` 搬出去作为依赖，基于 CuTe 手写主流 LLM 算子的 tensor core 实现（dense GEMM、量化 GEMM、FA fwd/bwd、GQA/decode attention、MoE grouped GEMM 等）。本课程每个 stage 的产出 kernel 都是该仓库的种子代码。

**"利用率 70%" 的度量口径**（统一标准，避免各 stage 各说各话）：
- 绝对口径：`ncu` 指标 `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` ≥ 70%，或等价地 实测 TFLOPS ÷ 该 dtype 硬件峰值 ≥ 70%
- 相对口径（日常用）：≥ 80% cuBLAS / 官方 example——cuBLAS 大形状约 85-90% 峰值，80% cuBLAS ≈ 70% 绝对利用率
- compute-bound 算子（大 GEMM、FA prefill）看此指标；memory-bound 场景（decode、routing）改看 `dram__throughput` 接近带宽 roofline

**范围**：只学 CuTe + CUTLASS 3.x，跳过 CUTLASS 2.x（`gemm/threadblock/`、`gemm/warp/`、`transform/threadblock/` 等）。

**约定**：本计划及所有练习代码都在 `study/` 目录下，与上游 `include/`、`examples/` 隔离，避免 rebase 冲突。

---

## 路线图

| 阶段 | 内容 | 周 | 预计时长 | 目录 |
|------|------|------|----------|------|
| Stage 1 | CuTe 张量代数 | W1–4 | 60h | [stage1_cute_algebra/](stage1_cute_algebra/) |
| Stage 2 | 硬件原语（SM90 → SM100 增量）| W5–8 | 60h | [stage2_primitives/](stage2_primitives/) |
| Stage 3 | 手写 GEMM（SM100）| W9–13 | 75h | [stage3_gemm/](stage3_gemm/) |
| Stage 4 | FlashAttention（SM100）| W14–18 | 75h | [stage4_flashattn/](stage4_flashattn/) |
| Stage 5 | Sparse MoE | W19–21 | 45h | [stage5_moe/](stage5_moe/) |
| Stage 6 | 源码精读 | W22–24 | 45h | [stage6_source_reading/](stage6_source_reading/) |
| Stage 7 | 极致调优 | 持续 | — | [stage7_tuning/](stage7_tuning/) |

**硬件**：全计划 **Blackwell-only，目标硬件只有 B200（SM100，租赁 ~$5-15/hr）**。W9 之后所有性能数字都在 B200 上测。Stage 2 已一次性消化 SM90 + SM100 硬件原语（SM90 WGMMA 是 UMMA 的概念地基，文档大量 "differs from WGMMA in X"）。

进度跟踪：[PROGRESS.md](PROGRESS.md)

---

## 每周统一模板

每个 `weekNN_xxx/README.md` 都按这个 schema 写：

```markdown
# Week N — <标题>

预计 ~15h ｜ 目标硬件：B200（SM100）

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
- 编译：`cmake -DCUTLASS_ENABLE_STUDY=ON -DCUTLASS_NVCC_ARCHS=100a ..`
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
├── THINKING.md              # 跨 stage 的洞察 / 踩坑 / 方法论
├── CMakeLists.txt           # 一键 build 所有练习
├── cutlass_reading_strategy.md  # include/cutlass/ 67 万行取舍清单
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
├── stage2_primitives/
│   ├── README.md
│   ├── week05_wgmma/
│   ├── week06_tma/
│   ├── week07_pipeline_cluster/
│   └── week08_tmem_umma/
│
├── stage3_gemm/
│   ├── README.md
│   ├── week09_3x_design/
│   ├── week10_warpspec_writeup/
│   ├── week11_warpspec_optimize/
│   ├── week12_pingpong_vs_coop/
│   └── week13_sm100_gemm/
│
├── stage4_flashattn/
│   ├── README.md
│   ├── week14_fa_algorithm/
│   ├── week15_fa_fwd_writeup/
│   ├── week16_fa_fwd_optimize/
│   ├── week17_fa_bwd/
│   └── week18_sm100_fa/
│
├── stage5_moe/
│   ├── README.md
│   ├── week19_grouped_gemm/
│   ├── week20_routing/
│   └── week21_fused_moe/
│
├── stage6_source_reading/
│   ├── README.md
│   ├── week22_pipeline_collective/
│   ├── week23_kernel_scheduler/
│   └── week24_evt_arch_compare/
│
└── stage7_tuning/
    ├── README.md
    ├── profiling_recipes.md
    └── b200_baselines.md
```

---

## 如何使用

### 编译练习

```bash
mkdir -p build && cd build
cmake .. -DCUTLASS_ENABLE_STUDY=ON -DCUTLASS_NVCC_ARCHS=100a  # B200（SM100）

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
- `pipeline/sm90_pipeline.hpp`（1388 行）→ Stage 2 W7 片段读（SM90 概念地基）
- `pipeline/sm100_pipeline.hpp`（1328 行）→ Stage 2 W8 片段读 + Stage 3 W10/W13 + **Stage 6 W22 全文精读**
- `gemm/collective/sm100_mma_warpspecialized.hpp` → Stage 3 W10/W11 片段读 + **Stage 6 W22 全文精读**
- `gemm/kernel/sm100_gemm_tma_warpspecialized.hpp` → Stage 3 W10/W12 片段读 + **Stage 6 W23 全文精读**
- `gemm/kernel/sm100_tile_scheduler*.hpp` → Stage 3 W11/W12、Stage 5 W19 片段读 + **Stage 6 W23 全文精读**
- EVT（`epilogue/collective/sm100_*` + `epilogue/fusion/sm100_*`）→ Stage 3 W11 选读 + **Stage 6 W24 全文精读**

**Stage 1-5 按需读片段，Stage 6 系统精读全文**。

---

## 务实建议

1. **不要一开始就用 `CollectiveBuilder`**。它把细节全藏起来。先用 CuTe 原语手写，跑出正确结果后，再用 Builder 对比。最终优化时回到手写。
2. B200（SM100）上 90% 的性能天花板取决于对 **UMMA（tcgen05.mma）发射节奏**和 **TMA 预取时序**的理解，这两个只有手写才能掌握。
3. 每个练习先求**正确**（CPU ref check），再求**快**（vs cuBLAS / 上游 example）。
4. 每周都跑一次 `ncu --set full`，把关键指标记到 [PROGRESS.md](PROGRESS.md) 形成自己的 baseline 表。
