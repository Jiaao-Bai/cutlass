# 学习进度跟踪

格式：每完成一周，在对应行打勾，记关键产出和 ncu 指标。

## 总览

| Stage | 周 | 状态 | 主要产出 | 完成日期 |
|-------|----|----|---------|---------|
| 1 | W1 — Layout basics | ☑ | `ex01_crd2idx_paper.md` + `ex01_verify.cu` + `ex02_layout_print.cu` + `ex03_coalesce.cu` + 5 道自检 | 2026-05-12 |
| 1 | W2 — Composition + Swizzle | ☑ | `ex04_composition_paper.md` + `ex04_verify.cu` + `ex05_swizzle_paper.md` + `ex06_hgemm_naive.cu` + 4 道自检（Q3 重写后） | 2026-05-12 |
| 1 | W3 — TiledMMA | 🔵 in-progress | | |
| 1 | W4 — TiledCopy | ☐ | | |
| 1 | **CHECKPOINT** | ☐ | sgemm_sm80 变体 + 5 道口答 | |
| 2 | W5 — WGMMA | ☐ | | |
| 2 | W6 — TMA | ☐ | | |
| 2 | W7 — Pipeline + Cluster | ☐ | | |
| 2 | **CHECKPOINT** | ☐ | minimal warpspec ping-pong 玩具 | |
| 3 | W8 — 3.x 分层设计 | ☐ | | |
| 3 | W9 — WarpSpec writeup | ☐ | | |
| 3 | W10 — WarpSpec optimize | ☐ | | |
| 3 | W11 — Pingpong vs Cooperative | ☐ | | |
| 3 | **CHECKPOINT** | ☐ | 自写 GEMM ≥ 70% cuBLAS（H20） | |
| 4 | W12 — FA 算法 | ☐ | | |
| 4 | W13 — FA fwd writeup | ☐ | | |
| 4 | W14 — FA fwd optimize | ☐ | | |
| 4 | W15 — FA bwd | ☐ | | |
| 4 | **CHECKPOINT** | ☐ | FA fwd ≥ 80% 88_hopper_fmha | |
| 5 | W16 — Grouped GEMM | ☐ | | |
| 5 | W17 — Routing | ☐ | | |
| 5 | W18 — Fused MoE | ☐ | | |
| 5 | **CHECKPOINT** | ☐ | end-to-end MoE forward 正确 | |
| 6 | W19 — TMEM + UMMA | ☐ | | |
| 6 | W20 — SM100 GEMM | ☐ | | |
| 6 | W21 — SM100 FA | ☐ | | |
| 6 | **CHECKPOINT** | ☐ | SM100 GEMM 跑过 + 与 SM90 对照笔记 | |
| 7 | 持续调优 | ☐ | | |

---

## ncu 关键指标基线（H20）

每跑通一个练习，就来这里记一行。便于横向对比。

| 练习 | M/N/K (or shape) | TFLOPS | sm__throughput | dram BW (GB/s) | wgmma 指令数 | 备注 |
|------|------------------|--------|----------------|----------------|--------------|------|
| `ex06_hgemm_naive` (W2) | 256/256/128 | (待测) | | | 0 (no tensor core) | 单线程一元素，仅做正确性 baseline |
| | | | | | | |

## ncu 关键指标基线（B200）

| 练习 | M/N/K | TFLOPS | sm__throughput | dram BW (GB/s) | umma 指令数 | tmem 占用 | 备注 |
|------|-------|--------|----------------|----------------|--------------|----------|------|
| | | | | | | | |

---

## 踩坑记录

每个坑一行，包含：现象 / 根因 / 修复 / 关联 commit。

| 日期 | 坑 | 根因 | 修复 | commit |
|------|----|----|------|---------|
| | | | | |

---

## 自检题失败记录

答不上来的题记在这里，回去重读对应 reading 后再来打勾。

| 周 | 题目 | 状态 |
|----|------|------|
| | | |
