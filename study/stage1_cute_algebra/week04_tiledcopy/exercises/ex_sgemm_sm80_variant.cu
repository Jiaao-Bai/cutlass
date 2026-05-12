/***************************************************************************************************
 * CHECKPOINT — sgemm_sm80 变体
 *
 * 任务：基于 examples/cute/tutorial/sgemm_sm80.cu 改 MMA atom layout，跑出正确结果。
 *
 * 原版（sgemm_sm80.cu:375-376）：
 *   TiledMMA mmaC = make_tiled_mma(SM80_16x8x16_F16F16F16F16_TN{},
 *                                  Layout<Shape<_2,_2>>{});  // 2x2 atoms = 4 warps
 *   CTA tile：BLK_M=128, BLK_N=128, BLK_K=16
 *
 * 你要改成（任选一种或都试）：
 *   - Layout<Shape<_4,_2>>{}   → 4x2 atoms = 8 warps，CTA 跑 64 行 M 方向
 *   - Layout<Shape<_2,_4>>{}   → 2x4 atoms = 8 warps，CTA 跑 32 列 N 方向
 *   - Layout<Shape<_4,_4>>{}   → 4x4 atoms = 16 warps（一个 CTA 跑 512 thread）
 *
 * 改动会影响：
 *   1. 每个 CTA 的 thread 数 = atoms × 32（必须 ≤ 1024）
 *   2. CTA tile 的 M/N 维（atom 16x8，乘以 atom layout 倍数）
 *   3. SMEM 用量（同 BLK_M / BLK_N → 不变）
 *   4. Register 用量（每 thread 持有的 C fragment 大小变化）
 *
 * 验证流程：
 *   1. 编译：make study_stage1_w04_ex_sgemm_sm80_variant -j 2>&1 | grep registers
 *   2. 正确性：./study_stage1_w04_ex_sgemm_sm80_variant 4096 4096 4096
 *      期望 max_diff < 1e-2（fp16 累加 fp16，精度损失正常）
 *   3. 记 register count + 性能差异到 study/PROGRESS.md
 *
 * 实现：直接复用 sgemm_sm80.cu 的整个 gemm_device 模板和 host 调用框架，
 *      只改 make_tiled_mma 那一行的 atom layout 参数。
 *
 * 这里只放骨架，**真正的工作是从 examples/cute/tutorial/sgemm_sm80.cu 复制 + 改一行**。
 **************************************************************************************************/

// TODO 第一步：把 examples/cute/tutorial/sgemm_sm80.cu 整个文件内容粘贴到这里
// （除了 main 函数的 cmd-line parsing 可以简化）。

// TODO 第二步：在 gemm_tn 函数里找到：
//   TiledMMA mmaC = make_tiled_mma(SM80_16x8x16_F16F16F16F16_TN{},
//                                  Layout<Shape<_2,_2>>{});
// 改成：
//   TiledMMA mmaC = make_tiled_mma(SM80_16x8x16_F16F16F16F16_TN{},
//                                  Layout<Shape<_4,_2>>{});    // ← 新 atom layout

// TODO 第三步：确认 __launch_bounds__ 参数不超过 1024：
//   __launch_bounds__(decltype(size(TiledMma{}))::value) ...
//   size(TiledMMA) = atom_count * 32 必须 ≤ 1024

// TODO 第四步：跑 + 记录：
//   ./study_stage1_w04_ex_sgemm_sm80_variant 4096 4096 4096
//   把 TFLOPS / max_diff / ptxas 寄存器数填回 study/PROGRESS.md 的 ncu 表格
//
// 思考：
//   - 8 warp 版本 vs 4 warp 版本，TFLOPS 谁高？为什么？
//   - 寄存器数为什么变化？（hint：CTA 总寄存器预算固定，warp 多 → 单 warp 少）
//   - SMEM 用量为啥不变？（hint：跟 atom layout 无关，跟 BLK_M/N/K 有关）

#include <cstdio>
int main(int argc, char** argv) {
  printf("Stub: 从 examples/cute/tutorial/sgemm_sm80.cu 复制实现，改 atom layout 后再跑。\n");
  printf("See TODO comments in this file.\n");
  return 0;
}
