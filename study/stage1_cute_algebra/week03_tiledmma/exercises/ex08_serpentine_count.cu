/***************************************************************************************************
 * Ex08 — Serpentine 遍历 vs 序列遍历的寄存器对比
 *
 * 背景：cute::gemm Dispatch 4 (gemm.hpp:260-390) 在 (V,M) × (V,N) → (V,M,N) 外积里
 *   按 m 在外、n 在内遍历。**顺序遍历**每次 m 换行后 n 从头开始读；
 *   **serpentine** 在奇数 m 行反向走 n，让相邻 m 之间的 b[n] 在寄存器里继续复用
 *   （上一行最后用的 b[N-1] 这一行从 b[N-1] 开始）。
 *
 * 这道题不真跑 GEMM，只 **比较两个等价 kernel 的 ptxas 寄存器报告**。
 *
 * 怎么用：
 *   make study_stage1_w03_ex08_serpentine_count -j 2>&1 | grep -E "registers|ex08"
 *   或编译命令加 -Xptxas=-v 看输出：
 *   ptxas info    : Compiling entry function '_Z..serpentine_kernel..' for 'sm_80'
 *   ptxas info    : Used XX registers, ...
 *
 * 期望：serpentine 版本的寄存器使用 < 顺序版本（数量级 ~几十 register）
 **************************************************************************************************/
#include <cstdio>
#include <cute/tensor.hpp>
#include "cutlass/half.h"

using namespace cute;

// 共用配置：单 thread 持 A_frag[M_V] = 8 fp16，B_frag[N_V] = 4 fp16
//          做 outer-product 累加到 C_frag[M_V][N_V] = 32 fp16
constexpr int M_V = 8;
constexpr int N_V = 4;
constexpr int K_TILE = 16;

using T = cutlass::half_t;
using AccT = float;

// ----- 顺序遍历 (sequential) ----------------------------------------------
__global__ void sequential_kernel(T const* gA, T const* gB, AccT* gC) {
  T a_frag[M_V];
  T b_frag[N_V];
  AccT c_frag[M_V][N_V] = {};

  // 从 gmem 装一份（简化：直接复制，不走 SMEM）
  #pragma unroll
  for (int i = 0; i < M_V; ++i) a_frag[i] = gA[threadIdx.x * M_V + i];
  #pragma unroll
  for (int j = 0; j < N_V; ++j) b_frag[j] = gB[threadIdx.x * N_V + j];

  // TODO：这里是关键——顺序遍历。注意 m 在外、n 在内、n 总是从 0 开始。
  #pragma unroll
  for (int m = 0; m < M_V; ++m) {
    #pragma unroll
    for (int n = 0; n < N_V; ++n) {
      c_frag[m][n] += AccT(a_frag[m]) * AccT(b_frag[n]);
    }
  }

  // store
  #pragma unroll
  for (int m = 0; m < M_V; ++m)
    #pragma unroll
    for (int n = 0; n < N_V; ++n)
      gC[(threadIdx.x * M_V + m) * N_V + n] = c_frag[m][n];
}

// ----- Serpentine 遍历 -----------------------------------------------------
__global__ void serpentine_kernel(T const* gA, T const* gB, AccT* gC) {
  T a_frag[M_V];
  T b_frag[N_V];
  AccT c_frag[M_V][N_V] = {};

  #pragma unroll
  for (int i = 0; i < M_V; ++i) a_frag[i] = gA[threadIdx.x * M_V + i];
  #pragma unroll
  for (int j = 0; j < N_V; ++j) b_frag[j] = gB[threadIdx.x * N_V + j];

  // TODO：关键差异在这里——奇数 m 反向走 n。
  //   m=0 时 n 走 0,1,2,3
  //   m=1 时 n 走 3,2,1,0  ← 直接接上 b_frag[3]，不用从 b[0] 重新读
  //   m=2 时 n 走 0,1,2,3 ...
  #pragma unroll
  for (int m = 0; m < M_V; ++m) {
    #pragma unroll
    for (int n = 0; n < N_V; ++n) {
      int ns = (m & 1) ? (N_V - 1 - n) : n;
      c_frag[m][ns] += AccT(a_frag[m]) * AccT(b_frag[ns]);
    }
  }

  #pragma unroll
  for (int m = 0; m < M_V; ++m)
    #pragma unroll
    for (int n = 0; n < N_V; ++n)
      gC[(threadIdx.x * M_V + m) * N_V + n] = c_frag[m][n];
}

int main() {
  // 这道题不跑 kernel，主要靠 ptxas -v 看寄存器使用差异。
  // 如果想跑做正确性 check 也可以：两个 kernel 输出应一致。
  printf("Compile this file with -Xptxas=-v and grep \"registers\" 看两个 kernel 的寄存器数。\n");
  printf("Expected: serpentine_kernel registers < sequential_kernel registers\n");
  printf("Why: serpentine 让 b_frag 在 m 切换时无需重新 spill/reload，活跃寄存器数下降\n");
  return 0;
}
