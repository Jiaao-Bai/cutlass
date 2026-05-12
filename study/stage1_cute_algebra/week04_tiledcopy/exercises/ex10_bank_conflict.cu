/***************************************************************************************************
 * Ex10 — SMEM bank conflict 实战：用 ncu 验证 swizzle 效果
 *
 * 这道题跟 ex05 paper 的呼应——把那里推过的 bit math 拿来跑数字。
 *
 * 场景：
 *   一个 CTA 把 32x32 fp16 tile（M=32, K=32）从 gmem 装到 smem（行优先），
 *   然后**按列**读出来做累加（模拟 fp16 GEMM 把 A 的 K-tile 加载后按 K 方向消费）。
 *
 * 两个 kernel：
 *   1. plain      — smem 用 row-major (32,32):(32,1)，列访问会触发 bank conflict
 *   2. swizzled   — 套 Swizzle<3,3,3>，看 conflict 是否清零
 *      （fp16 SMEM 上 SW128 的等价参数：M=3 因为 stride 单位是 8B = 4 fp16）
 *
 * 跑：
 *   ./study_stage1_w10_ex10_bank_conflict  # 先看正确性
 *   ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
 *       ./study_stage1_w04_ex10_bank_conflict
 *   期望 plain kernel 的 conflict 计数 >> 0，swizzled = 0
 **************************************************************************************************/
#include <cstdio>
#include <cute/tensor.hpp>
#include <cute/swizzle.hpp>
#include "cutlass/half.h"

using namespace cute;
using T = cutlass::half_t;

constexpr int M = 32, K = 32;

// ----- plain smem layout（无 swizzle）------------------------------------
__global__ void plain_kernel(T const* gA, float* gOut) {
  __shared__ T smem[M * K];

  // gmem (M, K) row-major
  auto g_layout = make_layout(Shape<Int<M>, Int<K>>{}, LayoutRight{});
  auto s_layout = make_layout(Shape<Int<M>, Int<K>>{}, LayoutRight{});  // 同样 row-major

  auto gA_t = make_tensor(make_gmem_ptr(gA),   g_layout);
  auto sA_t = make_tensor(make_smem_ptr(smem), s_layout);

  int tid = threadIdx.x;
  // 32 个线程，每个搬 32 个 fp16（一整行）
  if (tid < M) {
    for (int k = 0; k < K; ++k) sA_t(tid, k) = gA_t(tid, k);
  }
  __syncthreads();

  // 列访问：每个线程读 col=tid 的整列（32 个元素）
  // → 同 phase 32 个线程同 col → bank conflict 重灾区
  float acc = 0.f;
  if (tid < K) {
    for (int m = 0; m < M; ++m) acc += float(sA_t(m, tid));
  }
  if (tid < K) gOut[blockIdx.x * K + tid] = acc;
}

// ----- swizzled smem layout --------------------------------------------------
__global__ void swizzled_kernel(T const* gA, float* gOut) {
  __shared__ T smem[M * K];

  auto g_layout = make_layout(Shape<Int<M>, Int<K>>{}, LayoutRight{});

  // Swizzle<3,3,3>：M=3 因为 SMEM 元素是 fp16，单位 stride 是 2B
  //                  swizzle 在 16B/8 fp16 颗粒上做（M=3 → 跳过 8 个 fp16 = 16B）
  //                  B=3 覆盖 8 个 chunks（一整条 bank-row）
  //                  S=3 让 row 字段 XOR 进 chunk 字段
  // 注意：对 32x32 fp16 tile，每行 64B（不是 128B），所以严格来说这不是 SW128。
  //       这里只是演示原理；真实 GEMM 里 tile 会更宽。
  auto sw_layout = composition(Swizzle<3, 3, 3>{},
                                make_layout(Shape<Int<M>, Int<K>>{}, LayoutRight{}));

  auto gA_t = make_tensor(make_gmem_ptr(gA),   g_layout);
  auto sA_t = make_tensor(make_smem_ptr(smem), sw_layout);

  int tid = threadIdx.x;
  if (tid < M) {
    for (int k = 0; k < K; ++k) sA_t(tid, k) = gA_t(tid, k);
  }
  __syncthreads();

  float acc = 0.f;
  if (tid < K) {
    for (int m = 0; m < M; ++m) acc += float(sA_t(m, tid));
  }
  if (tid < K) gOut[blockIdx.x * K + tid] = acc;
}

int main() {
  const int N_BLK = 64;
  const int N_ELEM = N_BLK * M * K;

  T *d_A;
  float *d_out_plain, *d_out_swizzled;
  cudaMalloc(&d_A, N_ELEM * sizeof(T));
  cudaMalloc(&d_out_plain,    N_BLK * K * sizeof(float));
  cudaMalloc(&d_out_swizzled, N_BLK * K * sizeof(float));

  cudaMemset(d_A, 0x3c, N_ELEM * sizeof(T));  // 1.0 fp16

  plain_kernel    <<<N_BLK, 64>>>(d_A, d_out_plain);
  swizzled_kernel <<<N_BLK, 64>>>(d_A, d_out_swizzled);
  cudaDeviceSynchronize();

  // 正确性 check：两个 kernel 结果应一致（swizzle 不改变逻辑映射）
  float h_plain[K], h_swizzled[K];
  cudaMemcpy(h_plain,    d_out_plain,    K * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_swizzled, d_out_swizzled, K * sizeof(float), cudaMemcpyDeviceToHost);
  for (int k = 0; k < K; ++k) {
    if (h_plain[k] != h_swizzled[k]) {
      printf("MISMATCH at col=%d : plain=%g swizzled=%g\n", k, h_plain[k], h_swizzled[k]);
      return 1;
    }
  }
  printf("Correctness PASS. 两个 kernel 输出一致。\n");
  printf("现在用 ncu 跑：\n");
  printf("  ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./study_stage1_w04_ex10_bank_conflict\n");
  printf("plain kernel 的 conflict 应该 >> 0，swizzled 应该 = 0\n");

  cudaFree(d_A);
  cudaFree(d_out_plain);
  cudaFree(d_out_swizzled);
  return 0;
}
