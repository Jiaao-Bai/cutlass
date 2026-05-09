/***************************************************************************************************
 * Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

using namespace cute;

// TODO: 在这里实现你的 naive FP16 GEMM kernel
// 提示：
// 1. 使用 make_tensor 创建 global memory 的 Tensor 视图
// 2. 使用 local_tile 或直接索引来划分工作
// 3. 使用 Tensor 的 () 操作符来访问元素
// 4. 计算 C = A * B (不考虑 alpha/beta)
//
// 参数说明：
//   M, N, K: 矩阵维度
//   A: M x K 矩阵 (row-major)
//   B: K x N 矩阵 (row-major)
//   C: M x N 矩阵 (row-major)
__global__ void hgemm_naive_kernel(
    int M, int N, int K,
    cutlass::half_t const* A,
    cutlass::half_t const* B,
    cutlass::half_t* C)
{
  // TODO: 你的实现
  //
  // 建议的实现步骤：
  // 1. 计算当前 thread 负责的输出位置 (m, n)
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  int n = blockIdx.x * blockDim.x + threadIdx.x;

  if (m >= M || n >= N) return;

  // 2. 创建 Tensor 视图：
  Tensor mA = make_tensor(make_gmem_ptr(A), make_shape(M, K), make_stride(K, Int<1>{}));
  Tensor mB = make_tensor(make_gmem_ptr(B), make_shape(K, N), make_stride(N, Int<1>{}));
  Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), make_stride(N, Int<1>{}));
  
  // 3. 沿 K 维度累加：for (int k = 0; k < K; ++k) { ... }
  float res = 0.0f;
  for (int k = 0; k < K; k++) {
   res += float(mA(m, k)) * float(mB(k, n)); 
  }

  // 4. 写回结果
  mC(m, n) = cutlass::half_t(res);
}

///////////////////////////////////////////////////////////////////////////////
// CPU reference: 简单的三重循环 GEMM，用于验证正确性
///////////////////////////////////////////////////////////////////////////////
void gemm_cpu_ref(
    int M, int N, int K,
    cutlass::half_t const* A,
    cutlass::half_t const* B,
    cutlass::half_t* C)
{
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {
        acc += float(A[m * K + k]) * float(B[k * N + n]);
      }
      C[m * N + n] = cutlass::half_t(acc);
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// main: 初始化数据 -> 调用 kernel -> 验证结果
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
  // 默认用小矩阵方便调试，通过后可以改大
  int M = 256;
  int N = 256;
  int K = 128;

  if (argc >= 2) sscanf(argv[1], "%d", &M);
  if (argc >= 3) sscanf(argv[2], "%d", &N);
  if (argc >= 4) sscanf(argv[3], "%d", &K);

  printf("FP16 Naive GEMM: M=%d, N=%d, K=%d\n", M, N, K);
  printf("Layout: A [%d x %d] row-major, B [%d x %d] row-major, C [%d x %d] row-major\n",
         M, K, K, N, M, N);

  cute::device_init(0);

  // Host 数据
  thrust::host_vector<cutlass::half_t> h_A(M * K);
  thrust::host_vector<cutlass::half_t> h_B(K * N);
  thrust::host_vector<cutlass::half_t> h_C(M * N, cutlass::half_t(0));
  thrust::host_vector<cutlass::half_t> h_C_ref(M * N, cutlass::half_t(0));

  // 用小随机值初始化，避免 FP16 溢出
  srand(42);
  for (int i = 0; i < M * K; ++i) h_A[i] = cutlass::half_t(0.5f * (rand() / double(RAND_MAX)) - 0.25f);
  for (int i = 0; i < K * N; ++i) h_B[i] = cutlass::half_t(0.5f * (rand() / double(RAND_MAX)) - 0.25f);

  // Device 数据
  thrust::device_vector<cutlass::half_t> d_A = h_A;
  thrust::device_vector<cutlass::half_t> d_B = h_B;
  thrust::device_vector<cutlass::half_t> d_C = h_C;

  // =========================================================================
  // TODO: 设置 grid/block 维度并启动 kernel
  // 提示：最简单的方式是每个 thread 算一个 C 元素
  //   dim3 block(16, 16);
  //   dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
  // =========================================================================
  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

  hgemm_naive_kernel<<<grid, block>>>(
      M, N, K,
      d_A.data().get(),
      d_B.data().get(),
      d_C.data().get());

  CUTE_CHECK_LAST();

  // 拷回结果
  thrust::host_vector<cutlass::half_t> gpu_result = d_C;

  // CPU reference
  gemm_cpu_ref(M, N, K, h_A.data(), h_B.data(), h_C_ref.data());

  // 验证
  int errors = 0;
  for (int i = 0; i < M * N; ++i) {
    float gpu_val = float(gpu_result[i]);
    float ref_val = float(h_C_ref[i]);
    float diff = fabs(gpu_val - ref_val);
    // FP16 精度有限，用相对误差
    float rel = diff / (fabs(ref_val) + 1e-6f);
    if (rel > 0.05f) {  // 5% 相对误差阈值
      if (errors < 10) {
        printf("MISMATCH at [%d] (row=%d, col=%d): gpu=%.4f, ref=%.4f, rel_err=%.4f\n",
               i, i / N, i % N, gpu_val, ref_val, rel);
      }
      ++errors;
    }
  }

  if (errors == 0) {
    printf("\nPASSED! All %d elements match.\n", M * N);
  } else {
    printf("\nFAILED: %d / %d mismatches.\n", errors, M * N);
  }

  return errors > 0 ? 1 : 0;
}
