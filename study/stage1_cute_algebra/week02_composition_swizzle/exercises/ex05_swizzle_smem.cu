/***************************************************************************************************
 * Ex05 — Swizzle smem demo: 32x32 fp32 transpose, with vs without Swizzle<3,2,3>.
 *
 * 流程：
 *   1) 单 block 1024 线程
 *   2) 每个线程往 smem[tid/32, tid%32] 写一个值（行优先，0 conflict）
 *   3) syncthreads
 *   4) 每个线程从 smem[tid%32, tid/32] 读（**列优先转置**）
 *      - plain layout：32 个线程 lock-step 命中同一个 bank → 32-way conflict
 *      - 包了 Swizzle<3,2,3>：col 低 3 bit XOR row 低 3 bit → 0 conflict
 *   5) reduce 一下避免 dead-code elimination
 *
 * 验证：
 *   ./study_stage1_w02_ex05_swizzle_smem
 *   ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
 *       ./study_stage1_w02_ex05_swizzle_smem
 *   两个 kernel 的 conflict 计数应该差 ~1 个数量级。
 **************************************************************************************************/
#include <cstdio>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/swizzle.hpp>

using namespace cute;

constexpr int TILE = 32;

// ----- plain row-major smem layout: 32x32 fp32, no swizzle ----------------
__global__ void kernel_no_swizzle(float const* in, float* out) {
  __shared__ float smem[TILE * TILE];

  auto layout = make_layout(Shape<Int<TILE>, Int<TILE>>{},
                            Stride<Int<TILE>, _1>{});  // row-major
  Tensor S = make_tensor(make_smem_ptr(smem), layout);

  int tid = threadIdx.x;
  int r = tid / TILE;
  int c = tid % TILE;

  // store: row-major write (bank-conflict-free)
  S(r, c) = in[tid];
  __syncthreads();

  // transposed load: now r and c swap → 32-way conflict on plain layout
  float v = S(c, r);
  out[tid] = v;
}

// ----- swizzled smem layout: same logical layout wrapped in Swizzle<3,2,3>
__global__ void kernel_swizzled(float const* in, float* out) {
  __shared__ float smem[TILE * TILE];

  // Swizzle<B=3, M=2, S=3>:
  //   M=2: 4-byte (fp32) atom
  //   S=3: shift amount (skip 3 bits = col index region)
  //   B=3: 3 bits XOR'd
  auto raw = make_layout(Shape<Int<TILE>, Int<TILE>>{},
                         Stride<Int<TILE>, _1>{});
  auto sw_layout = composition(Swizzle<3, 2, 3>{}, raw);
  Tensor S = make_tensor(make_smem_ptr(smem), sw_layout);

  int tid = threadIdx.x;
  int r = tid / TILE;
  int c = tid % TILE;

  S(r, c) = in[tid];
  __syncthreads();

  float v = S(c, r);
  out[tid] = v;
}

int main() {
  const int N = TILE * TILE;
  float *d_in, *d_out;
  cudaMalloc(&d_in,  N * sizeof(float));
  cudaMalloc(&d_out, N * sizeof(float));

  float h_in[N];
  for (int i = 0; i < N; ++i) h_in[i] = float(i);
  cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

  printf("=== no swizzle  (expect ~32-way bank conflicts on transpose load) ===\n");
  kernel_no_swizzle<<<1, N>>>(d_in, d_out);
  cudaDeviceSynchronize();

  printf("=== swizzled    (expect 0 bank conflicts) ===\n");
  kernel_swizzled<<<1, N>>>(d_in, d_out);
  cudaDeviceSynchronize();

  // sanity: print a few output values to make sure compiler didn't dead-code it
  float h_out[8];
  cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost);
  printf("out[0..7] = ");
  for (int i = 0; i < 8; ++i) printf("%g ", h_out[i]);
  printf("\n");

  cudaFree(d_in);
  cudaFree(d_out);
  return 0;
}
