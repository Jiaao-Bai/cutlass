/***************************************************************************************************
 * Ex09 — cp.async G→S 拷贝
 *
 * 目标：
 *   1. 学会用 cute::copy + SM80_CP_ASYNC_CACHEGLOBAL 把 gmem → smem
 *   2. 用 cp_async_fence + cp_async_wait 同步
 *   3. 对比 sync 路径（普通 LDG → STS）和 async 路径的运行时间
 *
 * 配置：
 *   - tile shape (128, 64) fp16 = 16KB（smem 一块够大）
 *   - 用 32 个线程 / block 搬这一块（每线程 LDG.E.128 + STS.E.128 一次 = 16B）
 *   - 共 128*64*2 / 32 = 512B / thread = 32 次 LDG.E.128
 *
 * 跑：
 *   ./study_stage1_w04_ex09_async_copy
 *   期望：async 版本 ≥ sync 版本 1.5x 吞吐（H100 上更明显）
 **************************************************************************************************/
#include <cstdio>
#include <cute/tensor.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/copy_traits_sm80.hpp>
#include "cutlass/half.h"
#include "cutlass/util/GPU_Clock.hpp"

using namespace cute;
using T = cutlass::half_t;

constexpr int BLK_M = 128;
constexpr int BLK_K = 64;
constexpr int N_THR = 128;  // 一个 block 用 128 线程搬

// ------ sync 版本：直接 cute::copy 默认走 LDG + STS --------------------------
__global__ void sync_copy_kernel(T const* gA, T* gB_dummy) {
  __shared__ T smem[BLK_M * BLK_K];

  auto layout_gmem = make_layout(Shape<Int<BLK_M>, Int<BLK_K>>{}, LayoutRight{});
  auto layout_smem = make_layout(Shape<Int<BLK_M>, Int<BLK_K>>{}, LayoutRight{});

  auto g_tensor = make_tensor(make_gmem_ptr(gA), layout_gmem);
  auto s_tensor = make_tensor(make_smem_ptr(smem), layout_smem);

  // 简单 TiledCopy：每线程 16B（_8 个 fp16）
  auto tiled_copy = make_tiled_copy(
      Copy_Atom<DefaultCopy, T>{},          // 默认 LDG + STS（同步）
      Layout<Shape<_16, _8>>{},             // 128 个线程，按 16x8 排
      Layout<Shape< _1, _8>>{});            // 每线程一次 16B (8 fp16)

  auto thr_copy = tiled_copy.get_slice(threadIdx.x);
  auto tg = thr_copy.partition_S(g_tensor);
  auto ts = thr_copy.partition_D(s_tensor);

  copy(tiled_copy, tg, ts);
  __syncthreads();

  // 防 dead-code
  if (threadIdx.x == 0) gB_dummy[blockIdx.x] = smem[0];
}

// ------ async 版本：用 SM80_CP_ASYNC_CACHEGLOBAL ------------------------------
__global__ void async_copy_kernel(T const* gA, T* gB_dummy) {
  __shared__ T smem[BLK_M * BLK_K];

  auto layout_gmem = make_layout(Shape<Int<BLK_M>, Int<BLK_K>>{}, LayoutRight{});
  auto layout_smem = make_layout(Shape<Int<BLK_M>, Int<BLK_K>>{}, LayoutRight{});

  auto g_tensor = make_tensor(make_gmem_ptr(gA), layout_gmem);
  auto s_tensor = make_tensor(make_smem_ptr(smem), layout_smem);

  // 用 cp.async.cg 一次 16B（uint128_t）
  // SM80_CP_ASYNC_CACHEGLOBAL<TVecType> 的 TVecType 决定一次搬几字节，
  // 这里 uint128_t = 16B
  auto tiled_copy = make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, T>{},
      Layout<Shape<_16, _8>>{},
      Layout<Shape< _1, _8>>{});

  auto thr_copy = tiled_copy.get_slice(threadIdx.x);
  auto tg = thr_copy.partition_S(g_tensor);
  auto ts = thr_copy.partition_D(s_tensor);

  copy(tiled_copy, tg, ts);
  cp_async_fence();      // 提交这个 group
  cp_async_wait<0>();    // 等到所有 group 完成
  __syncthreads();

  if (threadIdx.x == 0) gB_dummy[blockIdx.x] = smem[0];
}

int main() {
  const int N_BLK = 256;  // 跑 256 个 block，吞吐有意义
  const int N_ELEM = N_BLK * BLK_M * BLK_K;

  T *d_in;
  T *d_out;
  cudaMalloc(&d_in,  N_ELEM * sizeof(T));
  cudaMalloc(&d_out, N_BLK   * sizeof(T));
  cudaMemset(d_in, 0x3c, N_ELEM * sizeof(T));  // 1.0 in fp16

  GPU_Clock timer;

  // sync
  timer.start();
  for (int i = 0; i < 10; ++i)
    sync_copy_kernel<<<N_BLK, N_THR>>>(d_in, d_out);
  cudaDeviceSynchronize();
  double t_sync = timer.seconds() / 10;
  printf("sync  copy:  %.3f ms / iter\n", t_sync * 1e3);

  // async
  timer.start();
  for (int i = 0; i < 10; ++i)
    async_copy_kernel<<<N_BLK, N_THR>>>(d_in, d_out);
  cudaDeviceSynchronize();
  double t_async = timer.seconds() / 10;
  printf("async copy:  %.3f ms / iter\n", t_async * 1e3);

  printf("\nspeedup: %.2fx\n", t_sync / t_async);
  // TODO：观察并思考——
  //   1. async 在哪一步省时间？（hint：LDG 和 STS 不再串行）
  //   2. 如果换 SM80_CP_ASYNC_CACHEALWAYS<uint128_t> 速度会变化吗？为啥？
  //   3. 把 wait<0> 改成 wait<1> 会发生什么？

  cudaFree(d_in);
  cudaFree(d_out);
  return 0;
}
