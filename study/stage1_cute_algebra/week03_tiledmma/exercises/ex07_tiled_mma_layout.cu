/***************************************************************************************************
 * Ex07 — TiledMMA partition：打印 thread 0 / 32 / 64 / 96 各自看到的 fragment 形状
 *
 * 目标：
 *   理解 TiledMMA 把 (M,N) tile 切给多个 warp / lane 的过程，亲眼看到
 *     - partition_C(gC) 给当前线程切出几个 C 元素
 *     - partition_A(gA) 给当前线程切出几个 A 元素
 *     - partition_B(gB) 给当前线程切出几个 B 元素
 *
 * 配置：SM80_16x8x16_F16F16F16F16_TN，atom layout (2,2,1) → 4 warps = 128 threads
 *       一个 CTA tile 32x16x16 fp16
 *
 * 跑：
 *   ./study_stage1_w03_ex07_tiled_mma_layout
 *
 * 预测先做（手算）：
 *   - 单个 atom 是 M16xN8xK16 → 每 thread 持 A=8, B=4, C=4 fp16
 *   - TiledMMA atom_layout=(2,2,1) → 拼成 M32xN16xK16，4 warps
 *   - thread 0 是 warp 0 的 lane 0，应该看到 atom (0,0) 的份额
 *   - thread 32 是 warp 1 的 lane 0（M 方向第 2 个 atom），fragment 形状跟 thread 0 一样、地址偏移不同
 *   - thread 64 是 warp 2 的 lane 0（N 方向第 2 个 atom）
 *   - thread 96 是 warp 3 的 lane 0（M+N 都偏移）
 **************************************************************************************************/
#include <cstdio>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include "cutlass/half.h"

using namespace cute;

template <class TiledMMA, class GMemTensor>
__global__ void print_partition_kernel(TiledMMA mma, GMemTensor gC,
                                        GMemTensor gA, GMemTensor gB)
{
  int tid = threadIdx.x;
  // 只让 4 个有代表性的 thread 打印
  if (tid != 0 && tid != 32 && tid != 64 && tid != 96) return;

  auto thr_mma = mma.get_thread_slice(tid);
  auto tCgC = thr_mma.partition_C(gC);
  auto tAgA = thr_mma.partition_A(gA);
  auto tBgB = thr_mma.partition_B(gB);

  printf("---- tid=%3d ----\n", tid);
  printf("  partition_C : "); print(tCgC); printf("\n");
  printf("  partition_A : "); print(tAgA); printf("\n");
  printf("  partition_B : "); print(tBgB); printf("\n");
  // TODO：解读这三个 layout 的 shape 维度。
  //   - 第一个 mode = (V) = 这个 thread 持有的"原子内"元素数
  //   - 第二/三 mode = (RestM, RestK) 或 (RestN, RestK) = 沿 tile 维度的重复
  //   - 三个 partition 加起来覆盖整个 32x16 (C) / 32x16 (A) / 16x16 (B) tile
}

int main() {
  using MMA_op = SM80_16x8x16_F16F16F16F16_TN;

  // atom layout (2,2,1) → 拼成 32 x 16 x 16 tile，用 4 warp = 128 thread
  auto tiled_mma = make_tiled_mma(MMA_op{}, Layout<Shape<_2,_2,_1>>{});
  static_assert(decltype(size(tiled_mma))::value == 128,
                "TiledMMA should span 128 threads (4 warps)");

  // CTA tile 形状（用 gmem dummy ptr 只是为了让 layout 走通；不实际访存）
  auto layout_gC = make_layout(Shape<_32, _16>{},   Stride<_16, _1>{});  // M=32, N=16, row-major
  auto layout_gA = make_layout(Shape<_32, _16>{},   Stride<_16, _1>{});  // M=32, K=16
  auto layout_gB = make_layout(Shape<_16, _16>{},   Stride<_16, _1>{});  // N=16, K=16

  // 用 device-side dummy pointer 0x10000，反正不解引用
  using T = cutlass::half_t;
  auto gC = make_tensor(make_gmem_ptr((T*)0x10000), layout_gC);
  auto gA = make_tensor(make_gmem_ptr((T*)0x20000), layout_gA);
  auto gB = make_tensor(make_gmem_ptr((T*)0x30000), layout_gB);

  print_partition_kernel<<<1, 128>>>(tiled_mma, gC, gA, gB);
  cudaDeviceSynchronize();

  printf("\n做完对照：\n");
  printf("  - 4 个 thread 的 partition_C shape 都应该是 (2,1,1)（V=2 是 atom 内, rest=1）\n");
  printf("  - thread 0 和 32 的 partition_C **同 shape 但 stride 不同**，因为它们属于不同 warp\n");
  return 0;
}
