/***************************************************************************************************
 * probe_shapes_sm100.cu — 纯 host 端打印 05 的 layout/shape，不 launch kernel、不跑 GEMM
 *
 * 目的：在 5060 Ti(SM120)甚至无 GPU 机器上，看清 tutorial 05 的：
 *   - TiledMMA / ThrLayoutVMNK / ALayout
 *   - partition_shape_A/B 的输出
 *   - tile_to_mma_shape 出来的 sA_layout（含真实 swizzle Sw<B,M,S>）
 *   - 任意坐标的真实 smem offset（验证 swizzle 到底打到哪）
 *
 * 全是编译期 layout 代数 + host print，不触发 tcgen05 指令，所以任何 arch 都能跑。
 *
 * 编译（5060 Ti 或任意机器都行，arch 无所谓因为不 launch）：
 *   nvcc -std=c++17 -I include -arch=sm_120a study/.../probe_shapes_sm100.cu -o probe
 *   ./probe
 **************************************************************************************************/
#include <cstdio>
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include "cutlass/half.h"

using namespace cute;
using TypeA = cutlass::half_t;
using TypeB = cutlass::half_t;
using TypeC = float;

int main() {
  // —— 和 05 完全一致的 2-SM UMMA atom ——
  TiledMMA tiled_mma = make_tiled_mma(
      SM100_MMA_F16BF16_2x1SM_SS<TypeA, TypeB, TypeC,
                                 256, 256,
                                 UMMA::Major::K, UMMA::Major::K>{});

  print("==== TiledMMA ====\n"); print(tiled_mma); print("\n\n");

  // —— mma_tiler 和 05 一致：bM=tile_size<0>, bK=tile_size<2>*4 ——
  auto bM = tile_size<0>(tiled_mma);
  auto bN = tile_size<1>(tiled_mma);
  auto bK = tile_size<2>(tiled_mma) * Int<4>{};
  auto mma_tiler = make_shape(bM, bN, bK);
  print("mma_tiler:\t"); print(mma_tiler); print("\n\n");

  // —— partition_shape_A/B：单 CTA 的数据份额形状 ——
  auto mma_shape_A = partition_shape_A(tiled_mma, make_shape(size<0>(mma_tiler), size<2>(mma_tiler)));
  auto mma_shape_B = partition_shape_B(tiled_mma, make_shape(size<1>(mma_tiler), size<2>(mma_tiler)));
  print("mma_shape_A:\t"); print(mma_shape_A); print("\n");
  print("mma_shape_B:\t"); print(mma_shape_B); print("\n\n");

  // —— tile_to_mma_shape：带 swizzle 的 smem 布局（看真实 Sw<B,M,S>）——
  auto sA_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TypeA>{}, mma_shape_A);
  auto sB_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TypeB>{}, mma_shape_B);
  print("sA_layout:\t"); print(sA_layout); print("\n");
  print("sB_layout:\t"); print(sB_layout); print("\n\n");

  // —— 上一轮的悬案：某坐标的真实 smem offset，看 swizzle 到底打到哪 ——
  // sA_layout 形如 ((MMA_M,MMA_K),M_MMAs,K_MMAs)；这里取内层 (m,k) 扫几个看 offset。
  print("==== sA offset 探针 (coord -> swizzled offset, 元素单位) ====\n");
  for (int m = 0; m < 8; ++m) {
    for (int k = 0; k < 16; k += 8) {
      // 坐标结构跟 sA_layout 对齐：((m,k), 0, 0)
      auto off = sA_layout(make_coord(make_coord(m, k), 0, 0));
      printf("(m=%d,k=%2d) -> %d\n", m, k, int(off));
    }
  }

  // 也单独看 swizzle 原子本身的真实参数
  print("\nLayout_K_SW128_Atom<half>:\t");
  print(UMMA::Layout_K_SW128_Atom<TypeA>{}); print("\n");

  return 0;
}
