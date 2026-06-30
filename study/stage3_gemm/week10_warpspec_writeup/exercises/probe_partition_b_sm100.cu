// 一锤定音:2x1SM 的 B 每个 CTA 到底是 N/2(分摊,1 份)还是 完整 N(复制,2 份)?
//
// 背景:example 05 的源码注释写 mma_shape_B = ((256,16),1,4) / tCsB = ((256,16)),
//       但 agent 用 g++ 跑 CUTLASS 自己的 partition_shape_B 得到的是 ((128,16),1,4)。
//       怀疑 example 注释 stale。此探针直接打印真值 + 构建 SMEM 布局看每 CTA 占用。
//
// 全是 host constexpr layout 代数,不需要 B200,任意能编 nvcc 的卡都能跑:
//   nvcc -std=c++17 -I /path/to/cutlass/include -arch=sm_90a \
//        probe_partition_b_sm100.cu -o probe_b && ./probe_b
// (有 B200 就 -arch=sm_100a;其实 -x cu 编 host 也行,不调用 fma 不触发 SM100 指令)

#include <cstdio>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/atom/mma_traits_sm100.hpp>

using namespace cute;

int main() {
  // 与 example 05 完全一致的 2x1SM TiledMMA(M=256,N=256,K-major)
  TiledMMA tiled_mma = make_tiled_mma(
    SM100_MMA_F16BF16_2x1SM_SS<half_t, half_t, float,
                               256, 256,
                               UMMA::Major::K, UMMA::Major::K>{});
  print("TiledMMA:\n"); print(tiled_mma); print("\n\n");

  auto bM = tile_size<0>(tiled_mma);
  auto bN = tile_size<1>(tiled_mma);
  auto bK = tile_size<2>(tiled_mma) * Int<4>{};
  auto mma_tiler = make_shape(bM, bN, bK);
  print("mma_tiler (bM,bN,bK): "); print(mma_tiler); print("\n\n");

  // —— 核心:partition_shape_A vs partition_shape_B ——
  auto mma_shape_A = partition_shape_A(tiled_mma, make_shape(size<0>(mma_tiler), size<2>(mma_tiler)));
  auto mma_shape_B = partition_shape_B(tiled_mma, make_shape(size<1>(mma_tiler), size<2>(mma_tiler)));
  print("partition_shape_A: "); print(mma_shape_A);
  print("   (example 注释: ((_128,_16),_1,_4))\n");
  print("partition_shape_B: "); print(mma_shape_B);
  print("   (example 注释: ((_256,_16),_1,_4))  <== 看这里到底是 128 还是 256\n\n");

  // —— 构建每 CTA 的 SMEM 布局(同 example 05),看占用元素数 ——
  auto sA_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<half_t>{}, mma_shape_A);
  auto sB_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<half_t>{}, mma_shape_B);
  print("sA_layout: "); print(sA_layout); print("\n");
  print("sB_layout: "); print(sB_layout); print("\n\n");

  int A_per_cta = cosize(sA_layout);
  int B_per_cta = cosize(sB_layout);
  printf("=========================================================\n");
  printf("A per CTA = %d half-elems\n", A_per_cta);
  printf("B per CTA = %d half-elems\n", B_per_cta);
  printf("---------------------------------------------------------\n");
  printf("判读:\n");
  printf("  B_per_cta == 8192  => B 沿 N 分摊(N/2=128),cluster 1 份  => 2-SM 省 B SMEM ✅\n");
  printf("  B_per_cta == 16384 => B 复制(完整 N=256),cluster 2 份    => 2-SM 不省 B\n");
  printf("=========================================================\n");
  return 0;
}
