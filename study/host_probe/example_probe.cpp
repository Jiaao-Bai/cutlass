// study/host_probe/example_probe.cpp —— 模板探针。复制我、改 main()、`make run`。
//
// 这是 host_probe 的"hello world":构造一个 SM100 2x1SM TiledMMA,打印它 +
// partition_shape_A/B。在纯 CPU(g++ + ../stub)上就能跑出真值。
//
//   cd study/host_probe && make run
//
// 想验别的(swizzle / TMA / 你自己的 tiler),照着加几行 print 即可。
// 真实用例见同目录 README 指向的两个 probe_*.cu。

#include <cstdio>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
using namespace cute;

int main() {
  TiledMMA tiled_mma = make_tiled_mma(
    SM100_MMA_F16BF16_2x1SM_SS<half_t, half_t, float, 256, 256, UMMA::Major::K, UMMA::Major::K>{});
  print(tiled_mma); printf("\n");

  auto mma_tiler = make_shape(tile_size<0>(tiled_mma),
                              tile_size<1>(tiled_mma),
                              tile_size<2>(tiled_mma) * Int<4>{});
  printf("mma_tiler:         "); print(mma_tiler); printf("\n");
  printf("partition_shape_A: "); print(partition_shape_A(tiled_mma, make_shape(size<0>(mma_tiler), size<2>(mma_tiler)))); printf("\n");
  printf("partition_shape_B: "); print(partition_shape_B(tiled_mma, make_shape(size<1>(mma_tiler), size<2>(mma_tiler)))); printf("\n");
  return 0;
}
