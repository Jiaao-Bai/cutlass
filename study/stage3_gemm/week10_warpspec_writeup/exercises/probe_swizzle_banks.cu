/***************************************************************************************************
 * probe_swizzle_banks.cu — 实测 Layout_K_SW128_Atom 对 half / float 的 bank 分布
 *
 * 纯 host，不 launch kernel，5060Ti 或任意有 CUDA toolkit 的机器都能编译运行。
 *   nvcc -std=c++17 -I include study/.../probe_swizzle_banks.cu -o probe && ./probe
 *
 * bank = (元素offset × sizeof(Type) / 4) % 32     // smem 32 banks × 4 字节
 *   half (2B): 每 bank 装 2 个 half，bank=(off*2/4)%32=(off/2)%32
 *   float(4B): 每 bank 装 1 个 float，bank=(off*4/4)%32=off%32
 **************************************************************************************************/
#include <cstdio>
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include "cutlass/half.h"

using namespace cute;

template <class Type, int COLS>
void dump(const char* name) {
  auto atom = UMMA::Layout_K_SW128_Atom<Type>{};
  printf("\n==== Layout_K_SW128_Atom<%s>  (sizeof=%d B) ====\n", name, int(sizeof(Type)));
  print("layout: "); print(atom); print("\n");

  // 表头
  printf("       ");
  for (int j = 0; j < COLS; ++j) printf("%3d", j);
  printf("   <- col(K)\n");

  for (int i = 0; i < 8; ++i) {            // 8 行(swizzle 周期)
    printf("row%d:  ", i);
    for (int j = 0; j < COLS; ++j) {
      auto off  = atom(make_coord(i, j)); // ComposedLayout: Sw(B((i,j)))，元素 offset
      int  bank = (int(off) * int(sizeof(Type)) / 4) % 32;
      printf("%3d", bank);
    }
    printf("\n");
  }
}

int main() {
  dump<cutlass::half_t, 64>("half ");   // half  atom = 8 × 64
  dump<float,           32>("float");   // float atom = 8 × 32
  return 0;
}
