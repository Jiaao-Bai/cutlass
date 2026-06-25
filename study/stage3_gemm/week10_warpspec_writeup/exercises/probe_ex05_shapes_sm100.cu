// 复刻 examples/cute/tutorial/blackwell/05_mma_tma_epi_sm100.cu 里的 *所有* layout print,
// 跑出真值,用于核对/修正源码里的注释。2025 起源码注释有 9 处 stale shape,本探针是改正依据。
//
// 全是 host constexpr 代数(MMA atom 的 fma() 不被调用,不触发 SM100 指令),
// 5060Ti / 任意能编 nvcc 的机器都能跑(甚至无 GPU):
//   nvcc -std=c++17 -I /path/to/cutlass/include -arch=sm_90a probe_ex05_shapes_sm100.cu -o p && ./p
//
// 真值表(对照 example 05 注释,❌=源码注释曾经写错的):
//   gA   (_256,_64,4)              ❌注释曾写 _128   (cluster MMA tile,partition 前 M=256)
//   gC/gD(_256,_256)               ❌注释曾写 _128
//   tCgB ((_128,_16),_1,_4,4)      ❌注释曾写 _256   (partition 后 per-CTA,B 沿 N 分摊→128)
//   tCsB ((_128,_16),_1,_4)        ❌注释曾写 _256
//   mma_shape_B ((_128,_16),_1,_4) ❌注释曾写 _256
//   sB   ((_128,_16),_1,_4)        ❌注释曾写 _256
//   tBgB (((_64,_128),_1),4)       ❌注释曾写 _256
//   tBsB ((_8192,_1))              ❌注释曾写 _16384
//   —— 其余(gB/tCgA/tCgC/tCsA/tCrA/tCrB/tCtAcc/tma_atom_*/tAgA/tAsA/C/D)注释本就正确。

#include <cstdio>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
using namespace cute;
template<class T> static void L(const char* n, T const& t){ printf("%-8s", n); print(t); printf("\n"); }

int main() {
  TiledMMA tiled_mma = make_tiled_mma(
    SM100_MMA_F16BF16_2x1SM_SS<half_t,half_t,float,256,256,UMMA::Major::K,UMMA::Major::K>{});
  auto bM = tile_size<0>(tiled_mma);
  auto bN = tile_size<1>(tiled_mma);
  auto bK = tile_size<2>(tiled_mma) * Int<4>{};
  auto mma_tiler = make_shape(bM, bN, bK);
  L("mma_tiler", mma_tiler);

  auto mma_shape_A = partition_shape_A(tiled_mma, make_shape(size<0>(mma_tiler), size<2>(mma_tiler)));
  auto mma_shape_B = partition_shape_B(tiled_mma, make_shape(size<1>(mma_tiler), size<2>(mma_tiler)));
  auto mma_shape_C = partition_shape_C(tiled_mma, make_shape(size<0>(mma_tiler), size<1>(mma_tiler)));
  L("mma_shA", mma_shape_A); L("mma_shB", mma_shape_B); L("mma_shC", mma_shape_C);

  auto sA = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<half_t>{}, mma_shape_A);
  auto sB = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<half_t>{}, mma_shape_B);
  L("sA", sA); L("sB", sB);

  // GMEM(真指针,TN=K-major)+ 坐标 tensor;问题规模 M512 N1024 K256
  auto mA = make_tensor(make_gmem_ptr((half_t*)nullptr), make_layout(make_shape(512,256),  make_stride(256,1)));
  auto mB = make_tensor(make_gmem_ptr((half_t*)nullptr), make_layout(make_shape(1024,256), make_stride(256,1)));
  auto cluster_shape = make_shape(Int<4>{}, Int<4>{}, Int<1>{});
  Layout cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape),
                                            make_tile(typename decltype(tiled_mma)::AtomThrID{}));
  Copy_Atom tma_A = make_tma_atom_A_sm100(SM100_TMA_2SM_LOAD_MULTICAST{}, mA, sA, mma_tiler, tiled_mma, cluster_layout_vmnk);
  Copy_Atom tma_B = make_tma_atom_B_sm100(SM100_TMA_2SM_LOAD_MULTICAST{}, mB, sB, mma_tiler, tiled_mma, cluster_layout_vmnk);

  auto mA_t = tma_A.get_tma_tensor(make_shape(512,256));
  auto mB_t = tma_B.get_tma_tensor(make_shape(1024,256));
  auto mma_coord = make_coord(Int<0>{}, Int<0>{}, _);   // (m,n,k=_);坐标只改 offset 不改 shape
  auto gA = local_tile(mA_t, mma_tiler, mma_coord, Step<_1, X,_1>{});
  auto gB = local_tile(mB_t, mma_tiler, mma_coord, Step< X,_1,_1>{});
  L("gA", gA); L("gB", gB);

  auto cta = tiled_mma.get_slice(Int<0>{});             // mma_v=0
  auto tCgA = cta.partition_A(gA);
  auto tCgB = cta.partition_B(gB);
  L("tCgA", tCgA); L("tCgB", tCgB);

  auto tCsA = make_tensor(make_smem_ptr((half_t*)nullptr), sA);
  auto tCsB = make_tensor(make_smem_ptr((half_t*)nullptr), sB);
  L("tCsA", tCsA); L("tCsB", tCsB);
  L("tCrA", cta.make_fragment_A(tCsA));
  L("tCrB", cta.make_fragment_B(tCsB));

  auto cvmnk = cluster_layout_vmnk.get_flat_coord(0);
  auto rA = tma_partition(tma_A, get<2>(cvmnk), make_layout(size<2>(cluster_layout_vmnk)),
                          group_modes<0,3>(tCsA), group_modes<0,3>(tCgA));
  auto rB = tma_partition(tma_B, get<1>(cvmnk), make_layout(size<1>(cluster_layout_vmnk)),
                          group_modes<0,3>(tCsB), group_modes<0,3>(tCgB));
  L("tAgA", get<0>(rA)); L("tAsA", get<1>(rA));
  L("tBgB", get<0>(rB)); L("tBsB", get<1>(rB));
  return 0;
}
