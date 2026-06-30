// 复刻 examples/cute/tutorial/blackwell/05_mma_tma_epi_sm100.cu 的 *全部* layout(含 epilogue),
// 在 CPU 上跑出真值,用于核对我给源码加的"具体值"注释。
//
// 全是 host constexpr 代数(MMA/TMA/TMEM 的真实指令不被调用),用 host_probe 的 stub 顶掉 CUDA 头:
//   cd study/host_probe && make run PROBE=../stage3_gemm/week10_warpspec_writeup/exercises/probe_ex05_full_sm100.cu
//
// 问题规模(与 example 05 默认一致):M=512, N=1024, K=256, cluster=(4,4,1), 2SM-MMA。
//
// 说明:打印里的指针/地址是假的((nil)/0x0),会有 cast_smem_ptr_to_uint 的 stderr 警告 —— 无视,shape/stride 是对的。
//       CuTe 的 print 给出的是 *未 coalesce* 的原始模,可能比源码注释多拆几层(如 _64 显示成 _16,_4),size 相同即等价。

#include <cstdio>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/atom/copy_traits_sm100.hpp>
using namespace cute;
template<class T> static void L(const char* n, T const& t){ printf("%-10s", n); print(t); printf("\n"); }

int main() {
  using TypeA = half_t; using TypeB = half_t; using TypeC = float; using TypeD = float;

  TiledMMA tiled_mma = make_tiled_mma(
    SM100_MMA_F16BF16_2x1SM_SS<TypeA,TypeB,TypeC,256,256,UMMA::Major::K,UMMA::Major::K>{});

  auto bM = tile_size<0>(tiled_mma);
  auto bN = tile_size<1>(tiled_mma);
  auto bK = tile_size<2>(tiled_mma) * Int<4>{};
  auto mma_tiler = make_shape(bM, bN, bK);
  L("mma_tiler", mma_tiler);

  // ---- cluster ----
  auto cluster_shape = make_shape(Int<4>{}, Int<4>{}, Int<1>{});
  Layout cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape),
                                            make_tile(typename decltype(tiled_mma)::AtomThrID{}));
  L("cluster_vmnk", cluster_layout_vmnk);
  printf("  size<0..3>(cluster_vmnk)=V%d M%d N%d K%d\n",
         int(size<0>(cluster_layout_vmnk)), int(size<1>(cluster_layout_vmnk)),
         int(size<2>(cluster_layout_vmnk)), int(size<3>(cluster_layout_vmnk)));

  // ---- A/B SMEM layouts ----
  auto mma_shape_A = partition_shape_A(tiled_mma, make_shape(size<0>(mma_tiler), size<2>(mma_tiler)));
  auto mma_shape_B = partition_shape_B(tiled_mma, make_shape(size<1>(mma_tiler), size<2>(mma_tiler)));
  auto mma_shape_C = partition_shape_C(tiled_mma, make_shape(size<0>(mma_tiler), size<1>(mma_tiler)));
  L("mma_shape_A", mma_shape_A); L("mma_shape_B", mma_shape_B); L("mma_shape_C", mma_shape_C);

  auto sA_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TypeA>{}, mma_shape_A);
  auto sB_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TypeB>{}, mma_shape_B);
  L("sA_layout", sA_layout); L("sB_layout", sB_layout);

  // ---- epilogue C/D SMEM layouts ----
  auto epi_tiler = make_tile(size<0,0>(mma_shape_C), size<0,1>(mma_shape_C) / Int<4>{});
  L("epi_tiler", epi_tiler);
  auto sC_layout_mn = tile_to_shape(UMMA::Layout_K_SW128_Atom<TypeC>{}, make_shape(size<0>(epi_tiler), size<1>(epi_tiler)));
  auto sD_layout_mn = tile_to_shape(UMMA::Layout_K_SW128_Atom<TypeD>{}, make_shape(size<0>(epi_tiler), size<1>(epi_tiler)));
  auto sC_layout = group<0,2>(sC_layout_mn);
  auto sD_layout = group<0,2>(sD_layout_mn);
  L("sC_layout_mn", sC_layout_mn); L("sC_layout", sC_layout);
  L("sD_layout_mn", sD_layout_mn); L("sD_layout", sD_layout);

  // ---- GMEM tensors (TN; C/D row-major N=1024) ----
  auto mA = make_tensor(make_gmem_ptr((TypeA*)nullptr), make_layout(make_shape(512,256),  make_stride(256,1)));
  auto mB = make_tensor(make_gmem_ptr((TypeB*)nullptr), make_layout(make_shape(1024,256), make_stride(256,1)));
  auto mC = make_tensor(make_gmem_ptr((TypeC*)nullptr), make_layout(make_shape(512,1024), make_stride(1024,1)));
  auto mD = make_tensor(make_gmem_ptr((TypeD*)nullptr), make_layout(make_shape(512,1024), make_stride(1024,1)));
  L("mC", mC); L("mD", mD);

  // ---- TMA atoms ----
  Copy_Atom tma_A = make_tma_atom_A_sm100(SM100_TMA_2SM_LOAD_MULTICAST{}, mA, sA_layout, mma_tiler, tiled_mma, cluster_layout_vmnk);
  Copy_Atom tma_B = make_tma_atom_B_sm100(SM100_TMA_2SM_LOAD_MULTICAST{}, mB, sB_layout, mma_tiler, tiled_mma, cluster_layout_vmnk);
  Copy_Atom tma_C = make_tma_atom(SM90_TMA_LOAD{},  mC, sC_layout, epi_tiler);
  Copy_Atom tma_D = make_tma_atom(SM90_TMA_STORE{}, mD, sD_layout, epi_tiler);

  auto mA_t = tma_A.get_tma_tensor(make_shape(512,256));
  auto mB_t = tma_B.get_tma_tensor(make_shape(1024,256));
  auto mC_t = tma_C.get_tma_tensor(make_shape(512,1024));
  auto mD_t = tma_D.get_tma_tensor(make_shape(512,1024));
  L("mC_t", mC_t); L("mD_t", mD_t);

  // ---- per-MMA-tile GMEM slices ----
  auto mma_coord = make_coord(Int<0>{}, Int<0>{}, _);
  auto gA = local_tile(mA_t, mma_tiler, mma_coord, Step<_1, X,_1>{});
  auto gB = local_tile(mB_t, mma_tiler, mma_coord, Step< X,_1,_1>{});
  auto gC = local_tile(mC_t, mma_tiler, mma_coord, Step<_1,_1, X>{});
  auto gD = local_tile(mD_t, mma_tiler, mma_coord, Step<_1,_1, X>{});
  L("gA", gA); L("gB", gB); L("gC", gC); L("gD", gD);

  // ---- MMA partition ----
  auto cta = tiled_mma.get_slice(Int<0>{});   // mma_v=0
  auto tCgA = cta.partition_A(gA);
  auto tCgB = cta.partition_B(gB);
  auto tCgC = cta.partition_C(gC);
  auto tCgD = cta.partition_C(gD);
  L("tCgA", tCgA); L("tCgB", tCgB); L("tCgC", tCgC); L("tCgD", tCgD);

  auto tCsA = make_tensor(make_smem_ptr((TypeA*)nullptr), sA_layout);
  auto tCsB = make_tensor(make_smem_ptr((TypeB*)nullptr), sB_layout);
  L("tCsA", tCsA); L("tCsB", tCsB);
  L("tCrA", cta.make_fragment_A(tCsA));
  L("tCrB", cta.make_fragment_B(tCsB));
  auto tCtAcc = cta.make_fragment_C(tCgC);
  L("tCtAcc", tCtAcc);

  // ---- mainloop TMA partition ----
  auto cvmnk = cluster_layout_vmnk.get_flat_coord(0);
  auto rA = tma_partition(tma_A, get<2>(cvmnk), make_layout(size<2>(cluster_layout_vmnk)),
                          group_modes<0,3>(tCsA), group_modes<0,3>(tCgA));
  auto rB = tma_partition(tma_B, get<1>(cvmnk), make_layout(size<1>(cluster_layout_vmnk)),
                          group_modes<0,3>(tCsB), group_modes<0,3>(tCgB));
  L("tAgA", get<0>(rA)); L("tAsA", get<1>(rA));
  L("tBgB", get<0>(rB)); L("tBsB", get<1>(rB));

  // ---- EPILOGUE ----
  auto epi_tiler_v = make_tile(epi_tiler);
  L("epi_tiler_v", epi_tiler_v);
  auto tAcc_epi = zipped_divide(tCtAcc, epi_tiler_v);
  auto gC_epi   = zipped_divide(tCgC,   epi_tiler_v);
  auto gD_epi   = zipped_divide(tCgD,   epi_tiler_v);
  L("tAcc_epi", tAcc_epi); L("gC_epi", gC_epi); L("gD_epi", gD_epi);

  auto sC_epi = make_tensor(make_smem_ptr((TypeC*)nullptr), sC_layout);
  auto sD_epi = make_tensor(make_smem_ptr((TypeD*)nullptr), sD_layout);
  L("sC_epi", sC_epi); L("sD_epi", sD_epi);

  auto rC = tma_partition(tma_C, sC_epi, gC_epi);
  auto rD = tma_partition(tma_D, sD_epi, gD_epi);
  L("tGS_gC", get<0>(rC)); L("tGS_sC", get<1>(rC));
  L("tSG_gD", get<0>(rD)); L("tSG_sD", get<1>(rD));

  TiledCopy t2r_copy = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tAcc_epi(_,Int<0>{}));
  auto thr_t2r = t2r_copy.get_slice(Int<0>{});
  auto tTR_tAcc = thr_t2r.partition_S(tAcc_epi);
  auto tTR_sC   = thr_t2r.partition_D(sC_epi);
  auto tTR_sD   = thr_t2r.partition_D(sD_epi);
  L("tTR_tAcc", tTR_tAcc); L("tTR_sC", tTR_sC); L("tTR_sD", tTR_sD);
  L("tTR_rC", make_tensor_like(tTR_sC));
  L("tTR_rD", make_fragment_like(tTR_sD));
  return 0;
}
