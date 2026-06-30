/***************************************************************************************************
 * Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

///////////////////////////////////////////////////////////////////////////////////////////////////
//
//                             CuTe Tutorial for SM100 Programming
// This tutorial series demonstrates CuTe Blackwell capabilities that are frequently used
// throughout CUTLASS. The goal is to familiarize developers with CuTe SM100 interfaces.
//
// The tutorial series is split into five stages:
// * 01_mma_sm100.cu: Simple Blackwell SM100 GEMM using a tcgen05.mma instruction.
// * 02_mma_tma_sm100.cu: Simple Blackwell SM100 GEMM using tcgen05.mma and TMA instructions.
// * 03_mma_tma_multicast_sm100.cu: Blackwell SM100 GEMM using tcgen05.mma and Multicast TMA.
// * 04_mma_tma_2sm_sm100.cu: Blackwell SM100 GEMM with 2SM tcgen05.mma and 2SM Multicast TMA.
// * 05_mma_tma_epi_sm100.cu: Blackwell SM100 GEMM with 2SM tcgen05.mma, 2SM TMA mainloop, and TMA epilogue.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <cstdio>

// Use Thrust to handle host/device allocations
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// Cutlass includes
#include <cutlass/half.h>                       // F16 data type
#include <cutlass/util/print_error.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>

// CuTe includes
#include <cute/tensor.hpp>                      // CuTe tensor implementation
#include <cute/arch/cluster_sm90.hpp>           // CuTe functions for querying the details of cluster launched
#include <cute/numeric/integral_constant.hpp>   // Compile time in constants such as _1, _256 etc.
#include <cute/algorithm/cooperative_copy.hpp>  // Auto vectorized copy operation
#include <cute/arch/tmem_allocator_sm100.hpp>   // TMEM allocator for SM100

// Tutorial helpers
#include "example_utils.hpp"

using namespace cute;

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Tutorial 05: Blackwell SM100 GEMM with 2SM tcgen05.mma, 2SM TMA mainloop, and TMA epilogue
//
///////////////////////////////////////////////////////////////////////////////////////////////////

// We will implement a GEMM operation: D (f32) = beta * C (F32) + alpha * A (F16) * B (F16) where:
// - Matrix A is MxK, K-major (BLAS transpose T, row-major)
// - Matrix B is NxK, K-major (BLAS transpose N, column-major)
// - Matrices C and D are MxN, N-major (BLAS row-major)
//
// Key extensions to tutorial 04_mma_tma_2sm_sm100.cu:
// 1. Demonstrate using TMA instructions in the epilogue
//
// This GEMM kernel will perform the following steps:
// 1. Load A and B matrices from GMEM to SMEM using Multicasted TMA.2SM load operations.
// 2. Perform matrix multiply-accumulate (MMA) operations using 2SM tcgen05.mma instruction.
// 3. Load completed accumulator from tensor memory (TMEM) to registers (RMEM) using tcgen05.ld.
// 4. Read C matrix from global memory (GMEM) to shared memory (SMEM) with TMA.
// 5. Apply alpha and beta scaling to the MMA accumulator and C matrix.
// 6. Store D matrix from shared memory (SMEM) to global memory (GMEM) with TMA.
//
// SM100 2SM tcgen05.mma instructions operate as follows:
// - Mma is launched by only one SM
//    With 2SM MMA instructions, only 1 of the 2 CTAs collaborating on MMA executes the instruction.
//    We call the collaborating CTAs, peer CTAs. And the CTA executing the MMA instruction is called leader CTA.
// - Read matrix A from SMEM or TMEM
// - Read matrix B from SMEM
// - Write accumulator to TMEM
// The accumulator in TMEM must then be loaded to registers before writing back to GMEM.
//
// The tcgen05.mma instruction requires an Instruction Descriptor that encodes A, B, and Accumulator types
//   and the MMA's M and N dimensions.
// The A and B matrices that are read from SMEM need to be provided to MMA instructions as SMEM Descriptors.
//   These are the A and B fragments of the tcgen05.mma in CuTe terminology.
// CuTe provides these descriptors transparently in the instruction and fragments, shown in this tutorial.
//
// The MMA details:
// We use the tcgen05.mma.f16 instruction (F16xF16 = F32) that performs a 256x256x16 MMA
// operation. F32 accumulator type is chosen since both C and D matrices use F32.
// This example uses F16xF16 = F32 MMA where:
// TypeA = cutlass::half_t;  // MMA A Data Type
// TypeB = cutlass::half_t;  // MMA B Data Type
// TypeC = float;            // MMA C Data Type
// TypeD = float;            // MMA D Data Type
// TypeAccumulator = float;  // Both TypeC and TypeD are float, so we use float accumulator type

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// The shared memory buffers for A, B, C, and D matrices.
template <class TypeA,           // Tensor A data type
          class TypeB,           // Tensor B data type
          class TypeC,           // Tensor C data type
          class TypeD,           // Tensor D data type
          class ASmemLayout,     // (MmaA, NumMma_M, NumMma_K, ...)
          class BSmemLayout,     // (MmaB, NumMma_N, NumMma_K, ...)
          class CSmemLayout,     // EpiTile_M, EpiTile_N
          class DSmemLayout>     // EpiTile_M, EpiTile_N
struct SharedStorage
{
  alignas(128) union {
    alignas(128) struct {
      alignas(128) cute::ArrayEngine<TypeA, cute::cosize_v<ASmemLayout>> A;
      alignas(128) cute::ArrayEngine<TypeB, cute::cosize_v<BSmemLayout>> B;
    } mainloop;
    alignas(128) cute::ArrayEngine<TypeC, cute::cosize_v<CSmemLayout>> C;
    alignas(128) cute::ArrayEngine<TypeD, cute::cosize_v<DSmemLayout>> D;
  } tensors;

  alignas(16) cute::uint64_t mma_barrier;  // Barrier to track MMA computation on SMEM
  alignas(16) cute::uint64_t tma_barrier;  // Barrier to track TMA data transfers to SMEM

  alignas(16) cute::uint32_t tmem_base_ptr; // Base pointer for TMEM allocation

  CUTE_DEVICE constexpr auto tensor_sA() { return make_tensor(make_smem_ptr(tensors.mainloop.A.begin()), ASmemLayout{}); }
  CUTE_DEVICE constexpr auto tensor_sB() { return make_tensor(make_smem_ptr(tensors.mainloop.B.begin()), BSmemLayout{}); }
  CUTE_DEVICE constexpr auto tensor_sC() { return make_tensor(make_smem_ptr(tensors.C.begin()), CSmemLayout{}); }
  CUTE_DEVICE constexpr auto tensor_sD() { return make_tensor(make_smem_ptr(tensors.D.begin()), DSmemLayout{}); }
};

// The device kernel
template <class SharedStorage,
          class ATensor, class BTensor, class CTensor, class DTensor,
          class MmaTiler_MNK, class EpiTiler_MN, class TiledMMA, class ClusterShape_MNK,
          class TmaAtomA, class TmaAtomB, class TmaAtomC, class TmaAtomD,
          class Alpha, class Beta>
__global__ static
void
gemm_device(ATensor mA,                      // (Gemm_M, Gemm_K)
            BTensor mB,                      // (Gemm_N, Gemm_K)
            CTensor mC,                      // (Gemm_M, Gemm_N)
            DTensor mD,                      // (Gemm_M, Gemm_N)
            MmaTiler_MNK mma_tiler,          // <MmaTile_M, MmaTile_N, MmaTile_K>
            EpiTiler_MN epi_tiler_mn,        // <EpiTile_M, EpiTile_N>
            TiledMMA tiled_mma,              // <    Mma_M,     Mma_N,     Mma_K>
            ClusterShape_MNK cluster_shape,  // (ClusterM, ClusterN, ClusterK)
            CUTE_GRID_CONSTANT TmaAtomA const tma_atom_A,
            CUTE_GRID_CONSTANT TmaAtomB const tma_atom_B,
            CUTE_GRID_CONSTANT TmaAtomC const tma_atom_C,
            CUTE_GRID_CONSTANT TmaAtomD const tma_atom_D,
            Alpha alpha, Beta beta)
{
  // Step 1: The Prologue.

  // The CTA layout within the Cluster: (V,M,N,K) -> CTA idx
  Layout cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape),
                                            make_tile(typename TiledMMA::AtomThrID{}));
  // 具体值(M=512,N=1024,K=256, cluster=(4,4,1), 2SM-MMA):
  //   cluster_shape                          = (_4,_4,_1)
  //   make_layout(cluster_shape)             = (_4,_4,_1):(_1,_4,_16)
  //   TiledMMA::AtomThrID                    = _2:_1            (2 个 peer CTA 协作一条 2SM MMA)
  //   make_tile(AtomThrID)                   = (_2)             (沿 mode0 切出 V=2 的子 tile)
  //   ⇒ cluster_layout_vmnk = (_2,_2,_4,_1):(_1,_2,_4,_16)     (V,M,N,K)=(2,2,4,1)
  //     mode0 的 4 被拆成 V=2(peer)·M=2(剩余 M 簇)，N=4 与 K=1 原样保留

  // Construct the MMA grid coordinate from the CTA grid coordinate
  auto mma_coord_vmnk = make_coord(blockIdx.x % size<0>(cluster_layout_vmnk), // Peer CTA coordinate
                                   blockIdx.x / size<0>(cluster_layout_vmnk), //    MMA-M coordinate
                                   blockIdx.y,                                //    MMA-N coordinate
                                   _);                                        //    MMA-K coordinate

  // Partition the GMEM tensors with the mma_tiler and mma_coord to get the slices processed
  //   by this mma tile.
  // CuTe provides local_tile partitioning function. local_tile accepts 4 parameters:
  //   * Tensor to partition
  //   * Tiler to use for partitioning
  //   * Coordinate to use for slicing the partitioned tensor
  //   * Projection to ignore unwanted modes of the Tiler and Coordinate
  auto mma_coord = select<1,2,3>(mma_coord_vmnk);
  Tensor gA = local_tile(mA, mma_tiler, mma_coord, Step<_1, X,_1>{});  // (MmaTile_M, MmaTile_K, Tiles_K)
  // 具体值: mA=(512,256); mma_tiler=(_256,_256,_64); Step<_1,X,_1> 投影掉 N 模 ⇒ 用 tiler(_256,_,_64)
  //   gA = (_256,_64,4):(_1@1,_1@0,_64@0)  (MmaTile_M=256, MmaTile_K=64, Tiles_K=256/64=4)
  Tensor gB = local_tile(mB, mma_tiler, mma_coord, Step< X,_1,_1>{});  // (MmaTile_N, MmaTile_K, Tiles_K)
  // 具体值: mB=(1024,256); Step<X,_1,_1> 投影掉 M 模 ⇒ 用 tiler(_,_256,_64)
  //   gB = (_256,_64,4):(_1@1,_1@0,_64@0)  (MmaTile_N=256, MmaTile_K=64, Tiles_K=4)
  Tensor gC = local_tile(mC, mma_tiler, mma_coord, Step<_1,_1, X>{});  // (MmaTile_M, MmaTile_N)
  // 具体值: mC=(512,1024); Step<_1,_1,X> 投影掉 K 模 ⇒ 用 tiler(_256,_256,_)
  //   gC = (_256,_256):(1024,_1)  (MmaTile_M=256, MmaTile_N=256)
  Tensor gD = local_tile(mD, mma_tiler, mma_coord, Step<_1,_1, X>{});  // (MmaTile_M, MmaTile_N)
  // 具体值: mD=(512,1024) ⇒ gD = (_256,_256):(1024,_1)  (MmaTile_M=256, MmaTile_N=256)

  if (thread0()) {
    print("mA:\t"); print(mA); print("\n");   // mA:   ArithTuple(_0,_0) o (512,256):(_1@1,_1@0)
    print("mB:\t"); print(mB); print("\n");   // mB:   ArithTuple(_0,_0) o (1024,256):(_1@1,_1@0)
    print("mC:\t"); print(mC); print("\n");   // mC:   gmem_ptr[32b](GMEM_ADDR_C) o (512,1024):(1024,_1)
    print("mD:\t"); print(mD); print("\n");   // mD:   gmem_ptr[32b](GMEM_ADDR_D) o (512,1024):(1024,_1)

    print("gA:\t"); print(gA); print("\n");   // gA:   ArithTuple(_0,0) o (_256,_64,4):(_1@1,_1@0,_64@0)   <- 改正,原误作 (_128,_64,4)
    print("gB:\t"); print(gB); print("\n");   // gB:   ArithTuple(_0,0) o (_256,_64,4):(_1@1,_1@0,_64@0)
    print("gC:\t"); print(gC); print("\n");   // gC:   gmem_ptr[32b](GMEM_ADDR_C + offset_for_mma_tile) o (_256,_256):(1024,_1)   <- 改正,shape 原误作 (_128,_256); stride 原误作 (256,_1)(那是 N=256 的捕获,本例 N=1024 行宽=1024)
    print("gD:\t"); print(gD); print("\n");   // gD:   gmem_ptr[32b](GMEM_ADDR_D + offset_for_mma_tile) o (_256,_256):(1024,_1)   <- 改正,同 gC
  } __syncthreads();

  // The SMEM tensors

  // Allocate SMEM
  extern __shared__ char shared_memory[];
  SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(shared_memory);

  // Represent the SMEM buffers for A and B
  Tensor tCsA = shared_storage.tensor_sA();         // (MmaA, NumMma_M, NumMma_K, Tiles_K)
  // 具体值: ASmemLayout = sA_layout = Sw<3,4,3> ((_128,_16),_1,_4):((_64,_1),_0,_16)
  //   tCsA 形状 ((_128,_16),_1,_4)  (MmaA=(128,16), NumMma_M=1, NumMma_K=4)  ※单 buffer,无 Tiles_K 维
  Tensor tCsB = shared_storage.tensor_sB();         // (MmaB, NumMma_M, NumMma_K, Tiles_K)
  // 具体值: BSmemLayout = sB_layout = Sw<3,4,3> ((_128,_16),_1,_4):((_64,_1),_0,_16)
  //   tCsB 形状 ((_128,_16),_1,_4)  (MmaB=(128,16), NumMma_N=1, NumMma_K=4)

  //
  // Mma partitioning for A and B
  //

  auto mma_v = get<0>(mma_coord_vmnk);
  ThrMMA cta_mma = tiled_mma.get_slice(mma_v);   // Use Peer CTA coordinate
  Tensor tCgA = cta_mma.partition_A(gA);         // (MmaA, NumMma_M, NumMma_K, Tiles_K)
  // 具体值: gA=(_256,_64,4); 2SM-MMA 把 M=256 拆给 2 个 peer CTA(每 CTA 128)，K=64 拆成 4 个 MMA(每 16)
  //   tCgA = ((_128,_16),_1,_4,4):((_1@1,_1@0),_0,_16@0,_64@0)  (MmaA=(128,16),NumMma_M=1,NumMma_K=4,Tiles_K=4)
  Tensor tCgB = cta_mma.partition_B(gB);         // (MmaB, NumMma_N, NumMma_K, Tiles_K)
  // 具体值: gB=(_256,_64,4) ⇒ tCgB = ((_128,_16),_1,_4,4):((_1@1,_1@0),_0,_16@0,_64@0)
  //   (MmaB=(128,16),NumMma_N=1,NumMma_K=4,Tiles_K=4)
  Tensor tCgC = cta_mma.partition_C(gC);         // (MmaC, NumMma_M, NumMma_N)
  // 具体值: gC=(_256,_256):(1024,_1); 2SM 把 M=256 拆成每 CTA 128 ⇒ tCgC = ((_128,_256),_1,_1):((1024,_1),_0,_0)
  Tensor tCgD = cta_mma.partition_C(gD);         // (MmaC, NumMma_M, NumMma_N)
  // 具体值: gD=(_256,_256):(1024,_1) ⇒ tCgD = ((_128,_256),_1,_1):((1024,_1),_0,_0)

  if (thread0()) {
    print("tCgA:\t"); print(tCgA); print("\n");  // tCgA:   ArithTuple(_0,0) o ((_128,_16),_1,_4,4):((_1@1,_1@0),_0,_16@0,_64@0)
    print("tCgB:\t"); print(tCgB); print("\n");  // tCgB:   ArithTuple(_0,0) o ((_128,_16),_1,_4,4):((_1@1,_1@0),_0,_16@0,_64@0)   <- 改正,原误作 ((_256,_16),_1,_4,4)
    print("tCgC:\t"); print(tCgC); print("\n");  // tCgC:   gmem_ptr[32b](GMEM_ADDR_C + offset_for_mma_tile + offset_for_mma) o ((_128,_256),_1,_1):((1024,_1),_0,_0)   <- 改正,stride 原误作 (256,_1)(N=1024 行宽=1024)
    print("tCgD:\t"); print(tCgD); print("\n");  // tCgD:   gmem_ptr[32b](GMEM_ADDR_D + offset_for_mma_tile + offset_for_mma) o ((_128,_256),_1,_1):((1024,_1),_0,_0)   <- 改正,同 tCgC
  } __syncthreads();

  // MMA Fragment Allocation
  // We allocate "fragments" which are SMEM descriptors that serve as inputs to cute::gemm operations.
  // For tcgen05.mma operations:
  // - Matrices A and B are sourced from SMEM
  // - tCrA and tCrB provide descriptor views of tCsA and tCsB respectively
  // - The first mode of each descriptor represents the SMEM for a single MMA operation
  Tensor tCrA = cta_mma.make_fragment_A(tCsA);      // (MmaA, NumMma_M, NumMma_K, Tiles_K)
  // 具体值: 把 tCsA 的 SMEM 包成 UMMA 描述符(不存数据,只存地址/swizzle 信息)
  //   tCrA = UMMA::DescriptorIterator o (_1,_1,_4):(_0,_0,_2)  (NumMma_M=1,NumMma_K=4;K 上 4 个描述符,步进 2)
  Tensor tCrB = cta_mma.make_fragment_B(tCsB);      // (MmaB, NumMma_M, NumMma_K, Tiles_K)
  // 具体值: tCrB = UMMA::DescriptorIterator o (_1,_1,_4):(_0,_0,_2)  (NumMma_N=1,NumMma_K=4)

  // TMEM Allocation
  // On SM100 architecture, accumulators are stored exclusively in tensor memory (TMEM).
  // ThrMma's make_fragment_C() creates a TMEM tensor with the appropriate layout for the accumulator.
  Tensor tCtAcc = cta_mma.make_fragment_C(tCgC);    // (MmaC, NumMma_M, NumMma_N)
  // 具体值: 在 TMEM 里开累加器,形状跟 tCgC 对齐
  //   tCtAcc = tmem_[32b] o ((_128,_256),_1,_1):((_65536,_1),_0,_0)
  //   (MmaC=(128,256): M 跨 TMEM 行 stride=65536=0x10000(列偏移在高位), N 连续 stride=1)

  uint32_t elect_one_thr  = cute::elect_one_sync();
  uint32_t elect_one_warp = (threadIdx.x / 32 == 0);

  using TmemAllocator = cute::TMEM::Allocator2Sm;
  TmemAllocator tmem_allocator{};

  if (elect_one_warp) {
    tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &shared_storage.tmem_base_ptr);
  }
  __syncthreads(); // Wait for all threads until warp0 allocates TMEM
  tCtAcc.data() = shared_storage.tmem_base_ptr;

  if (thread0()) {
    print("tCsA:\t"); print(tCsA); print("\n");     // tCsA:   Sw<3,4,3>_smem_ptr[16b](SMEM_ADDR_A) o ((_128,_16),_1,_4):((_64,_1),_0,_16)
    print("tCsB:\t"); print(tCsB); print("\n");     // tCsB:   Sw<3,4,3>_smem_ptr[16b](SMEM_ADDR_B) o ((_128,_16),_1,_4):((_64,_1),_0,_16)   <- 改正,原误作 ((_256,_16),_1,_4)
    print("tCrA:\t"); print(tCrA); print("\n");     // tCrA:   UMMA::DescriptorIterator o (_1,_1,_4):(_0,_0,_2)
    print("tCrB:\t"); print(tCrB); print("\n");     // tCrB:   UMMA::DescriptorIterator o (_1,_1,_4):(_0,_0,_2)
    print("tCtAcc:\t"); print(tCtAcc); print("\n"); // tCtAcc: tmem_[32b](TMEM_ADDR) o ((_128,_256),_1,_1):((_65536,_1),_0,_0)
  } __syncthreads();

  // TMA Setup
  //
  //   These are TMA partitionings, which have a dedicated custom partitioner.
  //   In this example, the TMA multicasts the loads across multiple CTAs.
  //   Loads of A are multicasted along the N dimension of the cluster_shape_VMNK and
  //   Loads of B are multicasted along the M dimension of the cluster_shape_VMNK.
  //      Any multicasting must be in conformance with tma_x constructed with make_tma_atom on host.
  //   For A tensor: The group_modes<0,3> transforms the (MmaA, NumMma_M, NumMma_K, Tiles_K)-shaped tensor
  //      into ((MmaA, NumMma_M, NumMma_K), Tiles_K). The partitioning only pays attention to mode-0, the MMA Tile MK.
  //   For B tensor: The group_modes<0,3> transforms the (MmaB, NumMma_M, NumMma_K, Tiles_K)-shaped tensor
  //      into ((MmaB, NumMma_M, NumMma_K), Tiles_K). The partitioning only pays attention to mode-0, the MMA Tile NK.
  //   Simply put, the TMA will be responsible for everything in mode-0 with a single call to cute::copy.
  //   The tma_partition reorders and offsets mode-0 according to the tma_x atom and the multicast info.

  // Each CTA with the same m-coord will load a portion of A
  // Each CTA with the same n-coord will load a portion of B
  // Computation of the multicast masks must take into account the Peer CTA for TMA.2SM

  // Construct the CTA-in-Cluster coordinate for multicasting
  auto cta_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(int(cute::block_rank_in_cluster()));
  // 具体值: cluster_layout_vmnk=(_2,_2,_4,_1); 把 rank∈[0,16) 反查成 (V,M,N,K) 坐标
  //   例如 block_rank=5 ⇒ (V,M,N,K)=(1,0,1,0)；V=get<0>=peer 内序号, M=get<1>, N=get<2>
  auto elect_one_cta  = get<0>(cta_in_cluster_coord_vmnk) == Int<0>{};

  // Project the cluster_layout for tma_A along the N-modes
  auto [tAgA, tAsA] = tma_partition(tma_atom_A,
                                    get<2>(cta_in_cluster_coord_vmnk),          // The CTA coordinate along N mode of the cluster
                                    make_layout(size<2>(cluster_layout_vmnk)),  // The CTA layout along N mode of the cluster
                                    group_modes<0,3>(tCsA), group_modes<0,3>(tCgA));
  // 具体值: size<2>(cluster_layout_vmnk)=_4 ⇒ N 模 CTA 布局 make_layout(_4)=(_4):(_1)  (A 沿 N 多播给 4 个 CTA)
  //   group_modes<0,3>(tCgA)=((_128,_16,_1,_4),4)=((_8192),4); group_modes<0,3>(tCsA)=((_8192))
  //   tAgA = (((_64,_128),_1),4):(((_1@0,_1@1),_0),_64@0)   (mode0=单次 TMA 搬的 MK 块, 末维 4=Tiles_K)
  //   tAsA = ((_8192,_1)):((_1,_0))                          (SMEM 落点, 一次搬 128*64=8192 个 half)

  // Project the cluster_layout for tma_B along the M-modes
  auto [tBgB, tBsB] = tma_partition(tma_atom_B,
                                    get<1>(cta_in_cluster_coord_vmnk),          // The CTA coordinate along M mode of the cluster
                                    make_layout(size<1>(cluster_layout_vmnk)),  // The CTA layout along M mode of the cluster
                                    group_modes<0,3>(tCsB), group_modes<0,3>(tCgB));
  // 具体值: size<1>(cluster_layout_vmnk)=_2 ⇒ M 模 CTA 布局 make_layout(_2)=(_2):(_1)  (B 沿 M 多播给 2 个 CTA)
  //   tBgB = (((_64,_128),_1),4):(((_1@0,_1@1),_0),_64@0)   (末维 4=Tiles_K)
  //   tBsB = ((_8192,_1)):((_1,_0))                          (一次搬 128*64=8192 个 half)

  // Project the cluster_layout and cta_coord along the N-mode to determine the multicast mask for A
  uint16_t tma_mcast_mask_a = create_tma_multicast_mask<2>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk);
  // Project the cluster_layout and cta_coord along the M-mode to determine the multicast mask for B
  uint16_t tma_mcast_mask_b = create_tma_multicast_mask<1>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk);
  // Project the cluster_layout and cta_coord along the VM + VN-modes to determine the multicast mask for C
  uint16_t mma_mcast_mask_c = create_tma_multicast_mask<0,1>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk) |
                              create_tma_multicast_mask<0,2>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk);

  // Calculate total bytes that TMA will transfer each tile to track completion, accounting for TMA.2SM
  int tma_transaction_bytes = size<0>(cluster_layout_vmnk) * sizeof(make_tensor_like(tAsA))
                            + size<0>(cluster_layout_vmnk) * sizeof(make_tensor_like(tBsB));

  if (thread0()) {
    print("tAgA:\t"); print(tAgA); print("\n");  // tAgA:   ArithTuple(_0,0) o (((_64,_128),_1),4):(((_1@0,_1@1),_0),_64@0)
    print("tAsA:\t"); print(tAsA); print("\n");  // tAsA:   Sw<3,4,3>_smem_ptr[16b](SMEM_ADDR_A) o ((_8192,_1)):((_1,_0))
    print("tBgB:\t"); print(tBgB); print("\n");  // tBgB:   ArithTuple(_0,0) o (((_64,_128),_1),4):(((_1@0,_1@1),_0),_64@0)   <- 改正,原误作 (((_64,_256),_1),4)
    print("tBsB:\t"); print(tBsB); print("\n");  // tBsB:   Sw<3,4,3>_smem_ptr[16b](SMEM_ADDR_B) o ((_8192,_1)):((_1,_0))   <- 改正,原误作 ((_16384,_1))
    printf("tma_transaction_bytes: %d\n", tma_transaction_bytes);
    printf("tma_mcast_mask_a: %x\n", tma_mcast_mask_a);
    printf("tma_mcast_mask_b: %x\n", tma_mcast_mask_b);
    printf("mma_mcast_mask_c: %x\n", mma_mcast_mask_c);
  } __syncthreads();

  // Barrier Initialization
  // Barriers in SMEM should be initialized by a single thread.
  if (elect_one_warp && elect_one_thr) {
    // The number of CTAs that participates in multicast operation with this CTA (for both A and B matrices)
    int num_mcast_participants = size<1>(cluster_layout_vmnk) + size<2>(cluster_layout_vmnk) - 1;
    cute::initialize_barrier(shared_storage.mma_barrier, /* num_ctas */ num_mcast_participants);
    cute::initialize_barrier(shared_storage.tma_barrier, /* num_threads */ 1);
  }
  int mma_barrier_phase_bit = 0;  // Each barrier has an associated phase_bit.
  int tma_barrier_phase_bit = 0;  // Each barrier has an associated phase_bit.
  cute::cluster_sync();           // Make sure all CTAs in Cluster observe barrier init and TMEM alloc.

  // Step 2: The Mainloop.

  // Set mma accumulate option to zero so that the first MMA instruction will clear the TMEM accumulator.
  tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

  // Execute a MmaTile_M x MmaTile_N x GEMM_K GEMM
  for (int k_tile = 0; k_tile < size<3>(tCgA); ++k_tile)
  {
    // Step 2a: Load A and B tiles

    // TMA Load Operations:
    // - Execute asynchronous TMA loads with single thread
    // - Both peer and leader CTAs initiate TMA loads
    // - Set expected transaction bytes. For 2SM TMA instructions, the transaction bytes counts both CTAs.
    // - Although TMAs are initiated by both peer and leader CTAs, the barrier is only set and waited by the leader CTA.
    // - Initiate asynchronous transfers with a multicast mask that includes all CTAs that participate in multicast.
    if (elect_one_warp && elect_one_thr) { // TMA loads are executed by one thread
      if (elect_one_cta) { // Only the leader CTA waits for TMA transactions
        cute::set_barrier_transaction_bytes(shared_storage.tma_barrier, tma_transaction_bytes); // Set the expected transaction bytes for the TMA loads
      }
      copy(tma_atom_A.with(shared_storage.tma_barrier,tma_mcast_mask_a), tAgA(_,k_tile), tAsA); // Load MmaTile_M x MmaTile_K A tile
      copy(tma_atom_B.with(shared_storage.tma_barrier,tma_mcast_mask_b), tBgB(_,k_tile), tBsB); // Load MmaTile_N x MmaTile_K B tile
    }

    // Step 2b: Execute the MMAs for this tile

    if (elect_one_cta) {
      // Wait for TMA loads to complete on leader CTAs
      cute::wait_barrier(shared_storage.tma_barrier, tma_barrier_phase_bit);
      tma_barrier_phase_bit ^= 1;

      // tcgen05.mma instructions require single-thread execution:
      // - Only one warp performs the MMA-related loop operations
      // - CuTe operations internally manage the single-thread execution of tcgen05.mma and tcgen05.cp
      // - No explicit elect_one_sync region is needed from the user
      if (elect_one_warp) {
        // Execute a MmaTile_M x MmaTile_N x MmaTile_K GEMM
        for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
            gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCtAcc);
            tiled_mma.accumulate_ = UMMA::ScaleOut::One;
        }
        // Ensure MMAs are completed, only then we can reuse the A and B SMEM.
        cutlass::arch::umma_arrive_multicast_2x1SM(&shared_storage.mma_barrier, mma_mcast_mask_c); // All multicasting CTAs encoded in mask.
      }
    }
    // Wait MMAs to complete to avoid overwriting the A and B SMEM.
    cute::wait_barrier(shared_storage.mma_barrier, mma_barrier_phase_bit);
    mma_barrier_phase_bit ^= 1;
  }

  // Step 3: The Epilogue.

  // Apply rank-2 epilogue tiler to rank-2 MMA_V mode
  auto epi_tiler_v = make_tile(epi_tiler_mn);               // (EpiTile)
  // 具体值: epi_tiler_mn=(_128,_64); epi_tiler_v=make_tile((_128,_64))  (把 (M,N) tile 包成作用于 MMA_V 模的单个 tile)
  Tensor tAcc_epi = zipped_divide(tCtAcc, epi_tiler_v);     // (EpiTile,NumTiles)
  // 具体值: tCtAcc 的 (_128,_256) 按 (128,64) 切 ⇒ M:128/128=1, N:256/64=4
  //   tAcc_epi = ((_128,_64),(_1,_4,_1,_1)):((_65536,_1),(_0,_64,_0,_0))   (EpiTile=(128,64), NumTiles=4)
  Tensor gC_epi   = zipped_divide(tCgC,   epi_tiler_v);     // (EpiTile,NumTiles)
  // 具体值: tCgC 的 (_128,_256):(1024,_1) 按 (128,64) 切 ⇒ gC_epi = ((_128,_64),(_1,_4,_1,_1)):((1024,_1),(_0,_64,_0,_0))
  Tensor gD_epi   = zipped_divide(tCgD,   epi_tiler_v);     // (EpiTile,NumTiles)
  // 具体值: gD_epi = ((_128,_64),(_1,_4,_1,_1)):((1024,_1),(_0,_64,_0,_0))  (与 gC_epi 同形, NumTiles=4)

  // Construct corresponding SMEM tensors
  Tensor sC_epi = shared_storage.tensor_sC();               // (EpiTile)
  // 具体值: CSmemLayout=sC_layout=Sw<3,4,3> ((_8,_16),(_32,_2)):((_32,_256),(_1,_4096))  (1 个 EpiTile=128*64=8192 floats)
  Tensor sD_epi = shared_storage.tensor_sD();               // (EpiTile)
  // 具体值: DSmemLayout=sD_layout=Sw<3,4,3> ((_8,_16),(_32,_2)):((_32,_256),(_1,_4096))  (同 sC_epi)

  // Partition for TMA
  auto [tGS_gC, tGS_sC] = tma_partition(tma_atom_C, sC_epi, gC_epi); // (GMEM -> SMEM)
  // 具体值(形状级): tma_atom_C 把 EpiTile 全塞进 mode0 ⇒ tGS_sC=((8192),...) 一次搬整个 EpiTile=8192 floats 到 SMEM;
  //   tGS_gC 末维=NumTiles=4(配合循环里的 tGS_gC(_,epi_tile_idx))。精确 stride 需在 B200 上 print 验证。
  auto [tSG_gD, tSG_sD] = tma_partition(tma_atom_D, sD_epi, gD_epi); // (SMEM -> GMEM)
  // 具体值(形状级): tSG_sD=((8192),...) SMEM 源; tSG_gD 末维=NumTiles=4, 方向是 SMEM->GMEM 存 D。

  // Reset transaction bytes for C load
  tma_transaction_bytes = sizeof(make_tensor_like(tGS_sC));

  // Partition for TMEM accumulators load (TMEM -> RMEM)
  TiledCopy t2r_copy = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tAcc_epi(_,_0{}));
  // 具体值: 用 tcgen05.ld(32 数据通路×32b)在 1 个 EpiTile=(128,64) 上铺 128 线程的 TiledCopy
  //   每线程每 EpiTile 搬 128*64/128 = 64 个 float (TMEM -> RMEM)
  ThrCopy   thr_t2r  = t2r_copy.get_slice(threadIdx.x);
  Tensor tTR_tAcc = thr_t2r.partition_S(tAcc_epi);          // (TmemCpy,NumTmemCpy,NumTiles)
  // 具体值: 本线程视角 ⇒ tTR_tAcc 形状 (TmemCpy, NumTmemCpy, NumTiles=4)，前两模乘积=64(每 tile 每线程 64 个 float)
  Tensor tTR_sC   = thr_t2r.partition_D(sC_epi);            // (TmemCpy,NumTmemCpy)
  // 具体值: tTR_sC 形状 (TmemCpy, NumTmemCpy)，元素数=64(本线程在 1 个 EpiTile 的 SMEM-C 落点)
  Tensor tTR_sD   = thr_t2r.partition_D(sD_epi);            // (TmemCpy,NumTmemCpy)
  // 具体值: tTR_sD 形状 (TmemCpy, NumTmemCpy)，元素数=64(本线程在 1 个 EpiTile 的 SMEM-D 落点)
  // Allocate register tensors
  Tensor tTR_rC = make_tensor_like(tTR_sC);                 // (TmemCpy,NumTmemCpy)
  // 具体值: 与 tTR_sC 同形的寄存器张量，64 个 float/线程(放 C)
  Tensor tTR_rD = make_fragment_like(tTR_sD);               // (TmemCpy,NumTmemCpy)
  // 具体值: 与 tTR_sD 同形的寄存器张量，64 个 float/线程(放累加器/最终 D)

  // Loop over the epilogue tiles
  CUTE_UNROLL
  for (int epi_tile_idx = 0; epi_tile_idx < size<2>(tTR_tAcc); ++epi_tile_idx) {
    // TMA Load C:  GMEM -> SMEM
    if (elect_one_warp && elect_one_thr) {
      cute::set_barrier_transaction_bytes(shared_storage.tma_barrier, tma_transaction_bytes);
      copy(tma_atom_C.with(shared_storage.tma_barrier, 0 /*no multicast*/), tGS_gC(_,epi_tile_idx), tGS_sC);
    }
    // All threads wait for C TMA load to complete
    cute::wait_barrier(shared_storage.tma_barrier, tma_barrier_phase_bit);
    tma_barrier_phase_bit ^= 1;

    // Load C:  SMEM -> RMEM
    copy_aligned(tTR_sC, tTR_rC);

    // Load Acc:  TMEM -> RMEM
    copy(t2r_copy, tTR_tAcc(_,_,epi_tile_idx), tTR_rD);

    // Compute D = beta * C + alpha * (A*B)
    axpby(beta, tTR_rC, alpha, tTR_rD);

    // Store D:  RMEM -> SMEM
    __syncthreads(); // Ensure C loads are finished before reusing smem (unnecessary if smem layouts match)
    copy_aligned(tTR_rD, tTR_sD);

    // TMA Store D:  SMEM -> GMEM
    tma_store_fence(); // Ensure D smem stores are visible to TMA
    __syncthreads(); // Ensure all threads have issued fence
    if (elect_one_warp && elect_one_thr) {
      copy(tma_atom_D, tSG_sD, tSG_gD(_,epi_tile_idx));
      tma_store_arrive(); // issuing thread commits D TMA store
      tma_store_wait<0>(); // issuing thread waits for D TMA store to complete
    }
    __syncthreads(); // All threads sync with issuing thread
  }
  __syncthreads();

  // Release the right to allocate before deallocations so that the next CTA can rasterize
  // Then deallocate TMEM
  if (elect_one_warp) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(shared_storage.tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
  }
}

template <class TypeA, class LayoutA,
          class TypeB, class LayoutB,
          class TypeC, class LayoutC,
          class TypeD, class LayoutD,
          class Alpha, class Beta>
void gemm_host_f16xf16_f32_f32_tnt(TypeA const* device_ptr_A, LayoutA layout_A,
                                   TypeB const* device_ptr_B, LayoutB layout_B,
                                   TypeC const* device_ptr_C, LayoutC layout_C,
                                   TypeD      * device_ptr_D, LayoutD layout_D,
                                   Alpha const alpha, Beta const beta)
{
  assert(shape<0>(layout_A) == shape<0>(layout_C));  // Gemm_M
  assert(shape<0>(layout_A) == shape<0>(layout_D));  // Gemm_M
  assert(shape<0>(layout_B) == shape<1>(layout_C));  // Gemm_N
  assert(shape<0>(layout_B) == shape<1>(layout_D));  // Gemm_N
  assert(shape<1>(layout_A) == shape<1>(layout_B));  // Gemm_K

  // Represent the full tensors in global memory
  Tensor mA = make_tensor(make_gmem_ptr(device_ptr_A), layout_A);      // (Gemm_M, Gemm_K)
  // 具体值: mA = gmem_ptr[16b] o (512,256):(256,_1)
  Tensor mB = make_tensor(make_gmem_ptr(device_ptr_B), layout_B);      // (Gemm_N, Gemm_K)
  // 具体值: mB = gmem_ptr[16b] o (1024,256):(256,_1)
  Tensor mC = make_tensor(make_gmem_ptr(device_ptr_C), layout_C);      // (Gemm_M, Gemm_N)
  // 具体值: mC = gmem_ptr[32b] o (512,1024):(1024,_1)
  Tensor mD = make_tensor(make_gmem_ptr(device_ptr_D), layout_D);      // (Gemm_M, Gemm_N)
  // 具体值: mD = gmem_ptr[32b] o (512,1024):(1024,_1)

  // Get M, N, K dimensions of the GEMM we are running
  auto Gemm_M = shape<0>(layout_A);
  auto Gemm_N = shape<0>(layout_B);
  auto Gemm_K = shape<1>(layout_A);
  std::cout << "Running for problem shape (MxNxK): " << Gemm_M << "x" << Gemm_N << "x" << Gemm_K << std::endl;

  ////////////////////////////////////////////////////////////
  //
  // Initialize the GEMM kernel parameters
  //
  ////////////////////////////////////////////////////////////

  // Create TiledMma. make_tiled_mma takes the target instructions and an (optional) instruction layout as parameters to create a
  // larger TiledMma from the given mma instruction.
  // See cute/arch/mma_sm100_umma.hpp for all tcgen05.mma instructions
  TiledMMA tiled_mma = make_tiled_mma(SM100_MMA_F16BF16_2x1SM_SS<TypeA, TypeB, TypeC,                 // Mma's A, B, and Accumulator types
                                                                 256, 256,                            // Mma M and N dimensions
                                                                 UMMA::Major::K, UMMA::Major::K>{});  // A and B layouts

  // We can also print and inspect the tiled_mma
  print(tiled_mma);
  // TiledMMA
  //   ThrLayoutVMNK:  (_2,_1,_1,_1):(_1,_0,_0,_0)
  //   PermutationMNK: (_,_,_)
  // MMA_Atom
  //   ThrID:      _2:_1
  //   Shape_MNK:  (_256,_256,_16)                      // MmaM, MmaN, MmaK (MmaK is constant for each instr.)
  //   LayoutA_TV: (_2,(_128,_16)):(_128,(_1,_256))     // TV -> MmaCoordinate mapping for A matrix
  //   LayoutB_TV: (_2,(_128,_16)):(_128,(_1,_256))     // TV -> MmaCoordinate mapping for B matrix
  //   LayoutC_TV: (_2,(_128,_256)):(_128,(_1,_256))    // TV -> MmaCoordinate mapping for C matrix

  // Define MMA tiler sizes (static)
  auto bM = tile_size<0>(tiled_mma);             // MMA Tile M. We'll use 1 MMAs per MMA Tile M.
  auto bN = tile_size<1>(tiled_mma);             // MMA Tile N. We'll use 1 MMAs per MMA Tile M.
  auto bK = tile_size<2>(tiled_mma) * Int<4>{};  // MMA Tile K. We'll use 4 MMAs per MMA Tile K. For 16b types, tcgen05.mma has K16.
  auto mma_tiler = make_shape(bM, bN, bK);       // (MMA_M, MMA_N, MMA_K)
  // 具体值: tiled_mma 的 Shape_MNK=(_256,_256,_16)
  //   bM=tile_size<0>=_256, bN=tile_size<1>=_256, bK=tile_size<2>*4=_16*4=_64
  //   ⇒ mma_tiler = (_256,_256,_64)

  // In SM90,  the MMAs are CTA-local and perform thread-level partitioning.
  // In SM100, the MMAs are Cluster-local and perform CTA-level partitioning.
  // Thus, SM90 uses a cta_tiler to extract portions of the Problem for the CTA
  //  and SM100 uses a mma_tiler to extract portions of the Problem for the MMA.
  //  The MMA's partitioning then yields the CTA-local work.

  if (not evenly_divides(shape(mma_tiler), tile_shape(tiled_mma))) {
    std::cerr << "The MMA Shape should evenly divide the MMA Tiler." << std::endl;
    return;
  }

  if (not evenly_divides(make_shape(Gemm_M, Gemm_N, Gemm_K), mma_tiler)) {
    std::cerr << "OOB accesses are not supported. MmaTiler_MNK should evenly divide ProblemShape_MNK." << std::endl;
    return;
  }

  //
  // Determine the SMEM layouts:
  //

  //  * SMEM layouts for A and B must match the post-partitioned (CTA-local) shapes expected by the MMA instructions.
  //  * CuTe provides partition_shape_[A|B] functions to determine the post-partitioned shape.
  //    These functions take the TiledMma, and the MMA Tile Shape as inputs and returns a shape that is at least rank-3
  //    where the first mode has the same shape as the MMA instruction, 2nd and 3rd mode expresses the number of time
  //    MMA instr is repeated in M/N mode and K mode of MMA tile, respectively.
  //  * Note that SMEM layouts are needed to determine SMEM allocation for kernel launch.

  // Pre-partitioned Tile Shape (MmaTile_M, MmaTile_K) to post-partitioned (MmaA, NumMma_M, NumMma_K)
  auto mma_shape_A = partition_shape_A(tiled_mma, make_shape(size<0>(mma_tiler), size<2>(mma_tiler)));
  // 具体值: 输入 (MmaTile_M,MmaTile_K)=(_256,_64); 2SM 把 M=256 分到 2 个 CTA(每 128), K=64 拆成 4 个 K16
  //   ⇒ mma_shape_A = ((_128,_16),_1,_4)  (MmaA=(128,16), NumMma_M=1, NumMma_K=4)
  // Pre-partitioned Tile Shape (MmaTile_N, MmaTile_K) to post-partitioned (MmaB, NumMma_N, NumMma_K)
  auto mma_shape_B = partition_shape_B(tiled_mma, make_shape(size<1>(mma_tiler), size<2>(mma_tiler)));
  // 具体值: 输入 (MmaTile_N,MmaTile_K)=(_256,_64) ⇒ mma_shape_B = ((_128,_16),_1,_4)  (MmaB=(128,16),NumMma_N=1,NumMma_K=4)

  // Print and inspect mma_shape_A, and mma_shape_B for this example.
  print("mma_shape_A:\t"); print(mma_shape_A); print("\n");  // mma_shape_A:  ((_128,_16),_1,_4)
  print("mma_shape_B:\t"); print(mma_shape_B); print("\n");  // mma_shape_B:  ((_128,_16),_1,_4)   <- 改正,原误作 ((_256,_16),_1,_4)

  // A and B tensors are swizzled in SMEM to improve MMA performance.
  //  * However, expressing swizzled layouts is very hard.
  //  * CuTe provides tile_to_mma_shape functions for SM100 to create swizzled layouts for post-partitioned Mma Shapes
  auto sA_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TypeA>{}, mma_shape_A);
  // 具体值: 把 SW128(128B swizzle, half) atom 平铺到 mma_shape_A=((_128,_16),_1,_4)
  //   ⇒ sA_layout = Sw<3,4,3> ((_128,_16),_1,_4):((_64,_1),_0,_16)  (cosize=128*16*4=8192 个 half)
  auto sB_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TypeB>{}, mma_shape_B);
  // 具体值: ⇒ sB_layout = Sw<3,4,3> ((_128,_16),_1,_4):((_64,_1),_0,_16)  (cosize=8192 个 half)

  // Print and inspect sA_layout and sB_layout for this example.
  print("sA_layout:\t"); print(sA_layout); print("\n");      // sA_layout:   Sw<3,4,3> o smem_ptr[16b](unset) o ((_128,_16),_1,_4):((_64,_1),_0,_16)
  print("sB_layout:\t"); print(sB_layout); print("\n");      // sB_layout:   Sw<3,4,3> o smem_ptr[16b](unset) o ((_128,_16),_1,_4):((_64,_1),_0,_16)   <- 改正,原误作 ((_256,_16),_1,_4)

  //
  // Epilogue parameters
  //

  // Pre-partitioned Tile Shape (MmaTile_M, MmaTile_N) to post-partitioned ((MmaM,MmaN), NumMma_M, NumMma_N)
  auto mma_shape_C = partition_shape_C(tiled_mma, make_shape(size<0>(mma_tiler), size<1>(mma_tiler)));
  // 具体值: 输入 (MmaTile_M,MmaTile_N)=(_256,_256); 2SM 把 M=256 分到 2 个 CTA(每 128)
  //   ⇒ mma_shape_C = ((_128,_256),_1,_1)  (MmaC=(128,256), NumMma_M=1, NumMma_N=1)

  // For TMA epilogue performance it may be beneficial to iterate over the output in smaller tiles than the MMA tile
  auto epi_tiler = make_tile(size<0,0>(mma_shape_C), size<0,1>(mma_shape_C) / Int<4>{});  // 4 TMA copies per CTA per MMA tile
  // 具体值: size<0,0>(mma_shape_C)=_128(M), size<0,1>(mma_shape_C)/4=_256/4=_64(N) ⇒ epi_tiler = (_128,_64)

  // SMEM layouts for C and D should match the epilogue tile
  auto sC_layout_mn = tile_to_shape(UMMA::Layout_K_SW128_Atom<TypeC>{}, // MMA K-major is equivalent to epilogue N-major
                                    make_shape(size<0>(epi_tiler), size<1>(epi_tiler)));
  // 具体值: 把 SW128(128B swizzle, float) atom 铺到 (M,N)=(_128,_64) ⇒ sC_layout_mn = Sw<3,4,3> (_128,_64):(...) (cosize=8192 floats)
  auto sC_layout = group<0,2>(sC_layout_mn); // Group modes for tma_partition
  // 具体值: group<0,2> 把前两模并起来供 tma_partition ⇒ sC_layout = Sw<3,4,3> ((_8,_16),(_32,_2)):((_32,_256),(_1,_4096))

  auto sD_layout_mn = tile_to_shape(UMMA::Layout_K_SW128_Atom<TypeD>{}, // MMA K-major is equivalent to epilogue N-major
                                    make_shape(size<0>(epi_tiler), size<1>(epi_tiler)));
  // 具体值: ⇒ sD_layout_mn = Sw<3,4,3> (_128,_64):(...) (cosize=8192 floats, 与 sC_layout_mn 同)
  auto sD_layout = group<0,2>(sD_layout_mn); // Group modes for tma_partition
  // 具体值: ⇒ sD_layout = Sw<3,4,3> ((_8,_16),(_32,_2)):((_32,_256),(_1,_4096))

  print("sC_layout:\t"); print(sC_layout); print("\n");      // sC_layout:   Sw<3,4,3> o smem_ptr[32b](unset) o ((_8,_16),(_32,_2)):((_32,_256),(_1,_4096))
  print("sD_layout:\t"); print(sD_layout); print("\n");      // sD_layout:   Sw<3,4,3> o smem_ptr[32b](unset) o ((_8,_16),(_32,_2)):((_32,_256),(_1,_4096))

  // Now we can find the SMEM allocation size
  using SMEMStorage = SharedStorage<TypeA, TypeB, TypeC, TypeD,
                                    decltype(sA_layout), decltype(sB_layout),
                                    decltype(sC_layout), decltype(sD_layout)>;

  //
  // TMA Descriptor Creation (Host Side)
  //

  // The cluster shape and layout
  auto cluster_shape = make_shape(Int<4>{}, Int<4>{}, Int<1>{});
  // 具体值: cluster_shape = (_4,_4,_1)  (ClusterM=4, ClusterN=4, ClusterK=1)
  Layout cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape),
                                            make_tile(typename decltype(tiled_mma)::AtomThrID{}));
  // 具体值: make_layout(cluster_shape)=(_4,_4,_1):(_1,_4,_16); AtomThrID=_2, make_tile(_2)=(_2)
  //   ⇒ cluster_layout_vmnk = (_2,_2,_4,_1):(_1,_2,_4,_16)  (V,M,N,K)=(2,2,4,1)  (与 kernel 内同一份)

  // SM100 interface for creating TMA loads.
  Copy_Atom tma_atom_A = make_tma_atom_A_sm100(
      SM100_TMA_2SM_LOAD_MULTICAST{}, // TMA load operation -- Multicasting 2SM instruction.
      mA,                             // Source GMEM tensor
      sA_layout,                      // Destination SMEM layout
      mma_tiler,                      // MmaTiler_MNK. Unlike Sm90 interface where the tiler only included M and K modes.
      tiled_mma,                      // Sm100 also requires the TiledMma to perform CTA-level partitioning.
      cluster_layout_vmnk);           // ClusterLayout_VMNK. Unlike Sm90 interface where only the multicasting mode is passed.
                                      //   We have make_tma_atom_[A|B]_sm100 and which determines the multicast mode.
  Tensor mA_tma = tma_atom_A.get_tma_tensor(shape(mA));   // (Gemm_M, Gemm_K)
  // 具体值: TMA 张量用 ArithTuple 坐标(非真实指针) ⇒ mA_tma = ArithTuple(_0,_0) o (512,256):(_1@1,_1@0)

  print("tma_atom_A:\t"); print(tma_atom_A); print("\n");
  // tma_atom_A:     Copy_Atom
  //  ThrID:        _2:_1
  //  ValLayoutSrc: (_2,_8192):(_8192,_1)
  //  ValLayoutDst: (_2,_8192):(_8192,_1)
  //  ValLayoutRef: (_2,_8192):(_8192,_1)
  //  ValueType:    16b

  // SM100 interface for creating TMA loads.
  Copy_Atom tma_atom_B = make_tma_atom_B_sm100(
    SM100_TMA_2SM_LOAD_MULTICAST{}, // TMA load operation -- Multicasting 2SM instruction.
    mB,                             // Source GMEM tensor
    sB_layout,                      // Destination SMEM layout
    mma_tiler,                      // MmaTiler_MNK. Unlike Sm90 interface where the tiler only included M and K modes.
    tiled_mma,                      // Sm100 also requires the TiledMma to perform CTA-level partitioning.
    cluster_layout_vmnk);           // ClusterLayout_VMNK. Unlike Sm90 interface where only the multicasting mode is passed.
                                    //   We have make_tma_atom_[A|B]_sm100 and which determines the multicast mode.
  Tensor mB_tma = tma_atom_B.get_tma_tensor(shape(mB));   // (Gemm_N, Gemm_K)
  // 具体值: mB_tma = ArithTuple(_0,_0) o (1024,256):(_1@1,_1@0)

  print("tma_atom_B:\t"); print(tma_atom_B); print("\n");
  // tma_atom_B:     Copy_Atom
  // ThrID:        _2:_1
  // ValLayoutSrc: (_2,_8192):(_8192,_1)
  // ValLayoutDst: (_2,_8192):(_8192,_1)
  // ValLayoutRef: (_2,_8192):(_8192,_1)
  // ValueType:    16b

  Copy_Atom tma_atom_C = make_tma_atom(
        SM90_TMA_LOAD{},            // TMA load operation
        mC,                         // Source GMEM tensor
        sC_layout,                  // Destination SMEM layout
        epi_tiler);                 // MN Tiler for epilogue
  Tensor mC_tma = tma_atom_C.get_tma_tensor(shape(mC));   // (Gemm_M, Gemm_N)
  // 具体值: mC_tma = ArithTuple(_0,_0) o (512,1024):(_1@1,_1@0)  (注:device 端 print 的 mC 显示真实 gmem_ptr+stride(1024,_1))

  print("tma_atom_C:\t"); print(tma_atom_C); print("\n");
  // tma_atom_C:     Copy_Atom
  //   ThrID:        _1:_0
  //   ValLayoutSrc: (_1,_4096):(_0,_1)
  //   ValLayoutDst: (_1,_4096):(_0,_1)
  //   ValLayoutRef: (_1,_4096):(_0,_1)
  //   ValueType:    32b

  Copy_Atom tma_atom_D = make_tma_atom(
        SM90_TMA_STORE{},           // TMA store operation
        mD,                         // Destination GMEM tensor
        sD_layout,                  // Source SMEM layout
        epi_tiler);                 // MN Tiler for epilogue
  Tensor mD_tma = tma_atom_D.get_tma_tensor(shape(mD));   // (Gemm_M, Gemm_N)
  // 具体值: mD_tma = ArithTuple(_0,_0) o (512,1024):(_1@1,_1@0)

  print("tma_atom_D:\t"); print(tma_atom_D); print("\n");
  // tma_atom_D:     Copy_Atom
  //   ThrID:        _1:_0
  //   ValLayoutSrc: (_1,_4096):(_0,_1)
  //   ValLayoutDst: (_1,_4096):(_0,_1)
  //   ValLayoutRef: (_1,_4096):(_0,_1)
  //   ValueType:    32b

  ////////////////////////////////////////////////////////////
  //
  // Launch GEMM kernel
  //
  ////////////////////////////////////////////////////////////

  dim3 dimBlock(128);
  dim3 dimCluster(size<0>(cluster_shape), size<1>(cluster_shape), size<2>(cluster_shape));
  dim3 dimGrid(size(ceil_div(Gemm_M, bM * size<1>(cluster_layout_vmnk))) * dimCluster.x,
               size(ceil_div(Gemm_N, bN * size<2>(cluster_layout_vmnk))) * dimCluster.y);
  int  smemBytes = sizeof(SMEMStorage);

  auto* kernel_ptr = &gemm_device<SMEMStorage,
                                  decltype(mA_tma), decltype(mB_tma), decltype(mC_tma), decltype(mD_tma),
                                  decltype(mma_tiler), decltype(epi_tiler), decltype(tiled_mma), decltype(cluster_shape),
                                  decltype(tma_atom_A), decltype(tma_atom_B), decltype(tma_atom_C), decltype(tma_atom_D), // Includes the TMA descriptor.
                                  Alpha, Beta>;

  // Set kernel attributes (set SMEM)
  CUTE_CHECK_ERROR(cudaFuncSetAttribute(kernel_ptr,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smemBytes));

  printf("Grid launched: %d, %d, %d\n", dimGrid.x, dimGrid.y, dimGrid.z);
  printf("Cluster launched: %d, %d, %d\n", dimCluster.x, dimCluster.y, dimCluster.z);

  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smemBytes};
  cutlass::Status status = cutlass::launch_kernel_on_cluster(params, (void const*) kernel_ptr,
                                                             mA_tma, mB_tma, mC_tma, mD_tma,
                                                             mma_tiler, epi_tiler, tiled_mma, cluster_shape,
                                                             tma_atom_A, tma_atom_B, tma_atom_C, tma_atom_D,
                                                             alpha, beta);
  CUTE_CHECK_LAST();

  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Error: Failed at kernel Launch" << std::endl;
  }
}

#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

int main(int argc, char** argv)
{
  cudaDeviceProp props;
  int current_device_id;
  cudaGetDevice(&current_device_id);
  cudaGetDeviceProperties(&props, current_device_id);
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if ((props.major != 10) || (props.major == 10 && props.minor > 1)) {
    std::cerr << "This example requires NVIDIA's Blackwell Architecture GPU with compute capability 100a." << std::endl;
    std::cerr << "  Found " << props.major << "." << props.minor << std::endl;
    return -1;
  }

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

  int Gemm_M = 512;
  if (argc >= 2)
    sscanf(argv[1], "%d", &Gemm_M);

  int Gemm_N = 1024;
  if (argc >= 3)
    sscanf(argv[2], "%d", &Gemm_N);

  int Gemm_K = 256;
  if (argc >= 4)
    sscanf(argv[3], "%d", &Gemm_K);

  ////////////////////////////////////////////////////////////
  //
  // Create A, B, C, and D tensors
  //
  ////////////////////////////////////////////////////////////
  // Define the data types. A and B types are same for MMA instruction.
  using TypeA = cutlass::half_t; // MMA A Data Type
  auto type_str_a = "half_t";
  using TypeB = cutlass::half_t; // MMA B Data Type
  auto type_str_b = "half_t";
  using TypeC = float;           // MMA C Data Type
  [[maybe_unused]] auto type_str_c = "float";
  using TypeD = float;           // MMA D Data Type
  auto type_str_d = "float";
  using TypeAccumulator = float; // Both TypeC and TypeD are float, use float accumulator type.

  // A tensor MxK K-major (Layout T = Row-Major)
  Layout layout_A = make_layout(make_shape (Gemm_M,   Gemm_K),
                                make_stride(Gemm_K, Int<1>{}));   // (Gemm_M,Gemm_K):(Gemm_K,_1)
  // 具体值(默认 M=512,N=1024,K=256): layout_A = (512,256):(256,_1)  (行主序, 每行 K=256 个 half)
  // B tensor NxK K-major (Layout N = Column-Major)
  Layout layout_B = make_layout(make_shape (Gemm_N,   Gemm_K),
                                make_stride(Gemm_K, Int<1>{}));   // (Gemm_N,Gemm_K):(Gemm_K,_1)
  // 具体值: layout_B = (1024,256):(256,_1)  (每行 K=256 个 half)
  // C tensor MxN N-major (Layout T = Row-Major)
  Layout layout_C = make_layout(make_shape (Gemm_M,   Gemm_N),
                                make_stride(Gemm_N, Int<1>{}));   // (Gemm_M,Gemm_N):(Gemm_N,_1)
  // 具体值: layout_C = (512,1024):(1024,_1)  (行主序, 每行 N=1024 个 float)
  // D tensor MxN N-major (Layout T = Row-Major)
  Layout layout_D = make_layout(make_shape (Gemm_M,   Gemm_N),
                                make_stride(Gemm_N, Int<1>{}));   // (Gemm_M,Gemm_N):(Gemm_N,_1)
  // 具体值: layout_D = (512,1024):(1024,_1)  (与 layout_C 同)

  // Host allocations and host CuTe tensors for A, B, and C tensors.
  thrust::host_vector<TypeA>   host_A(Gemm_M * Gemm_K);
  Tensor host_tensor_A = make_tensor(host_A.data(), layout_A);
  print("host_tensor_A:\t"); print(host_tensor_A); print("\n"); // host_tensor_A:	ptr[16b](ADDR_A) o (512,256):(256,_1)

  thrust::host_vector<TypeB>   host_B(Gemm_N * Gemm_K);
  Tensor host_tensor_B = make_tensor(host_B.data(), layout_B);
  print("host_tensor_B:\t"); print(host_tensor_B); print("\n"); // host_tensor_B:	ptr[16b](ADDR_B) o (1024,256):(256,_1)

  thrust::host_vector<TypeC>   host_C(Gemm_M * Gemm_N);
  Tensor host_tensor_C = make_tensor(host_C.data(), layout_C);
  print("host_tensor_C:\t"); print(host_tensor_C); print("\n"); // host_tensor_C:	ptr[32b](ADDR_C) o (512,1024):(1024,_1)

  // Note that we don't need a host_tensor for D yet.
  thrust::device_vector<TypeD> device_D(Gemm_M * Gemm_N);

  // Initialize A, B, and C tensors with random values.
  initialize_tensor(host_tensor_A);
  initialize_tensor(host_tensor_B);
  initialize_tensor(host_tensor_C);

  // Copy A, B, and C tensors from host memory to device memory
  thrust::device_vector<TypeA> device_A = host_A;
  thrust::device_vector<TypeB> device_B = host_B;
  thrust::device_vector<TypeC> device_C = host_C;

  using Alpha = float;
  using Beta = float;
  Alpha alpha = 1.0f;
  Beta beta = 0.0f;
  // Setup input and output tensors, and the kernel parameters; and execute the kernel on device
  gemm_host_f16xf16_f32_f32_tnt(device_A.data().get(), layout_A,
                                device_B.data().get(), layout_B,
                                device_C.data().get(), layout_C,
                                device_D.data().get(), layout_D,
                                alpha, beta);
  // Host allocation for D tensor and transfer D tensor from device to host
  thrust::host_vector<TypeD> host_D = device_D;
  // Create a non-owning CuTe tensor for D tensor
  Tensor host_tensor_D = make_tensor(host_D.data(), layout_D);
  // 具体值: host_tensor_D = ptr[32b](ADDR_D) o (512,1024):(1024,_1)

  ////////////////////////////////////////////////////////////
  //
  // Execute reference GEMM kernel
  //
  ////////////////////////////////////////////////////////////

  thrust::host_vector<TypeD> host_reference_D(Gemm_M*Gemm_N);
  auto host_reference_tensor_D = make_tensor(host_reference_D.data(), layout_D);
  reference_gemm<TypeAccumulator>(host_tensor_A, host_tensor_B, host_tensor_C, host_reference_tensor_D, alpha, beta);

  ////////////////////////////////////////////////////////////
  //
  // Compare results
  //
  ////////////////////////////////////////////////////////////

  auto relative_error = print_matrix_multiply_mollified_relative_error(type_str_a, host_tensor_A,
                                                                       type_str_b, host_tensor_B,
                                                                       type_str_d, host_tensor_D, host_reference_tensor_D);
  bool success = relative_error <= 0.0;
  std::cout << "Execution is " << ((success) ? "successful." : "failed.") << std::endl;
#else
  std::cout << "CUTLASS_ARCH_MMA_SM100_SUPPORTED must be enabled, but it is not. Test is waived \n" << std::endl;
#endif

  return 0;
}
