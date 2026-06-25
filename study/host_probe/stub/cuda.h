#pragma once
// Minimal host stub of CUDA driver TMA-descriptor types for layout probes.
#include <cstdint>
struct CUtensorMap { alignas(64) uint64_t opaque[16]; };
typedef int CUresult;
typedef unsigned long long CUdeviceptr;
typedef enum CUtensorMapDataType_enum {
  CU_TENSOR_MAP_DATA_TYPE_UINT8=0, CU_TENSOR_MAP_DATA_TYPE_UINT16, CU_TENSOR_MAP_DATA_TYPE_UINT32,
  CU_TENSOR_MAP_DATA_TYPE_INT32, CU_TENSOR_MAP_DATA_TYPE_UINT64, CU_TENSOR_MAP_DATA_TYPE_INT64,
  CU_TENSOR_MAP_DATA_TYPE_FLOAT16, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, CU_TENSOR_MAP_DATA_TYPE_FLOAT64,
  CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_DATA_TYPE_TFLOAT32,
  CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,
  CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B
} CUtensorMapDataType;
typedef enum CUtensorMapSwizzle_enum {
  CU_TENSOR_MAP_SWIZZLE_NONE=0, CU_TENSOR_MAP_SWIZZLE_32B, CU_TENSOR_MAP_SWIZZLE_64B,
  CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B, CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B
} CUtensorMapSwizzle;
typedef enum CUtensorMapL2promotion_enum {
  CU_TENSOR_MAP_L2_PROMOTION_NONE=0, CU_TENSOR_MAP_L2_PROMOTION_L2_64B,
  CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B
} CUtensorMapL2promotion;
typedef enum CUtensorMapFloatOOBfill_enum {
  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE=0, CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
} CUtensorMapFloatOOBfill;
typedef enum CUtensorMapInterleave_enum {
  CU_TENSOR_MAP_INTERLEAVE_NONE=0, CU_TENSOR_MAP_INTERLEAVE_16B, CU_TENSOR_MAP_INTERLEAVE_32B
} CUtensorMapInterleave;
inline CUresult cuTensorMapEncodeTiled(CUtensorMap*, CUtensorMapDataType, unsigned int,
  void*, const unsigned long long*, const unsigned long long*, const unsigned int*,
  const unsigned int*, CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion,
  CUtensorMapFloatOOBfill){ return 0; }
inline CUresult cuGetErrorString(CUresult, const char**){ return 0; }
typedef CUresult (*PFN_cuTensorMapEncodeTiled)(...);
inline CUresult cuDriverGetVersion(int* v){ if(v)*v=12000; return 0; }
