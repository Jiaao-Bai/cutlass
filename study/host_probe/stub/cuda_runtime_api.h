#pragma once
#include <cstddef>
#include <vector_types.h>
typedef int cudaError_t;
// CUDA execution-space keywords -> no-ops on host (g++)
#ifndef __CUDACC__
  #define __device__
  #define __host__
  #define __global__
  #define __forceinline__ inline
  #define __shared__
  #define __constant__
  #define __grid_constant__
  #define __launch_bounds__(...)
#endif
