# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CUTLASS is NVIDIA's CUDA C++ template library for high-performance matrix-matrix multiplication (GEMM) and related computations. It provides abstractions for hierarchical decomposition and data movement within CUDA, supporting Volta, Turing, Ampere, Ada, Hopper, and Blackwell architectures.

The project consists of:
- **CUTLASS C++**: Header-only template library in `include/cutlass/`
- **CuTe**: Core library for tensor/layout abstractions in `include/cute/`
- **CuTe DSL**: Python DSL for writing CUDA kernels in `python/CuTeDSL/`

## Build Commands

CUTLASS is header-only and does not require building to use. To build tests, examples, and profiler:

```bash
# Configure (from project root)
mkdir build && cd build
cmake .. -DCUTLASS_NVCC_ARCHS=90a  # For Hopper; use 80 for Ampere, 100a for Blackwell

# Build unit tests
make test_unit -j

# Build profiler
make cutlass_profiler -j16

# Build specific example
make 00_basic_gemm -j
```

### Key CMake Options

- `CUTLASS_NVCC_ARCHS`: Target architecture(s) - e.g., "80", "90a", "100a", "75;80"
- `CUTLASS_LIBRARY_KERNELS`: Kernel subset to build - e.g., "all" or specific patterns
- `CUTLASS_ENABLE_GTEST_UNIT_TESTS`: Enable unit tests (default ON)

### Running Tests

```bash
# Run all unit tests
./build/test/unit/test_unit

# Run Python tests (requires CuTe DSL setup)
pytest test/python/cutlass/gemm/
pytest test/examples/CuTeDSL/
```

## CuTe DSL (Python)

Setup the Python DSL:

```bash
# CUDA 12 (default)
python/CuTeDSL/setup.sh --cu12

# CUDA 13
python/CuTeDSL/setup.sh --cu13
```

This installs `nvidia-cutlass-dsl` package. Examples are in `examples/python/CuTeDSL/` organized by architecture:
- `ampere/` - SM80 examples
- `hopper/` - SM90 examples
- `blackwell/` - SM100 examples
- `experimental/` - Higher-level composable APIs

## Code Architecture

### C++ Template Library (`include/`)

```
include/
  cutlass/           # Main CUTLASS templates
    arch/            # Architecture-specific instructions (MMA, etc.)
    gemm/            # GEMM kernel templates
      collective/    # Collective operations (mainloop)
      device/        # Device-level GEMM entry points
      kernel/        # Kernel-level implementations
    conv/            # Convolution kernels
    epilogue/        # Epilogue operations
    layout/          # Memory layout definitions
  cute/              # CuTe core library
    algorithm/       # copy, gemm, fill, etc.
    arch/            # PTX wrappers for copy/MMA
    atom/            # Copy_Atom, Mma_Atom, TiledCopy, TiledMma
    container/       # tuple, array, etc.
    numeric/         # IntTuple, Layout, Tensor types
```

### Key Abstractions

1. **CuTe Layout/Tensor**: `cute::Layout` describes mapping from logical coordinates to memory. `cute::Tensor` combines a pointer with a layout.

2. **MMA Atoms**: Architecture-specific tensor core operations defined in `cute/atom/mma_traits_sm*.hpp`

3. **Collective Builders**: `cutlass::gemm::collective::CollectiveBuilder` constructs kernel components from high-level parameters.

4. **Device Adapters**: `cutlass::gemm::device::GemmUniversalAdapter` provides the user-facing API.

### Python DSL (`python/CuTeDSL/cutlass/`)

```
cutlass/
  cute/              # CuTe Python bindings
    arch/            # Architecture operations
    nvgpu/            # GPU-specific helpers (warpgroup, tcgen05, etc.)
    experimental/    # Higher-level APIs (pipeline, memory, algorithm)
  base_dsl/          # DSL infrastructure (compiler, JIT, runtime)
```

## Architecture Targeting

| Architecture | Compute Capability | CMake Flag | Notes |
|-------------|-------------------|------------|-------|
| Volta | 7.0 | 70 | Minimum supported |
| Turing | 7.5 | 75 | |
| Ampere | 8.0, 8.6, 8.9 | 80, 86, 89 | |
| Hopper | 9.0 | 90a | Use "a" suffix for arch-accelerated features |
| Blackwell | 10.0, 10.3, 12.0 | 100a, 103a, 120a | SM100 for datacenter, SM120 for GeForce |

## Profiler Usage

```bash
# Profile GEMM
./tools/profiler/cutlass_profiler --kernels=cutlass_tensorop_s*gemm_f16_* --m=3456 --n=4096 --k=4096

# Profile convolution
./tools/profiler/cutlass_profiler --kernels=cutlass_tensorop_s*fprop_optimized_f16 --n=8 --h=224 --w=224 --c=128 --k=128 --r=3 --s=3
```

## Documentation

- Quickstart: `media/docs/cpp/quickstart.md`
- CuTe Tutorial: `media/docs/cpp/cute/00_quickstart.md`
- GEMM API 3.x: `media/docs/cpp/gemm_api_3x.md`
- Code Organization: `media/docs/cpp/code_organization.md`
