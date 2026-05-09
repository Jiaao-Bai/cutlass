# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> ## ⚡ READ FIRST — 用户的当前主线任务
>
> **这是一个 fork**。owner 不是在为 CUTLASS 开发新特性，而是在**用这个 fork 学习** CuTe + Hopper(SM90)/Blackwell(SM100) 的内核编程，目标是能手写并极致优化 GEMM / FlashAttention / Sparse MoE。
>
> **所有学习计划、笔记、练习都在 `study/` 目录下**（与 `include/`、`examples/` 物理隔离，避免 rebase 冲突）。
>
> 进入这个仓库后，agent 的默认行为应该是 **指导/陪伴学习**，不是开发上游特性。除非用户明确要改 `include/` `examples/` 等上游目录，否则改动都应该落在 `study/` 下。
>
> ### 必看入口（按顺序）
> 1. **[`study/README.md`](study/README.md)** — 主计划，含 6 个 stage + 21 周 + 统一周模板 + 目录结构
> 2. **[`study/PROGRESS.md`](study/PROGRESS.md)** — 用户的进度跟踪，看这里就知道当前进行到第几周
> 3. **[`study/cutlass_reading_strategy.md`](study/cutlass_reading_strategy.md)** — `include/cutlass/` 67 万行的取舍清单（必读 5000 行 / 跳过 38 万行 2.x 遗产）
>
> ### 目录速查
> ```
> study/
> ├── README.md, PROGRESS.md, cutlass_reading_strategy.md
> ├── CMakeLists.txt                # -DCUTLASS_ENABLE_STUDY=ON 才编译
> ├── stage1_cute_algebra/          W1-4   CuTe 张量代数
> ├── stage2_sm90_primitives/       W5-7   WGMMA / TMA / Pipeline
> ├── stage3_hopper_gemm/           W8-11  手写 Hopper WarpSpec GEMM
> ├── stage4_flashattn/             W12-15 FlashAttention fwd/bwd
> ├── stage5_moe/                   W16-18 Sparse MoE
> ├── stage6_b200_increment/        W19-21 SM100 (TMEM/UMMA) 增量
> └── stage7_tuning/                持续：profiling + baselines
> ```
> 每个 `stageN/` 有 README + CHECKPOINT；每个 `weekNN/` 有 README（统一模板：目标 / 读 / 写 / 跑 / 自检）+ `exercises/`。
>
> ### Agent 行为指南
> - **默认假设是学习模式**。开 session 先看 `study/PROGRESS.md` 确认当前周次，再看对应的 `weekNN/README.md`。
> - **写练习代码**放在 `study/stageX_xxx/weekNN_xxx/exercises/exNN_xxx.cu`，命名沿用现有规范（见 `ex06_hgemm_naive.cu`）。
> - **不要改上游** `include/` `examples/` `tools/`，除非用户明确要 fix bug 或 contribute。
> - 用户提交了 ncu 数据 → 帮他记到 `study/stage7_tuning/h20_baselines.md` 或 `b200_baselines.md`。
> - 用户问"`include/cutlass/` 里 X 文件怎么读" → 先查 `study/cutlass_reading_strategy.md` 的映射，给出"在第 N 周读"的答复。
> - 用户答完一周的"自检题" → 帮他在 `study/PROGRESS.md` 打勾。
> - 周次推进：完成 weekNN 的练习 + 自检，且 stage CHECKPOINT 通过，才进下一周。
> - 用户的硬件是 H20（SM90）和 B200（SM100）；编译命令 `cmake .. -DCUTLASS_ENABLE_STUDY=ON -DCUTLASS_NVCC_ARCHS=90a`（或 `100a`）。

---

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
