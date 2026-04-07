# CUTLASS 学习路径指南

本指南帮助你系统地学习 NVIDIA CUTLASS 仓库。根据你的背景和目标，可以选择 **C++ 路径**或 **Python DSL 路径**，也可以两者结合。

---

## 前置知识

开始之前，建议具备以下基础：

| 领域 | 要求 | 推荐资源 |
|------|------|----------|
| **C/C++** | 熟悉模板、constexpr、类型推导 | C++ Templates: The Complete Guide |
| **CUDA** | 了解线程层级、共享内存、同步机制 | CUDA C++ Programming Guide |
| **线性代数** | 理解矩阵乘法（GEMM）的数学原理 | - |

> 如果你只想使用 Python DSL，C++ 模板知识可以降低要求，但仍需理解 CUDA 编程模型。

---

## 学习路线概览

```
阶段一：环境搭建与初识
    │
    ▼
阶段二：核心概念（类型、布局、工具）
    │
    ▼
阶段三：CuTe 深入（Layout → Tensor → Algorithm）
    │
    ▼
阶段四：GEMM 实现原理
    │
    ├──────────────────┐
    ▼                  ▼
阶段五：架构专精      Python DSL 路径（可独立学习）
    │
    ▼
阶段六：高级主题（卷积、融合、稀疏、分布式）
```

---

## 阶段一：环境搭建与初识

**目标**：成功构建项目并运行第一个 GEMM 示例。

### 阅读材料

1. [`media/docs/cpp/quickstart.md`](media/docs/cpp/quickstart.md) — 构建环境配置、CMake 选项
2. [`media/docs/cpp/terminology.md`](media/docs/cpp/terminology.md) — CUTLASS 核心术语（Tile、Warp、ThreadBlock 等）
3. [`media/docs/cpp/code_organization.md`](media/docs/cpp/code_organization.md) — 代码目录结构与模块划分

### 动手实践

```bash
# 构建并运行第一个示例
mkdir build && cd build
cmake .. -DCUTLASS_NVCC_ARCHS=80  # 根据你的 GPU 调整
make 00_basic_gemm -j
./examples/00_basic_gemm/00_basic_gemm
```

- **示例**: [`examples/00_basic_gemm`](examples/00_basic_gemm) — 最简单的 GEMM 调用，理解 CUTLASS 的基本使用模式

### 检查点

- [ ] 能成功编译和运行 `00_basic_gemm`
- [ ] 能解释 CUTLASS 中 Tile、Warp Tile、Thread Tile 的含义

---

## 阶段二：核心概念

**目标**：理解 CUTLASS 的基础数据类型和内存布局抽象。

### 阅读材料

1. [`media/docs/cpp/fundamental_types.md`](media/docs/cpp/fundamental_types.md) — 数值类型（half_t、bfloat16_t、tfloat32_t）、容器类型
2. [`media/docs/cpp/layout.md`](media/docs/cpp/layout.md) — 内存布局（RowMajor、ColumnMajor、TensorNHWC 等）
3. [`media/docs/cpp/tile_iterator_concept.md`](media/docs/cpp/tile_iterator_concept.md) — Tile 迭代器的概念与使用
4. [`media/docs/cpp/utilities.md`](media/docs/cpp/utilities.md) — 工具函数（内存分配、调试打印）

### 动手实践

- [`examples/01_cutlass_utilities`](examples/01_cutlass_utilities) — CUTLASS 工具函数的使用
- [`examples/03_visualize_layout`](examples/03_visualize_layout) — **重要**：可视化不同内存布局，帮助直观理解数据在内存中的排列
- [`examples/04_tile_iterator`](examples/04_tile_iterator) — Tile 迭代器的实际使用

### 检查点

- [ ] 理解 RowMajor 和 ColumnMajor 布局的区别及其内存映射
- [ ] 能用 `03_visualize_layout` 工具生成布局可视化

---

## 阶段三：CuTe 深入

**目标**：掌握 CuTe（CUTLASS 3.x 的核心抽象层），理解 Layout、Tensor、Algorithm 三大概念。

> CuTe 是 CUTLASS 3.x 的基础。理解 CuTe 对后续学习至关重要。

### 阅读材料（按顺序）

1. [`media/docs/cpp/cute/00_quickstart.md`](media/docs/cpp/cute/00_quickstart.md) — CuTe 概述、调试技巧
2. [`media/docs/cpp/cute/01_layout.md`](media/docs/cpp/cute/01_layout.md) — **核心**：Layout = Shape + Stride，CuTe 最基本的抽象
3. [`media/docs/cpp/cute/02_layout_algebra.md`](media/docs/cpp/cute/02_layout_algebra.md) — Layout 的代数运算（组合、补集、逆等）
4. [`media/docs/cpp/cute/03_tensor.md`](media/docs/cpp/cute/03_tensor.md) — Tensor = 指针 + Layout，多维数组抽象
5. [`media/docs/cpp/cute/04_algorithms.md`](media/docs/cpp/cute/04_algorithms.md) — CuTe 内置算法（copy、gemm、fill 等）

### 动手实践

- [`examples/cute/tutorial/sgemm_1.cu`](examples/cute/tutorial/sgemm_1.cu) — 用 CuTe 从零实现 SGEMM
- [`examples/cute/tutorial/sgemm_2.cu`](examples/cute/tutorial/sgemm_2.cu) — SGEMM 的优化版本
- [`examples/cute/tutorial/tiled_copy.cu`](examples/cute/tutorial/tiled_copy.cu) — TiledCopy 操作

### 检查点

- [ ] 能解释 `Layout<Shape, Stride>` 如何将逻辑坐标映射到内存偏移
- [ ] 能阅读并理解 CuTe 风格的 GEMM kernel 代码

---

## 阶段四：GEMM 实现原理

**目标**：理解 CUTLASS 3.x GEMM 的完整实现流程，从 MMA 原子操作到设备级 API。

### 阅读材料

1. [`media/docs/cpp/cute/0t_mma_atom.md`](media/docs/cpp/cute/0t_mma_atom.md) — MMA 原子操作：Tensor Core 的底层接口
2. [`media/docs/cpp/cute/0x_gemm_tutorial.md`](media/docs/cpp/cute/0x_gemm_tutorial.md) — **重要**：从零构建一个完整的 GEMM
3. [`media/docs/cpp/cute/0y_predication.md`](media/docs/cpp/cute/0y_predication.md) — 边界处理（当矩阵尺寸不能被 Tile 整除时）
4. [`media/docs/cpp/gemm_api_3x.md`](media/docs/cpp/gemm_api_3x.md) — CUTLASS 3.x GEMM API 全貌
5. [`media/docs/cpp/cutlass_3x_design.md`](media/docs/cpp/cutlass_3x_design.md) — 3.x 设计哲学
6. [`media/docs/cpp/efficient_gemm.md`](media/docs/cpp/efficient_gemm.md) — 高性能 GEMM 编写指南

### 动手实践

- [`examples/05_batched_gemm`](examples/05_batched_gemm) — 批量 GEMM
- [`examples/06_splitK_gemm`](examples/06_splitK_gemm) — Split-K 并行策略
- [`examples/07_volta_tensorop_gemm`](examples/07_volta_tensorop_gemm) — Tensor Core GEMM（Volta）
- [`examples/12_gemm_bias_relu`](examples/12_gemm_bias_relu) — 带融合 Bias+ReLU 的 GEMM（理解 Epilogue）
- [`examples/02_dump_reg_shmem`](examples/02_dump_reg_shmem) — 调试工具：查看寄存器和共享内存内容

### 检查点

- [ ] 理解 GEMM 的分层分解：Device → Kernel → Collective → Warp → Thread
- [ ] 理解 Mainloop（主循环）和 Epilogue（收尾操作）的职责
- [ ] 能解释 Split-K 策略的优势和适用场景

---

## 阶段五：架构专精

**目标**：针对你的目标 GPU 架构，深入学习架构特有的优化技术。

### 通用参考

- [`media/docs/cpp/functionality.md`](media/docs/cpp/functionality.md) — 各架构支持的功能矩阵
- [`media/docs/cpp/programming_guidelines.md`](media/docs/cpp/programming_guidelines.md) — 编程最佳实践

### Ampere (SM80) 路径

| 阅读 | 实践 |
|------|------|
| `functionality.md` 中 SM80 部分 | [`examples/14_ampere_tf32_tensorop_gemm`](examples/14_ampere_tf32_tensorop_gemm) — TF32 Tensor Core |
| | [`examples/15_ampere_sparse_tensorop_gemm`](examples/15_ampere_sparse_tensorop_gemm) — 结构化稀疏 |
| | [`examples/16_ampere_tensorop_conv2dfprop`](examples/16_ampere_tensorop_conv2dfprop) — 卷积 |
| | [`examples/27_ampere_3xtf32_fast_accurate_tensorop_gemm`](examples/27_ampere_3xtf32_fast_accurate_tensorop_gemm) — 3xTF32 |
| | [`examples/cute/tutorial/sgemm_sm80.cu`](examples/cute/tutorial/sgemm_sm80.cu) — CuTe SM80 SGEMM |

### Hopper (SM90) 路径

| 阅读 | 实践 |
|------|------|
| [`media/docs/cpp/pipeline.md`](media/docs/cpp/pipeline.md) — 异步流水线 | [`examples/48_hopper_warp_specialized_gemm`](examples/48_hopper_warp_specialized_gemm) — Warp 专用化 |
| [`media/docs/cpp/cute/0z_tma_tensors.md`](media/docs/cpp/cute/0z_tma_tensors.md) — TMA | [`examples/49_hopper_gemm_with_collective_builder`](examples/49_hopper_gemm_with_collective_builder) — CollectiveBuilder |
| | [`examples/54_hopper_fp8_warp_specialized_gemm`](examples/54_hopper_fp8_warp_specialized_gemm) — FP8 |
| | [`examples/55_hopper_mixed_dtype_gemm`](examples/55_hopper_mixed_dtype_gemm) — 混合精度 |
| | [`examples/57_hopper_grouped_gemm`](examples/57_hopper_grouped_gemm) — 分组 GEMM |

### Blackwell (SM100) 路径

| 阅读 | 实践 |
|------|------|
| [`media/docs/cpp/blackwell_functionality.md`](media/docs/cpp/blackwell_functionality.md) | [`examples/70_blackwell_gemm`](examples/70_blackwell_gemm) — 基础 Blackwell GEMM |
| [`media/docs/cpp/blackwell_cluster_launch_control.md`](media/docs/cpp/blackwell_cluster_launch_control.md) | [`examples/71_blackwell_gemm_with_collective_builder`](examples/71_blackwell_gemm_with_collective_builder) — CollectiveBuilder |
| | [`examples/72_blackwell_narrow_precision_gemm`](examples/72_blackwell_narrow_precision_gemm) — 低精度 |
| | [`examples/74_blackwell_gemm_streamk`](examples/74_blackwell_gemm_streamk) — StreamK |
| | [`examples/75_blackwell_grouped_gemm`](examples/75_blackwell_grouped_gemm) — 分组 GEMM |

---

## 阶段六：高级主题

根据兴趣和需求选学。

### 卷积

- [`media/docs/cpp/implicit_gemm_convolution.md`](media/docs/cpp/implicit_gemm_convolution.md) — 卷积的 Implicit GEMM 实现原理
- [`examples/09_turing_tensorop_conv2dfprop`](examples/09_turing_tensorop_conv2dfprop) — 2D 卷积前向传播
- [`examples/34_transposed_conv2d`](examples/34_transposed_conv2d) — 转置卷积
- [`examples/76_blackwell_conv`](examples/76_blackwell_conv) — Blackwell 卷积

### 融合操作

- [`examples/13_two_tensor_op_fusion`](examples/13_two_tensor_op_fusion) — 两个 GEMM 的 kernel 融合
- [`examples/35_gemm_softmax`](examples/35_gemm_softmax) — GEMM + Softmax 融合
- [`examples/37_gemm_layernorm_gemm_fusion`](examples/37_gemm_layernorm_gemm_fusion) — GEMM + LayerNorm + GEMM
- [`examples/41_fused_multi_head_attention`](examples/41_fused_multi_head_attention) — 融合多头注意力
- [`examples/77_blackwell_fmha`](examples/77_blackwell_fmha) / [`examples/88_hopper_fmha`](examples/88_hopper_fmha) — Flash Attention

### 稀疏与分布式

- [`examples/62_hopper_sparse_gemm`](examples/62_hopper_sparse_gemm) — 稀疏 GEMM
- [`examples/65_distributed_gemm`](examples/65_distributed_gemm) — 多 GPU 分布式 GEMM
- [`examples/82_blackwell_distributed_gemm`](examples/82_blackwell_distributed_gemm) — Blackwell 分布式

### 性能分析

- [`media/docs/cpp/profiler.md`](media/docs/cpp/profiler.md) — CUTLASS Profiler 使用指南
- [`media/docs/cpp/heuristics.md`](media/docs/cpp/heuristics.md) — 参数选择启发式

```bash
# 使用 Profiler 测试 kernel 性能
./tools/profiler/cutlass_profiler --kernels=cutlass_tensorop_s*gemm_f16_* \
  --m=3456 --n=4096 --k=4096
```

---

## Python DSL 路径（可独立学习）

如果你更熟悉 Python，可以通过 CuTe DSL 来学习和使用 CUTLASS，无需深入 C++ 模板。

### 环境搭建

```bash
python/CuTeDSL/setup.sh --cu12  # CUDA 12
# 或
python/CuTeDSL/setup.sh --cu13  # CUDA 13
```

### 学习顺序

1. [`media/docs/pythonDSL/quick_start.rst`](media/docs/pythonDSL/quick_start.rst) — 快速上手
2. [`media/docs/pythonDSL/overview.rst`](media/docs/pythonDSL/overview.rst) — 功能概览
3. [`media/docs/pythonDSL/cute_dsl_general/dsl_introduction.rst`](media/docs/pythonDSL/cute_dsl_general/dsl_introduction.rst) — DSL 核心概念
4. [`media/docs/pythonDSL/cute_dsl_general/dsl_code_generation.rst`](media/docs/pythonDSL/cute_dsl_general/dsl_code_generation.rst) — 代码生成机制
5. [`media/docs/pythonDSL/cute_dsl_general/dsl_control_flow.rst`](media/docs/pythonDSL/cute_dsl_general/dsl_control_flow.rst) — 控制流
6. [`media/docs/pythonDSL/cute_dsl_api/cute.rst`](media/docs/pythonDSL/cute_dsl_api/cute.rst) — CuTe Python API 参考

### 示例（按架构分组）

| 架构 | 目录 | 说明 |
|------|------|------|
| Ampere | [`examples/python/CuTeDSL/ampere/`](examples/python/CuTeDSL/ampere/) | 基础和中级示例 |
| Hopper | [`examples/python/CuTeDSL/hopper/`](examples/python/CuTeDSL/hopper/) | Dense GEMM、Grouped GEMM、注意力 |
| Blackwell | [`examples/python/CuTeDSL/blackwell/`](examples/python/CuTeDSL/blackwell/) | 最新架构的完整示例 |
| 分布式 | [`examples/python/CuTeDSL/distributed/`](examples/python/CuTeDSL/distributed/) | 多 GPU 集合操作 |

### 进阶

- [`media/docs/pythonDSL/cute_dsl_general/autotuning_gemm.rst`](media/docs/pythonDSL/cute_dsl_general/autotuning_gemm.rst) — 自动调优
- [`media/docs/pythonDSL/cute_dsl_general/framework_integration.rst`](media/docs/pythonDSL/cute_dsl_general/framework_integration.rst) — 与 PyTorch/JAX 集成
- [`media/docs/pythonDSL/cute_dsl_general/debugging.rst`](media/docs/pythonDSL/cute_dsl_general/debugging.rst) — 调试技巧
- [`media/docs/pythonDSL/limitations.rst`](media/docs/pythonDSL/limitations.rst) — 已知限制

---

## 附录：示例索引速查

<details>
<summary>点击展开完整示例列表</summary>

### 基础 (00-06)
| # | 名称 | 主题 |
|---|------|------|
| 00 | basic_gemm | 最简 GEMM |
| 01 | cutlass_utilities | 工具函数 |
| 02 | dump_reg_shmem | 寄存器/共享内存调试 |
| 03 | visualize_layout | 布局可视化 |
| 04 | tile_iterator | Tile 迭代器 |
| 05 | batched_gemm | 批量 GEMM |
| 06 | splitK_gemm | Split-K 策略 |

### Tensor Core GEMM (07-08, 14, 19-20)
| # | 名称 | 主题 |
|---|------|------|
| 07 | volta_tensorop_gemm | Volta Tensor Core |
| 08 | turing_tensorop_gemm | Turing Tensor Core |
| 14 | ampere_tf32_tensorop_gemm | TF32 (Ampere) |
| 19 | tensorop_canonical | 标准 Tensor Core GEMM |
| 20 | simt_canonical | 标准 SIMT GEMM |

### Hopper (48-68)
| # | 名称 | 主题 |
|---|------|------|
| 48 | hopper_warp_specialized_gemm | Warp 专用化 |
| 49 | hopper_gemm_with_collective_builder | CollectiveBuilder |
| 50 | hopper_gemm_with_epilogue_swizzle | Epilogue Swizzle |
| 54 | hopper_fp8_warp_specialized_gemm | FP8 |
| 55 | hopper_mixed_dtype_gemm | 混合精度 |
| 57 | hopper_grouped_gemm | 分组 GEMM |

### Blackwell (70-93)
| # | 名称 | 主题 |
|---|------|------|
| 70 | blackwell_gemm | 基础 GEMM |
| 71 | blackwell_gemm_with_collective_builder | CollectiveBuilder |
| 72 | blackwell_narrow_precision_gemm | 低精度 |
| 74 | blackwell_gemm_streamk | StreamK |
| 75 | blackwell_grouped_gemm | 分组 GEMM |
| 77 | blackwell_fmha | Flash Attention |
| 92 | blackwell_moe_gemm | MoE GEMM |

</details>
