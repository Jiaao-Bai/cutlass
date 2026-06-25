# host_probe —— 在 CPU 上跑 CuTe layout 代数(不用 GPU / 不用 CUDA toolkit)

> **一句话**:CuTe 的 `Layout` / `Shape` / `TiledMMA` / `partition_shape_*` / `tile_to_mma_shape` / swizzle / 甚至 `make_tma_atom` + `tma_partition` **全是 host `constexpr` 代数**。只要用本目录的 `stub/` 顶掉 CUDA 头,`g++` 就能在纯 CPU 上把这些 **shape / stride 真值** 打出来 —— 5060Ti、甚至无 GPU 的机器都行。

## 为什么需要它

1. **验 layout 不用开 GPU**。想知道某个 `TiledMMA` 的 `partition_shape_B` 到底是 `((128,16),..)` 还是 `((256,16),..)`?跑出来看,别脑推。
2. **源码注释会骗人**。`examples/cute/tutorial/blackwell/05` 的注释曾有 9 处 stale shape,把人(和 agent)带歪两轮。教训见 [`../THINKING.md` 的 O42](../THINKING.md)。**口诀:实跑 > 脑推 > 源码注释。**
3. **B200 很贵**。UMMA/TMEM 的 *性能* 要 B200,但 *shape/layout 对不对* 是编译期的事,CPU 上就能定。

## 用法

```bash
cd study/host_probe

make run                       # 跑模板 example_probe.cpp(开箱即用)
make run PROBE=你的探针.cpp     # 跑指定探针(.cpp 或 .cu 都行)

# 跑已有的真实探针(核对 example 05 的全部 shape):
make run PROBE=../stage3_gemm/week10_warpspec_writeup/exercises/probe_ex05_shapes_sm100.cu
make run PROBE=../stage3_gemm/week10_warpspec_writeup/exercises/probe_partition_b_sm100.cu
```

不想用 make 也行,直接:
```bash
g++ -std=c++17 -I stub -I ../../include -x c++ 你的探针.cu -o /tmp/p && /tmp/p
```

写新探针:复制 `example_probe.cpp`,在 `main()` 里 `print(你想看的 layout)` 即可。

## 能跑什么 / 不能跑什么

| ✅ 能(host constexpr) | ❌ 不能(需要真 GPU) |
|---|---|
| `make_tiled_mma` / `print(tiled_mma)` | 真正执行 MMA(`fma()` 里是 SM100 PTX) |
| `partition_shape_A/B/C`、`thrfrg_*` | kernel launch、跑出数值结果 |
| `tile_to_mma_shape` / swizzle atom / `tile_to_shape` | ncu 性能数字 |
| `local_tile` / `partition_A/B/C` / `make_fragment_*` | `cp.async` / `tcgen05` / TMA 真实搬运 |
| `make_tma_atom_*` + `tma_partition` 的 **layout**(descriptor 内容是垃圾但 shape 对) | TMA 真实拷贝 |

**注意**:打印里的指针/地址是假的(`(nil)` / `0x0000`),会有 `cast_smem_ptr_to_uint not supported` 之类的 **运行时 stderr 警告 —— 无视它**,shape/stride 是对的(`make run` 已重定向,或自己 `2>/dev/null`)。

## `stub/` 是什么(下次的我别删)

`g++` 没有 `cuda_runtime.h` / `cuda_fp16.h` / `cuda/std/*` / `cuda.h`(driver TMA 类型)这些 NVIDIA 头,但 CuTe 经 `cutlass.h` 硬 `#include` 它们。`stub/` 提供**最小可编译替身**:

- `cuda_runtime_api.h`:把 `__device__/__host__/__global__/__forceinline__` 等执行空间关键字在 host 定义成 no-op(否则 `tmem_allocator_sm100.hpp` 里裸 `__device__` 编不过)。
- `cuda_fp16.h` / `cuda_bf16.h`:最小 `__half`/`__nv_bfloat16`(16-bit 存储)+ `half`/`half2` 别名 + 占位转换函数(不被调用)。
- `vector_types.h`:`dim3`/`float4`/... ;`cuComplex.h`:`cuFloatComplex` + `make_*`。
- `cuda.h`:TMA descriptor 类型(`CUtensorMap` 不透明 128B + `CU_TENSOR_MAP_*` 枚举)+ `cuTensorMapEncodeTiled` no-op 桩(`tma_partition` 的 layout 不依赖 map 内容)。
- `cuda/std/*`:**转发桩** —— 每个 `cuda/std/X` 就是 `#include <X>` 再 `using namespace ::std`,把 libcu++ 的 `cuda::std::` 接到宿主 STL。

这些只为"让 layout 代数编译通过",**不模拟任何 CUDA 运行时行为**。要加新头桩时:按 g++ 报的 `fatal error: xxx.h: No such file` 逐个补,几轮就齐。

## 这套东西的来历

诞生于一次"2-SM UMMA 的 B 到底分摊还是复制"的连环翻车(见 `../THINKING.md` O42)。当时无 B200、5060Ti 也跑不了 SM100 kernel,就靠这套 stub 在 CPU 上跑出 `partition_shape_B = ((128,16),..)`,推翻了源码 stale 注释。**以后凡涉及"shape 到底是多少",先来这里跑。**
