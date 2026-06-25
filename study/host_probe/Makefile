# study/host_probe/Makefile —— 用 g++ 在 CPU 上跑 CuTe layout 代数(无需 GPU / CUDA toolkit)
#
# 用法:
#   make run                      # 编译并跑 example_probe.cpp(模板,开箱即用)
#   make run PROBE=foo.cpp        # 跑指定探针(相对/绝对路径都行;.cu 也当 C++ 编)
#   make run PROBE=../stage3_gemm/week10_warpspec_writeup/exercises/probe_ex05_shapes_sm100.cu
#   make clean
#
# 原理见 README.md。核心:CuTe 的 layout/shape/partition/swizzle 全是 host constexpr,
# 只要用 stub/ 顶掉 CUDA 头(g++ 没有 cuda_runtime.h 等),就能在纯 CPU 上跑出真值。

CUTLASS_INC ?= $(abspath $(CURDIR)/../../include)   # 仓库根的 include/
STUB        := $(CURDIR)/stub
CXX         ?= g++
CXXFLAGS    := -std=c++17 -I$(STUB) -I$(CUTLASS_INC)

PROBE ?= example_probe.cpp
BIN   := /tmp/$(basename $(notdir $(PROBE)))

.PHONY: run clean
run: $(BIN)
	@echo "=== run $(notdir $(PROBE)) ===" && $(BIN)

# -x c++ 让 .cu 也按 C++ 编(我们只跑 host layout 代数,不调用任何 device 指令)
$(BIN): $(PROBE)
	$(CXX) $(CXXFLAGS) -x c++ $(PROBE) -o $(BIN)

clean:
	rm -f /tmp/example_probe /tmp/probe_*
