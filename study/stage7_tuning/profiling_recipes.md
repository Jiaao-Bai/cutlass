# Profiling 命令收藏

## ncu — Nsight Compute（kernel 级）

### 全量收集（生成 .ncu-rep）
```bash
ncu --set full -o profile_name ./your_kernel ARGS
```

### 关键指标快速看
```bash
ncu --metrics \
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  l1tex__t_bytes.sum,\
  dram__bytes_read.sum,\
  smsp__sass_thread_inst_executed_op_hmma.sum,\
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
  ./your_kernel ARGS
```

### Roofline
```bash
ncu --section SpeedOfLight --section MemoryWorkloadAnalysis \
    --section ComputeWorkloadAnalysis ./your_kernel ARGS
```

### 只 profile 某次 kernel launch
```bash
ncu --launch-skip 0 --launch-count 1 ./your_kernel ARGS
```

### Source view 找瓶颈行
```bash
ncu --set full --import-source on -o profile ./your_kernel ARGS
ncu-ui profile.ncu-rep
```

## nsys — Nsight Systems（系统级 timeline）

```bash
nsys profile -o trace_name --stats=true ./your_kernel ARGS
nsys-ui trace_name.nsys-rep
```

观察：kernel launch 间隔、CPU/GPU 同步、TMA / WGMMA 时序

## ptxas 寄存器使用

```bash
nvcc -Xptxas=-v -arch=sm_90a your_kernel.cu 2>&1 | grep -E "registers|stack|spill"
```

期望：
- 没有 stack frame
- 没有 spill stores / loads
- registers/thread 在 128~255 之间（255 是上界，超过会自动 spill）

## cuobjdump 看 SASS
```bash
cuobjdump --dump-sass your_kernel | less
# 找 HMMA / WGMMA / cp.async.bulk
```

## 常见指标解读

| 指标 | 意义 | 期望值 |
|------|------|--------|
| `sm__throughput` | SM 利用率 | GEMM > 80%，FA > 60% |
| `dram__throughput` | HBM 利用率 | memory bound > 80% |
| `l1tex__data_bank_conflicts_*` | bank conflict | 0（swizzle 正确）|
| `smsp__warps_active.avg.pct_of_peak_sustained_active` | occupancy | 50-100% 都正常 |
| `smsp__sass_thread_inst_executed_op_hmma.sum` | tensor core 指令数 | 应等于理论值（验证算对） |

## 反向工程上游 example 的命令

跑你的 kernel 之前，先跑上游的对照 kernel 拿数据：

```bash
ncu --set full -o cublas_ref \
  ./cublas_gemm_bench M=4096 N=4096 K=4096
ncu --set full -o ours \
  ./study_stage3_w11_ex_warpspec_gemm_v3_pingpong 4096 4096 4096
```

然后在 ncu-ui 里 `File → Compare` 两份 report，逐项找差距。
