# Week 18 — FA 变体：GQA / decode（SM100）

预计 ~15h ｜ 目标硬件：B200（SM100）
> 本周承接 W15-W17 的 FA 主线——把 prefill 形态的 FA fwd 改造成 **LLM 推理真实场景的两个变体**：GQA（多 Q head 共享 KV head）与 decode（q_len=1，memory bound）。这两个变体是开源算子库 attention 家族的必备成员。

## 目标
- 在 FA fwd 上支持 GQA（H_q ≠ H_kv），理解 KV head 复用对 tile 划分和 TMA 的影响
- 写 decode attention（q_len=1 / 短 q）：识别它是 memory bound，优化方向从 TC 利用率切换到 KV 加载带宽
- 跑通 `examples/93_blackwell_low_latency_gqa` 作对照基线
- 写完 Stage 4 CHECKPOINT 文档

## 读
- `examples/python/CuTeDSL/blackwell/fmha.py` — SM100 FA fwd 参考实现（Python，逻辑清晰）
- `examples/93_blackwell_low_latency_gqa/tgv_gqa.cu` / `tgv_gqa.cuh` — 官方 GQA 低延迟实现
- `examples/77_blackwell_fmha` 的 varlen / GQA 选项 — prefill 侧 GQA 参照

## 写
- `exercises/ex_fa_gqa.cu` — W16 v2 加 GQA 支持：H_q/H_kv 比例 8:1（Llama 风格），正确性 + B200 性能
- `exercises/ex_fa_decode.cu` — decode 变体：q_len=1，KV cache S=4k/32k 两档，对比 example 93
- `exercises/VARIANTS_NOTES.md` — 记录：GQA 的 KV 复用怎么省带宽；decode 的 roofline 位置（算术强度 ≈ 1）；split-KV 在长 S 下的收益

## 跑
```bash
cmake -DCUTLASS_ENABLE_STUDY=ON -DCUTLASS_NVCC_ARCHS=100a ..
make study_stage4_w18_ex_fa_gqa study_stage4_w18_ex_fa_decode -j
ncu --set full -o fa_decode ./study_stage4_w18_ex_fa_decode
```
- 期望：GQA prefill 仍 ≥ 80% example 77；decode 看 `dram__throughput` 接近带宽 roofline（不看 TC 利用率）

## 自检
1. GQA 下 KV tile 被几个 Q head 复用？这个复用应该放在哪一层（同 CTA 多 Q head vs TMA multicast）？
2. decode（q_len=1）的算术强度是多少？为什么 TC 利用率指标在这里失效、该看什么指标？
3. split-KV（FlashDecoding）把 S 维拆给多个 CTA，最后怎么合并 partial softmax（log-sum-exp 合并公式）？
4. decode 场景下 persistent kernel / CLC 调度还有收益吗？瓶颈是 compute 还是 launch/latency？
5. paged KV cache（vLLM 风格非连续 KV block）对 TMA 意味着什么——还能用 TMA 吗，还是退回 cp.async？
