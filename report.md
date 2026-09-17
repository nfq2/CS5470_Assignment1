# CS5470 Assignment 1 Report (Draft)

## Task 1: Single-GPU Benchmark

I served Llama-3.1-8B with tensor parallel size 1 and ran the assigned benchmark workload. All 200 requests completed successfully.

| Metric | p50 | p99 |
| --- | ---: | ---: |
| Time to first token (TTFT) | 94.98 ms | 1190.52 ms |
| Time per output token (TPOT) | 42.60 ms | 56.50 ms |

TTFT measures the delay from sending a prompt until its first output token arrives. TPOT is the average gap between successive output tokens **within one request**. The benchmark stores the first-token wait as the first entry of each `itls` list, so I excluded that entry before averaging the remaining gaps. I then took the p50 and p99 across requests. A p50 TPOT of 42.60 ms means that half of the requests had an average gap of about 43 ms or less between output tokens.

The p99 TTFT was much higher than the median. The slowest request had a 16,000-token prompt, and several requests arriving soon afterward also had long TTFTs. This pattern suggests that processing the long prompt may have delayed nearby requests, although the benchmark data alone does not establish the cause.

The run completed 3.79 requests per second and generated 1609.82 output tokens per second across concurrent requests. The server reported 10,203 GPU KV-cache blocks.

## Task 2: Multi-GPU Benchmark

I ran the same assigned workload with tensor parallel sizes 2 and 4. All 200 requests completed in each run. The model, request rate, prompt count, and server batching settings stayed the same.

| GPUs | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | TPOT p99 (ms) | Output tokens/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 94.98 | 1190.52 | 42.60 | 56.50 | 1609.82 |
| 2 | 59.09 | 717.15 | 25.06 | 33.55 | 1868.15 |
| 4 | 46.73 | 388.95 | 20.33 | 22.29 | 1939.08 |

The required ratios are TTFT p99(TP2)/p99(TP1) = **0.602**, TTFT p99(TP4)/p99(TP1) = **0.327**, and TPOT p50(TP4)/p50(TP1) = **0.477**. Lower ratios mean shorter delays. Output throughput increased by about 16.0% from one to two GPUs, then 3.8% from two to four GPUs; latency improved more than aggregate throughput in the four-GPU run.

## Task 3: Performance Visualization

Each curve sorts that configuration's 200 request latencies independently, so the x-axis shows latency rank rather than the requests' original order.

![Sorted time to first token for 1, 2, and 4 GPUs](figures/ttft_sorted.png)

![Sorted time per output token for 1, 2, and 4 GPUs](figures/tpot_sorted.png)

TTFT is lower with more GPUs for most requests. All three curves rise sharply at the slowest few requests, but the TP4 tail is much shorter: its p99 TTFT is 388.95 ms versus 1190.52 ms for TP1. TPOT also falls with more GPUs. The TP4 curve stays near 20 ms for most requests, while TP1 is mostly between about 30 and 50 ms and reaches a higher tail. These plots show improved latency across the workload; they do not identify the cause of individual slow requests.

## Task 4: NVIDIA Nsight Profiling

I profiled Llama-3.1-8B with tensor parallel size 4 while serving three sequential, identical completion requests. The trace recorded CUDA kernels. Nsight's CUDA GPU Kernel Summary ranked these kernels by cumulative GPU execution time across the trace:

| Kernel | Total time (ns) | Share of kernel time |
| --- | ---: | ---: |
| `vllm::cross_device_reduce_1stage<__nv_bfloat16, 4>` | 66,847,922,806 | 69.6% |
| `ampere_bf16_s16816gemm_bf16_128x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | 9,590,177,620 | 10.0% |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | 5,396,774,850 | 5.6% |

The first kernel is vLLM's AllReduce operation for communication among the four tensor-parallel ranks. It accounts for **69.6% of total recorded kernel time**. Kernel time sums execution across GPUs and can exceed the trace's elapsed wall time. This percentage describes the three profiled requests, not the 200-request benchmark from Task 2.

![Nsight Systems CUDA GPU Kernel Summary showing the AllReduce kernel at 69.6% of recorded kernel time](figures/allreduce_kernel.png)

## Task 5: Experiment

Output throughput improved only 3.8% from TP2 to TP4, the weakest step in scaling. At TP4, p99 TTFT was 388.95 ms versus a 46.73 ms median; the 16,000-token prompt and requests immediately after it had the highest TTFTs. I chose H2: the 2,048-token prefill batch limit may spread long prompts over too many scheduler steps. I ran a TP4 probe with `--max-num-batched-tokens 4096`, keeping the assigned client workload and other server settings fixed. All 200 requests completed. The probe reduced p99 TTFT from 388.95 to 321.58 ms (17.3%), while p50 TPOT rose from 20.33 to 20.79 ms (2.3%) and output throughput fell from 1,939.08 to 1,913.58 tokens/s (1.3%). This supports H2 as a contributor to the TTFT tail, but the change did not improve the weak throughput scaling. One run per setting cannot establish how much of the difference was due to run-to-run variation.

The most difficult part was collecting a usable multi-GPU Nsight trace. Earlier captures produced reports without CUDA kernel data; starting collection before sending requests and checking its state yielded a trace with GPU kernels. The resulting AllReduce kernel dominated recorded GPU kernel time, even though the probe showed that increasing the prefill batch limit helped the TTFT tail.
