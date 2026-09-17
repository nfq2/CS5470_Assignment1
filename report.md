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

To be completed after the 2-GPU and 4-GPU runs.

## Task 3: Performance Visualization

To be completed after all three baseline runs.

## Task 4: NVIDIA Nsight Profiling

To be completed after profiling the 4-GPU server.

## Task 5: Experiment

To be completed after comparing the scaling results.
