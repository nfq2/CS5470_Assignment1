#!/bin/bash
# CS5470 A1 H2 probe -- netid nfq2, cfg 7034f168, TP=$1
TP=${1:?usage: ./probe.sh <4>}
python3 -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B \
    --tensor-parallel-size $TP \
    --swap-space 16 \
    --enforce-eager \
    --enable-chunked-prefill \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 128 \
    --disable-sliding-window \
    2>&1 | tee vllm_server_probe.log
