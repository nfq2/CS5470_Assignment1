#!/bin/bash
# CS5470 A1 server -- netid nfq2, cfg 7034f168, TP=$1
TP=${1:?usage: ./server.sh <1|2|4>}
python3 -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B \
    --tensor-parallel-size $TP \
    --swap-space 16 \
    --enforce-eager \
    --enable-chunked-prefill \
    --max-num-batched-tokens 2048 \
    --max-num-seqs 128 \
    --disable-sliding-window \
    2>&1 | tee vllm_server_tp$TP.log
