#!/bin/bash
# CS5470 A1 client -- netid nfq2, cfg 7034f168
# usage: ./client.sh <tag>   e.g. ./client.sh tp1  or  ./client.sh probe
TAG=${1:?usage: ./client.sh <tag>}
python3 benchmark.py --backend vllm \
    --model meta-llama/Llama-3.1-8B \
    --seed 39795 \
    --request-rate 8 \
    --num-prompts 200 \
    --dataset-name dummy \
    --long-prompts 5 \
    --long-prompt-len 16000 \
    --save-result --result-dir results \
    --metadata netid=nfq2 cfg=7034f168 run=$TAG
