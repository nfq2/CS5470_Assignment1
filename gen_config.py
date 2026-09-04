#!/usr/bin/env python3
"""Derive A1 configuration from student netid."""
import argparse, hashlib, json

SALT = "cs5470-fa26-a1-v1"      # bump to reshuffle the whole class

def config(netid: str) -> dict:
    key = f"{SALT}:{netid.strip().lower()}".encode()
    digest = hashlib.sha256(key).hexdigest()
    h = int(digest, 16)
    return {
        "netid": netid.strip().lower(),
        "cfg_hash": digest[:8],
        "seed":                   h % 100000,
        "request_rate":           [4, 8, 12, 20][(h >> 17) % 4],
        "num_prompts":            [100, 150, 200][(h >> 23) % 3],
        "long_prompts":           [0, 2, 5][(h >> 29) % 3],
        "long_prompt_len":        16000,
        "max_num_batched_tokens": [512, 1024, 2048][(h >> 35) % 3],
        "max_num_seqs":           [64, 128, 256][(h >> 41) % 3],
        "model": "meta-llama/Llama-3.1-8B",
    }

SERVER = """#!/bin/bash
# CS5470 A1 server -- netid {netid}, cfg {cfg_hash}, TP=$1
TP=${{1:?usage: ./server.sh <1|2|4>}}
python3 -m vllm.entrypoints.openai.api_server \\
    --model {model} \\
    --tensor-parallel-size $TP \\
    --swap-space 16 \\
    --enforce-eager \\
    --enable-chunked-prefill \\
    --max-num-batched-tokens {max_num_batched_tokens} \\
    --max-num-seqs {max_num_seqs} \\
    --disable-sliding-window \\
    2>&1 | tee vllm_server_tp$TP.log
"""

CLIENT = """#!/bin/bash
# CS5470 A1 client -- netid {netid}, cfg {cfg_hash}
# usage: ./client.sh <tag>   e.g. ./client.sh tp1  or  ./client.sh probe
TAG=${{1:?usage: ./client.sh <tag>}}
python3 benchmark.py --backend vllm \\
    --model {model} \\
    --seed {seed} \\
    --request-rate {request_rate} \\
    --num-prompts {num_prompts} \\
    --dataset-name dummy \\
    --long-prompts {long_prompts} \\
    --long-prompt-len {long_prompt_len} \\
    --save-result --result-dir results \\
    --metadata netid={netid} cfg={cfg_hash} run=$TAG
"""

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--netid", required=True)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    c = config(a.netid)
    if a.write:
        for name, tmpl in (("server.sh", SERVER), ("client.sh", CLIENT)):
            open(name, "w").write(tmpl.format(**c))
        open("config.json", "w").write(json.dumps(c, indent=2))
        print("wrote server.sh, client.sh, config.json")
    else:
        print(json.dumps(c, indent=2))
        print(SERVER.format(**c))
        print(CLIENT.format(**c))
