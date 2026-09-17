#!/usr/bin/env python3
"""Summarize a CS5470 benchmark run and update answers.json.

Usage: python3 analyze.py --run tp1
The result file is selected by its `run` metadata, not by its filename alone.
"""

import argparse
import json
import math
import re
from pathlib import Path


def percentile(values, percent):
    """Linear interpolation, matching numpy.percentile's default method."""
    if not values:
        raise ValueError("Cannot calculate a percentile from an empty list")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def select_result(directory, run, config):
    matches = []
    for path in directory.glob("*.json"):
        with path.open() as file:
            data = json.load(file)
        if data.get("run") != run:
            continue
        if data.get("netid") != config["netid"] or data.get("cfg") != config["cfg_hash"]:
            raise ValueError(f"{path} has the wrong NetID or configuration hash")
        matches.append((path, data))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {run} result in {directory}; found {len(matches)}. "
            "Move extra runs aside before analyzing."
        )
    return matches[0]


def gpu_blocks(log_path):
    if not log_path.exists():
        return None
    log = log_path.read_text(errors="replace")
    patterns = (
        r"# GPU blocks:\s*([\d,]+)",
        r"num_gpu_blocks(?:\s+is)?\s*[=:]\s*([\d,]+)",
        r"GPU KV cache blocks:\s*([\d,]+)",
    )
    for pattern in patterns:
        matches = re.findall(pattern, log, flags=re.IGNORECASE)
        if matches:
            return int(matches[-1].replace(",", ""))
    return None


def summarize(data):
    ttfts = data["ttfts"]
    itls = data["itls"]
    errors = data["errors"]
    if not (len(ttfts) == len(itls) == len(errors) == data["num_prompts"]):
        raise ValueError("Per-request arrays do not match num_prompts")
    successful = [i for i, error in enumerate(errors) if not error]
    if len(successful) != data["completed"]:
        raise ValueError("completed count disagrees with errors")
    if len(successful) != data["num_prompts"]:
        raise ValueError(f"Only {len(successful)} of {data['num_prompts']} requests completed")
    ttft_ms = [ttfts[i] * 1000 for i in successful]
    # The benchmark records TTFT as itls[i][0]; only later entries are
    # gaps between successive output tokens.
    tpot_ms = [
        sum(itls[i][1:]) / (len(itls[i]) - 1) * 1000
        for i in successful if len(itls[i]) > 1
    ]
    if not tpot_ms:
        raise ValueError("No request has inter-token latencies")
    return {
        "ttft_p50_ms": percentile(ttft_ms, 50),
        "ttft_p99_ms": percentile(ttft_ms, 99),
        "tpot_p50_ms": percentile(tpot_ms, 50),
        "tpot_p99_ms": percentile(tpot_ms, 99),
        "output_throughput_tok_s": data["output_throughput"],
        "request_throughput_req_s": data["request_throughput"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=("tp1", "tp2", "tp4"), required=True)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--answers", type=Path, default=Path("answers.json"))
    parser.add_argument("--log", type=Path, help="Server log (default: vllm_server_<run>.log)")
    args = parser.parse_args()

    with args.config.open() as file:
        config = json.load(file)
    path, data = select_result(args.results, args.run, config)
    metrics = summarize(data)
    log_path = args.log or Path(f"vllm_server_{args.run}.log")
    blocks = gpu_blocks(log_path)
    if blocks is not None:
        metrics["gpu_kv_cache_blocks"] = blocks

    if args.answers.exists():
        with args.answers.open() as file:
            answers = json.load(file)
        if (answers.get("netid"), answers.get("cfg_hash")) != (config["netid"], config["cfg_hash"]):
            raise ValueError(f"{args.answers} belongs to another configuration")
    else:
        answers = {"netid": config["netid"], "cfg_hash": config["cfg_hash"]}
    answers.setdefault("baseline", {})[args.run] = metrics
    if all(run in answers["baseline"] for run in ("tp1", "tp2", "tp4")):
        baseline = answers["baseline"]
        answers["scaling"] = {
            "ttft_p99_ratio_tp2_over_tp1": baseline["tp2"]["ttft_p99_ms"] / baseline["tp1"]["ttft_p99_ms"],
            "ttft_p99_ratio_tp4_over_tp1": baseline["tp4"]["ttft_p99_ms"] / baseline["tp1"]["ttft_p99_ms"],
            "tpot_p50_ratio_tp4_over_tp1": baseline["tp4"]["tpot_p50_ms"] / baseline["tp1"]["tpot_p50_ms"],
        }
    args.answers.write_text(json.dumps(answers, indent=2) + "\n")
    print(f"Analyzed {path}")
    print(json.dumps(metrics, indent=2))
    if blocks is None:
        print(f"GPU KV cache block count was not found in {log_path}; add it when available.")


if __name__ == "__main__":
    main()
