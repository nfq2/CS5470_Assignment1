#!/usr/bin/env python3
"""Summarize CS5470 benchmark runs and generate the latency figures.

Usage: python3 analyze.py --run tp1
       python3 analyze.py --run tp4 --plots
       python3 analyze.py --run tp4 --profile-csv profile/cuda_gpu_kern_sum.csv
       python3 analyze.py --run probe
The result file is selected by its `run` metadata, not by its filename alone.
"""

import argparse
import csv
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


def request_latencies_ms(data):
    """Return successful requests' TTFT and per-request TPOT in milliseconds."""
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
    return ttft_ms, tpot_ms


def summarize(data):
    ttft_ms, tpot_ms = request_latencies_ms(data)
    return {
        "ttft_p50_ms": percentile(ttft_ms, 50),
        "ttft_p99_ms": percentile(ttft_ms, 99),
        "tpot_p50_ms": percentile(tpot_ms, 50),
        "tpot_p99_ms": percentile(tpot_ms, 99),
        "output_throughput_tok_s": data["output_throughput"],
        "request_throughput_req_s": data["request_throughput"],
    }


def summarize_profile(path):
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No GPU kernels found in {path}")
    kernels = sorted(rows, key=lambda row: int(row["Total Time (ns)"]), reverse=True)
    total_time = sum(int(row["Total Time (ns)"]) for row in kernels)
    if total_time <= 0:
        raise ValueError(f"No GPU kernel time found in {path}")

    def kernel_record(row):
        time_ns = int(row["Total Time (ns)"])
        return {
            "name": row["Name"],
            "total_time_ns": time_ns,
            "pct_of_kernel_time": round(100 * time_ns / total_time, 1),
        }

    allreduce = next(
        (row for row in kernels if re.search(r"all.?reduce|cross_device_reduce", row["Name"], re.I)),
        None,
    )
    if allreduce is None:
        raise ValueError(f"No AllReduce kernel found in {path}")
    return {
        "top3_kernels": [kernel_record(row) for row in kernels[:3]],
        "allreduce_kernel_name": allreduce["Name"],
        "allreduce_pct_of_kernel_time": kernel_record(allreduce)["pct_of_kernel_time"],
    }


def plot_latencies(results_dir, config, figure_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plotting requires matplotlib; install it or run without --plots") from exc

    series = {}
    for run in ("tp1", "tp2", "tp4"):
        _, data = select_result(results_dir, run, config)
        series[run] = request_latencies_ms(data)

    figure_dir.mkdir(parents=True, exist_ok=True)
    styles = (
        ("tp1", "1 GPU (TP1)", "#225ea8"),
        ("tp2", "2 GPUs (TP2)", "#d95f0e"),
        ("tp4", "4 GPUs (TP4)", "#238b45"),
    )
    for metric_index, filename, ylabel, title in (
        (0, "ttft_sorted.png", "TTFT (ms)", "Sorted time to first token"),
        (1, "tpot_sorted.png", "TPOT (ms)", "Sorted time per output token"),
    ):
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        for run, label, color in styles:
            values = sorted(series[run][metric_index])
            ax.plot(range(1, len(values) + 1), values, label=label,
                    color=color, linewidth=2)
        ax.set(title=title, xlabel="Prompt index (sorted by latency)", ylabel=ylabel)
        ax.set_xlim(left=1)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper left", frameon=False)
        path = figure_dir / filename
        fig.savefig(path, dpi=240)
        plt.close(fig)
        print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=("tp1", "tp2", "tp4", "probe"), required=True)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--answers", type=Path, default=Path("answers.json"))
    parser.add_argument("--log", type=Path, help="Server log (default: vllm_server_<run>.log)")
    parser.add_argument("--plots", action="store_true", help="Generate sorted TTFT and TPOT plots from TP1, TP2, and TP4")
    parser.add_argument("--figure-dir", type=Path, default=Path("figures"))
    parser.add_argument("--profile-csv", type=Path, help="Nsight CUDA GPU kernel summary CSV")
    parser.add_argument("--probe-script", type=Path, default=Path("probe.sh"),
                        help="H2 server script used for the probe run")
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
    if args.run == "probe":
        baseline_tp4 = answers.get("baseline", {}).get("tp4")
        if baseline_tp4 is None:
            raise ValueError("Analyze the TP4 baseline before the probe")
        script = args.probe_script.read_text()
        match = re.search(r"--max-num-batched-tokens\s+(\d+)", script)
        if match is None:
            raise ValueError(f"No max-num-batched-tokens setting found in {args.probe_script}")
        answers["probe"] = {
            "hypothesis_id": "H2",
            "tp": 4,
            "knob_changed": "max_num_batched_tokens",
            "baseline_value": config["max_num_batched_tokens"],
            "new_value": int(match.group(1)),
            "ttft_p99_ms": metrics["ttft_p99_ms"],
            "tpot_p50_ms": metrics["tpot_p50_ms"],
            "supports_hypothesis": metrics["tpot_p50_ms"] < baseline_tp4["tpot_p50_ms"],
        }
    else:
        answers.setdefault("baseline", {})[args.run] = metrics
        if all(run in answers["baseline"] for run in ("tp1", "tp2", "tp4")):
            baseline = answers["baseline"]
            answers["scaling"] = {
                "ttft_p99_ratio_tp2_over_tp1": baseline["tp2"]["ttft_p99_ms"] / baseline["tp1"]["ttft_p99_ms"],
                "ttft_p99_ratio_tp4_over_tp1": baseline["tp4"]["ttft_p99_ms"] / baseline["tp1"]["ttft_p99_ms"],
                "tpot_p50_ratio_tp4_over_tp1": baseline["tp4"]["tpot_p50_ms"] / baseline["tp1"]["tpot_p50_ms"],
            }
    if args.profile_csv:
        answers["profile"] = summarize_profile(args.profile_csv)
    args.answers.write_text(json.dumps(answers, indent=2) + "\n")
    print(f"Analyzed {path}")
    print(json.dumps(metrics, indent=2))
    if blocks is None:
        print(f"GPU KV cache block count was not found in {log_path}; add it when available.")
    if args.plots:
        plot_latencies(args.results, config, args.figure_dir)


if __name__ == "__main__":
    main()
