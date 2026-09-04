# CS5470 - HW1: vLLM Benchmarking and Profiling
**Due Date:** Sep 18 11:59PM 

Please start this assignment as early as possible!

## Overview

This homework focuses on setting up and benchmarking the vLLM inference server ([GitHub repository](https://github.com/vllm-project/vllm/)). You will:

- Set up the inference server
- Benchmark generation performance
- Understand the breakdown of GPU execution time during LLM inference

## Prerequisites

### Hardware Access
You will be provided access to the **Perlmutter HPC** where you can reserve a server with 4 A100 GPUs, each having 40 GB of memory.

### Setup Instructions

#### Step 1: Setup Conda and VLLM

Set up the conda environment and clone the vLLM repository with the specified commit. 

<!-- Install conda if it is not available on the GPU server using the [Linux terminal installer](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-terminal-installer). -->

**Setup Commands:**
```bash
# On login node, load conda
module load conda
conda create --name sysml python=3.10.12
conda activate sysml

git clone https://github.com/vllm-project/vllm/
cd vllm

# Use the version we are using for this homework
git reset --hard 2f13319f47eb9a78b471c5ced0fcf90862cd16a2

# Install vLLM with precompiled binaries
VLLM_USE_PRECOMPILED=1 python3 -m pip install -e .
python3 -m pip install 'transformers>=4.55.2,<5'
```

<!-- If you run into disk quota issue, you could try 

https://docs.nersc.gov/filesystems/perlmutter-scratch/

/pscratch/sd/FirstLetterOfUserName/YourUserName

```
conda create -p /pscratch/sd/e/$USER/sysml python=3.10.12


``` -->
#### Step 2: Huggingface access
1. Create Huggingface account and request for model access.
- `meta-llama/Llama-3.1-8B`
2. Log in your account from command line
   ```bash
   hf auth login
   # Create token through the url and copy-paste
   ```
#### Step 3: Download model and tokenizer
This is required as vLLM could not download models on-the-fly on Perlmutter GPU nodes due to restrictions on large file download.

```
python download.py --model-id meta-llama/Llama-3.1-8B --cache-dir $PSCRATCH/huggingface
```

**Note**: Due to disk space limitation on `home` file system, we encourage using `pscratch` for storing model weights. More information could be found [here](https://docs.nersc.gov/filesystems/perlmutter-scratch/).

**Note**: remember to set environment variable `HF_HOME=$SCRATCH/huggingface` in your running environment such that HF library could select locally-cached model weights instead of pulling from remote repository.

<!-- TODO Do we need this 

Copy the homework zip file into the vLLM source's root directory and uncompress it. -->

## Homework Tasks

This homework is designed to teach how to serve LLMs on modern GPUs. You will learn how to serve models that fit on a single GPU and how to serve models using tensor parallelism on multiple GPUs.

### Task 1: Single GPU Benchmarking

Within the homework folder, we have provided `gen_config.py`, a script that generates `server.sh` and `client.sh` from your assigned workload and server configuration based on your netid. `server.sh` is a script that starts a `meta-llama/Llama-3.1-8B` instance on GPU 0 and exposes a vLLM API.

**Requirements:**
1. Allocate a node with 1 GPU.
2. Generate `server.sh` and `client.sh` by running `gen_config.py --netid [your netid here] --write`
3. Start the serving workload on 1 GPU using the command provided in `server.sh`
4. Execute the benchmark script using the command in `client.sh`
5. After benchmarking, write a script `analyze.py` to calculate the following metrics from results/\*.json:

`ttft`: delay between LLM recieving input and the output of the first token. Calculate the p50 and p99 in milliseconds.

`tpot`: the average of all inter token latencies, or, `itls`. Calculate the p50 and p99 in milliseconds. 

**Hint:**
1. Allocate interactive GPU node using `salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 1 --account <projectID>`.
2. Remember to make those scripts executable.
3. Remember to activate your conda environment.
4. Spawning two terminals would be useful, one for server and one for client. They should be on the same GPU node. Access the allocated node using `ssh <GPU_node_ID>` if the other terminal is on the login node.
5. You need to login to Huggingface again on GPU nodes, and set `HF_HOME` to `HF_HOME=$SCRATCH/huggingface` for both server and client terminal.
6. You might wait for several minutes until the serving engine is up and running.

### Task 2: Multi-GPU Benchmarking

**Requirements:**
1. Serve `meta-llama/Llama-3.1-8B` on 2 GPUs.
2. Serve `meta-llama/Llama-3.1-8B` on 4 GPUs.
3. Use `server.sh` to execute these configurations.
   - Review the vLLM documentation to understand what CLI arguments you need to add or modify for multi-GPU inference
4. Calculate the same metrics you did in task 1 for the multi-GPU setup in addition to the following ratios:

ttft p99_tp2 / p99_tp1

ttft p99_tp4 / p99_tp1

tpot p50_tp4 / p50_tp1

### Task 3: Performance Visualization

Create two plots showing:
1. Sorted TTFTs of prompts in the benchmark from all three configurations: serving with 1, 2, and 4 GPUs.
2. Sorted TPOTs of prompts in the benchmark from all three configurations

X-axis is the prompt index (not the original index of prompt) and the y-axis is TTFT or TPOT. Add labels to indicate the configuration.

### Task 4: NVIDIA Nsight Profiling

**Requirements:**
1. Install NVIDIA Nsight Systems if not available from the [official website](https://developer.nvidia.com/nsight-systems/get-started)
   - Use the CLI-only deb installer for Linux
   - The current version on Perlmutter may not work for us.
2. Generate an Nsight trace using the `nsys` profiler for Llama-3.1-8B on 4 GPUs
3. Open the generated `.nsys-rep` file in Nsight desktop client on your laptop
4. Use the Stats System View to identify the top 3 kernels that consumed the most time
5. Identify one AllReduce kernel, record its name, and take a screenshot.
6. Calculate the percentage of total kernel time from the AllReduce operation.
7. Export the kernel summary using `nsys stats --report cuda_gpu_kern_sum --format csv`

**Hint**
1. Use interactive profiling mode.
2. Store the `nsys-rep` file under `pscratch`.

### Task 5: Experiment

**Requirements:**
1. Using the metrics you calculated in Tasks 1 - 3, identify where the scaling behaved worst for your configuration between the different TP settings. 
2. Pick a hypothesis for why the scaling underperformed out of the following, and adjust the permitted knob to test your hypothesis:

| ID | Hypothesis | Permitted knob |
|---|---|---|
| H1 | Queueing dominates because concurrency is capped | `--max-num-seqs` |
| H2 | Prefill chunking is too fine, so prefill is spread over too many scheduler steps | `--max-num-batched-tokens` |
| H3 | The long prompts dominate the tail | `--long-prompts` |
| H4 | The server is under-loaded, so additional GPUs cannot help | `--request-rate` |

Do not change any other parameters from your assigned configuration other than the permitted knob.

Create a `probe.sh` file by copying the generated script that contains your permitted knob.

The client command must use the tag `probe` so that the result
filename and JSON metadata identify the probe run.

You must submit the `probe.sh` that you modified. It may be based on either
`server.sh` or `client.sh` depending on the hypothesis you chose

3. Was your hypothesis correct? Write a paragraph describing why you picked this hypothesis, the results of the experiment, and justify why or why not you think it was correct. 


## Deliverables

#### 1. Report (PDF format)
- All requested plots (TTFT and TPOT visualizations) and a brief analysis of what's going on.
- A table with Nsight profiling results showing top 3 kernel names and times
- Identification (Screenshot) of the All-reduce kernel responsible for tensor parallel communication during multi-GPU inference
- One paragraph describing the results of your experiment.
- One paragraph discussing the difficulties or surprising results you have in this assignment.

#### 2. Code
- `probe.sh` file you modified to launch the experiment probe in Task 5.
- `analyze.py` python script for data parsing and visualization. This script should also generate your results in the json format described in the submission requirements. 

#### 3. Data Files
- `config.json` file that describes your generated configuration.
- Benchmark results
- Profiling outputs

### Submission Requirements

`answers.json` will be where your answers to tasks 1-5 will be reported using the following format.

```json
{
  "netid": "abc123",
  "cfg_hash": "1a2b3c4d",

  "baseline": {
    "tp1": {"ttft_p50_ms": 0.0, "ttft_p99_ms": 0.0,
            "tpot_p50_ms": 0.0, "tpot_p99_ms": 0.0,
            "output_throughput_tok_s": 0.0,
            "request_throughput_req_s": 0.0,
            "gpu_kv_cache_blocks": 0},
    "tp2": {"...": "same keys"},
    "tp4": {"...": "same keys"}
  },

  "scaling": {
    "ttft_p99_ratio_tp2_over_tp1": 0.0,
    "ttft_p99_ratio_tp4_over_tp1": 0.0,
    "tpot_p50_ratio_tp4_over_tp1": 0.0
  },

  "profile": {
    "top3_kernels": [
      {"name": "...", "total_time_ns": 0, "pct_of_kernel_time": 0.0},
      {"name": "...", "total_time_ns": 0, "pct_of_kernel_time": 0.0},
      {"name": "...", "total_time_ns": 0, "pct_of_kernel_time": 0.0}
    ],
    "allreduce_kernel_name": "...",
    "allreduce_pct_of_kernel_time": 0.0
  },

  "probe": {
    "hypothesis_id": "H2",
    "tp": 4,
    "knob_changed": "max_num_batched_tokens",
    "baseline_value": 512,
    "new_value": 2048,
    "ttft_p99_ms": 0.0,
    "tpot_p50_ms": 0.0,
    "supports_hypothesis": true
  }
}
```

Again, please keep aware of your units. `ttfts` and `itls` are stored in seconds, while your reported aggregates of these metrics are in milliseconds. 

Your submission will be one archive named `<netid>_a1.zip` in the following format:

```text
<netid>_a1.zip
├── config.json                 produced by gen_config.py, unmodified
├── answers.json                schema containing your answers
├── report.pdf                  2 pages maximum
├── results/
│   ├── vllm-*-tp1-*.json       raw benchmark output, run=tp1
│   ├── vllm-*-tp2-*.json       run=tp2
│   ├── vllm-*-tp4-*.json       run=tp4
│   └── vllm-*-probe-*.json     run=probe
├── logs/
│   ├── vllm_server_tp1.log
│   ├── vllm_server_tp2.log
│   ├── vllm_server_tp4.log
│   └── vllm_server_probe.log
├── profile/
│   ├── tp4.nsys-rep
│   └── cuda_gpu_kern_sum.csv
└── code/
    ├── analyze.py              produced answers.json and the figures
    └── probe.sh                the modified launch used for the probe
```
Ensure that your submission matches the format and file names exactly.

## Chatting with the LLM

One of the main use cases of LLMs is the ability to provide a chat user interface to end-users. However, LLMs are trained to generate the next token given a series of tokens in the prompt without any inherent attribution to each token's source (user or LLM).

To generate meaningful responses, it is critical to annotate which parts of the prompts are generated by which entity. LLMs are fine-tuned to follow a specific template to distinguish between user and assistant messages. Conversations need to follow this template for generating prompts for every user's new messages (see [Llama 3.1 documentation](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/)).

We have provided a simple Python script called `complete.py` to test this functionality. We encourage you to:
- Go through the file and test custom messages
- Build on top of it to manage conversation history
- Understand how LLM applications like ChatGPT and Claude manage conversation history

---

