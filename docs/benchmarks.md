# Serving benchmarks

This page describes how Metallix is compared with other LLM servers on one
Apple Silicon Mac, and will hold the results. No results are published yet:
the table below stays empty until a baseline run meets the protocol's own
gate.

All engines run on the same machine, with the same model, prompts and output
length, one engine at a time. The numbers describe that machine and those
versions only.

## What is measured

The client streams requests to each server's OpenAI-compatible API and records
for each request:

- time to first token (TTFT), from sending the request to the first streamed
  content or reasoning token;
- time per output token (TPOT), the decode time after the first token divided
  by the remaining output tokens;
- output tokens, as reported in the server's usage block (the streamed chunk
  count when a server sends none);
- the outcome: completed, HTTP error, or stream error.

Each cell (engine, prompt set, prefix-cache arm, concurrency) reports output
throughput (completed output tokens over the cell's wall time), TTFT and TPOT
percentiles, completed and failed counts, and the peak memory sampled once a
second: system memory in use (app, wired and compressed pages, as Activity
Monitor counts them), GPU-resident memory, and the server's process-group
resident size.

## Matrix

| Axis | Values |
|---|---|
| Model | Qwen3-0.6B BF16; larger models as they are supported |
| Prompt sets | short; shared-prefix (2,048-token agent preamble, distinct tasks); long (2K to 8K unique tokens); the SiliconBench agent split |
| Concurrency | 1, 2, 4, 8, 16, 32 requests in flight (closed loop) |
| Prefix cache | on and off as separate arms, for engines that can switch it |
| Output | 128 tokens, temperature 0, thinking off; the agent split uses each prompt's 256-token cap |

The agent split is the 100-prompt set from SiliconBench ([paper](https://arxiv.org/abs/2609.19169),
[prompts](https://github.com/WindChimeRan/SiliconBench/blob/616aa51c450383ee9309d2149d6765f9bd117119/prompts/agent_benchmark_prompts.json),
MIT licensed): recorded multi-turn tool-use conversations of 0.3K to 8.9K
tokens. It is used unchanged from that commit (SHA-256
`76d3ff55f4026a50679fd60becd07a9cad93549dc46d98a59bcd642c90d0f81e`).

## Protocol

1. Idle gate before every arm, with its readings stored in the report:
   1-minute load average below 2.0, no `cargo` or `rustc` running, AC power
   (`pmset -g batt`), no thermal or performance warning (`pmset -g therm`),
   and GPU utilization at most 5% over five one-second samples
   (`ioreg -r -c AGXAccelerator -d 1`). An arm that cannot pass within 30
   minutes is skipped and reported.
2. A load average above 4 during an arm aborts it, and so does machine-wide
   GPU memory in use above a cap sized to the model (24 GiB for Qwen3-0.6B,
   where the largest engine peaked at 12.6 GiB). Each arm records the GPU
   memory already in use before its server started. Aborted arms report no
   numbers.
3. A fresh server for every engine at every concurrency level, so no level
   inherits another level's prefix cache or allocator state.
4. Discarded warmup requests at the level's width: at least three, and at
   least twice the concurrency. A rate level warms at its expected number of
   requests in flight, the rate times the time two serial requests took.
5. 64 measured requests per cell up to concurrency 8, 100 above.
6. Three passes. Each pass visits the engines in an order shuffled from a
   recorded seed, with a 60-second cooldown between arms.
7. Equal work. Every engine gets the same prompts and output cap, with
   `ignore_eos` where the server supports it. When engines produce different
   output token counts for the same prompts, the report says so, because
   throughput then compares unequal work.
8. Recorded with every run: macOS version and build, chip and memory, the MLX
   version Metallix links, every engine's version or commit, and each
   server's exact command line.

## Reading results and claims

Each cell shows the median pass with the minimum and maximum of the three
passes.

- Faster: Metallix's worst pass beats the other engine's best pass on that
  metric.
- Significantly faster: faster, and the median ratio is at least 1.5.
- Parity: medians within 10%.
- Overlapping: pass ranges overlap. No claim either way.

A cell supports a claim only when every pass in it passed the idle gate. Two
shapes are expected to show a real difference: an agent fan-out where
concurrent requests share a long prefix (concurrency 4 and up), and a small
model with a single request in flight. Elsewhere, the expected and honest
outcome is parity. Large-model single-stream decode, for example, already
runs close to the memory-bandwidth limit in the fastest engines.

## Engine configuration

Each engine runs with settings that make the comparison fair to it.

- Metallix: `mx serve` built with `cargo build -p server --release
  --all-features`. The cache-off arm passes `--prefix-cache-mib 0`.
- vllm-metal: the same maximum model length; prefix caching set explicitly
  with `--enable-prefix-caching` or `--no-enable-prefix-caching`;
  `ignore_eos`; thinking off through `chat_template_kwargs`.
- mlx-lm: `mlx_lm.server` with thinking off through `--chat-template-args`.
  No request seed, which would disable batching. The cache-off arm passes
  `--prompt-cache-size 0`.
- MTPLX: `ar_batch` scheduler with the `throughput` batching preset. It has no
  in-memory prefix-cache switch, so it runs only its default arm.
- llama.cpp, Ollama and other servers are started by hand and measured
  through `--url`, with their command line and version recorded. llama.cpp
  needs one slot per concurrent request (`-np`) and a context of slots times
  the per-request context.
- Ollama: one `ollama serve` per level with `OLLAMA_NUM_PARALLEL` set to the
  concurrency, not the background service. Requests carry
  `--request-extra '{"reasoning_effort": "none"}'` to turn Qwen3 thinking off.
  Ollama's library has no BF16 build of Qwen3-0.6B, so its rows use the F16
  build and say so with `--note`.

Engines and versions used to prepare the protocol (each report records the
versions it ran):

| Engine | Version | Install | Weights |
|---|---|---|---|
| vllm-metal | 0.30.0 (vllm 0.30.0+cpu, mlx 0.32.1) | release wheels of vllm and vllm-metal v0.30.0 | Hugging Face Qwen3-0.6B, BF16 |
| mlx-lm | 0.32.0 (mlx 0.32.3) | `uv pip install mlx-lm==0.32.0` | Hugging Face Qwen3-0.6B, BF16 |
| MTPLX | 2.12.2 (mlx 0.32.2) | `uv pip install mtplx==2.12.2` | Hugging Face Qwen3-0.6B, BF16 |
| llama.cpp | b11146 (commit 7fe450e19) | `brew install llama.cpp` | BF16 GGUF from the same checkpoint, `convert_hf_to_gguf.py --outtype bf16` at that commit |
| Ollama | 0.35.1 | `brew install ollama` | `ollama pull qwen3:0.6b-fp16` (digest `626c9556a80f`), GGUF F16, not BF16 |

## Reproducing

The scripts need [uv](https://docs.astral.sh/uv/). The Python servers run
from one virtual environment each (`vllm-metal`, `mlx-lm`, `mtplx`) under the
directory given by `--envs`.

```sh
cargo build -p server --release --all-features

# Download the agent split pinned above.
curl -LO https://raw.githubusercontent.com/WindChimeRan/SiliconBench/616aa51c450383ee9309d2149d6765f9bd117119/prompts/agent_benchmark_prompts.json

# Three passes over the managed engines, both cache arms.
uv run scripts/bench_campaign.py \
  --server metallix,vllm-metal,mlx-lm --model-path MODEL_DIR --envs ENVS_DIR \
  --sets shared-prefix,file --prompts-file agent_benchmark_prompts.json \
  --concurrency 1,4,8,16 --cache-arms on,off --context-tokens 16384 \
  --abort-gpu-gib 24 --json campaign.json

# A hand-started server, once per pass, then one summary for everything.
uv run scripts/bench_load.py --url http://127.0.0.1:8080 --label llama.cpp \
  --launch-argv "llama-server -m qwen3-0.6b-bf16.gguf -ngl 99 -np 16 -c 262144" \
  --version llama.cpp=BUILD --model-path MODEL_DIR --sets shared-prefix \
  --concurrency 1,4,8,16 --json llama-pass1.json
uv run scripts/bench_campaign.py summarize campaign.json llama-pass*.json
```

`summarize` counts each `bench_load.py` report as one pass. Such a report
records the idle gate once, at its start, and cannot restart a hand-started
server between levels.

To check the harness without a model, run the campaign against the built-in
stub servers, which stream fixed-speed tokens. Their numbers mean nothing:

```sh
uv run scripts/bench_campaign.py --server stub,stub-slow --subject stub \
  --model-path MODEL_DIR --sets short --concurrency 1,4 --passes 3 \
  --cooldown 0 --skip-idle-gate --requests 8
```

## Comparing two builds on a busy machine

`scripts/bench_pair.py` answers a narrower question than the protocol: is
configuration B faster or slower than A? It runs the two arms at one level in
the order A,B, B,A, A,B, ..., each on a fresh server, and reports the median
of the per-pair ratios B/A with a bootstrap 95% confidence interval, plus the
load when each run started. Slow changes in machine load affect both runs of
a pair about equally, and alternating the order cancels a steady trend, so an
interval that excludes 1 shows a difference even under load. Its absolute
numbers are not results; those come only from the idle protocol above.

```sh
uv run scripts/bench_pair.py --server metallix --model-path MODEL_DIR \
  --sets short --concurrency 1 --pairs 8 \
  --a-args "--mx build-a/mx" --b-args "--mx build-b/mx"
```

## Results

No runs have passed the protocol yet.

| Model | Prompt set | Cache | c | Engine | Output tok/s, median [min-max] | TTFT p50 ms | TPOT p50 ms | Completed | Peak memory GiB | Claim vs Metallix |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |

Machine, OS build, engine versions and the order seed for each published run
will be listed here with the table.
