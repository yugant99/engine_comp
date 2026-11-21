# Engine Arena: spec v1 (frozen 2026-11-20)

Each team is one AI model driving one coding-agent harness. Every team gets this spec, the same
starter kit and the same box, and must build an LLM inference server **from scratch**. A grader
measures every engine on the same GPU against vLLM.

This file is the contract. It does not change after the first team starts. Anything a later
version adds will be a new, separately labelled track.

## 1. Target

| | |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0), Hugging Face safetensors layout, at `$MODEL_DIR` |
| GPU | one NVIDIA T4, 16 GB |
| CPU | the engine process tree is pinned to 4 cores |
| Precision | weights and activations in 16-bit float or wider. No quantization below 16 bits, no pruning, no distillation, no changed weights. |
| Context | requests never exceed 4,096 tokens (prompt + output) |
| Base image | Python 3.13, `torch 2.14.1+cu130`, `triton 3.8`, `safetensors`, `tokenizers`, `numpy`, `fastapi`, `uvicorn`, `aiohttp` |

## 2. What "from scratch" means

Allowed: `torch` (all of it, including `torch.compile`, CUDA graphs and
`scaled_dot_product_attention`), `triton`, `flash-attn`, `cupy`, `numba`, your own CUDA/C++/Rust
extensions, `safetensors`, `tokenizers` (with the model's `tokenizer.json`), `numpy`, and any
general-purpose library such as an HTTP server or a JSON parser.

Banned, as a dependency **and** as a source of code:

- any existing LLM inference engine or model runtime: vLLM, nano-vllm, SGLang, llama.cpp,
  llama-cpp-python, Text Generation Inference, TensorRT and TensorRT-LLM, ONNX Runtime,
  CTranslate2, ExLlama, MLC, LMDeploy, LightLLM, mistral.rs, candle, Ollama;
- `transformers`, `optimum`, `accelerate`, `peft`, `bitsandbytes`, `xformers`;
- copying, vendoring or porting source code from any of the above, or from any other inference
  engine, in any language.

You may use ideas that are public knowledge (papers, documentation, blog posts). You may not paste
or translate someone else's implementation. The grader scans imports and installed packages on
every submission, and the final code of every team is audited for copied code before results are
published. A team that breaks this rule is disqualified.

The engine must not use the network at run time. The grader starts it with `HF_HUB_OFFLINE=1` and
no proxy.

## 3. Entry contract

The submission is a git commit. At its root:

| File | Required | Purpose |
|---|---|---|
| `run.sh` | yes | Starts the server in the foreground. Reads `MODEL_DIR`, `HOST`, `PORT` from the environment. Must exit on SIGTERM within 10 s. |
| `requirements.txt` | no | Extra Python packages. Installed with `uv pip install -r` into a virtualenv that already sees the base image. 10-minute limit. |
| `build.sh` | no | Runs once after the install, before `run.sh`. For compiling extensions. 10-minute limit. |

The server is "ready" when `GET /health` returns 200. It must be ready within 300 s of `run.sh`
starting.

## 4. API

All endpoints are plain HTTP on `$HOST:$PORT`.

### `GET /health`
200 once the model is loaded and requests will be served.

### `POST /v1/completions`

Request fields the grader uses:

| Field | Type | Meaning |
|---|---|---|
| `prompt` | string | Raw prompt. Tokenized with the model's tokenizer, no template, no added special tokens. |
| `max_tokens` | int | Upper bound on generated tokens. |
| `temperature` | float | `0` means greedy (argmax). Values above 0 sample. |
| `top_p` | float | Nucleus sampling, used only when `temperature > 0`. |
| `seed` | int | Same seed, same request, same engine state: should give the same sample. |
| `stop` | list of strings | Stop before emitting any of these. The stop string is not included in the output. |
| `stream` | bool | If true, reply as server-sent events. |
| `stream_options.include_usage` | bool | If true, the last event before `[DONE]` carries `usage`. |
| `ignore_eos` | bool | If true, never stop on an end-of-sequence token; generate exactly `max_tokens` tokens. |

Non-streaming response:

```json
{"id": "...", "object": "text_completion", "model": "...",
 "choices": [{"index": 0, "text": "...", "finish_reason": "length"}],
 "usage": {"prompt_tokens": 12, "completion_tokens": 64, "total_tokens": 76}}
```

`finish_reason` is `"length"` when `max_tokens` was reached and `"stop"` when the model emitted an
end-of-sequence token or a stop string matched.

Streaming response: `data: {json}\n\n` events, each with `choices[0].text` holding new text, then
`data: [DONE]\n\n`. The last event that has a `choices` entry carries the `finish_reason`. Each
generated token must be flushed as soon as it is decodable; an engine may not hold back output to
send it in bursts.

Text is decoded with special tokens skipped. Incomplete UTF-8 sequences are held until complete.

### `POST /v1/chat/completions`

Same fields, with `messages` (list of `{role, content}`) in place of `prompt`. The prompt is built
with the model's own chat template (`tokenizer_config.json`), with the generation prompt added.
`usage.prompt_tokens` must equal the token count of that templated prompt.

Non-streaming: `choices[0].message = {"role": "assistant", "content": "..."}`.
Streaming: `choices[0].delta.content` holds new text; the first event may carry
`delta.role = "assistant"`.

### Errors

Malformed JSON, a missing prompt, or a request longer than 4,096 tokens: HTTP 400 with a JSON body.
The server must keep serving afterwards.

## 5. How a submission is graded

Every run draws **new** prompts from a fresh random seed and computes its own reference outputs, so
there is nothing to memorise. The grader source is public; only the seed is hidden.

1. **Scan.** Banned imports or packages fail the run.
2. **Install and build**, within the limits above.
3. **Cold start.** Seconds from launching `run.sh` to the first successful 1-token completion.
   Weights are on local disk.
4. **Conformance.** Every behaviour in section 4 is checked. Any failure fails the run.
5. **Workloads.** Closed loop: a fixed number of clients, each sending its next request as soon as
   the previous one finishes. All requests stream, use `temperature: 0`, and are checked for
   correctness.

| Workload | Clients | Requests | Prompt | Output |
|---|---|---|---|---|
| `single` | 1 | 8 | about 512 tokens, completions API | 128 tokens, `ignore_eos` |
| `c8` | 8 | 48 | 64 to 1,024 tokens, completions API | 128 tokens, `ignore_eos` |
| `c32` | 32 | 160 | 64 to 1,024 tokens, completions API | 128 tokens, `ignore_eos` |
| `chat16` | 16 | 64 | chat API, one shared system prompt of about 300 tokens, different user turns | up to 96 tokens, natural stop |

Any failed request, HTTP error or timeout (120 s per request) fails the run.

### Correctness

The reference is greedy decoding of the same model in float16 by a plain PyTorch implementation
(Hugging Face `transformers`) that the grader runs itself on the same GPU. Prompts are generated at
exact token lengths and the reference only batches prompts of equal length, so it never pads.

For every workload request the grader compares the engine's text with the reference text:

- **Head match:** the engine's text must start with the reference's first 16 tokens. At least 98%
  of requests must pass.
- **Prefix fraction:** matched prefix length divided by reference length, averaged over requests.
  It must be at least the gate in `BASELINES.md`. The gate is set from what vLLM itself scores
  against the same reference on the same box, minus a small margin, and is the same for every team.

Float16 arithmetic differs slightly between implementations, so a late divergence on a near-tie is
tolerated. Garbage, another request's text, or systematic drift is not.

The grader never trusts the engine's own token counts. Tokens credited to a request are the
grader's own tokenization of the returned text, capped at `max_tokens`. When the text matches the
reference in full, `usage.completion_tokens` must also equal the reference token count within 2%.

### Metrics

For each workload: output tokens per second (total credited tokens divided by wall time from first
request sent to last response finished), time to first token p50 and p95, time per output token
p50 and p95, and end-to-end latency p50 and p95.

### Score

For each workload, `ratio = engine tokens/s ÷ vLLM tokens/s`, with vLLM measured on the same box
with the same grader (median of three runs, recorded in `BASELINES.md`).

**Arena score = geometric mean of the four ratios.** A score of 1.00 means "as fast as vLLM".
Cold start, time to first token and the per-workload numbers are published next to it and are not
folded into the score.

A run that fails the scan, conformance, correctness, or any request scores 0.

## 6. Team rules

- One model and one harness per team. The team may run several agents in parallel.
- No human edits to a team's code. Human actions are limited to starting, stopping and restarting
  agents and infrastructure, and every such action is logged.
- Agents may not read another team's code or results.
- Every agent session is logged in full, with tokens and wall-clock time, and published.
- Agents may use the GPU for their own experiments only through `arena gpu`, which queues behind
  the grader so measurements are not disturbed.
- A team's result is its best fully passing `arena submit` (full mode) on its main branch.
