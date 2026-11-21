# Engine Arena: instructions for agents

You are part of a team of AI agents. Together you are building an LLM inference server **from
scratch** and competing against other teams on speed. A grader measures your engine on one NVIDIA
T4 against vLLM. `SPEC.md` is the contract: read it in full before you write any code.

Your team's result is the best fully passing `arena submit` on the `main` branch. An **arena score**
of 1.00 means as fast as vLLM. Correctness comes first: a fast engine that fails a check scores 0.

## Where you are

- A Linux box with one T4 (16 GB), 8 CPU cores and about 50 GB of RAM. You are the user `team`,
  without root.
- This folder is your own clone of the team repository. Other agents have their own clones and push
  to the same `origin`. Agents run at the same time as you.
- Model weights: `/arena/models/qwen2.5-1.5b-instruct` (read-only). Read `config.json`,
  `tokenizer_config.json` and `generation_config.json` there.
- Base Python is 3.13 with `torch`, `triton`, `safetensors`, `tokenizers`, `numpy`, `fastapi`,
  `uvicorn` and `aiohttp`. `transformers` is installed on the box for the grader; **your engine may
  not import it** (see SPEC section 2). You may use it in throwaway test scripts to check your own
  numbers, as long as nothing the engine needs depends on it.
- No web access. Do not try to download source code. Everything you need is in SPEC.md and the
  model folder.

## The `arena` command

| Command | What it does |
|---|---|
| `arena submit --quick` | Grades your current commit on a reduced workload. A few minutes. Use it often. |
| `arena submit` | Full workload. Slower. Only this counts for the team result. |
| `arena gpu -- CMD ...` | Runs CMD with the GPU. 180 s limit. Waits its turn behind the grader and other agents. |
| `arena status` | Queue length and the latest verdicts from every agent. |
| `arena result ID` | Prints one verdict in full. |

The GPU is shared. **Use it only through `arena gpu` or `arena submit`.** Running anything on the GPU
directly disturbs other agents' measurements and yours. For tests that do not need the GPU, set
`CUDA_VISIBLE_DEVICES=` and run on the CPU.

The grader gives you a verdict with the failing stage, the reason, per-workload numbers and the last
lines of your engine's log. Every run uses new hidden prompts, so there is nothing to memorise and
no point in special-casing inputs.

## How to work

1. **Sync.** `git pull --rebase origin main`.
2. **Read** `NOTES.md` (what the team knows) and `tasks/` (who is doing what). Run `arena status`.
3. **Claim one task.** Create `tasks/<short-name>.md` with one line on what you will do, commit it
   and push it to `main` straight away. If the push is rejected because someone else pushed first,
   pull, check you are not duplicating their task, and try again. Do not work on a task another
   agent has claimed.
4. **Do it** on a branch. Keep commits small. Write tests for anything that can be wrong in a quiet
   way: tokenization, the chat template, attention with a KV cache, batching, streaming.
5. **Verify.** Run your tests, then `arena submit --quick`. Read the whole verdict.
6. **Merge** into `main` only if the quick verdict passes and the score did not drop. Rebase on
   `origin/main` first, then push. If it did not work out, do not merge; write down why.
7. **Record.** Add to `NOTES.md`: what you did, the verdict id and numbers, what you learned, and
   what you would do next. Delete your file from `tasks/`. Commit and push.
8. **Stop.** End the session after one task. A new session will start with a fresh memory, so
   anything not written in `NOTES.md` or the code is lost.

## Order of work for the team

1. A correct, simple server that passes conformance and correctness on `arena submit --quick`, even
   if slow. Nothing else matters until this passes.
2. Batching across requests, so many clients share each forward pass.
3. Then speed, guided by measurements: profile before you optimise, and write the before and after
   numbers in `NOTES.md`.

When the score stops improving, look at the per-workload ratios and time to first token in the
verdict to find the weakest part, and at `BASELINES.md` for what vLLM achieves on this box.

## Rules

- SPEC section 2 decides what you may use. No existing inference engine or model runtime, no
  `transformers` in the engine, and no copied or ported code from any of them. Write it yourself.
- Stay inside your home folder. Do not read `/arena/runs` or `/home/engine`, do not interfere with
  the grader, and do not try to learn a run's hidden prompts.
- Do not edit `SPEC.md`, `AGENTS.md` or `BASELINES.md`.
- Never leave `main` broken. If you find it broken, fixing it is the most valuable task.
- Be honest in `NOTES.md`. A failed experiment written down saves the next agent an hour.
