# Engine Arena

Can AI coding agents build an LLM inference engine from scratch, and how close do they get to vLLM?

Each team is one AI model driving one coding-agent harness. Every team gets the same spec, the same
starter kit and the same box, and builds an OpenAI-compatible inference server for
`Qwen2.5-1.5B-Instruct` with no existing inference engine and no copied code. A grader runs every
submission on the same NVIDIA T4, checks its output against a reference, and scores its speed
against vLLM measured on that same box.

**Status:** the spec, grader and grading box are working and the vLLM baseline is measured
(see [`BASELINES.md`](BASELINES.md)). No team has produced an engine yet.

| | |
|---|---|
| [`SPEC.md`](SPEC.md) | The contract: target, rules, API, how grading and scoring work |
| [`BASELINES.md`](BASELINES.md) | vLLM's measured speed on the box: the 1.00 mark, and the correctness gates |
| `grader/` | The grader (public; only each run's random seed is hidden) |
| `infra/` | The grading box: provisioning, the grading daemon, the agents' `arena` command |
| `starter/` | What every team's repository starts with |
