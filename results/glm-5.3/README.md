# Team GLM 5.3 (opencode): night 1

Model `glm-5.3` through the opencode harness, 2 agents for the first 85 minutes and 6 after that.
Agents started at 04:18 UTC on 2026-11-21 and were halted at 06:32 UTC with the allowance far from spent,
so this is an interim result, not the team's final one.

## Full-mode verdicts (the ones that count)

| Minutes after start | Arena score | 1 client | 8 clients | 32 clients | 16 chat clients | Head match | Cold start |
|---|---|---|---|---|---|---|---|
| 20 | 0.641 | 0.49 | 0.64 | 0.91 | 0.59 | 0.946 | 4.8 s |
| 47 | 0.782 | 0.78 | 0.82 | 0.95 | 0.62 | 0.989 | 10.1 s |
| 73 | 0.787 | 0.78 | 0.80 | 0.95 | 0.65 | 0.946 | 10.1 s |
| 85 | 0.863 | 0.90 | 0.90 | 1.03 | 0.67 | 0.954 | 11.6 s |
| 92 | 0.942 | 0.90 | 0.91 | 1.04 | 0.92 | 0.971 | 12.1 s |
| 115 | 0.999 | 0.94 | 0.93 | 1.13 | 1.00 | 0.950 | 11.3 s |
| 126 | 1.002 | 0.94 | 0.92 | 1.10 | 1.05 | 0.975 | 11.6 s |

Workload columns are tokens per second as a share of vLLM 0.33.0 on the same box (see `BASELINES.md`).
vLLM's own cold start on this box is about 130 s.

## How to read this

- Each row is a single grading run. vLLM's own numbers move 2 to 6% from run to run, so 1.002 means
  "matches vLLM within noise", not "beats vLLM". The official figure will be the median of three runs.
- The baseline is vLLM with default settings on a T4 with 4 CPU cores. It says nothing about larger GPUs.
- 23 verdicts were issued in total, quick and full; all are in `verdicts/`.
- Reported agent spend for the night was about $39 (from opencode's own session logs).
- The team's code and session transcripts are held back until every team has finished, because the
  rules forbid a team from reading another team's code.
