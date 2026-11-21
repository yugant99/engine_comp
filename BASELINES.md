# Baselines: vLLM 0.33.0 on this box

vLLM serves the same model on the same T4, pinned to the same 4 cores, graded by the same grader.
Launcher: `infra/box/baselines/vllm/run.sh` (float16, 4,096-token context, repetition penalty off).
An arena score of 1.00 means the geometric mean of your four tokens-per-second ratios equals 1.

## Full mode (`arena submit`), 3 runs

| Workload | Baseline tokens/s (median) | Lowest run | Highest run | Spread |
|---|---|---|---|---|
| 1 client (`single`) | 62.4 | 61.8 | 62.8 | 2% |
| 8 clients (`c8`) | 332.0 | 322.9 | 336.6 | 4% |
| 32 clients (`c32`) | 554.1 | 549.6 | 583.6 | 6% |
| 16 chat clients (`chat16`) | 409.9 | 406.1 | 420.4 | 3% |

| Correctness gate | Your run needs at least | vLLM's own runs |
|---|---|---|
| Head-match rate | 0.907 | 0.957, 0.971, 0.968 |
| Prefix fraction | 0.864 | 0.921, 0.914, 0.918 |

## Quick mode (`arena submit --quick`), 3 runs

| Workload | Baseline tokens/s (median) | Lowest run | Highest run | Spread |
|---|---|---|---|---|
| 1 client (`single`) | 61.7 | 61.5 | 61.7 | 0% |
| 8 clients (`c8`) | 326.1 | 314.1 | 365.8 | 16% |
| 32 clients (`c32`) | 595.9 | 591.3 | 634.5 | 7% |
| 16 chat clients (`chat16`) | 337.0 | 325.9 | 341.9 | 5% |

| Correctness gate | Your run needs at least | vLLM's own runs |
|---|---|---|
| Head-match rate | 0.874 | 0.985, 0.924, 0.970 |
| Prefix fraction | 0.846 | 0.920, 0.896, 0.897 |

The spread column is run-to-run noise in vLLM's own numbers with nothing changed. A difference
smaller than that between two of your submissions is noise, not progress.
