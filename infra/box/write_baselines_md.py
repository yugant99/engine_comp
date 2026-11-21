#!/usr/bin/env python3
"""write_baselines_md.py BASELINES_JSON OUT_MD: the human-readable record of the vLLM baseline."""
import json
import sys

b = json.load(open(sys.argv[1]))
modes = b.get("modes", b)
names = [("single", "1 client"), ("c8", "8 clients"), ("c32", "32 clients"), ("chat16", "16 chat clients")]
out = ["# Baselines: vLLM 0.33.0 on this box", "",
       "vLLM serves the same model on the same T4, pinned to the same 4 cores, graded by the same grader.",
       "Launcher: `infra/box/baselines/vllm/run.sh` (float16, 4,096-token context, repetition penalty off).",
       "An arena score of 1.00 means the geometric mean of your four tokens-per-second ratios equals 1.", ""]
for mode in ("full", "quick"):
    if mode not in modes:
        continue
    m = modes[mode]
    runs = m["source"]["runs"]
    out += [f"## {mode.capitalize()} mode (`arena submit{' --quick' if mode == 'quick' else ''}`), {len(runs)} runs", "",
            "| Workload | Baseline tokens/s (median) | Lowest run | Highest run | Spread |", "|---|---|---|---|---|"]
    for key, label in names:
        vals = [r["tok_s"][key] for r in runs]
        spread = (max(vals) - min(vals)) / m["tok_s"][key] * 100
        out.append(f"| {label} (`{key}`) | {m['tok_s'][key]:.1f} | {min(vals):.1f} | {max(vals):.1f} | {spread:.0f}% |")
    hm = [r["head_match_rate"] for r in runs]
    pf = [r["prefix_fraction"] for r in runs]
    out += ["", "| Correctness gate | Your run needs at least | vLLM's own runs |", "|---|---|---|",
            f"| Head-match rate | {m['head_match_gate']:.3f} | {', '.join(f'{v:.3f}' for v in hm)} |",
            f"| Prefix fraction | {m['prefix_fraction_gate']:.3f} | {', '.join(f'{v:.3f}' for v in pf)} |", ""]
out += ["The spread column is run-to-run noise in vLLM's own numbers with nothing changed. A difference",
        "smaller than that between two of your submissions is noise, not progress.", ""]
open(sys.argv[2], "w").write("\n".join(out))
print("\n".join(out))
