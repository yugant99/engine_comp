"""Build or update the baselines JSON from calibration runs of vLLM (SPEC section 5: median of three).

    python -m grader.make_baselines --margin 0.01 --out baselines.json run1.json run2.json run3.json

All runs must be passing calibration runs of one mode, and every correctness gate must have passed
in each of them (calibration runs report gates without enforcing them; the baseline must not be
set from a run that would have failed them). tok_s per workload is the median over the runs;
prefix_fraction_gate is the median of the runs' pooled prefix fraction minus --margin.

One file holds both modes, because graderd.py passes the same file to quick and full runs. If
--out exists, only the entry for the runs' mode is replaced; the other mode is kept:

    {"spec_version": "1",
     "modes": {"full":  {"tok_s": {"single": ..., "c8": ..., "c32": ..., "chat16": ...},
                         "prefix_fraction_gate": 0.97, "source": {...}},
               "quick": {...}}}
"""

import argparse
import json
import os
import statistics
import sys

from . import SPEC_VERSION
from .prompts import WORKLOAD_ORDER


def build(results, margin):
    """(mode, entry) from a list of calibration result dicts."""
    modes = {r.get("mode") for r in results}
    if len(modes) != 1:
        raise ValueError("runs mix modes: %s" % sorted(map(str, modes)))
    for r in results:
        who = "run with seed %s" % r.get("seed")
        if r.get("status") != "pass":
            raise ValueError("%s did not pass (%s: %s)" % (who, r.get("fail_stage"), r.get("fail_reason")))
        if not r.get("calibration"):
            raise ValueError("%s is not a calibration run" % who)
        gates = r.get("gates") or {}
        if gates.get("failed") is None:
            raise ValueError("%s has no correctness gates" % who)
        if gates["failed"]:
            raise ValueError("%s failed the gates %s; vLLM itself must pass them" % (who, gates["failed"]))
    tok_s = {w: statistics.median(r["workloads"][w]["output_tok_s"] for r in results) for w in WORKLOAD_ORDER}
    pf = statistics.median(r["gates"]["prefix_fraction"]["value"] for r in results)
    entry = {
        "tok_s": tok_s,
        "prefix_fraction_gate": max(0.0, pf - margin),
        "source": {
            "runs": [{"seed": r.get("seed"), "commit": r.get("commit"), "finished_at": r.get("finished_at"),
                      "baseline_engine": r.get("baseline_engine"),
                      "tok_s": {w: r["workloads"][w]["output_tok_s"] for w in WORKLOAD_ORDER},
                      "prefix_fraction": r["gates"]["prefix_fraction"]["value"],
                      "head_match_rate": r["gates"]["head_match"]["value"],
                      "min_events_per_credited_token":
                          r["gates"].get("streaming_granularity", {}).get("min_events_per_credited_token")}
                     for r in results],
            "median_prefix_fraction": pf,
            "margin": margin,
        },
    }
    return modes.pop(), entry


def main(argv=None):
    ap = argparse.ArgumentParser(description="Baselines JSON from vLLM calibration runs.")
    ap.add_argument("results", nargs="+")
    ap.add_argument("--margin", type=float, required=True, help="subtracted from the median prefix fraction")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    results = []
    for path in args.results:
        with open(path, encoding="utf-8") as f:
            results.append(json.load(f))
    if not all(r.get("baseline_engine") for r in results):
        print("warning: not every run was graded with --baseline-engine", file=sys.stderr)
    mode, entry = build(results, args.margin)
    data = {"spec_version": SPEC_VERSION, "modes": {}}
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            data = json.load(f)
        if str(data.get("spec_version")) != SPEC_VERSION:
            sys.exit("%s is for spec %r, not %r" % (args.out, data.get("spec_version"), SPEC_VERSION))
        data.setdefault("modes", {})
    kept = sorted(m for m in data["modes"] if m != mode)
    data["modes"][mode] = entry
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, args.out)
    print("wrote %s mode to %s (other modes kept: %s)" % (mode, args.out, ", ".join(kept) or "none"))
    print(json.dumps({"tok_s": entry["tok_s"], "prefix_fraction_gate": entry["prefix_fraction_gate"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
