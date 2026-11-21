#!/usr/bin/env python3
"""report.py: every verdict with its numbers, and each agent's token use from its session logs."""
import glob
import json

rows = [json.loads(x) for x in open("/srv/arena/scoreboard.jsonl").read().splitlines()]
for s in rows:
    try:
        r = json.load(open("/srv/arena/results/%s.json" % s["id"]))
    except OSError:
        continue
    print(s["time"], s["id"], s.get("agent"), r.get("mode"), r.get("status"), "score", round(r.get("score") or 0, 3),
          "cold_start", r.get("cold_start_s"), "| fail:", r.get("fail_stage"), (r.get("fail_reason") or "")[:160])
    g = r.get("gates") or {}
    if g.get("head_match", {}).get("value") is not None:
        print("    head match %.3f (needs %s) | prefix %.3f (needs %s)" % (
            g["head_match"]["value"], g["head_match"]["threshold"],
            g["prefix_fraction"]["value"], g["prefix_fraction"]["threshold"]))
    for k, w in (r.get("workloads") or {}).items():
        if isinstance(w, dict) and w.get("tok_s") is not None:
            print("    %-7s %7.1f tok/s  ratio %s  ttft p50 %.0f ms  p95 %.0f ms" % (
                k, w["tok_s"], "%.2f" % w["ratio"] if w.get("ratio") else "-", 1000 * w["ttft_p50"], 1000 * w["ttft_p95"]))

total = {}
for f in glob.glob("/home/team/logs/a*/*.jsonl"):
    agent = f.split("/")[4]
    d = total.setdefault(agent, {"sessions": 0, "steps": 0, "input": 0, "output": 0, "reasoning": 0, "cache_read": 0, "cost": 0.0})
    d["sessions"] += 1
    for line in open(f, errors="replace"):
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        part = ev.get("part") or {}
        t = part.get("tokens")
        if isinstance(t, dict):
            d["steps"] += 1
            d["input"] += t.get("input", 0)
            d["output"] += t.get("output", 0)
            d["reasoning"] += t.get("reasoning", 0)
            d["cache_read"] += (t.get("cache") or {}).get("read", 0)
            d["cost"] += part.get("cost") or 0
for agent in sorted(total):
    d = total[agent]
    print("usage %s: %d sessions, %d steps, input %d, cached %d, output %d, reasoning %d, cost reported $%.2f" % (
        agent, d["sessions"], d["steps"], d["input"], d["cache_read"], d["output"], d["reasoning"], d["cost"]))
