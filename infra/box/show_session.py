#!/usr/bin/env python3
"""show_session.py AGENT [LINES]: print the readable part of an agent's latest session log.

opencode writes one JSON event per line. This pulls out what a person wants to read: the agent's
text, the commands it ran and the files it touched.
"""
import glob
import json
import os
import sys

agent = sys.argv[1] if len(sys.argv) > 1 else "a1"
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 40
logs = sorted(glob.glob(f"/home/team/logs/{agent}/*.jsonl"), key=os.path.getmtime)
if not logs:
    sys.exit(f"no sessions yet for {agent}")
path = logs[-1]
out = []
for line in open(path, errors="replace"):
    try:
        ev = json.loads(line)
    except ValueError:
        continue
    part = ev.get("part") or {}
    if part.get("type") == "text" and part.get("text"):
        out.append("SAYS  " + " ".join(part["text"].split())[:300])
    elif part.get("type") == "tool":
        state = part.get("state") or {}
        inp = state.get("input") or {}
        what = inp.get("command") or inp.get("filePath") or inp.get("path") or inp.get("pattern") or ""
        out.append(f"TOOL  {part.get('tool')}: {' '.join(str(what).split())[:200]}")
    elif ev.get("type") == "error":
        out.append("ERROR " + json.dumps(ev.get("error"))[:300])
print(f"{agent}: session {os.path.basename(path)} ({len(logs)} so far)")
print("\n".join(out[-limit:]))
