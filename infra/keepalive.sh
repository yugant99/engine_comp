#!/usr/bin/env bash
# keepalive.sh: keep the Colab box alive and back it up. Colab shuts a box down after roughly
# 20 minutes without a connection, and everything on it is lost.
#
#   caffeinate -i ./infra/keepalive.sh      (caffeinate stops the Mac from sleeping while this runs)
#
# Every 2 minutes it touches the box. Every 10 minutes it copies results, logs and every team's
# git history to results/box/ on the Mac. It stops with a loud message if the box disappears.
set -u
HERE="$(cd "$(dirname "$0")" && pwd -P)"
BOX="$HERE/box.sh"
n=0; misses=0
while :; do
  if out=$("$BOX" sh 'echo "$(date -u +%H:%M:%S) up load=$(cut -d" " -f1 /proc/loadavg) agents=$(pgrep -u team -f run_agent.sh | wc -l) graderd=$(pgrep -f graderd.py | wc -l)"' 2>&1 | grep ' up load=' | tail -1) && [ -n "$out" ]; then
    misses=0; echo "box: $out"
    if [ $((n % 5)) -eq 4 ]; then "$BOX" pull 2>&1 | tail -1; fi
  else
    misses=$((misses + 1)); echo "$(date '+%H:%M:%S') no answer from the box ($misses): $out"
    if [ "$misses" -ge 4 ]; then
      echo "THE BOX IS GONE. Last backup is in results/box/. Rebuild with: ./infra/box.sh up, then provision."
      exit 1
    fi
  fi
  n=$((n + 1)); sleep 120
done
