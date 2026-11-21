#!/usr/bin/env bash
# run_agent.sh N: one agent's loop. Runs on the box as the `team` user, inside /home/team/aN.
# Each pass is a fresh agent session; the team's memory is the repository itself (NOTES.md, tasks/).
# Stop every agent by creating /home/team/STOP.
set -u
N=$1
source /home/team/team.env                 # HARNESS_CMD, e.g.: opencode run --standalone --auto -m opencode-go/glm-5.3 --format json
export ARENA_AGENT="a$N" PATH="/home/team/.opencode/bin:/home/team/.local/bin:$PATH"
cd "/home/team/a$N" || exit 1
mkdir -p "/home/team/logs/a$N"
while [ ! -f /home/team/STOP ]; do
  ts=$(date -u +%Y%m%dT%H%M%SZ)
  git fetch -q origin && { git checkout -q main 2>/dev/null; git reset -q --hard origin/main; git clean -qfd; }
  start=$(date +%s)
  timeout -k 30 "${SESSION_LIMIT_S:-5400}" $HARNESS_CMD "$(cat /home/team/PROMPT.txt)" \
      > "/home/team/logs/a$N/$ts.jsonl" 2> "/home/team/logs/a$N/$ts.err" < /dev/null
  rc=$?
  echo "$ts agent=a$N exit=$rc seconds=$(( $(date +%s) - start ))" >> /home/team/logs/sessions.log
  [ $rc -ne 0 ] && sleep 60                # back off after a failure (quota window, network)
  sleep 5
done
