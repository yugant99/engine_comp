#!/usr/bin/env bash
# box.sh: drive the Colab grading box from the Mac with Google's Colab CLI.
#
#   ./infra/box.sh up              get a high-memory T4 (starts using compute units)
#   ./infra/box.sh sh 'CMD'        run one shell command on the box as root
#   ./infra/box.sh long NAME 'CMD' run a long command detached; follow its log until it exits
#   ./infra/box.sh push            upload grader/, infra/box/ and starter/ to /arena/src on the box
#   ./infra/box.sh pull            bring results, logs and every team's git history back to the Mac
#   ./infra/box.sh login           open an interactive root shell on the box
#   ./infra/box.sh tty 'CMD'       run one interactive command on the box (keeps your terminal attached)
#   ./infra/box.sh status          sessions and compute units left
#   ./infra/box.sh down            pull, then release the box (stops using compute units)
#
# Works with the macOS default bash 3.2.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

SESSION=${ARENA_SESSION:-arena}
HERE="$(cd "$(dirname "$0")" && pwd -P)"
ROOT="$(dirname "$HERE")"
TMP="${TMPDIR:-/tmp}"

# Colab allows ONE SSH connection to a box at a time, so every call takes a lock on the Mac first.
LOCK="$TMP/arena_ssh_$SESSION.lock"
cssh() {
  local waited=0
  until mkdir "$LOCK" 2>/dev/null; do
    # A lock older than 15 minutes belongs to a dead process.
    if [ -n "$(find "$LOCK" -maxdepth 0 -mmin +15 2>/dev/null)" ]; then rmdir "$LOCK" 2>/dev/null || true; continue; fi
    sleep 2; waited=$((waited + 2))
    [ "$waited" -lt 600 ] || { echo "box.sh: another connection has held the box for 10 minutes" >&2; return 1; }
  done
  local rc=0
  colab ssh -s "$SESSION" "$@" || rc=$?
  rmdir "$LOCK" 2>/dev/null || true
  return $rc
}

# Run a shell command on the box as root over SSH (the script goes in on stdin).
# The notebook channel (`colab exec`) loses its connection too often to rely on.
# Never leave a process running in the foreground of this connection: start daemons with
# `setsid ... > log 2>&1 < /dev/null &`, or the connection (and the lock) never closes.
rsh() { printf '%s\n' "$*" | cssh -- bash -s; }

session_exists() { colab sessions 2>/dev/null | grep -q "^\[$SESSION\]"; }

cmd_up() {
  if session_exists; then echo "Session '$SESSION' already exists."; else colab new -s "$SESSION" --gpu T4 --high-mem; fi
  rsh 'nproc; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'
}

# Long commands run detached on the box; the Mac polls the log. Colab drops long quiet connections.
cmd_long() {
  local name=$1 cmd=$2
  local log="/arena/logs/$name.log" done="/arena/logs/$name.exit"
  printf '%s\n' "$cmd" | cssh -- bash -c "cat > /tmp/long_$name.sh"
  rsh "mkdir -p /arena/logs; rm -f $log $done; setsid nohup bash -c 'bash /tmp/long_$name.sh; echo \$? > $done' > $log 2>&1 < /dev/null &" >/dev/null
  local seen=0 out status code
  while :; do
    sleep "${POLL:-10}"
    out=$(rsh "d=\$(cat $done 2>/dev/null || echo running); n=\$(wc -l < $log 2>/dev/null || echo 0); tail -n +$((seen + 1)) $log | head -n \$((n - $seen)); echo \"@@ \$n \$d\"" 2>/dev/null) || continue
    printf '%s\n' "$out" | grep -v '^@@ ' || true
    status=$(printf '%s\n' "$out" | grep '^@@ ' | tail -1)
    [[ -n "$status" ]] || continue
    seen=$(echo "$status" | awk '{print $2}')
    code=$(echo "$status" | awk '{print $3}')
    case "$code" in
      running|"") ;;
      0) return 0 ;;
      *) echo "'$name' failed with exit code $code. Log on the box: $log"; return 1 ;;
    esac
  done
}

cmd_push() {
  local tarball="$TMP/arena_src.tgz"
  ( cd "$ROOT" && COPYFILE_DISABLE=1 tar --exclude '__pycache__' --exclude '.pytest_cache' -czf "$tarball" SPEC.md grader infra/box starter )
  cssh -- bash -c 'mkdir -p /arena/src && tar -xzf - -C /arena/src && ls /arena/src' < "$tarball"
}

cmd_pull() {
  mkdir -p "$ROOT/results/box"
  cssh -- bash -c 'cd / && tar -czf - --ignore-failed-read srv/arena/results srv/arena/scoreboard.jsonl arena/logs home/*/repo.git home/*/logs 2>/dev/null' > "$TMP/arena_pull.tgz"
  tar -xzf "$TMP/arena_pull.tgz" -C "$ROOT/results/box"
  echo "pulled into $ROOT/results/box at $(date '+%H:%M:%S')"
}

cmd_status() { colab sessions; colab usage; }

cmd_down() {
  cmd_pull || { echo "Pull failed, so the box was NOT released."; exit 1; }
  colab stop -s "$SESSION"
  cmd_status
}

case "${1:-}" in
  up)     cmd_up ;;
  sh)     shift; rsh "$@" ;;
  long)   shift; cmd_long "$1" "$2" ;;
  push)   cmd_push ;;
  pull)   cmd_pull ;;
  login)  cssh ;;
  tty)    shift; cssh -- bash -c "$*" ;;
  status) cmd_status ;;
  down)   cmd_down ;;
  *)      sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
