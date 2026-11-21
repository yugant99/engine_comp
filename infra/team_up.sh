#!/usr/bin/env bash
# team_up.sh: put an opencode team on the grading box and start it. Run this yourself in a terminal.
#
#   ./infra/team_up.sh install      install opencode on the box for the `team` user
#   ./infra/team_up.sh login        sign opencode in on the box (you paste your OpenCode Go key)
#   ./infra/team_up.sh start N      start the grading daemon and N agents (default 2)
#   ./infra/team_up.sh watch        sessions so far, latest verdicts, GPU and memory
#   ./infra/team_up.sh tail N       what agent N is doing right now: its words, commands and files (default 1)
#   ./infra/team_up.sh stop         stop every agent after its current step (the grader keeps running)
#
# The agents run on the box as an unprivileged user. They never run on this Mac and never see
# anything on it.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd -P)"
BOX="$HERE/box.sh"
OPENCODE_VERSION=2.0.27
AS_TEAM='setpriv --reuid team --regid team --init-groups env -i HOME=/home/team USER=team TERM=xterm PATH=/home/team/.opencode/bin:/usr/local/bin:/usr/bin:/bin'

case "${1:-}" in
  install)
    "$BOX" sh "cd /home/team && $AS_TEAM bash -c 'curl -fsSL https://opencode.ai/v2/install | bash -s -- --version $OPENCODE_VERSION' | tail -3; $AS_TEAM opencode --version"
    ;;
  login)
    echo "Paste a key from the OpenCode Console when asked. A separate key just for this box is easiest to revoke later."
    "$BOX" tty "cd /home/team && $AS_TEAM opencode auth login"
    ;;
  start)
    N=${2:-2}
    "$BOX" push
    "$BOX" sh "chmod -R a+rX /arena/src && bash /arena/src/infra/box/setup_team.sh $N
      pgrep -f graderd.py >/dev/null || { cd /arena/src && setsid nohup python3 infra/box/graderd.py >> /arena/logs/graderd.log 2>&1 < /dev/null & }
      rm -f /home/team/STOP
      for i in \$(seq 1 $N); do
        pgrep -u team -f \"run_agent.sh \$i\" >/dev/null && { echo \"agent a\$i already running\"; continue; }
        cd /home/team && setsid $AS_TEAM nohup bash /arena/src/infra/box/run_agent.sh \$i >> /home/team/logs/runner_a\$i.log 2>&1 < /dev/null &
        echo \"started agent a\$i\"
      done
      sleep 2; pgrep -fa 'graderd.py|run_agent.sh' | cut -c1-80"
    ;;
  watch)
    "$BOX" sh "echo '== sessions'; tail -8 /home/team/logs/sessions.log 2>/dev/null
      echo '== verdicts'; setpriv --reuid team --regid team --init-groups arena status
      echo '== box'; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; free -g | sed -n 2p; uptime
      echo '== notes (last lines)'; git -c safe.directory='*' --git-dir=/home/team/repo.git show main:NOTES.md 2>/dev/null | tail -15"
    ;;
  tail)
    "$BOX" sh "python3 /arena/src/infra/box/show_session.py a${2:-1} ${3:-40}"
    ;;
  stop)
    "$BOX" sh "touch /home/team/STOP; chown team:team /home/team/STOP; echo 'Agents will stop after their current session. To stop them now: pkill -u team'"
    ;;
  *) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
