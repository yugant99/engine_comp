#!/usr/bin/env bash
# halt_team.sh: stop every agent now, without losing unfinished work. Runs on the box as root.
# Each agent's uncommitted changes are committed and pushed to refs/wip/<agent>-<time> in the team
# repository, which the Mac backs up. Nothing is merged; main is left exactly as the agents left it.
set -u
touch /home/team/STOP && chown team:team /home/team/STOP
pkill -u team -f run_agent.sh 2>/dev/null
pkill -u team -f "opencode" 2>/dev/null
sleep 3
pkill -KILL -u team 2>/dev/null
ts=$(date -u +%Y%m%dT%H%M%SZ)
for d in /home/team/a*; do
  a=$(basename "$d")
  setpriv --reuid team --regid team --init-groups env -i HOME=/home/team PATH=/usr/local/bin:/usr/bin:/bin bash -c "
    cd $d || exit 0
    rm -f .git/index.lock
    git add -A >/dev/null 2>&1
    git -c user.name=$a -c user.email=$a@localhost commit -q -m 'Work in progress saved at shutdown ($ts)' >/dev/null 2>&1
    git push -q origin HEAD:refs/wip/$a-$ts 2>/dev/null && echo \"$a: saved \$(git rev-parse --short HEAD) on branch \$(git rev-parse --abbrev-ref HEAD) as refs/wip/$a-$ts\"
  "
done
echo "agents running now: $(pgrep -u team | wc -l)"
echo "main is at: $(git -c safe.directory='*' --git-dir=/home/team/repo.git log --oneline -1 main)"
