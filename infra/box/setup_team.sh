#!/usr/bin/env bash
# setup_team.sh N_AGENTS: prepare the `team` user's workspace on the box. Runs as root. Safe to run again.
set -euo pipefail
N=${1:-2}
install -m 755 /arena/src/infra/box/arena /usr/local/bin/arena
T=/home/team
as_team() { sudo -u team -H bash -lc "$*"; }

if [ ! -d $T/repo.git ]; then
  as_team "git init -q --bare -b main $T/repo.git"
  rm -rf /tmp/seed && cp -r /arena/src/starter /tmp/seed && cp /arena/src/SPEC.md /tmp/seed/
  [ -f /arena/baselines/BASELINES.md ] && cp /arena/baselines/BASELINES.md /tmp/seed/
  chown -R team:team /tmp/seed
  as_team "cd /tmp/seed && git init -q -b main && git config user.name arena && git config user.email arena@localhost \
           && git add -A && git commit -q -m 'Starter kit' && git push -q $T/repo.git main"
fi
for i in $(seq 1 "$N"); do
  [ -d $T/a$i ] || as_team "git clone -q $T/repo.git $T/a$i && cd $T/a$i && git config user.name a$i && git config user.email a$i@localhost"
done
as_team "mkdir -p $T/logs $T/.config/opencode"
cp /arena/src/infra/box/PROMPT.txt $T/PROMPT.txt
cp /arena/src/infra/box/opencode.json $T/.config/opencode/opencode.json
chown team:team $T/PROMPT.txt $T/.config/opencode/opencode.json
[ -f $T/team.env ] || { echo 'HARNESS_CMD="opencode run --standalone --auto -m opencode-go/glm-5.3 --format json"' > $T/team.env; chown team:team $T/team.env; }
echo "team workspace ready: $(ls -d $T/a* | tr '\n' ' ')"
