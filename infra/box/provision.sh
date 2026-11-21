#!/usr/bin/env bash
# provision.sh: turn a fresh Colab T4 box into the Engine Arena grading box. Safe to run again.
# Runs on the box as root.
set -euo pipefail

MODEL=Qwen/Qwen2.5-1.5B-Instruct
MODEL_DIR=/arena/models/qwen2.5-1.5b-instruct

mkdir -p /arena/{models,logs,venvs,src,runs,baselines} /srv/arena/{queue,results}

# Two unprivileged users: `team` runs the coding agents, `engine` runs submitted engines.
# Neither can read /arena/runs (grader working files) or the other's home.
id team   >/dev/null 2>&1 || useradd -m -s /bin/bash team
id engine >/dev/null 2>&1 || useradd -m -s /bin/bash engine
chmod 750 /home/team /home/engine
chmod 700 /arena/runs
chown team:team /srv/arena/queue && chmod 755 /srv/arena/queue      # team writes requests, root reads
chmod 755 /srv/arena/results                                        # root writes results, team reads
touch /srv/arena/gpu.lock && chmod 666 /srv/arena/gpu.lock

# The measured vLLM baseline travels with the repository, so a rebuilt box grades against the same
# numbers. (A new box is a different physical machine: re-check with one vLLM run before trusting it.)
if [ -f /arena/src/baselines/baselines.json ] && [ ! -f /arena/baselines/baselines.json ]; then
  cp /arena/src/baselines/baselines.json /arena/src/BASELINES.md /arena/baselines/ 2>/dev/null || true
  chmod a+r /arena/baselines/* 2>/dev/null || true
fi

# Colab exposes the GPU driver libraries only through root's LD_LIBRARY_PATH. Register them
# system-wide so the unprivileged users can use the GPU too.
echo /usr/lib64-nvidia > /etc/ld.so.conf.d/arena-nvidia.conf && ldconfig 2>/dev/null || true

# Record the base image before anything else is installed, so later changes can be detected.
if [ ! -f /arena/base_freeze.txt ]; then
  pip freeze 2>/dev/null | sort > /arena/base_freeze.txt
  python3 -c "import sys, torch, triton, tokenizers, safetensors, numpy, fastapi, uvicorn, aiohttp
print('python', sys.version.split()[0])
for m in (torch, triton, tokenizers, safetensors, numpy, fastapi, uvicorn, aiohttp): print(m.__name__, m.__version__)
print('cuda', torch.version.cuda, torch.cuda.get_device_name(0))" > /arena/base_image.txt
fi
cat /arena/base_image.txt

# Model weights.
if [ ! -f "$MODEL_DIR/model.safetensors" ]; then
  hf download "$MODEL" --local-dir "$MODEL_DIR" > /arena/logs/model_dl.log 2>&1
fi
chmod -R a+rX /arena/models
du -sh "$MODEL_DIR"

# Text corpus that prompts are cut from (WikiText-103 test split, raw).
if [ ! -s /arena/corpus.txt ]; then
  hf download Salesforce/wikitext --repo-type dataset --include "wikitext-103-raw-v1/test-*.parquet" \
     --local-dir /arena/wikitext > /arena/logs/corpus_dl.log 2>&1
  python3 - <<'EOF'
import glob, pandas as pd
rows = []
for f in sorted(glob.glob('/arena/wikitext/wikitext-103-raw-v1/test-*.parquet')):
    rows += [t.strip() for t in pd.read_parquet(f)['text'] if len(t.strip()) > 300 and not t.strip().startswith('=')]
open('/arena/corpus.txt', 'w').write('\n\n'.join(rows))
print('corpus paragraphs', len(rows), 'chars', sum(map(len, rows)))
EOF
fi
chmod a+r /arena/corpus.txt

# vLLM, the baseline, in its own virtualenv. Colab sets UV_SYSTEM_PYTHON, which would make uv install
# into the base image, so name the target interpreter explicitly and unset it.
if [ ! -x /arena/venvs/vllm/bin/vllm ]; then
  env -u UV_SYSTEM_PYTHON uv venv /arena/venvs/vllm --python 3.12 > /arena/logs/vllm_install.log 2>&1
  env -u UV_SYSTEM_PYTHON uv pip install --python /arena/venvs/vllm/bin/python vllm==0.33.0 --torch-backend=auto \
     >> /arena/logs/vllm_install.log 2>&1
fi
/arena/venvs/vllm/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"
# Only root and the `baseline` user may read vLLM. Otherwise a team could launch it from run.sh, or
# copy from its source.
id baseline >/dev/null 2>&1 || useradd -m -s /bin/bash baseline
chgrp -R baseline /arena/venvs && chmod -R g+rX,o-rwx /arena/venvs

# The base image must be untouched.
if pip freeze 2>/dev/null | sort | diff -q - /arena/base_freeze.txt >/dev/null; then
  echo "base image unchanged"
else
  echo "BASE IMAGE CHANGED:"; pip freeze 2>/dev/null | sort | diff - /arena/base_freeze.txt | head -20; exit 1
fi
echo "provision done"
