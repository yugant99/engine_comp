#!/usr/bin/env bash
# vLLM 0.33.0 with its default settings, as the baseline every engine is compared with.
# Graded by the same grader, on the same cores, as the same Unix user.
exec /arena/venvs/vllm/bin/vllm serve "$MODEL_DIR" \
  --host "$HOST" --port "$PORT" \
  --dtype float16 --max-model-len 4096 --gpu-memory-utilization 0.90
