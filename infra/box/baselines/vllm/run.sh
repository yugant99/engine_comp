#!/usr/bin/env bash
# vLLM 0.33.0 as the baseline every engine is compared with.
# Graded by the same grader, on the same cores, as the same Unix user.
#
# One setting differs from a bare `vllm serve`: --generation-config vllm. Without it vLLM applies the
# sampling defaults shipped in the model's generation_config.json, which for Qwen2.5 include a
# repetition penalty of 1.1 that stays on even at temperature 0. The spec defines temperature 0 as
# plain greedy decoding, so the baseline must not apply that penalty.
exec /arena/venvs/vllm/bin/vllm serve "$MODEL_DIR" \
  --host "$HOST" --port "$PORT" \
  --dtype float16 --max-model-len 4096 --gpu-memory-utilization 0.90 \
  --generation-config vllm
