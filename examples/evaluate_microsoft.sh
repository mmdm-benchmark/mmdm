#!/usr/bin/env bash
set -euo pipefail
: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY before running}"
mmdm-eval --provider microsoft --output runs/microsoft \
  --context-limit 32768 --service-overhead 1024 \
  --tokenizer Qwen/Qwen3.5-4B --tokenizer-role proxy \
  --option-cap 255 --concurrency 16 "$@"
