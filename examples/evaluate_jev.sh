#!/usr/bin/env bash
set -euo pipefail
: "${JEV_API_KEY:?Set JEV_API_KEY before running}"
mmdm-eval --provider jev --output runs/jev \
  --context-limit 32768 --service-overhead 1024 \
  --tokenizer Qwen/Qwen3.5-4B --tokenizer-role proxy \
  --option-cap 255 --concurrency 16 "$@"
