#!/usr/bin/env bash
# Prompt heads on layer-11 V. Separate from the locked response-head protocol.
set -euo pipefail
source "$(dirname "$0")/env.sh"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
mkdir -p artifacts/full/prompt_head/logs
LOG=artifacts/full/prompt_head/logs/prompt_head.log
exec >>"$LOG" 2>&1
echo "=== prompt head $(date -Is) ==="
"$PY" -m kv_safety.prompt_head --config configs/full.yaml
echo "=== prompt head done $(date -Is) ==="
