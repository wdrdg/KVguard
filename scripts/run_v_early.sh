#!/usr/bin/env bash
# v_early: same v_mean features as v_fuse, with early labels only on harmful answers.
set -euo pipefail
source "$(dirname "$0")/env.sh"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
mkdir -p artifacts/full/v_early/logs
LOG=artifacts/full/v_early/logs/v_early.log
exec >>"$LOG" 2>&1
echo "=== v_early $(date -Is) ==="
"$PY" -m kv_safety.v_early --config configs/full.yaml
echo "=== v_early done $(date -Is) ==="
