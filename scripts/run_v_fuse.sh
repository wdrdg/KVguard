#!/usr/bin/env bash
# v_fuse: response v_mean with fused prompt-or-response labels. Separate from the locked protocol.
set -euo pipefail
source "$(dirname "$0")/env.sh"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
mkdir -p artifacts/full/v_fuse/logs
LOG=artifacts/full/v_fuse/logs/v_fuse.log
exec >>"$LOG" 2>&1
echo "=== v_fuse $(date -Is) ==="
"$PY" -m kv_safety.v_fuse --config configs/full.yaml
echo "=== v_fuse done $(date -Is) ==="
