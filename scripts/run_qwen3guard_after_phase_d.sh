#!/usr/bin/env bash
# Score the held-out FineHarm test split with Qwen3Guard-Stream-4B.
# Run this after scripts/run_layer11.sh has written artifacts/full/protocol_lock.json.
set -euo pipefail
source "$(dirname "$0")/env.sh"
LOG=artifacts/full/logs/qwen3guard_stream.log
mkdir -p artifacts/full/logs
exec >>"$LOG" 2>&1

if [[ ! -f artifacts/full/protocol_lock.json ]]; then
  echo "artifacts/full/protocol_lock.json is missing. Finish the FineHarm test first." >&2
  exit 1
fi

echo "=== download Qwen/Qwen3Guard-Stream-4B $(date -Is) ==="
"$PY" - <<'PY'
import os
from huggingface_hub import snapshot_download
path = snapshot_download("Qwen/Qwen3Guard-Stream-4B", cache_dir=os.environ["HF_HOME"])
print(f"[download] {path}", flush=True)
PY

echo "=== evaluate held-out test $(date -Is) ==="
"$PY" -m kv_safety.qwen3guard_stream --config configs/full.yaml
echo "=== qwen3guard stream done $(date -Is) ==="
