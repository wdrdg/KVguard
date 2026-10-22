#!/usr/bin/env bash
# Full pipeline for one config: environment, prepare, extract, train, evaluate, benchmark, report.
set -euo pipefail
CONFIG="${1:?config path required}"
source "$(dirname "$0")/env.sh"
echo "[pipeline] config=$CONFIG start=$(date -Is)"
"$PY" -m kv_safety.environment --config "$CONFIG"
"$PY" -m kv_safety.data.prepare --config "$CONFIG"
"$PY" -m kv_safety.extract --config "$CONFIG"
"$PY" -m kv_safety.train --config "$CONFIG" --all-features
"$PY" -m kv_safety.evaluate --config "$CONFIG" --all-features
"$PY" -m kv_safety.benchmark --config "$CONFIG"
"$PY" -m kv_safety.report --config "$CONFIG"
echo "[pipeline] done=$(date -Is)"
