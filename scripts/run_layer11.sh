#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
LOG=artifacts/full/logs/full_train.log
mkdir -p artifacts/full/logs
exec >>"$LOG" 2>&1

echo "=== layer 11 run $(date -Is) ==="
df -h .

echo "=== extract layer 11 $(date -Is) ==="
"$PY" -m kv_safety.extract --config configs/full.yaml --layers 11
echo "=== train layer 11 $(date -Is) ==="
"$PY" -m kv_safety.train --config configs/full.yaml --all-features --layers 11
echo "=== drop layer 11 features $(date -Is) ==="
rm -rf artifacts/full/features
df -h .

echo "=== calibrate $(date -Is) ==="
"$PY" -m kv_safety.calibrate --config configs/full.yaml --split calibration --all-features
echo "=== evaluate dev $(date -Is) ==="
"$PY" -m kv_safety.evaluate --config configs/full.yaml --split dev --all-features

echo "=== protocol lock $(date -Is) ==="
"$PY" - <<'PY'
import hashlib
import subprocess
from kv_safety.config import REPO_ROOT, artifacts_dir, load_config, write_json

cfg = load_config("configs/full.yaml")
out = artifacts_dir(cfg)
manifest = out / "split_manifest.jsonl"
code_files = [
    "src/kv_safety/evaluate.py",
    "src/kv_safety/calibrate.py",
    "src/kv_safety/replay.py",
    "src/kv_safety/metrics.py",
    "src/kv_safety/train.py",
    "src/kv_safety/data/labels.py",
    "src/kv_safety/data/splits.py",
    "configs/full.yaml",
]
code = {rel: hashlib.sha256((REPO_ROOT / rel).read_bytes()).hexdigest() for rel in code_files}
try:
    git = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
except Exception:
    git = None
thr = list((out / "checkpoints").rglob("threshold.json"))
if not (out / "checkpoints" / "train_summary.json").exists() or len(thr) < 9:
    raise SystemExit(f"refusing to lock: thresholds={len(thr)}")
lock = {
    "purpose": "Phase D held-out FineHarm test on pilot-selected block 11. Thresholds and the layer were fixed before test scoring.",
    "allow_test_eval": True,
    "run_name": cfg["run_name"],
    "artifacts_dir": cfg["artifacts_dir"],
    "split_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    "code_sha256": code,
    "git_head": git,
    "model_id": cfg["model"]["id"],
    "deployment_layer": cfg["model"]["layer_index"],
    "seeds": cfg["train"]["seeds"],
    "target_fpr": cfg["evaluate"]["target_fpr"],
    "decision": "first logit strictly greater than the calibration threshold",
    "layer_selection": "pilot dev selected block 11 for h_mean, v_mean, kv_mean; test does not change the layer",
    "n_thresholds": len(thr),
}
write_json(out / "protocol_lock.json", lock)
print(f"[lock] wrote protocol_lock.json thresholds={len(thr)}")
PY

echo "=== evaluate test $(date -Is) ==="
"$PY" -m kv_safety.evaluate --config configs/full.yaml --split test \
  --protocol-lock artifacts/full/protocol_lock.json --all-features
echo "=== workshop benchmark $(date -Is) ==="
"$PY" -m kv_safety.benchmark --config configs/full.yaml
echo "=== report $(date -Is) ==="
"$PY" -m kv_safety.report --config configs/full.yaml
echo "=== full phase D done $(date -Is) ==="
