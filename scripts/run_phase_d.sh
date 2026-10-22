#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
LOG=artifacts/full/logs/full_train.log
mkdir -p artifacts/full/logs
exec >>"$LOG" 2>&1

echo "=== phase D resume $(date -Is) ==="
df -h .

echo "=== prepare $(date -Is) ==="
"$PY" -m kv_safety.data.prepare --config configs/full.yaml

"$PY" - <<'PY'
import json
from pathlib import Path
audit = json.loads(Path("artifacts/full/data_audit.json").read_text())
train = audit["splits"]["train"]["retained"]
dev = audit["splits"]["dev"]["retained"]
cal = audit["splits"]["calibration"]["retained_safe"]
print(f"[check] train={train['n']} safe={train['safe']} harmful={train['harmful']} dev={dev['n']} cal={cal['n']}")
assert train["n"] == 20319 and train["safe"] == 11039 and train["harmful"] == 9280, train
assert dev["n"] == 1353 and dev["safe"] == 742 and dev["harmful"] == 611, dev
assert cal["n"] == 750 and cal["harmful"] == 0, cal
assert not Path("artifacts/full/data/test.jsonl").exists()
print("[check] counts ok")
PY

echo "=== materialise held-out test rows $(date -Is) ==="
"$PY" - <<'PY'
import json
from kv_safety.config import artifacts_dir, load_config, write_json
from kv_safety.data.fineharm import load_fineharm
from kv_safety.data.prepare import _align_pool, _write_split, fineharm_paths
from kv_safety.data.splits import class_counts, resolve_leakage
from kv_safety.modeling import load_tokenizer

cfg = load_config("configs/full.yaml")
out = artifacts_dir(cfg)
dest = out / "data" / "test.jsonl"
tok = load_tokenizer(cfg)
raw, _ = load_fineharm(fineharm_paths(cfg), cfg["data"]["sentence_joiner"])
resolved, _ = resolve_leakage(raw, cfg["data"]["split_priority"])
with (out / "alignment_errors.jsonl").open("a", encoding="utf-8") as errors_f:
    aligned, info = _align_pool(resolved.get("test", []), tok, cfg, "test", errors_f)
manifest_rows = []
_write_split(dest, aligned, "test", manifest_rows)
with (out / "split_manifest.jsonl").open("a", encoding="utf-8") as f:
    for row in manifest_rows:
        f.write(json.dumps(row) + "\n")
audit_path = out / "data_audit.json"
audit = json.loads(audit_path.read_text())
retained = class_counts([r for r, _ in aligned])
audit["splits"]["test"]["retained"] = retained
audit["heldout_test"] = {
    "status": "prepared_not_scored",
    "eligible": info["eligible"],
    "class_eligible": info["class_eligible"],
    "excluded_too_long": info["excluded_too_long"],
    "alignment_failures": info["alignment_failures"],
    "scored": False,
    "reason": "rows materialised for Phase D; scoring waits for protocol_lock after calibration",
}
audit["manifest_counts"]["test"] = len(manifest_rows)
write_json(audit_path, audit)
print(f"[test-rows] wrote {len(manifest_rows)} safe={retained['safe']} harmful={retained['harmful']}")
assert retained["n"] == 2898 and retained["safe"] == 1623 and retained["harmful"] == 1275, retained
PY

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
    "purpose": "Phase D held-out FineHarm test. Written after full-train checkpoints and calibration thresholds exist. Test labels were not used to choose layers, thresholds, or filters.",
    "allow_test_eval": True,
    "run_name": cfg["run_name"],
    "artifacts_dir": cfg["artifacts_dir"],
    "split_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    "code_sha256": code,
    "git_head": git,
    "model_id": cfg["model"]["id"],
    "reference_layer": cfg["model"]["layer_index"],
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
