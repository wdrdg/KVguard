#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
LOG=artifacts/full_train_test/logs/run.log
mkdir -p artifacts/full_train_test/logs
exec > >(tee -a "$LOG") 2>&1

echo "=== full_train_test start $(date -Is) ==="
df -h .

echo "=== prepare $(date -Is) ==="
"$PY" -m kv_safety.data.prepare --config configs/full_train_test.yaml

"$PY" - <<'PY'
import json
from pathlib import Path
audit = json.loads(Path("artifacts/full_train_test/data_audit.json").read_text())
train = audit["splits"]["train"]
dev = audit["splits"]["dev"]["retained"]
cal = audit["splits"]["calibration"]["retained_safe"]
test = audit["splits"]["test"]
print(f"[check] train processed={train['processed_n']} official={train['official_n']} coverage={train['id_coverage']}")
print(f"[check] train retained={train['retained']}")
print(f"[check] dev={dev} cal={cal}")
print(f"[check] test processed={test['processed_n']} official={test['official_n']} coverage={test['id_coverage']} retained={test.get('retained')}")
print(f"[check] unprocessed train={len(train['unprocessed_ids'])} test={len(test['unprocessed_ids'])}")
print(f"[check] val withheld={audit['val_withheld_prompt_overlap']['n']}")
print(f"[check] pilot groups dev={audit['val_partition']['pilot_dev_groups_verified']} cal={audit['val_partition']['pilot_calibration_groups_verified']}")
assert audit["val_partition"]["pilot_dev_groups_verified"] > 0
assert not Path("artifacts/full_train_test/metrics/test_quality_summary.csv").exists()
print("[check] prepare ok")
PY

echo "=== extract layer 11 $(date -Is) ==="
"$PY" -m kv_safety.extract --config configs/full_train_test.yaml
echo "=== train $(date -Is) ==="
"$PY" -m kv_safety.train --config configs/full_train_test.yaml --all-features
echo "=== drop features after train $(date -Is) ==="
rm -rf artifacts/full_train_test/features
df -h .
echo "=== calibrate $(date -Is) ==="
"$PY" -m kv_safety.calibrate --config configs/full_train_test.yaml --split calibration --all-features
echo "=== evaluate dev $(date -Is) ==="
"$PY" -m kv_safety.evaluate --config configs/full_train_test.yaml --split dev --all-features
echo "=== lock $(date -Is) ==="
"$PY" -m kv_safety.lock_protocol --config configs/full_train_test.yaml
echo "=== evaluate test $(date -Is) ==="
"$PY" -m kv_safety.evaluate --config configs/full_train_test.yaml --split test \
  --protocol-lock artifacts/full_train_test/protocol_lock.json --all-features
echo "=== benchmark $(date -Is) ==="
"$PY" -m kv_safety.benchmark --config configs/full_train_test.yaml
echo "=== report $(date -Is) ==="
"$PY" -m kv_safety.report --config configs/full_train_test.yaml
echo "=== full_train_test done $(date -Is) ==="
