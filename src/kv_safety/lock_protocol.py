"""Write protocol_lock.json before the first test score.

python -m kv_safety.lock_protocol --config configs/full_train_test.yaml

Refuses to replace a lock after test scores exist. The evaluate command checks
the split manifest, code files and checkpoint/scaler/threshold hashes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .config import REPO_ROOT, artifacts_dir, base_parser, classifier_jobs, load_config, read_manifest, write_json


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_hashes() -> dict[str, str]:
    files = sorted(REPO_ROOT.joinpath("src", "kv_safety").rglob("*.py"))
    return {str(path.relative_to(REPO_ROOT)): _sha256(path) for path in files}


def main() -> None:
    ap = base_parser(__doc__)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if not cfg.get("allow_test_eval", False):
        raise SystemExit("refusing to lock a config with allow_test_eval false")
    out = artifacts_dir(cfg)
    if (out / "metrics" / "test_quality_summary.csv").exists() or (out / "scores" / "test").exists():
        raise SystemExit("test scores already exist; refusing to rewrite the protocol lock")
    manifest = out / "split_manifest.jsonl"
    if not manifest.exists():
        raise SystemExit("split_manifest.jsonl is missing; run prepare first")
    jobs = classifier_jobs(cfg)
    seeds = [int(s) for s in cfg["train"]["seeds"]]
    checkpoint_hashes = {}
    for job in jobs:
        for seed in seeds:
            root = out / "checkpoints" / str(job["layer"]) / job["name"] / str(seed)
            for name in ("model.pt", "scaler.npz", "threshold.json", "meta.json"):
                path = root / name
                if not path.exists():
                    raise SystemExit(f"missing {path}; train and calibrate before locking")
                rel = str(path.relative_to(out))
                checkpoint_hashes[rel] = _sha256(path)
    resources = read_manifest(cfg)
    lock = {
        "run_name": cfg["run_name"],
        "allow_test_eval": True,
        "config_path": str(Path(cfg["_config_path"]).relative_to(REPO_ROOT)),
        "config_sha256": _sha256(Path(cfg["_config_path"])),
        "split_manifest_sha256": _sha256(manifest),
        "code_sha256": _code_hashes(),
        "checkpoint_sha256": checkpoint_hashes,
        "model_revision": (resources.get("model") or {}).get("revision"),
        "fineharm_revision": (resources.get("fineharm") or {}).get("revision"),
        "layer": int(cfg["model"]["layer_index"]),
        "features": [job["name"] for job in jobs],
        "seeds": seeds,
        "hidden": int(cfg["classifier"]["hidden"]),
        "max_train_positions": int(cfg["features"]["max_train_positions"]),
        "chunk_size": int(cfg["features"]["chunk_size"]),
        "context_limit": cfg["data"].get("context_limit"),
        "max_length_filter": cfg["data"].get("max_length"),
        "label_rule": "t<a -> 0; a<=t<b -> -100; t>=b -> 1; safe responses are 0",
        "threshold_rule": "lowest threshold with calibration response FPR <= target, strict >",
        "target_fpr": cfg["evaluate"]["target_fpr"],
        "bootstrap_samples": cfg["evaluate"]["bootstrap_samples"],
        "bootstrap_seed": cfg["evaluate"]["bootstrap_seed"],
        "test_scored": False,
    }
    write_json(out / "protocol_lock.json", lock)
    print(f"[lock] wrote {out / 'protocol_lock.json'} checkpoints={len(checkpoint_hashes)}", flush=True)


if __name__ == "__main__":
    main()
