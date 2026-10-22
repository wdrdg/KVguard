"""Chunked causal replay that keeps scores and drops high-dimensional features.

Several classifier seeds share one backbone pass. Scores are FP32 logits aligned with every
response token, including positions the training loss ignores.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .classifier import load_checkpoint
from .config import stored_by_layer
from .extract import assemble_feature, extract_response_features_layers
from .train import predict


def score_path(out: Path, split: str, layer: int, name: str) -> Path:
    return out / "scores" / split / f"layer_{layer}" / f"{name}.npz"


def score_key(record_ids: list[str], jobs: list[dict]) -> str:
    payload = {
        "ids": record_ids,
        "jobs": [{k: j[k] for k in ("layer", "name", "seed", "feature", "hidden")} | {
            "checkpoint": j.get("checkpoint_stamp")} for j in jobs],
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def checkpoint_stamp(path: Path) -> str:
    meta = path / "meta.json"
    model = path / "model.pt"
    raw = meta.read_bytes() + str(model.stat().st_size).encode() + str(int(model.stat().st_mtime)).encode()
    return hashlib.sha1(raw).hexdigest()[:12]


def load_jobs(cfg: dict, out: Path, jobs: list[dict], seeds: list[int], device: str) -> list[dict]:
    loaded = []
    for job in jobs:
        for seed in seeds:
            path = out / "checkpoints" / str(job["layer"]) / job["name"] / str(seed)
            if not (path / "meta.json").exists():
                raise SystemExit(f"missing checkpoint {path}")
            module, meta = load_checkpoint(path, device)
            loaded.append({
                **job,
                "seed": int(seed),
                "module": module,
                "meta": meta,
                "checkpoint": str(path),
                "checkpoint_stamp": checkpoint_stamp(path),
            })
    return loaded


def _existing(path: Path, key: str) -> dict | None:
    done = path.with_suffix(".key")
    if path.exists() and done.exists() and done.read_text() == key:
        with np.load(path) as z:
            return {k: np.array(z[k]) for k in z.files}
    return None


def replay_scores(model, records: list[dict], cfg: dict, loaded: list[dict], device: str,
                  out: Path, split: str) -> dict[tuple, dict]:
    """Score every response token. Returns {(layer, name, seed): {sample_idx, position, logit}}."""
    if not loaded:
        return {}
    groups: dict[tuple, list] = {}
    for job in loaded:
        groups.setdefault((job["layer"], job["name"]), []).append(job)
    spec = {}
    for layer, feats in stored_by_layer(cfg).items():
        if any(job["layer"] == layer for job in loaded):
            spec[layer] = feats
    # Only layers that this call actually scores.
    needed_layers = {job["layer"] for job in loaded}
    spec = {layer: feats for layer, feats in spec.items() if layer in needed_layers}
    ids = [r["id"] for r in records]
    results = {}
    pending_groups = {}
    for key, jobs in groups.items():
        layer, name = key
        path = score_path(out, split, layer, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        cached = _existing(path, score_key(ids, jobs))
        if cached is not None:
            for job in jobs:
                col = f"logit_{job['seed']}"
                results[(layer, name, job["seed"])] = {
                    "sample_idx": cached["sample_idx"],
                    "position": cached["position"],
                    "logit": cached[col],
                }
            continue
        pending_groups[key] = (jobs, path)
    if not pending_groups:
        return results
    columns = {key: {job["seed"]: [] for job in jobs} for key, (jobs, _) in pending_groups.items()}
    sample_idx, position = [], []
    for i, rec in enumerate(tqdm(records, desc=f"replay {split}", leave=False)):
        feats = extract_response_features_layers(
            model, rec["prompt_ids"], rec["response_ids"], spec, cfg["features"]["chunk_size"], None)
        pos = np.arange(1, rec["T"] + 1, dtype=np.int32)
        sample_idx.append(np.full(rec["T"], i, np.int32))
        position.append(pos)
        built: dict[tuple, np.ndarray] = {}
        for key, (jobs, _) in pending_groups.items():
            layer, name = key
            feature = jobs[0]["feature"]
            cache_key = (layer, feature)
            if cache_key not in built:
                built[cache_key] = assemble_feature(feats[layer], feature)
            X = built[cache_key]
            for job in jobs:
                columns[key][job["seed"]].append(predict(job["module"], X, device))
        del feats, built
    sample_idx_a = np.concatenate(sample_idx)
    position_a = np.concatenate(position)
    for key, (jobs, path) in pending_groups.items():
        payload = {"sample_idx": sample_idx_a, "position": position_a}
        for job in jobs:
            payload[f"logit_{job['seed']}"] = np.concatenate(columns[key][job["seed"]]).astype(np.float32)
        np.savez(path, **payload)
        path.with_suffix(".key").write_text(score_key(ids, jobs))
        for job in jobs:
            results[(job["layer"], job["name"], job["seed"])] = {
                "sample_idx": sample_idx_a,
                "position": position_a,
                "logit": payload[f"logit_{job['seed']}"],
            }
    return results


def trajectories(scores: dict) -> dict[int, np.ndarray]:
    """Map sample index -> logit[T], requiring a complete 1..T sequence."""
    order = np.lexsort((scores["position"], scores["sample_idx"]))
    sid = scores["sample_idx"][order]
    pos = scores["position"][order]
    logit = np.asarray(scores["logit"])[order]
    out = {}
    bounds = np.flatnonzero(np.diff(sid)) + 1
    for s, p, lg in zip(np.split(sid, bounds), np.split(pos, bounds), np.split(logit, bounds)):
        if not np.array_equal(p, np.arange(1, len(p) + 1)):
            raise ValueError(f"sample {int(s[0])}: positions are not the complete 1..T sequence")
        out[int(s[0])] = lg
    return out


def assert_split_allowed(cfg: dict, split: str, protocol_lock: str | None) -> None:
    if split == "test":
        if not cfg.get("allow_test_eval", False):
            raise SystemExit("refusing split=test because allow_test_eval is false")
        if not protocol_lock:
            raise SystemExit("split=test requires --protocol-lock")
        _check_protocol_lock(cfg, protocol_lock)
        return
    if split not in ("dev", "calibration"):
        raise SystemExit(f"unsupported split {split}")


def _check_protocol_lock(cfg: dict, path: str) -> None:
    from .config import REPO_ROOT, artifacts_dir

    lock = json.loads(Path(path).read_text(encoding="utf-8"))
    manifest = artifacts_dir(cfg) / "split_manifest.jsonl"
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    expected = lock.get("split_manifest_sha256")
    if expected and expected != digest:
        raise SystemExit("protocol lock split_manifest hash does not match the current manifest")
    for rel, expected_hash in (lock.get("code_sha256") or {}).items():
        got = hashlib.sha256((REPO_ROOT / rel).read_bytes()).hexdigest()
        if got != expected_hash:
            raise SystemExit(f"protocol lock code hash mismatch for {rel}")
    if lock.get("allow_test_eval") is False:
        raise SystemExit("protocol lock itself forbids test evaluation")
    art = artifacts_dir(cfg)
    for rel, expected_hash in (lock.get("checkpoint_sha256") or {}).items():
        target = art / rel
        if not target.exists():
            raise SystemExit(f"protocol lock checkpoint missing: {rel}")
        got = hashlib.sha256(target.read_bytes()).hexdigest()
        if got != expected_hash:
            raise SystemExit(f"protocol lock checkpoint hash mismatch for {rel}")
