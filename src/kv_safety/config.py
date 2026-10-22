"""Config loading and shared path helpers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

BASE_FEATURES = ("h_last", "h_mean", "k_mean", "v_mean")
ALL_FEATURES = ("h_last", "h_mean", "k_mean", "v_mean", "kv_mean")


def load_config(path: str | os.PathLike) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(Path(path).resolve())
    return cfg


def resolve_path(p: str | os.PathLike) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def artifacts_dir(cfg: dict) -> Path:
    d = resolve_path(cfg["artifacts_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def manifest_path(cfg: dict) -> Path:
    return resolve_path(cfg["resource_manifest"])


def read_manifest(cfg: dict) -> dict:
    p = manifest_path(cfg)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def pinned_revision(cfg: dict, key: str) -> str | None:
    """Revision for a resource: explicit config value, else the SHA recorded at download."""
    if key == "model":
        explicit = cfg["model"].get("revision")
    else:
        explicit = cfg["data"][key].get("revision")
    if explicit:
        return explicit
    entry = read_manifest(cfg).get(key)
    return entry.get("revision") if entry else None


def apply_offline(cfg: dict) -> None:
    if cfg.get("offline"):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def feature_matrix(cfg: dict) -> list[dict]:
    """Explicit layer/feature blocks. No implicit cartesian product across layers."""
    block = cfg["features"].get("matrix")
    if block:
        return [{"layer": int(b["layer"]), "features": list(b["features"])} for b in block]
    return [{"layer": int(cfg["model"]["layer_index"]), "features": list(cfg["features"]["types"])}]


def classifier_jobs(cfg: dict) -> list[dict]:
    """One trainable head per configured (layer, feature), plus the optional matched-width control."""
    jobs = []
    seen = set()
    hidden = int(cfg["classifier"]["hidden"])
    for block in feature_matrix(cfg):
        for feat in block["features"]:
            key = (block["layer"], feat)
            if key in seen:
                continue
            seen.add(key)
            jobs.append({
                "layer": block["layer"],
                "name": feat,
                "feature": feat,
                "hidden": hidden,
                "capacity_control": False,
            })
    matched = cfg["classifier"].get("matched") or {}
    if matched.get("enabled"):
        jobs.append({
            "layer": int(matched.get("layer", cfg["model"]["layer_index"])),
            "name": matched["name"],
            "feature": matched["base_feature"],
            "hidden": int(matched["hidden"]),
            "capacity_control": True,
        })
    return jobs


def select_jobs(cfg: dict, all_features: bool, features: list[str] | None,
                layers: list[int] | None) -> list[dict]:
    jobs = classifier_jobs(cfg)
    if not all_features and features:
        wanted = set(features)
        jobs = [j for j in jobs if j["name"] in wanted]
    if layers:
        wanted_layers = {int(x) for x in layers}
        jobs = [j for j in jobs if j["layer"] in wanted_layers]
    if not jobs:
        raise SystemExit("no classifier jobs match the requested features/layers")
    return jobs


def stored_feature_parts(feature: str) -> tuple[str, ...]:
    if feature == "kv_mean":
        return ("k_mean", "v_mean")
    if feature not in BASE_FEATURES:
        raise ValueError(f"unknown feature {feature}")
    return (feature,)


def stored_by_layer(cfg: dict) -> dict[int, list[str]]:
    """Base arrays that must be materialised. kv_mean is concatenated at load/score time."""
    need: dict[int, list[str]] = {}
    for job in classifier_jobs(cfg):
        for part in stored_feature_parts(job["feature"]):
            bucket = need.setdefault(job["layer"], [])
            if part not in bucket:
                bucket.append(part)
    return need


def configured_layers(cfg: dict) -> list[int]:
    layers = {int(cfg["model"]["layer_index"])}
    layers.update(stored_by_layer(cfg))
    return sorted(layers)


def feature_list(cfg: dict, include_matched: bool = True) -> list[str]:
    names = []
    for job in classifier_jobs(cfg):
        if job["capacity_control"] and not include_matched:
            continue
        if job["name"] not in names:
            names.append(job["name"])
    return names


def classifier_spec(cfg: dict, feature: str) -> tuple[str, int]:
    """Map a classifier name to (input feature, hidden width). Layer is not encoded here."""
    for job in classifier_jobs(cfg):
        if job["name"] == feature:
            return job["feature"], job["hidden"]
    raise KeyError(feature)


def base_parser(desc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--config", required=True)
    return ap


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
