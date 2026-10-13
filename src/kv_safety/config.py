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


def feature_list(cfg: dict, include_matched: bool = True) -> list[str]:
    feats = list(cfg["features"]["types"])
    m = cfg["classifier"].get("matched", {})
    if include_matched and m.get("enabled"):
        feats.append(m["name"])
    return feats


def classifier_spec(cfg: dict, feature: str) -> tuple[str, int]:
    """Map a classifier name to (input feature, hidden width)."""
    m = cfg["classifier"].get("matched", {})
    if m.get("enabled") and feature == m["name"]:
        return m["base_feature"], int(m["hidden"])
    return feature, int(cfg["classifier"]["hidden"])


def base_parser(desc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--config", required=True)
    return ap


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
