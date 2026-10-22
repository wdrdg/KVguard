"""Optional Weights & Biases logging for train, evaluate, and benchmark.

Runs are skipped under pytest and when ``WANDB_MODE=disabled`` or ``wandb.enabled`` is false,
so a missing login never blocks the experiment.
"""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np


def enabled(cfg: dict | None = None) -> bool:
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    if os.environ.get("WANDB_MODE") == "disabled":
        return False
    wb = (cfg or {}).get("wandb") or {}
    return wb.get("enabled", True) is not False


def scalars(data: dict) -> dict[str, int | float]:
    out: dict[str, int | float] = {}
    for key, value in data.items():
        if isinstance(value, bool) or isinstance(value, (dict, list, tuple)):
            continue
        if isinstance(value, (np.integer, int)):
            out[key] = int(value)
            continue
        if isinstance(value, (np.floating, float)):
            number = float(value)
            if math.isnan(number) or math.isinf(number):
                continue
            out[key] = number
    return out


def start(cfg: dict, *, job: str, name: str, extra: dict[str, Any] | None = None):
    if not enabled(cfg):
        return None
    try:
        import wandb
    except ImportError:
        print("[wandb] not installed; skipping logs", flush=True)
        return None
    if wandb.run is not None:
        wandb.run.finish()
    wb = cfg.get("wandb") or {}
    kwargs: dict[str, Any] = {
        "project": wb.get("project") or "kvguard",
        "group": cfg.get("run_name"),
        "job_type": job,
        "name": name,
        "config": {
            "run_name": cfg.get("run_name"),
            "config_path": cfg.get("_config_path"),
            "model_id": cfg.get("model", {}).get("id"),
            "layer_index": cfg.get("model", {}).get("layer_index"),
            **(extra or {}),
        },
        "reinit": "finish_previous",
    }
    if wb.get("entity"):
        kwargs["entity"] = wb["entity"]
    if wb.get("mode"):
        kwargs["mode"] = wb["mode"]
    try:
        return wandb.init(**kwargs)
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] init failed: {type(exc).__name__}: {exc}", flush=True)
        return None


def log(run, data: dict, step: int | None = None) -> None:
    if run is None:
        return
    try:
        payload = scalars(data)
        if step is None:
            run.log(payload)
        else:
            run.log(payload, step=step)
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] log failed: {type(exc).__name__}: {exc}", flush=True)


def log_table(run, key: str, frame) -> None:
    if run is None or frame is None or len(frame) == 0:
        return
    try:
        import wandb

        numeric = frame.select_dtypes(include=["number", "bool"])
        keep = [c for c in frame.columns if c not in numeric.columns
                and frame[c].map(lambda v: isinstance(v, str)).all()]
        table = wandb.Table(dataframe=frame[list(numeric.columns) + keep].reset_index(drop=True))
        run.log({key: table})
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] table log failed: {type(exc).__name__}: {exc}", flush=True)


def finish(run, summary: dict | None = None) -> None:
    if run is None:
        return
    try:
        for key, value in scalars(summary or {}).items():
            run.summary[key] = value
        run.finish()
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] finish failed: {type(exc).__name__}: {exc}", flush=True)
