"""Train the scaler + MLP for each configured layer, feature and seed.

python -m kv_safety.train --config configs/pilot.yaml --all-features

Early stopping uses the fixed dev position list (at most 32 positions per response), with equal
weight per response. The backbone is not loaded.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .classifier import ScaledMLP, fit_weighted_scaler, save_checkpoint
from .config import artifacts_dir, base_parser, load_config, select_jobs, write_json
from .data.labels import IGNORE
from .extract import feature_disk_bytes, load_split_features
from .wandb_log import finish as wandb_finish
from .wandb_log import log as wandb_log
from .wandb_log import start as wandb_start


def weighted_bce(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """sum_i w_i * BCE_i / sum_i w_i over non-ignored rows; ignored rows contribute nothing."""
    mask = labels != IGNORE
    w = weights * mask
    y = torch.where(mask, labels, torch.zeros_like(labels)).float()
    per = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
    return (per * w).sum() / w.sum().clamp_min(1e-12)


def response_equal_bce(logits: np.ndarray, labels: np.ndarray, sample_idx: np.ndarray) -> float:
    """Mean over responses of the mean BCE over that response's non-ignored positions."""
    mask = labels != IGNORE
    if not np.any(mask):
        return float("nan")
    lg = torch.as_tensor(logits[mask], dtype=torch.float64)
    y = torch.as_tensor(labels[mask], dtype=torch.float64)
    per = nn.functional.binary_cross_entropy_with_logits(lg, y, reduction="none").numpy()
    sid = sample_idx[mask]
    uniq, inv = np.unique(sid, return_inverse=True)
    del uniq
    sums = np.bincount(inv, weights=per)
    cnt = np.bincount(inv)
    return float((sums / cnt).mean())


@torch.inference_mode()
def predict(model: ScaledMLP, X: np.ndarray, device: str, bs: int = 8192) -> np.ndarray:
    out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(np.ascontiguousarray(X[i:i + bs])).to(device)
        out.append(model(xb).float().cpu())
    return torch.cat(out).numpy() if out else np.zeros(0, np.float32)


def train_one(Xtr, ytr, wtr, Xva, yva, sva, hidden: int, seed: int, tc: dict, device: str,
              tag: str = "") -> tuple[ScaledMLP, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    keep = ytr != IGNORE
    Xtr, ytr, wtr = Xtr[keep], ytr[keep], wtr[keep]
    mean, scale = fit_weighted_scaler(Xtr, wtr)
    model = ScaledMLP(Xtr.shape[1], hidden, mean, scale).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    g = torch.Generator().manual_seed(seed)
    Xt = torch.from_numpy(np.ascontiguousarray(Xtr))
    yt = torch.from_numpy(ytr.astype(np.int64))
    wt = torch.from_numpy(wtr.astype(np.float32))
    best, best_state, bad, hist = float("inf"), None, 0, []
    for epoch in range(1, tc["max_epochs"] + 1):
        model.train()
        perm = torch.randperm(len(Xt), generator=g)
        tot, tw = 0.0, 0.0
        for i in range(0, len(perm), tc["batch_size"]):
            idx = perm[i:i + tc["batch_size"]]
            xb, yb, wb = Xt[idx].to(device), yt[idx].to(device), wt[idx].to(device)
            loss = weighted_bce(model(xb), yb, wb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * float(wb.sum())
            tw += float(wb.sum())
        model.eval()
        dev = response_equal_bce(predict(model, Xva, device), yva, sva)
        train_bce = tot / max(tw, 1e-12)
        hist.append({"epoch": epoch, "train_weighted_bce": train_bce, "dev_response_bce": dev})
        prefix = f"[train] {tag} " if tag else "[train] "
        print(f"{prefix}epoch={epoch} train_bce={train_bce:.6f} dev_bce={dev:.6f}", flush=True)
        if dev < best - 1e-6:
            best, bad = dev, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= tc["patience"]:
                break
    model.load_state_dict(best_state)
    best_epoch = min(hist, key=lambda h: h["dev_response_bce"])["epoch"]
    return model.eval(), {"history": hist, "best_dev_response_bce": best, "best_epoch": best_epoch}


def _cache_get(cache: dict, out: Path, split: str, feature: str, layer: int):
    key = (split, layer, feature)
    if key not in cache:
        cache[key] = load_split_features(out, split, feature, layer)
    return cache[key]


def _append_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--all-features", action="store_true")
    ap.add_argument("--features", nargs="+", default=None)
    ap.add_argument("--layers", nargs="+", type=int, default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = artifacts_dir(cfg)
    tc = cfg["train"]
    device = tc.get("device", "auto")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    jobs = select_jobs(cfg, args.all_features or not args.features, args.features, args.layers)
    ref_layer = int(cfg["model"]["layer_index"])
    jobs.sort(key=lambda j: (j["layer"] != ref_layer, j["layer"], j["capacity_control"], j["name"]))
    seeds = [int(s) for s in (args.seeds or tc["seeds"])]
    disk_bytes = feature_disk_bytes(out)
    log_path = out / "logs" / "train_metrics.csv"
    summary_path = out / "checkpoints" / "train_summary.json"
    # A layer-wise run appends. A full run replaces the log so a restarted matrix is not doubled.
    partial = args.layers is not None
    if partial and summary_path.exists():
        summary = json.loads(summary_path.read_text())
        active = {(j["layer"], j["name"], s) for j in jobs for s in seeds}
        summary = [row for row in summary if (row["layer"], row["method"], row["seed"]) not in active]
    else:
        summary = []
        if log_path.exists():
            log_path.unlink()
    cache: dict = {}
    log_rows = []
    for job in jobs:
        Xtr, mtr = _cache_get(cache, out, "train", job["feature"], job["layer"])
        Xdv, mdv = _cache_get(cache, out, "dev", job["feature"], job["layer"])
        n_nonfinite = int((~np.isfinite(Xtr)).sum() + (~np.isfinite(Xdv)).sum())
        if n_nonfinite:
            raise FloatingPointError(f"non-finite features for {job['name']} layer {job['layer']}: {n_nonfinite}")
        for seed in seeds:
            tag = f"L{job['layer']} {job['name']} seed={seed}"
            dest = out / "checkpoints" / str(job["layer"]) / job["name"] / str(seed)
            if (dest / "meta.json").exists() and (dest / "model.pt").exists():
                prev = json.loads((dest / "meta.json").read_text())
                summary.append({
                    "method": job["name"], "feature": job["feature"], "layer": job["layer"], "seed": seed,
                    "best_epoch": prev.get("best_epoch"), "best_dev_response_bce": prev.get("best_dev_response_bce"),
                    "n_parameters": prev.get("n_parameters"), "seconds": prev.get("seconds"),
                    "n_nonfinite": prev.get("n_nonfinite", 0), "feature_disk_bytes": disk_bytes,
                    "capacity_control": job["capacity_control"], "resumed": True,
                })
                print(f"[train] {tag} resume existing checkpoint", flush=True)
                continue
            t0 = time.time()
            model, info = train_one(
                Xtr, mtr["label"], mtr["weight"], Xdv, mdv["label"], mdv["sample_idx"],
                job["hidden"], seed, tc, device, tag=tag)
            seconds = round(time.time() - t0, 1)
            meta = {
                "classifier": job["name"], "method": job["name"], "feature": job["feature"],
                "seed": seed, "layer_index": job["layer"], "hidden": job["hidden"],
                "capacity_control": job["capacity_control"],
                "train_rows": int(len(Xtr)),
                "train_responses": int(len(np.unique(mtr["sample_idx"]))),
                "dev_rows": int(len(Xdv)),
                "dev_responses": int(len(np.unique(mdv["sample_idx"]))),
                "n_nonfinite": n_nonfinite,
                "feature_disk_bytes": disk_bytes,
                "train_config": {k: tc[k] for k in ("lr", "weight_decay", "batch_size", "max_epochs", "patience")},
                "seconds": seconds,
                "best_epoch": info["best_epoch"],
                "best_dev_response_bce": info["best_dev_response_bce"],
            }
            save_checkpoint(model.cpu(), out / "checkpoints" / str(job["layer"]) / job["name"] / str(seed), meta)
            summary.append({
                "method": job["name"], "feature": job["feature"], "layer": job["layer"], "seed": seed,
                "best_epoch": info["best_epoch"], "best_dev_response_bce": info["best_dev_response_bce"],
                "n_parameters": model.n_parameters(), "seconds": seconds, "n_nonfinite": n_nonfinite,
                "feature_disk_bytes": disk_bytes, "capacity_control": job["capacity_control"],
            })
            for row in info["history"]:
                log_rows.append({
                    "layer": job["layer"], "method": job["name"], "feature": job["feature"], "seed": seed,
                    "epoch": row["epoch"], "train_weighted_bce": row["train_weighted_bce"],
                    "dev_response_bce": row["dev_response_bce"],
                    "selected": row["epoch"] == info["best_epoch"],
                    "n_parameters": model.n_parameters(), "train_rows": int(len(Xtr)),
                    "train_responses": int(len(np.unique(mtr["sample_idx"]))),
                    "n_nonfinite": n_nonfinite, "seconds": seconds, "feature_disk_bytes": disk_bytes,
                })
            print(f"[train] {tag} best_epoch={info['best_epoch']} "
                  f"dev_bce={info['best_dev_response_bce']:.4f} params={model.n_parameters()}", flush=True)
            run = wandb_start(cfg, job="train", name=f"{cfg['run_name']}-L{job['layer']}-{job['name']}-s{seed}",
                              extra={"classifier": job["name"], "feature": job["feature"], "seed": seed,
                                     "layer": job["layer"], "hidden": job["hidden"]})
            for row in info["history"]:
                wandb_log(run, {"epoch": row["epoch"], "train_weighted_bce": row["train_weighted_bce"],
                                "dev_response_bce": row["dev_response_bce"]}, step=row["epoch"])
            wandb_finish(run, {"best_epoch": info["best_epoch"],
                               "best_dev_response_bce": info["best_dev_response_bce"],
                               "n_parameters": model.n_parameters(), "seconds": seconds})
        del Xtr, Xdv
        cache.pop(("train", job["layer"], job["feature"]), None)
        cache.pop(("dev", job["layer"], job["feature"]), None)
    write_json(summary_path, summary)
    _append_rows(out / "logs" / "train_metrics.csv", log_rows)
    _append_rows(out / "metrics" / "run_status.csv", [{
        "stage": "train", "status": "ok", "n_jobs": len(summary),
        "detail": f"{len(jobs)} heads x {len(seeds)} seeds",
    }])


if __name__ == "__main__":
    main()
