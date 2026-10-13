"""Train the scaler + MLP classifier for each feature and seed (backbone not needed).

python -m kv_safety.train --config configs/smoke.yaml --all-features
python -m kv_safety.train --config configs/pilot.yaml --features kv_mean --seeds 17
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .classifier import ScaledMLP, fit_weighted_scaler, save_checkpoint
from .config import artifacts_dir, base_parser, classifier_spec, feature_list, load_config, write_json
from .data.labels import IGNORE
from .extract import load_split_features


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
    lg = torch.as_tensor(logits[mask], dtype=torch.float64)
    y = torch.as_tensor(labels[mask], dtype=torch.float64)
    per = nn.functional.binary_cross_entropy_with_logits(lg, y, reduction="none").numpy()
    sid = sample_idx[mask]
    uniq, inv = np.unique(sid, return_inverse=True)
    sums = np.bincount(inv, weights=per)
    cnt = np.bincount(inv)
    return float((sums / cnt).mean())


@torch.inference_mode()
def predict(model: ScaledMLP, X: np.ndarray, device: str, bs: int = 8192) -> np.ndarray:
    out = []
    for i in range(0, len(X), bs):
        out.append(model(torch.from_numpy(X[i:i + bs]).to(device)).float().cpu())
    return torch.cat(out).numpy() if out else np.zeros(0, np.float32)


def train_one(Xtr, ytr, wtr, Xva, yva, sva, hidden: int, seed: int, tc: dict, device: str) -> tuple[ScaledMLP, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    keep = ytr != IGNORE  # training positions are already chosen among valid ones; guard anyway
    Xtr, ytr, wtr = Xtr[keep], ytr[keep], wtr[keep]
    mean, scale = fit_weighted_scaler(Xtr, wtr)  # train only
    model = ScaledMLP(Xtr.shape[1], hidden, mean, scale).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    g = torch.Generator().manual_seed(seed)
    Xt = torch.from_numpy(Xtr)
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
        val = response_equal_bce(predict(model, Xva, device), yva, sva)
        hist.append({"epoch": epoch, "train_weighted_bce": tot / max(tw, 1e-12), "val_response_bce": val})
        if val < best - 1e-6:
            best, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= tc["patience"]:
                break
    model.load_state_dict(best_state)
    return model.eval(), {"history": hist, "best_val_response_bce": best,
                          "best_epoch": min(hist, key=lambda h: h["val_response_bce"])["epoch"]}


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--all-features", action="store_true")
    ap.add_argument("--features", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = artifacts_dir(cfg)
    tc = cfg["train"]
    device = tc.get("device", "auto")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    feats = feature_list(cfg) if args.all_features or not args.features else args.features
    seeds = args.seeds or tc["seeds"]
    fm = json.loads((out / "feature_manifest.json").read_text())
    summary = []
    for name in feats:
        feat, hidden = classifier_spec(cfg, name)
        Xtr, mtr = load_split_features(out, "train", feat)
        Xva, mva = load_split_features(out, "val", feat)
        for seed in seeds:
            t0 = time.time()
            model, info = train_one(Xtr, mtr["label"], mtr["weight"], Xva, mva["label"], mva["sample_idx"],
                                    hidden, seed, tc, device)
            meta = {"classifier": name, "feature": feat, "seed": seed, "layer_index": fm["layer_index"],
                    "train_rows": int(len(Xtr)), "train_responses": int(len(np.unique(mtr["sample_idx"]))),
                    "train_config": tc, "seconds": round(time.time() - t0, 1), **info}
            save_checkpoint(model.cpu(), out / "checkpoints" / name / str(seed), meta)
            summary.append({k: meta[k] for k in ("classifier", "feature", "seed", "best_epoch",
                                                 "best_val_response_bce")} | {"n_parameters": model.n_parameters()})
            print(f"[train] {name} seed={seed} best_epoch={info['best_epoch']} "
                  f"val_bce={info['best_val_response_bce']:.4f} params={model.n_parameters()}")
    write_json(out / "checkpoints" / "train_summary.json", summary)


if __name__ == "__main__":
    main()
