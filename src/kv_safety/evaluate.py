"""Score val/test per token, calibrate on val safe responses, report streaming metrics on test.

python -m kv_safety.evaluate --config configs/smoke.yaml --all-features

Test never influences checkpoints or thresholds. Main metrics use the FIRST alarm only.
Balanced sampling means precision/F1 are not natural-traffic numbers.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from .classifier import load_checkpoint
from .config import artifacts_dir, base_parser, classifier_spec, feature_list, load_config, write_json
from .extract import load_prepared, load_split_features
from .metrics import bootstrap, calibrate_threshold, first_alarm, outcome, response_level_prf, summarize
from .train import predict


def per_response_scores(logits: np.ndarray, meta: dict) -> dict[int, np.ndarray]:
    order = np.lexsort((meta["position"], meta["sample_idx"]))
    sid, pos, lg = meta["sample_idx"][order], meta["position"][order], logits[order]
    out = {}
    bounds = np.flatnonzero(np.diff(sid)) + 1
    for s, p, l in zip(np.split(sid, bounds), np.split(pos, bounds), np.split(lg, bounds)):
        if not np.array_equal(p, np.arange(1, len(p) + 1)):
            raise ValueError(f"sample {s[0]}: positions are not the complete 1..T sequence")
        out[int(s[0])] = l
    return out


def evaluate_one(name: str, seed: int, cfg: dict, out, device: str, cache: dict) -> tuple[dict, list[dict]]:
    ec = cfg["evaluate"]
    windows = tuple(ec["recall_windows"])
    feat, _ = classifier_spec(cfg, name)
    model, meta = load_checkpoint(out / "checkpoints" / name / str(seed), device)
    traj = {}
    for split in ("val", "test"):
        if (split, feat) not in cache:
            cache[(split, feat)] = load_split_features(out, split, feat)
        X, m = cache[(split, feat)]
        logits = predict(model, X, device)
        traj[split] = per_response_scores(logits, m)
        tdir = out / "metrics" / "trajectories"  # full trajectories, diagnostics only
        tdir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(tdir / f"{name}_{seed}_{split}.npz", sample_idx=m["sample_idx"],
                            position=m["position"], logit=logits)
    recs = {s: load_prepared(out, s) for s in ("val", "test")}
    safe_val_max = [traj["val"][i].max() for i, r in enumerate(recs["val"]) if r["response_label"] == 0]
    thr, cal = calibrate_threshold(np.array(safe_val_max), ec["target_fpr"])
    rows = []
    for split in ("val", "test"):
        for i, r in enumerate(recs[split]):
            sc = traj[split][i]
            assert len(sc) == r["T"]
            ta = first_alarm(sc, thr)
            o = outcome(ta, r["response_label"], r["T"], r["a"], r["b"], windows)
            rows.append({"classifier": name, "feature": feat, "seed": seed, "split": split, "id": r["id"],
                         "label": r["response_label"], "T": r["T"], "a": r["a"], "b": r["b"],
                         "threshold": thr, "max_logit": float(sc.max()), "final_logit": float(sc[-1]), **o})
    test = [r for r in rows if r["split"] == "test"]
    s = summarize(test, windows)
    s.update(bootstrap(test, ["safe_response_fpr", "timely_recall", f"recall_by_end_plus_{windows[-1]}",
                              "premature_alarm_rate"], ec["bootstrap_samples"], ec["bootstrap_seed"], windows))
    lab = np.array([r["label"] for r in test])
    fin = np.array([r["final_logit"] for r in test])
    s["final_prefix_auroc"] = float(roc_auc_score(lab, fin)) if len(set(lab)) == 2 else float("nan")
    s.update(response_level_prf(np.array([r["alarm"] for r in test]), lab))
    val_rows = [r for r in rows if r["split"] == "val"]
    summary = {"classifier": name, "feature": feat, "seed": seed, "layer_index": meta["layer_index"],
               "dim": meta["in_dim"], "hidden": meta["hidden"], "n_parameters": meta["n_parameters"],
               "threshold_logit": thr, "val_calibration": cal,
               "val_fpr": summarize(val_rows, windows)["safe_response_fpr"], **s}
    return summary, rows


def length_table(rows: list[dict], bins: list[int]) -> pd.DataFrame:
    df = pd.DataFrame([r for r in rows if r["split"] == "test"])
    df["length_bin"] = pd.cut(df["T"], bins=bins + [10**9], right=True).astype(str)
    out = []
    for (c, s, lb), g in df.groupby(["classifier", "seed", "length_bin"], observed=True):
        safe, harm = g[g.label == 0], g[g.label == 1]
        out.append({"classifier": c, "seed": s, "length_bin": lb, "n_safe": len(safe), "n_harmful": len(harm),
                    "fpr": safe["false_alarm"].mean() if len(safe) else np.nan,
                    "timely_recall": harm["timely"].mean() if len(harm) else np.nan,
                    "nonpremature_detection_rate": harm["nonpremature_detection"].mean() if len(harm) else np.nan})
    return pd.DataFrame(out)


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--all-features", action="store_true")
    ap.add_argument("--features", nargs="+", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = artifacts_dir(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feats = feature_list(cfg) if args.all_features or not args.features else args.features
    summaries, all_rows, cache = [], [], {}
    for name in feats:
        for seed in cfg["train"]["seeds"]:
            s, rows = evaluate_one(name, seed, cfg, out, device, cache)
            summaries.append(s)
            all_rows.extend(rows)
            print(f"[evaluate] {name} seed={seed} thr={s['threshold_logit']:.3f} test_fpr={s['safe_response_fpr']:.3f} "
                  f"timely={s['timely_recall']:.3f} premature={s['premature_alarm_rate']:.3f} "
                  f"no_alarm={s['no_alarm_rate']:.3f}")
    mdir = out / "metrics"
    mdir.mkdir(exist_ok=True)
    pd.DataFrame(all_rows).to_parquet(mdir / "per_response.parquet", index=False)
    flat = []
    for s in summaries:
        f = {k: v for k, v in s.items() if not isinstance(v, (dict, list))}
        f["val_n_safe"] = s["val_calibration"]["n_safe"]
        f["calibration_unstable"] = s["val_calibration"]["unstable"]
        for k, v in s.items():
            if k.endswith("_ci95"):
                f[k + "_lo"], f[k + "_hi"] = v
        flat.append(f)
    pd.DataFrame(flat).to_csv(mdir / "quality_summary.csv", index=False)
    length_table(all_rows, cfg["evaluate"]["length_bins"]).to_csv(mdir / "quality_by_length.csv", index=False)
    write_json(mdir / "quality_summary.json", summaries)


if __name__ == "__main__":
    main()
