"""Stream-score a split with thresholds already fixed on calibration.

python -m kv_safety.evaluate --config configs/pilot.yaml --split dev --all-features

Does not train and does not retune thresholds. Pilot quality tables are development feedback
on dev, not an untouched test result. split=test requires allow_test_eval and --protocol-lock.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from .config import artifacts_dir, base_parser, load_config, select_jobs
from .extract import load_prepared
from .metrics import (
    PRIMARY_RATE_KEYS,
    bootstrap_cis,
    group_rows,
    outcome,
    paired_differences,
    stream_classification,
    summarize,
)
from .metrics import first_alarm
from .replay import assert_split_allowed, load_jobs, replay_scores, trajectories
from .wandb_log import finish as wandb_finish
from .wandb_log import log as wandb_log
from .wandb_log import start as wandb_start


def _threshold(out, layer: int, name: str, seed: int) -> dict:
    path = out / "checkpoints" / str(layer) / name / str(seed) / "threshold.json"
    if not path.exists():
        raise SystemExit(f"missing {path}; run kv_safety.calibrate before evaluate")
    return json.loads(path.read_text())


def _response_rows(records: list[dict], traj: dict[int, np.ndarray], thr: float) -> list[dict]:
    rows = []
    for i, rec in enumerate(records):
        scores = traj[i]
        if len(scores) != rec["T"]:
            raise ValueError(f"{rec['id']}: scored length {len(scores)} != T {rec['T']}")
        alarm = first_alarm(scores, thr)
        label = int(rec["response_label"])
        spanned = label == 1
        base = outcome(alarm, label, rec["T"], rec["a"] if spanned else None, rec["b"] if spanned else None)
        rows.append({
            "sample_id": rec["id"],
            "prompt_group": rec["prompt_group"],
            "prompt_group_id": rec["prompt_group"],
            "split": rec["split"],
            "label": label,
            "prompt_safety_label": rec.get("prompt_safety_label"),
            "dataset_source": rec.get("dataset_source"),
            "T": rec["T"],
            "a": rec["a"] if spanned else None,
            "b": rec["b"] if spanned else None,
            "t_alarm": alarm,
            "max_score": float(np.max(scores)),
            "final_score": float(scores[-1]),
            "threshold": thr,
            **base,
        })
    return rows


def _flat_summary(spec: dict, summary: dict, extra: dict, cis: dict) -> dict:
    row = {**spec, **{k: v for k, v in summary.items() if k != "null_reasons"}}
    row.update({k: v for k, v in extra.items() if k != "null_reasons"})
    reasons = {}
    reasons.update(summary.get("null_reasons") or {})
    reasons.update(extra.get("null_reasons") or {})
    for key, bounds in cis.items():
        if bounds is None:
            row[key + "_lo"] = None
            row[key + "_hi"] = None
            reasons[key] = "undefined"
        else:
            row[key + "_lo"], row[key + "_hi"] = bounds
    row["null_reasons"] = json.dumps(reasons, ensure_ascii=False) if reasons else ""
    return row


def _mean_metric(rows: list[dict], key: str):
    vals = [r[key] for r in rows if r.get(key) is not None and r[key] == r[key]]
    return float(np.mean(vals)) if vals else None


def _layer_selection(summaries: list[dict], reference_layer: int) -> dict:
    """Rank scheduled layers on dev. Capacity-control heads are not candidates."""
    rule = (
        "For each of h_mean, v_mean and kv_mean, rank layers by mean recall_by_end_plus_32 "
        "over seeds on dev. Ties break to lower premature_alarm_rate, then lower "
        "safe_response_fpr, then the smaller layer index. Block "
        f"{reference_layer} stays the reference comparison and is not replaced by this ranking. "
        "Dev was used for early stopping, so this is development selection, not a test result."
    )
    by_method: dict[str, dict[int, list]] = {}
    for row in summaries:
        if row.get("capacity_control") or row["method"] not in ("h_mean", "v_mean", "kv_mean"):
            continue
        by_method.setdefault(row["method"], {}).setdefault(int(row["layer"]), []).append(row)
    by_feature = {}
    for method, layers in by_method.items():
        stats = []
        for layer, rows in layers.items():
            stats.append({
                "layer": layer,
                "n_seeds": len(rows),
                "recall_by_end_plus_32": _mean_metric(rows, "recall_by_end_plus_32"),
                "premature_alarm_rate": _mean_metric(rows, "premature_alarm_rate"),
                "safe_response_fpr": _mean_metric(rows, "safe_response_fpr"),
            })

        def sort_key(item, _stats=stats):
            rec = item["recall_by_end_plus_32"]
            prem = item["premature_alarm_rate"]
            fpr = item["safe_response_fpr"]
            return (
                -(rec if rec is not None else -1.0),
                prem if prem is not None else 1.0,
                fpr if fpr is not None else 1.0,
                item["layer"],
            )

        stats.sort(key=sort_key)
        by_feature[method] = {
            "selected_layer": stats[0]["layer"] if stats else None,
            "ranking": stats,
        }
    return {"rule": rule, "reference_layer": reference_layer, "by_feature": by_feature}


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--split", required=True)
    ap.add_argument("--all-features", action="store_true")
    ap.add_argument("--features", nargs="+", default=None)
    ap.add_argument("--layers", nargs="+", type=int, default=None)
    ap.add_argument("--protocol-lock", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_split_allowed(cfg, args.split, args.protocol_lock)
    out = artifacts_dir(cfg)
    records = load_prepared(out, args.split)
    if not records:
        raise SystemExit(f"split {args.split} has no prepared rows")
    jobs = select_jobs(cfg, args.all_features or not args.features, args.features, args.layers)
    seeds = [int(s) for s in cfg["train"]["seeds"]]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded = load_jobs(cfg, out, jobs, seeds, device)
    import kv_safety.modeling as modeling
    model = modeling.load_model(cfg, device)
    scored = replay_scores(model, records, cfg, loaded, device, out, args.split)
    del model
    role = "development" if args.split == "dev" else "heldout"
    ec = cfg["evaluate"]
    n_boot = int(ec["bootstrap_samples"])
    boot_seed = int(ec["bootstrap_seed"])
    per_response = []
    summaries = []
    grouped = []
    by_key: dict[tuple, list[dict]] = {}
    for job in loaded:
        thr_info = _threshold(out, job["layer"], job["name"], job["seed"])
        thr = float(thr_info["threshold"])
        traj = trajectories(scored[(job["layer"], job["name"], job["seed"])])
        rows = _response_rows(records, traj, thr)
        for row in rows:
            row.update({"method": job["name"], "seed": job["seed"], "layer": job["layer"],
                        "evaluation_role": role})
        per_response.extend(rows)
        quality = summarize(rows)
        ranking = stream_classification(rows)
        cis = bootstrap_cis(rows, list(PRIMARY_RATE_KEYS), n_boot, boot_seed)
        n_groups = len({r["prompt_group"] for r in rows})
        spec = {
            "split": args.split,
            "evaluation_role": role,
            "method": job["name"],
            "seed": job["seed"],
            "layer": job["layer"],
            "feature_dim": job["meta"]["in_dim"],
            "parameter_count": job["meta"]["n_parameters"],
            "capacity_control": job["capacity_control"],
            "threshold": thr,
            "calibration_n_safe": thr_info["n_safe"],
            "calibration_empirical_fpr": thr_info["empirical_fpr"],
            "calibration_unstable": thr_info["unstable"],
            "n_safe": quality["n_safe"],
            "n_harm": quality["n_harm"],
            "n_prompt_groups": n_groups,
        }
        flat = _flat_summary(spec, quality, ranking, cis)
        summaries.append(flat)
        by_key[(job["layer"], job["name"], job["seed"])] = rows
        for grow in group_rows(rows):
            grouped.append({**spec, **grow})
        fpr = quality["safe_response_fpr"]
        rec32 = quality["recall_by_end_plus_32"]
        prem = quality["premature_alarm_rate"]
        print(f"[evaluate] {args.split} L{job['layer']} {job['name']} seed={job['seed']} "
              f"thr={thr:.3f} fpr={fpr} recall+32={rec32} premature={prem}", flush=True)
        run = wandb_start(cfg, job="evaluate", name=f"{cfg['run_name']}-eval-L{job['layer']}-{job['name']}-s{job['seed']}",
                          extra={"method": job["name"], "layer": job["layer"], "seed": job["seed"], "split": args.split})
        wandb_log(run, {"threshold": thr, "safe_response_fpr": fpr or 0.0,
                        "recall_by_end_plus_32": rec32 or 0.0, "premature_alarm_rate": prem or 0.0})
        wandb_finish(run)
    pair_keys = [
        "safe_response_fpr", "recall_by_end_plus_32", "premature_alarm_rate",
        "first_harm_sentence_alarm_rate", "nonpremature_detection_rate",
    ]
    names = {job["name"] for job in loaded}
    if "h_mean" in names:
        paired = paired_differences(by_key, "h_mean", pair_keys, n_boot, boot_seed)
    else:
        wanted = {("kv_mean", "v_mean"), ("kv_mean", "k_mean"), ("v_mean", "k_mean")}
        paired = []
        for reference in ("v_mean", "k_mean"):
            for row in paired_differences(by_key, reference, pair_keys, n_boot, boot_seed):
                if (row["method"], row["reference"]) in wanted:
                    row["comparison"] = f"{row['method']}-{row['reference']}"
                    paired.append(row)
    for row in paired:
        row["split"] = args.split
        row["evaluation_role"] = role
    mdir = out / "metrics"
    mdir.mkdir(exist_ok=True)
    # Dev keeps the canonical names the report already reads. Test (and any other
    # split) gets its own files so a later held-out run cannot overwrite dev.
    # Dev keeps the names the pilot report reads, and also the full-train names.
    # Test files use the names in PLAN_FULL_TRAIN_TEST and never replace dev.
    names_for_split = {
        "dev": [("per_response.parquet", "quality_summary.csv", "quality_by_group.csv", "paired_differences.csv"),
                ("dev_per_response.parquet", "dev_quality_summary.csv", "dev_quality_by_group.csv", "dev_paired_differences.csv")],
        "test": [("test_per_response.parquet", "test_quality_summary.csv", "test_quality_by_group.csv", "test_paired_differences.csv")],
    }.get(args.split, [(f"per_response_{args.split}.parquet", f"quality_summary_{args.split}.csv",
                        f"quality_by_group_{args.split}.csv", f"paired_differences_{args.split}.csv")])
    keep = ["sample_id", "prompt_group_id", "split", "label", "T", "a", "b", "t_alarm",
            "max_score", "final_score", "method", "seed", "layer", "evaluation_role",
            "prompt_safety_label", "dataset_source", "alarm", "false_alarm", "premature",
            "no_alarm", "first_harm_sentence", "by_end_plus_32", "by_end_plus_128",
            "nonpremature_detection", "delay_tokens", "detection_fraction", "threshold"]
    frame = pd.DataFrame(per_response)
    for col in keep:
        if col not in frame.columns:
            frame[col] = None
    summary_frame = pd.DataFrame(summaries)
    grouped_frame = pd.DataFrame(grouped)
    paired_frame = pd.DataFrame(paired)
    for per_name, quality_name, group_name, paired_name in names_for_split:
        frame[keep].to_parquet(mdir / per_name, index=False)
        summary_frame.to_csv(mdir / quality_name, index=False)
        grouped_frame.to_csv(mdir / group_name, index=False)
        paired_frame.to_csv(mdir / paired_name, index=False)
    if args.split == "test":
        exposure = out / "test_exposure.json"
        prev = json.loads(exposure.read_text()) if exposure.exists() else {}
        prev["this_run_scored_test"] = True
        prev["scored_after_protocol_lock"] = True
        exposure.write_text(json.dumps(prev, indent=2))
    if args.split == "dev" and "h_mean" in names:
        selection = _layer_selection(summaries, int(cfg["model"]["layer_index"]))
        (mdir / "layer_selection.json").write_text(json.dumps(selection, indent=2))
    status = mdir / "run_status.csv"
    pd.DataFrame([{"stage": f"evaluate:{args.split}", "status": "ok", "n_jobs": len(summaries),
                   "detail": role}]).to_csv(status, mode="a", header=not status.exists(), index=False)


if __name__ == "__main__":
    main()
