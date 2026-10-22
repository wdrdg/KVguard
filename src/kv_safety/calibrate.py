"""Replay calibration responses and set a response-level FPR threshold per checkpoint.

python -m kv_safety.calibrate --config configs/pilot.yaml --split calibration --all-features

Uses every safe response in the calibration split. Does not read dev or test labels to choose
the threshold. Streaming scores are stored; high-dimensional features are not.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from .config import artifacts_dir, base_parser, load_config, select_jobs, write_json
from .extract import load_prepared
from .metrics import calibrate_threshold
from .replay import assert_split_allowed, load_jobs, replay_scores, trajectories


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--split", default="calibration")
    ap.add_argument("--all-features", action="store_true")
    ap.add_argument("--features", nargs="+", default=None)
    ap.add_argument("--layers", nargs="+", type=int, default=None)
    ap.add_argument("--protocol-lock", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.split != "calibration":
        raise SystemExit("calibrate only fits thresholds on the calibration split")
    assert_split_allowed(cfg, args.split, args.protocol_lock)
    out = artifacts_dir(cfg)
    records = load_prepared(out, "calibration")
    if any(r["response_label"] != 0 for r in records):
        raise SystemExit("calibration jsonl must contain only safe responses")
    if not records:
        raise SystemExit("calibration split is empty")
    jobs = select_jobs(cfg, args.all_features or not args.features, args.features, args.layers)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded = load_jobs(cfg, out, jobs, [int(s) for s in cfg["train"]["seeds"]], device)
    import kv_safety.modeling as modeling
    model = modeling.load_model(cfg, device)
    scored = replay_scores(model, records, cfg, loaded, device, out, "calibration")
    del model
    rows = []
    target = float(cfg["evaluate"]["target_fpr"])
    for job in loaded:
        key = (job["layer"], job["name"], job["seed"])
        traj = trajectories(scored[key])
        safe_max = np.array([traj[i].max() for i in range(len(records))], dtype=np.float64)
        thr, info = calibrate_threshold(safe_max, target)
        info = {**info, "target_fpr": target, "split": "calibration"}
        thr_path = out / "checkpoints" / str(job["layer"]) / job["name"] / str(job["seed"]) / "threshold.json"
        write_json(thr_path, info)
        groups = {r["prompt_group"] for r in records}
        rows.append({
            "split": "calibration",
            "method": job["name"],
            "layer": job["layer"],
            "seed": job["seed"],
            "feature": job["feature"],
            "feature_dim": job["meta"]["in_dim"],
            "parameter_count": job["meta"]["n_parameters"],
            "capacity_control": job["capacity_control"],
            "n_safe": info["n_safe"],
            "n_prompt_groups": len(groups),
            "n_false_alarms": info["n_false_alarms"],
            "threshold": info["threshold"],
            "empirical_fpr": info["empirical_fpr"],
            "wilson95_lo": info["wilson95_lo"],
            "wilson95_hi": info["wilson95_hi"],
            "unstable": info["unstable"],
            "unstable_reason": info["unstable_reason"],
            "target_fpr": target,
        })
        print(f"[calibrate] L{job['layer']} {job['name']} seed={job['seed']} "
              f"thr={thr:.4f} fpr={info['empirical_fpr']:.4f} n={info['n_safe']} "
              f"unstable={info['unstable']}", flush=True)
    mdir = out / "metrics"
    mdir.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(mdir / "calibration_summary.csv", index=False)
    status = mdir / "run_status.csv"
    extra = pd.DataFrame([{"stage": "calibrate", "status": "ok", "n_jobs": len(rows),
                           "detail": "calibration safe responses only"}])
    extra.to_csv(status, mode="a", header=not status.exists(), index=False)
    print(json.dumps({"n_rows": len(rows), "n_safe": len(records)}, indent=2))


if __name__ == "__main__":
    main()
