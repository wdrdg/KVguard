"""v_early: response v_mean with Guard-style early labels. Does not replace v_fuse.

- harmful prompt, harmful response: every response token is 1
- harmful prompt, safe response: every response token is 0
- benign prompt, safe response: every response token is 0
- benign prompt, harmful response: original span labels

Features are the layer-11 response v_mean already extracted for v_fuse. Only the stored
labels change, because the selected positions are the same under this rule. Results go to
artifacts/full/v_early. The locked response heads and the previous v_fuse run stay put.

python -m kv_safety.v_early --config configs/full.yaml
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .classifier import save_checkpoint
from .config import artifacts_dir, base_parser, load_config, pinned_revision, write_json
from .data.labels import prefix_labels, select_train_positions
from .extract import load_prepared
from .metrics import auroc_score, calibrate_threshold, wilson_interval
from .train import train_one
from .v_fuse import _f1, _load_features, _load_heads, _score_rows, _span, ensure_disk

METHOD = "v_early"
SCHEME = "v_early_v1"


def early_labels(T: int, a: int | None, b: int | None, prompt_label: int, response_label: int) -> np.ndarray:
    """Early positive labels only when both the prompt and the response are harmful."""
    if int(prompt_label) not in (0, 1) or int(response_label) not in (0, 1):
        raise ValueError(f"labels must be 0 or 1, got prompt={prompt_label} response={response_label}")
    if T <= 0:
        raise ValueError("empty response")
    if int(response_label) == 0:
        return np.zeros(T, dtype=np.int64)
    if int(prompt_label) == 1:
        return np.ones(T, dtype=np.int64)
    if a is None or b is None:
        raise ValueError("benign prompt with a harmful response is missing the span [a, b]")
    return prefix_labels(T, a, b)


def _out_dir(cfg: dict) -> Path:
    path = artifacts_dir(cfg) / METHOD
    path.mkdir(parents=True, exist_ok=True)
    return path


def _key(ids: list[str]) -> str:
    return hashlib.sha1(json.dumps({"scheme": SCHEME, "ids": ids}).encode()).hexdigest()[:16]


def relabel_split(cfg: dict, out: Path, split: str) -> None:
    """Copy v_fuse shards and replace labels. V itself is unchanged."""
    rows = load_prepared(artifacts_dir(cfg), split)
    src = artifacts_dir(cfg) / "v_fuse" / "features" / split
    dest = out / "features" / split
    dest.mkdir(parents=True, exist_ok=True)
    shards = sorted(src.glob("shard_*.npz"))
    if not shards:
        raise SystemExit(f"missing v_fuse features for {split}; the V cache was not re-extracted")
    fc = cfg["features"]
    changed = 0
    checked = 0
    for path in shards:
        key_path = path.with_suffix(".key")
        old_key = key_path.read_text() if key_path.exists() else ""
        with np.load(path, allow_pickle=False) as blob:
            arrays = {name: blob[name] for name in blob.files}
        ids = [str(x) for x in arrays["id"].tolist()]
        new_key = _key(ids)
        target = dest / path.name
        target_key = dest / key_path.name
        if target.exists() and target_key.exists() and target_key.read_text() == new_key:
            checked += len(ids)
            continue
        if not old_key:
            raise SystemExit(f"{path} has no scheme key")
        labels = arrays["label"].astype(np.int16, copy=True)
        sample_idx = arrays["sample_idx"]
        position = arrays["position"]
        # Positions are grouped by sample because extraction appends one response at a time.
        cuts = np.flatnonzero(np.diff(sample_idx)) + 1
        bounds = np.r_[0, cuts, len(sample_idx)]
        seen = []
        for start, stop in zip(bounds[:-1], bounds[1:]):
            rec = rows[int(sample_idx[start])]
            if rec["id"] != ids[len(seen)]:
                raise SystemExit(f"{split} shard order does not match prepared ids at {rec['id']}")
            seen.append(rec["id"])
            lab = early_labels(rec["T"], _span(rec["a"]), _span(rec["b"]),
                               rec["prompt_safety_label"], rec["response_label"])
            expect = select_train_positions(
                lab, _span(rec["a"]), _span(rec["b"]), fc["max_train_positions"], fc["priority_after_end"])
            got = position[start:stop]
            if not np.array_equal(got, expect):
                raise SystemExit(f"{rec['id']} stored positions differ from the early-label selection")
            labels[start:stop] = lab[got - 1].astype(np.int16)
        if seen != ids:
            raise SystemExit(f"{split} shard {path.name} did not cover its ids")
        changed += int((labels != arrays["label"]).sum())
        arrays["label"] = labels
        np.savez(target, **arrays)
        target_key.write_text(new_key)
        checked += len(ids)
    print(f"[v_early] relabel {split} responses={checked} positions_changed={changed}", flush=True)


def train_heads(cfg: dict, out: Path, device: str) -> None:
    data_root = artifacts_dir(cfg)
    train_rows = load_prepared(data_root, "train")
    dev_rows = load_prepared(data_root, "dev")
    train = _load_features(out, "train", train_rows)
    dev = _load_features(out, "dev", dev_rows)
    tc = cfg["train"]
    hidden = int(cfg["classifier"]["hidden"])
    summary = []
    for seed in [int(s) for s in tc["seeds"]]:
        dest = out / "checkpoints" / str(seed)
        tag = f"{METHOD} seed={seed}"
        if (dest / "meta.json").exists() and (dest / "model.pt").exists():
            prev = json.loads((dest / "meta.json").read_text())
            print(f"[v_early] resume {tag} epoch={prev.get('best_epoch')}", flush=True)
            summary.append({"method": METHOD, "seed": seed, "resumed": True,
                            "best_epoch": prev.get("best_epoch"),
                            "best_dev_group_bce": prev.get("best_dev_response_bce"),
                            "n_parameters": prev.get("n_parameters")})
            continue
        t0 = time.time()
        model, info = train_one(
            train["v_mean"], train["label"].astype(np.int64), train["weight"].astype(np.float32),
            dev["v_mean"], dev["label"].astype(np.int64), dev["sample_idx"].astype(np.int64),
            hidden, seed, tc, device, tag=tag)
        seconds = round(time.time() - t0, 1)
        meta = {
            "classifier": METHOD, "method": METHOD, "feature": "v_mean", "scheme": SCHEME,
            "task": "HP+HR all tokens 1; safe responses all 0; BP+HR original span",
            "layer_index": int(cfg["model"]["layer_index"]), "seed": seed, "hidden": hidden,
            "train_responses": len(train_rows), "dev_responses": len(dev_rows),
            "seconds": seconds, "best_epoch": info["best_epoch"],
            "best_dev_response_bce": info["best_dev_response_bce"],
            "model_id": cfg["model"]["id"], "revision": pinned_revision(cfg, "model"),
        }
        save_checkpoint(model.cpu(), dest, meta)
        summary.append({"method": METHOD, "seed": seed, "resumed": False,
                        "best_epoch": info["best_epoch"],
                        "best_dev_group_bce": info["best_dev_response_bce"],
                        "n_parameters": model.n_parameters(), "seconds": seconds})
        print(f"[v_early] {tag} best_epoch={info['best_epoch']} "
              f"dev_bce={info['best_dev_response_bce']:.4f} seconds={seconds}", flush=True)
    write_json(out / "metrics" / "train_summary.json", summary)


def _rates(part: list[dict]) -> dict:
    y = np.array([r["response_label"] for r in part], dtype=bool)
    alarm = np.array([r["t_alarm"] is not None for r in part])
    scores = np.array([r["max_logit"] for r in part], dtype=np.float64)
    prompt = np.array([r["prompt_label"] for r in part])
    tp = int((alarm & y).sum())
    fp = int((alarm & ~y).sum())
    tn = int((~alarm & ~y).sum())
    fn = int((~alarm & y).sum())
    harm = [r for r in part if r["response_label"] == 1 and r["prompt_label"] == 0]
    hp_harm = [r for r in part if r["response_label"] == 1 and r["prompt_label"] == 1]

    def mean_flag(pool, key):
        if not pool:
            return None
        return float(np.mean([bool(r[key]) for r in pool]))

    def alarm_rate(mask):
        n = int(mask.sum())
        if n == 0:
            return n, None
        return n, float(alarm[mask].mean())

    hp_safe_n, hp_safe = alarm_rate((prompt == 1) & ~y)
    bp_safe_n, bp_safe = alarm_rate((prompt == 0) & ~y)
    hp_harm_n, hp_harm_rate = alarm_rate((prompt == 1) & y)
    bp_harm_n, bp_harm_rate = alarm_rate((prompt == 0) & y)
    early = [r for r in hp_harm if r["t_alarm"] is not None and r["a"] is not None and r["t_alarm"] < r["a"]]
    at1 = [r for r in hp_harm if r["t_alarm"] == 1]
    alarmed = [r["t_alarm"] for r in hp_harm if r["t_alarm"] is not None]
    fpr_w = wilson_interval(fp, int((~y).sum()))
    rec_w = wilson_interval(tp, int(y.sum()))
    return {
        "n": len(part), "n_pos": int(y.sum()), "n_neg": int((~y).sum()),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": _f1(tp, fp, fn),
        "fpr": fp / int((~y).sum()) if (~y).any() else None,
        "auroc": auroc_score(y, scores),
        "fpr_wilson95_lo": None if fpr_w is None else fpr_w[0],
        "fpr_wilson95_hi": None if fpr_w is None else fpr_w[1],
        "recall_wilson95_lo": None if rec_w is None else rec_w[0],
        "recall_wilson95_hi": None if rec_w is None else rec_w[1],
        "n_hp_safe": hp_safe_n, "fpr_hp_safe": hp_safe,
        "n_bp_safe": bp_safe_n, "fpr_bp_safe": bp_safe,
        "n_hp_harm": hp_harm_n, "recall_hp_harm": hp_harm_rate,
        "hp_harm_before_sentence": len(early) / hp_harm_n if hp_harm_n else None,
        "hp_harm_t1": len(at1) / hp_harm_n if hp_harm_n else None,
        "hp_harm_t_p50": None if not alarmed else float(np.median(alarmed)),
        "n_bp_harm": bp_harm_n, "recall_bp_harm": bp_harm_rate,
        "bp_harm_premature": mean_flag(harm, "premature"),
        "bp_harm_timely": mean_flag(harm, "first_harm_sentence"),
        "bp_harm_plus32": mean_flag(harm, "by_end_plus_32"),
        "bp_harm_plus64": mean_flag(harm, "by_end_plus_64"),
        "bp_harm_plus128": mean_flag(harm, "by_end_plus_128"),
        "bp_harm_none": mean_flag(harm, "no_alarm"),
    }


def calibrate_and_evaluate(cfg: dict, out: Path, device: str) -> None:
    import kv_safety.modeling as modeling
    from .v_fuse import _alarm_stats

    data_root = artifacts_dir(cfg)
    seeds = [int(s) for s in cfg["train"]["seeds"]]
    heads = _load_heads(out, seeds, device)
    model = modeling.load_model(cfg, device)
    layer = int(cfg["model"]["layer_index"])
    chunk = int(cfg["features"]["chunk_size"])
    target = float(cfg["evaluate"]["target_fpr"])
    cal_rows = load_prepared(data_root, "calibration")
    if any(r["response_label"] != 0 for r in cal_rows):
        raise SystemExit("calibration split must contain only safe responses")
    print(f"[v_early] score calibration n={len(cal_rows)}", flush=True)
    cal_scores = _score_rows(model, cal_rows, heads, device, layer, chunk)
    thresholds = {}
    cal_summary = []
    for seed in seeds:
        # Safe answers are negatives even when the prompt is harmful.
        safe_max = np.array([float(np.max(scores)) for scores in cal_scores[seed]], dtype=np.float64)
        thr, info = calibrate_threshold(safe_max, target)
        info.update({
            "target_fpr": target, "split": "calibration", "unit": "safe_response",
            "method": METHOD, "seed": seed,
            "n_harmful_prompt_safe": int(sum(r["prompt_safety_label"] == 1 for r in cal_rows)),
            "note": "harmful-prompt safe responses are negatives under v_early",
        })
        write_json(out / "checkpoints" / str(seed) / "threshold.json", info)
        thresholds[seed] = thr
        cal_summary.append({
            "method": METHOD, "seed": seed, "threshold": thr, "n_safe": info["n_safe"],
            "n_false_alarms": info["n_false_alarms"], "empirical_fpr": info["empirical_fpr"],
            "wilson95_lo": info["wilson95_lo"], "wilson95_hi": info["wilson95_hi"],
            "unstable": info["unstable"], "target_fpr": target,
        })
        print(f"[v_early] calibrate seed={seed} thr={thr:.4f} fpr={info['empirical_fpr']:.4f} "
              f"n={info['n_safe']}", flush=True)
    pd.DataFrame(cal_summary).to_csv(out / "metrics" / "calibration.csv", index=False)

    test_rows = load_prepared(data_root, "test")
    print(f"[v_early] score test n={len(test_rows)}", flush=True)
    test_scores = _score_rows(model, test_rows, heads, device, layer, chunk)
    frames, detail = [], []
    for seed in seeds:
        detailed = _alarm_stats(test_rows, test_scores[seed], thresholds[seed])
        packed = _rates(detailed)
        packed.update({"split": "test", "method": METHOD, "seed": seed, "threshold": thresholds[seed]})
        frames.append(packed)
        for row in detailed:
            detail.append({**row, "method": METHOD, "seed": seed, "threshold": thresholds[seed]})
        print(f"[v_early] test seed={seed} precision={packed['precision']:.4f} "
              f"recall={packed['recall']:.4f} f1={packed['f1']:.4f} fpr={packed['fpr']:.4f} "
              f"hp_safe_fpr={packed['fpr_hp_safe']:.4f} hp_harm_before={packed['hp_harm_before_sentence']:.4f}",
              flush=True)
    summary = pd.DataFrame(frames)
    summary.to_csv(out / "metrics" / "summary.csv", index=False)
    pd.DataFrame(detail).to_csv(out / "metrics" / "test_per_row.csv", index=False)
    means = {"method": METHOD, "n_seeds": len(seeds)}
    for col in ("precision", "recall", "f1", "fpr", "auroc", "fpr_hp_safe", "fpr_bp_safe",
                "recall_hp_harm", "hp_harm_before_sentence", "hp_harm_t1", "hp_harm_t_p50",
                "recall_bp_harm", "bp_harm_premature", "bp_harm_timely", "bp_harm_plus32",
                "bp_harm_plus64", "bp_harm_plus128", "bp_harm_none"):
        vals = summary[col].astype(float).to_numpy()
        means[f"{col}_mean"] = float(np.nanmean(vals))
        means[f"{col}_sd"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
    pd.DataFrame([means]).to_csv(out / "metrics" / "test_mean.csv", index=False)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_note(cfg: dict, out: Path) -> None:
    write_json(out / "note.json", {
        "method": METHOD,
        "feature": "layer-11 response v_mean copied from the v_fuse extract; prompt V is not averaged",
        "label": {
            "harmful_prompt_harmful_response": "1 from the first response token",
            "safe_response": "0 on every response token, including a harmful prompt",
            "benign_prompt_harmful_response": "original span: 0 before a, ignore on [a, b), 1 at and after b",
        },
        "threshold": "all calibration safe responses, including harmful prompts, target FPR from the config",
        "positive_at_test": "response_label == 1. An alarm before the harm sentence on a harmful prompt counts as a hit.",
        "does_not_change": ["v_fuse checkpoints", "response heads", "artifacts/full/protocol_lock.json"],
        "model_id": cfg["model"]["id"],
        "revision": pinned_revision(cfg, "model"),
        "layer_index": int(cfg["model"]["layer_index"]),
    })


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config)
    if int(cfg["model"]["layer_index"]) != 11:
        raise SystemExit("v_early is for the locked deployment layer 11")
    out = _out_dir(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    disk = ensure_disk(out, need_bytes=4 * 1024 ** 3)
    print(f"[v_early] disk {disk}", flush=True)
    write_note(cfg, out)
    for split in ("train", "dev"):
        relabel_split(cfg, out, split)
    train_heads(cfg, out, device)
    calibrate_and_evaluate(cfg, out, device)
    print("[v_early] done", flush=True)


if __name__ == "__main__":
    main()
