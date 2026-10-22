"""v_fuse: response v_mean trained with a fused prompt-or-response label.

Label at each response token t:
- harmful prompt: 1 for every generated token, including a safe answer
- benign prompt: the original span rule (0 before the harm sentence, ignore inside it, 1 at and after its end; a safe answer is 0 everywhere)

The input is the layer-11 FP32 prefix mean of response V. Prompt tokens are not averaged in.
This run writes artifacts/full/v_fuse and does not change the locked response heads.

python -m kv_safety.v_fuse --config configs/full.yaml
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .classifier import load_checkpoint, save_checkpoint
from .config import REPO_ROOT, artifacts_dir, base_parser, load_config, pinned_revision, write_json
from .data.labels import prefix_labels, position_weights, select_train_positions
from .extract import extract_response_features_layers, load_prepared
from .metrics import auroc_score, calibrate_threshold, first_alarm, outcome, wilson_interval
from .train import predict, train_one

METHOD = "v_fuse"
SCHEME = "v_fuse_v1"
LAYER_LOCKED = 11
_SCRATCH = ("artifacts/full/features", "wandb", ".pip-cache")


def fuse_labels(T: int, a: int | None, b: int | None, prompt_label: int) -> np.ndarray:
    """Harmful prompt -> every response token is positive. Benign prompt -> span labels."""
    if int(prompt_label) == 1:
        if T <= 0:
            raise ValueError("empty response")
        return np.ones(T, dtype=np.int64)
    if int(prompt_label) != 0:
        raise ValueError(f"prompt label must be 0 or 1, got {prompt_label}")
    return prefix_labels(T, a, b)


def fuse_dir(cfg: dict) -> Path:
    path = artifacts_dir(cfg) / METHOD
    path.mkdir(parents=True, exist_ok=True)
    return path


def _span(value) -> int | None:
    if value is None:
        return None
    return int(value)


def ensure_disk(path: Path, need_bytes: int) -> dict:
    usage = shutil.disk_usage(path if path.exists() else REPO_ROOT)
    report = {"free_bytes_before": usage.free, "need_bytes": need_bytes, "deleted": []}
    if usage.free >= need_bytes:
        report["action"] = "enough_space"
        return report
    if os.environ.get("KVGUARD_ALLOW_SCRATCH_DELETE") != "1":
        raise SystemExit(
            f"disk short ({usage.free} free, {need_bytes} needed). "
            "Set KVGUARD_ALLOW_SCRATCH_DELETE=1 to delete recomputable scratch "
            "under this repo (feature shards, wandb, .pip-cache)."
        )
    for rel in _SCRATCH:
        target = REPO_ROOT / rel
        if target.exists() and not path.resolve().is_relative_to(target.resolve()):
            report["deleted"].append(str(target.relative_to(REPO_ROOT)))
            shutil.rmtree(target)
    usage = shutil.disk_usage(path if path.exists() else REPO_ROOT)
    report["free_bytes_after"] = usage.free
    if usage.free < need_bytes:
        raise SystemExit(
            f"disk still short ({usage.free} free, {need_bytes} needed); "
            "refusing to delete the model cache, FineHarm, pilot, prompt heads, or locked full-run artifacts")
    report["action"] = "deleted_scratch"
    return report


def _shard_key(ids: list[str]) -> str:
    payload = {"scheme": SCHEME, "ids": ids}
    return hashlib.sha1(json.dumps(payload).encode()).hexdigest()[:16]


def _feature_dir(out: Path, split: str) -> Path:
    path = out / "features" / split
    path.mkdir(parents=True, exist_ok=True)
    return path


def extract_split(model, rows: list[dict], out: Path, split: str, layer: int, cfg: dict) -> None:
    fc = cfg["features"]
    spec = {layer: ["v_mean"]}
    ss = int(fc["shard_size"])
    fdir = _feature_dir(out, split)
    done = 0
    t0 = time.time()
    for si, start in enumerate(range(0, len(rows), ss)):
        chunk = rows[start:start + ss]
        ids = [r["id"] for r in chunk]
        path = fdir / f"shard_{si:05d}.npz"
        key_path = fdir / f"shard_{si:05d}.key"
        key = _shard_key(ids)
        if path.exists() and key_path.exists() and key_path.read_text() == key:
            done += len(chunk)
            continue
        cols, sample_idx, position, label, weight = [], [], [], [], []
        for j, rec in enumerate(tqdm(chunk, desc=f"{split} shard {si}", leave=False)):
            lab = fuse_labels(rec["T"], _span(rec["a"]), _span(rec["b"]), rec["prompt_safety_label"])
            pos = select_train_positions(
                lab, _span(rec["a"]), _span(rec["b"]), fc["max_train_positions"], fc["priority_after_end"])
            if len(pos) == 0:
                raise RuntimeError(f"{rec['id']} has no supervised positions")
            feats = extract_response_features_layers(
                model, rec["prompt_ids"], rec["response_ids"], spec, fc["chunk_size"], pos)
            values = feats[layer]["v_mean"]
            if values.shape[0] != len(pos):
                raise RuntimeError(f"{rec['id']} feature rows {values.shape[0]} != positions {len(pos)}")
            if not np.isfinite(values).all():
                raise FloatingPointError(f"non-finite v_mean for {rec['id']}")
            cols.append(values.astype(np.float32, copy=False))
            sample_idx.append(np.full(len(pos), start + j, np.int32))
            position.append(pos.astype(np.int32))
            label.append(lab[pos - 1].astype(np.int16))
            weight.append(position_weights(len(pos)))
        np.savez(
            path,
            id=np.array(ids),
            sample_idx=np.concatenate(sample_idx),
            position=np.concatenate(position),
            label=np.concatenate(label),
            weight=np.concatenate(weight),
            v_mean=np.concatenate(cols),
        )
        key_path.write_text(key)
        done += len(chunk)
        rate = done / max(time.time() - t0, 1e-6)
        print(f"[v_fuse] extract {split} {done}/{len(rows)} ({rate:.2f} resp/s)", flush=True)


def _load_features(out: Path, split: str, rows: list[dict]) -> dict[str, np.ndarray]:
    fdir = _feature_dir(out, split)
    shards = sorted(fdir.glob("shard_*.npz"))
    if not shards:
        raise SystemExit(f"missing features for {split}")
    parts = {k: [] for k in ("id", "sample_idx", "position", "label", "weight", "v_mean")}
    for path in shards:
        with np.load(path, allow_pickle=False) as blob:
            for key in parts:
                parts[key].append(blob[key])
    arrays = {key: np.concatenate(val) for key, val in parts.items()}
    if list(arrays["id"]) != [r["id"] for r in rows]:
        raise SystemExit(f"{split} feature ids do not match the prepared split")
    return arrays


def _features_ready(out: Path, split: str, rows: list[dict], shard_size: int) -> bool:
    fdir = out / "features" / split
    n_shards = (len(rows) + shard_size - 1) // shard_size
    for si in range(n_shards):
        ids = [r["id"] for r in rows[si * shard_size:(si + 1) * shard_size]]
        key_path = fdir / f"shard_{si:05d}.key"
        path = fdir / f"shard_{si:05d}.npz"
        if not path.exists() or not key_path.exists() or key_path.read_text() != _shard_key(ids):
            return False
    return True


def extract_all(cfg: dict, out: Path, device: str) -> None:
    # 32 positions of 1024-d float32 for train and dev, plus margin.
    disk = ensure_disk(out, need_bytes=4 * 1024 ** 3)
    print(f"[v_fuse] disk {disk}", flush=True)
    write_json(out / "disk.json", disk)
    layer = int(cfg["model"]["layer_index"])
    data_root = artifacts_dir(cfg)
    pending = []
    for split in ("train", "dev"):
        rows = load_prepared(data_root, split)
        if _features_ready(out, split, rows, int(cfg["features"]["shard_size"])):
            print(f"[v_fuse] reuse {split} features ({len(rows)} rows)", flush=True)
        else:
            pending.append((split, rows))
    if not pending:
        return
    import kv_safety.modeling as modeling

    model = modeling.load_model(cfg, device)
    for split, rows in pending:
        print(f"[v_fuse] extract {split} n={len(rows)} layer={layer}", flush=True)
        extract_split(model, rows, out, split, layer, cfg)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _label_census(rows: list[dict]) -> dict:
    counts = {"hp_harm": 0, "hp_safe": 0, "bp_harm": 0, "bp_safe": 0}
    for rec in rows:
        key = ("h" if rec["prompt_safety_label"] == 1 else "b") + "p_" + ("harm" if rec["response_label"] == 1 else "safe")
        counts[key] += 1
    return counts


def train_heads(cfg: dict, out: Path, device: str) -> None:
    data_root = artifacts_dir(cfg)
    train_rows = load_prepared(data_root, "train")
    dev_rows = load_prepared(data_root, "dev")
    train = _load_features(out, "train", train_rows)
    dev = _load_features(out, "dev", dev_rows)
    if not np.isfinite(train["v_mean"]).all() or not np.isfinite(dev["v_mean"]).all():
        raise FloatingPointError("non-finite v_fuse features")
    tc = cfg["train"]
    hidden = int(cfg["classifier"]["hidden"])
    layer = int(cfg["model"]["layer_index"])
    summary = []
    print(f"[v_fuse] train label census { _label_census(train_rows) }", flush=True)
    print(f"[v_fuse] dev label census { _label_census(dev_rows) }", flush=True)
    for seed in [int(s) for s in tc["seeds"]]:
        dest = out / "checkpoints" / str(seed)
        tag = f"{METHOD} seed={seed}"
        if (dest / "meta.json").exists() and (dest / "model.pt").exists():
            prev = json.loads((dest / "meta.json").read_text())
            print(f"[v_fuse] resume {tag} epoch={prev.get('best_epoch')}", flush=True)
            summary.append({
                "method": METHOD, "seed": seed, "resumed": True,
                "best_epoch": prev.get("best_epoch"),
                "best_dev_group_bce": prev.get("best_dev_response_bce"),
                "n_parameters": prev.get("n_parameters"),
            })
            continue
        t0 = time.time()
        model, info = train_one(
            train["v_mean"], train["label"].astype(np.int64), train["weight"].astype(np.float32),
            dev["v_mean"], dev["label"].astype(np.int64), dev["sample_idx"].astype(np.int64),
            hidden, seed, tc, device, tag=tag)
        seconds = round(time.time() - t0, 1)
        meta = {
            "classifier": METHOD,
            "method": METHOD,
            "feature": "v_mean",
            "scheme": SCHEME,
            "task": "harmful prompt -> all response tokens positive; benign prompt -> original span labels",
            "layer_index": layer,
            "seed": seed,
            "hidden": hidden,
            "train_rows": int(len(train["v_mean"])),
            "train_responses": len(train_rows),
            "dev_rows": int(len(dev["v_mean"])),
            "dev_responses": len(dev_rows),
            "train_census": _label_census(train_rows),
            "dev_census": _label_census(dev_rows),
            "seconds": seconds,
            "best_epoch": info["best_epoch"],
            "best_dev_response_bce": info["best_dev_response_bce"],
            "model_id": cfg["model"]["id"],
            "revision": pinned_revision(cfg, "model"),
        }
        save_checkpoint(model.cpu(), dest, meta)
        summary.append({
            "method": METHOD, "seed": seed, "resumed": False,
            "best_epoch": info["best_epoch"],
            "best_dev_group_bce": info["best_dev_response_bce"],
            "n_parameters": model.n_parameters(),
            "seconds": seconds,
        })
        print(f"[v_fuse] {tag} best_epoch={info['best_epoch']} "
              f"dev_bce={info['best_dev_response_bce']:.4f} seconds={seconds}", flush=True)
    write_json(out / "metrics" / "train_summary.json", summary)


@torch.inference_mode()
def _score_rows(model, rows: list[dict], heads: dict[int, object], device: str, layer: int, chunk: int
                ) -> dict[int, list[np.ndarray]]:
    spec = {layer: ["v_mean"]}
    out = {seed: [] for seed in heads}
    for rec in tqdm(rows, desc="score", leave=False):
        feats = extract_response_features_layers(
            model, rec["prompt_ids"], rec["response_ids"], spec, chunk, None)
        values = feats[layer]["v_mean"]
        if len(values) != rec["T"] or not np.isfinite(values).all():
            raise RuntimeError(f"bad score features for {rec['id']}")
        for seed, head in heads.items():
            out[seed].append(predict(head, values, device).astype(np.float64))
    return out


def _load_heads(out: Path, seeds: list[int], device: str) -> dict[int, object]:
    heads = {}
    for seed in seeds:
        module, _ = load_checkpoint(out / "checkpoints" / str(seed), device)
        heads[seed] = module
    return heads


def _f1(tp: int, fp: int, fn: int) -> float | None:
    if tp + fp == 0 or tp + fn == 0:
        return None
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def _alarm_stats(rows: list[dict], scores: list[np.ndarray], thr: float) -> list[dict]:
    detailed = []
    for rec, logit in zip(rows, scores):
        t_alarm = first_alarm(logit, thr)
        fused = int(rec["prompt_safety_label"] == 1 or rec["response_label"] == 1)
        item = {
            "id": rec["id"],
            "prompt_group": rec["prompt_group"],
            "prompt_label": int(rec["prompt_safety_label"]),
            "response_label": int(rec["response_label"]),
            "fused_label": fused,
            "T": int(rec["T"]),
            "a": _span(rec["a"]),
            "b": _span(rec["b"]),
            "t_alarm": t_alarm,
            "max_logit": float(np.max(logit)),
        }
        item.update(outcome(t_alarm, int(rec["response_label"]), int(rec["T"]), _span(rec["a"]), _span(rec["b"]),
                            windows=(32, 64, 128)))
        detailed.append(item)
    return detailed


def _pack(detailed: list[dict], method: str, seed: int, split: str, thr: float) -> dict:
    fused = np.array([r["fused_label"] for r in detailed], dtype=bool)
    alarm = np.array([r["t_alarm"] is not None for r in detailed])
    scores = np.array([r["max_logit"] for r in detailed], dtype=np.float64)
    tp = int((alarm & fused).sum())
    fp = int((alarm & ~fused).sum())
    tn = int((~alarm & ~fused).sum())
    fn = int((~alarm & fused).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    prompt = np.array([r["prompt_label"] for r in detailed])
    response = np.array([r["response_label"] for r in detailed])
    harm = [r for r in detailed if r["response_label"] == 1 and r["prompt_label"] == 0]

    def rate(pool, key):
        if not pool:
            return None
        return float(np.mean([bool(r[key]) for r in pool]))

    def alarm_rate(mask):
        n = int(mask.sum())
        if n == 0:
            return n, None
        return n, float(alarm[mask].mean())

    hp = prompt == 1
    hp_safe_n, hp_safe = alarm_rate(hp & (response == 0))
    hp_harm_n, hp_harm = alarm_rate(hp & (response == 1))
    bp_safe_n, bp_safe = alarm_rate((prompt == 0) & (response == 0))
    bp_harm_n, bp_harm = alarm_rate((prompt == 0) & (response == 1))
    alarmed_hp = [r["t_alarm"] for r in detailed if r["prompt_label"] == 1 and r["t_alarm"] is not None]
    fpr_w = wilson_interval(fp, int((~fused).sum())) if (~fused).any() else None
    rec_w = wilson_interval(tp, int(fused.sum())) if fused.any() else None
    return {
        "split": split, "method": method, "seed": seed, "threshold": thr,
        "n": len(detailed), "n_pos": int(fused.sum()), "n_neg": int((~fused).sum()),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall, "f1": _f1(tp, fp, fn),
        "fpr": fp / int((~fused).sum()) if (~fused).any() else None,
        "auroc": auroc_score(fused, scores),
        "fpr_wilson95_lo": None if fpr_w is None else fpr_w[0],
        "fpr_wilson95_hi": None if fpr_w is None else fpr_w[1],
        "recall_wilson95_lo": None if rec_w is None else rec_w[0],
        "recall_wilson95_hi": None if rec_w is None else rec_w[1],
        "n_hp_safe_response": hp_safe_n, "alarm_rate_hp_safe_response": hp_safe,
        "n_hp_harm_response": hp_harm_n, "alarm_rate_hp_harm_response": hp_harm,
        "n_bp_safe_response": bp_safe_n, "alarm_rate_bp_safe_response": bp_safe,
        "n_bp_harm_response": bp_harm_n, "alarm_rate_bp_harm_response": bp_harm,
        "hp_alarm_t_p50": None if not alarmed_hp else float(np.median(alarmed_hp)),
        "bp_harm_premature": rate(harm, "premature"),
        "bp_harm_timely": rate(harm, "first_harm_sentence"),
        "bp_harm_plus32": rate(harm, "by_end_plus_32"),
        "bp_harm_plus64": rate(harm, "by_end_plus_64"),
        "bp_harm_plus128": rate(harm, "by_end_plus_128"),
        "bp_harm_none": rate(harm, "no_alarm"),
        "bp_harm_n": len(harm),
    }


def calibrate_and_evaluate(cfg: dict, out: Path, device: str) -> None:
    import kv_safety.modeling as modeling

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
    print(f"[v_fuse] score calibration n={len(cal_rows)}", flush=True)
    cal_scores = _score_rows(model, cal_rows, heads, device, layer, chunk)
    thresholds = {}
    cal_summary = []
    for seed in seeds:
        benign = [i for i, rec in enumerate(cal_rows) if rec["prompt_safety_label"] == 0]
        safe_max = np.array([float(np.max(cal_scores[seed][i])) for i in benign], dtype=np.float64)
        thr, info = calibrate_threshold(safe_max, target)
        info.update({
            "target_fpr": target, "split": "calibration", "unit": "benign_prompt_safe_response",
            "method": METHOD, "seed": seed,
            "n_calibration_rows": len(cal_rows),
            "n_harmful_prompt_excluded": int(sum(r["prompt_safety_label"] == 1 for r in cal_rows)),
            "excluded": "harmful-prompt rows are positives under v_fuse and are not calibration negatives",
        })
        write_json(out / "checkpoints" / str(seed) / "threshold.json", info)
        thresholds[seed] = thr
        cal_summary.append({
            "method": METHOD, "seed": seed, "threshold": thr,
            "n_benign": info["n_safe"], "n_false_alarms": info["n_false_alarms"],
            "empirical_fpr": info["empirical_fpr"],
            "wilson95_lo": info["wilson95_lo"], "wilson95_hi": info["wilson95_hi"],
            "unstable": info["unstable"], "unstable_reason": info["unstable_reason"],
            "target_fpr": target,
        })
        print(f"[v_fuse] calibrate seed={seed} thr={thr:.4f} fpr={info['empirical_fpr']:.4f} "
              f"n={info['n_safe']}", flush=True)
    pd.DataFrame(cal_summary).to_csv(out / "metrics" / "calibration.csv", index=False)

    frames = []
    test_rows = load_prepared(data_root, "test")
    print(f"[v_fuse] score test n={len(test_rows)}", flush=True)
    test_scores = _score_rows(model, test_rows, heads, device, layer, chunk)
    detail = []
    for seed in seeds:
        detailed = _alarm_stats(test_rows, test_scores[seed], thresholds[seed])
        packed = _pack(detailed, METHOD, seed, "test", thresholds[seed])
        frames.append(packed)
        for row in detailed:
            detail.append({**row, "method": METHOD, "seed": seed, "threshold": thresholds[seed]})
        print(f"[v_fuse] test seed={seed} precision={packed['precision']:.4f} "
              f"recall={packed['recall']:.4f} f1={packed['f1']:.4f} fpr={packed['fpr']:.4f}", flush=True)
    summary = pd.DataFrame(frames)
    summary.to_csv(out / "metrics" / "summary.csv", index=False)
    pd.DataFrame(detail).to_csv(out / "metrics" / "test_per_row.csv", index=False)
    means = {"method": METHOD, "n_seeds": len(seeds)}
    for col in ("precision", "recall", "f1", "fpr", "auroc",
                "alarm_rate_hp_safe_response", "alarm_rate_hp_harm_response",
                "alarm_rate_bp_safe_response", "alarm_rate_bp_harm_response",
                "bp_harm_premature", "bp_harm_timely", "bp_harm_plus32",
                "bp_harm_plus64", "bp_harm_plus128", "bp_harm_none", "hp_alarm_t_p50"):
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
        "feature": "layer-11 FP32 prefix mean of response V; prompt V is not included",
        "label": {
            "harmful_prompt": "every response token is 1",
            "benign_prompt": "original span labels: 0 before a, ignore on [a, b), 1 at and after b; safe response is 0",
        },
        "train": "all train rows, equal weight per response, at most 32 positions",
        "early_stopping": "dev loss under the fused labels. Test does not choose the epoch.",
        "threshold": "calibration rows with a benign prompt and a safe response, target FPR from the config. "
                     "Harmful-prompt calibration rows are excluded.",
        "alarm": "first response token with logit > threshold",
        "fused_positive": "prompt harmful or response harmful",
        "does_not_change": ["response heads", "artifacts/full/protocol_lock.json", "prompt_head checkpoints"],
        "model_id": cfg["model"]["id"],
        "revision": pinned_revision(cfg, "model"),
        "layer_index": int(cfg["model"]["layer_index"]),
    })


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config)
    if int(cfg["model"]["layer_index"]) != LAYER_LOCKED:
        raise SystemExit(f"v_fuse is for the locked deployment layer {LAYER_LOCKED}")
    out = fuse_dir(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    write_note(cfg, out)
    extract_all(cfg, out, device)
    train_heads(cfg, out, device)
    calibrate_and_evaluate(cfg, out, device)
    print("[v_fuse] done", flush=True)


if __name__ == "__main__":
    main()
