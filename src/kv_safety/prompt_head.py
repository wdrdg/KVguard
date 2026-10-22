"""Prompt-level heads on layer-11 V. Does not change the response heads or protocol lock.

Two features, both from the V cache at model.layer_index after a prompt-only prefill
(the stored prompt_ids: chat template plus generation prompt, thinking off):

- prompt_v_last: V of the last prompt token
- prompt_v_mean: FP32 mean of V over every prompt token

Train uses every train row. Rows that share a prompt_group split weight 1, so a prompt
with several responses is not counted several times. Early stopping is the same
group-equal loss on dev. The threshold is fit only on calibration rows whose prompt
label is benign, at evaluate.target_fpr. Test does not choose the checkpoint or the
threshold.

python -m kv_safety.prompt_head --config configs/full.yaml
"""

from __future__ import annotations

import json
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .classifier import save_checkpoint
from .config import REPO_ROOT, artifacts_dir, base_parser, load_config, pinned_revision, write_json
from .features.cache_reader import check_cache, new_kv
from .metrics import auroc_score, average_precision, calibrate_threshold, wilson_interval
from .train import predict, train_one

METHODS = ("prompt_v_last", "prompt_v_mean")
# Recomputable leftovers only. The backbone cache, FineHarm, pilot, and the finished
# full-run checkpoints/metrics are not in this list.
_SCRATCH_DIRS = (
    "artifacts/full/features",
    "wandb",
    ".pip-cache",
)


def prompt_head_dir(cfg: dict) -> Path:
    path = artifacts_dir(cfg) / "prompt_head"
    path.mkdir(parents=True, exist_ok=True)
    return path


def group_weights(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Integer group codes and per-row weights 1/count(group). Each group sums to 1."""
    codes, inverse = np.unique(groups.astype(str), return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    weights = (1.0 / counts[inverse]).astype(np.float32)
    return inverse.astype(np.int64), weights


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
    for rel in _SCRATCH_DIRS:
        target = REPO_ROOT / rel
        if not target.exists():
            continue
        if path.resolve() == target.resolve() or path.resolve().is_relative_to(target.resolve()):
            continue
        report["deleted"].append(str(target.relative_to(REPO_ROOT)))
        shutil.rmtree(target)
    usage = shutil.disk_usage(path if path.exists() else REPO_ROOT)
    report["free_bytes_after"] = usage.free
    if usage.free < need_bytes:
        raise SystemExit(
            f"disk still short after deleting scratch ({usage.free} free, {need_bytes} needed); "
            "refusing to delete the model cache, FineHarm, pilot, or finished full-run artifacts")
    report["action"] = "deleted_scratch"
    return report


def load_split_rows(cfg: dict, split: str) -> list[dict]:
    path = artifacts_dir(cfg) / "data" / f"{split}.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"{split} is empty")
    missing = [r["id"] for r in rows if r.get("prompt_safety_label") not in (0, 1)]
    if missing:
        raise SystemExit(f"{split} has {len(missing)} rows without a prompt label, e.g. {missing[:3]}")
    return rows


def _batches(lengths: np.ndarray, token_budget: int) -> list[tuple[int, np.ndarray]]:
    """Groups of row indices that share a prompt length, capped by a token budget."""
    order = np.argsort(lengths, kind="mergesort")
    batches = []
    start = 0
    while start < len(order):
        length = int(lengths[order[start]])
        width = max(1, token_budget // max(length, 1))
        stop = start + 1
        while stop < len(order) and int(lengths[order[stop]]) == length and stop - start < width:
            stop += 1
        batches.append((length, order[start:stop]))
        start = stop
    return batches


@torch.inference_mode()
def _prompt_v(model, batch_ids: torch.Tensor, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    from transformers import DynamicCache

    base = model.model if hasattr(model, "model") else model
    cache = DynamicCache(config=model.config)
    base(input_ids=batch_ids, past_key_values=cache, use_cache=True)
    check_cache(cache)
    _, values = new_kv(cache, layer, batch_ids.shape[1])
    values = values.float()
    last = values[:, -1, :].contiguous()
    mean = values.sum(dim=1) / values.shape[1]
    del cache
    return last, mean


@torch.inference_mode()
def _batch_matches_single(model, layer: int, sequences: list[list[int]]) -> float:
    device = next(model.parameters()).device
    singles = []
    for seq in sequences:
        last, mean = _prompt_v(model, torch.tensor([seq], device=device), layer)
        singles.append((last[0].cpu(), mean[0].cpu()))
    batched_last, batched_mean = _prompt_v(model, torch.tensor(sequences, device=device), layer)
    diffs = []
    for i, (last, mean) in enumerate(singles):
        diffs.append(float((batched_last[i].cpu() - last).abs().max()))
        diffs.append(float((batched_mean[i].cpu() - mean).abs().max()))
    return max(diffs)


def extract_split(model, rows: list[dict], layer: int, token_budget: int) -> dict[str, np.ndarray]:
    device = next(model.parameters()).device
    ids = [r["prompt_ids"] for r in rows]
    lengths = np.array([len(x) for x in ids], dtype=np.int32)
    if (lengths < 1).any():
        raise RuntimeError("empty prompt")
    n = len(rows)
    last = np.empty((n, 0), np.float32)
    mean = np.empty((n, 0), np.float32)
    dim = None
    done = 0
    t0 = time.time()
    for length, index in _batches(lengths, token_budget):
        batch = torch.tensor([ids[int(i)] for i in index], device=device)
        batch_last, batch_mean = _prompt_v(model, batch, layer)
        if dim is None:
            dim = int(batch_last.shape[-1])
            last = np.empty((n, dim), np.float32)
            mean = np.empty((n, dim), np.float32)
        last[index] = batch_last.cpu().numpy()
        mean[index] = batch_mean.cpu().numpy()
        done += len(index)
        if done == len(index) or done % 2048 < len(index):
            rate = done / max(time.time() - t0, 1e-6)
            print(f"[prompt-head] extracted {done}/{n} ({rate:.1f} prompts/s)", flush=True)
    if not np.isfinite(last).all() or not np.isfinite(mean).all():
        raise FloatingPointError("non-finite prompt V")
    return {
        "id": np.array([r["id"] for r in rows]),
        "prompt_group": np.array([r["prompt_group"] for r in rows]),
        "prompt_label": np.array([int(r["prompt_safety_label"]) for r in rows], np.int64),
        "response_label": np.array([int(r["response_label"]) for r in rows], np.int64),
        "n_prompt_tokens": lengths,
        "v_last": last,
        "v_mean": mean,
    }


def _feature_path(out: Path, split: str) -> Path:
    return out / "features" / f"{split}.npz"


def _save_features(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def _load_features(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as blob:
        return {key: blob[key] for key in blob.files}


def _features_match(path: Path, rows: list[dict]) -> bool:
    if not path.exists():
        return False
    arrays = _load_features(path)
    return (
        len(arrays["id"]) == len(rows)
        and list(arrays["id"]) == [r["id"] for r in rows]
        and arrays["v_last"].shape == arrays["v_mean"].shape
        and arrays["v_last"].shape[0] == len(rows)
        and np.isfinite(arrays["v_last"]).all()
        and np.isfinite(arrays["v_mean"]).all()
    )


def extract_all(cfg: dict, out: Path, device: str) -> None:
    # Two float32 vectors of 1024 plus a small margin. The check refuses to touch
    # the backbone cache or the finished response-head artifacts.
    ensure_disk(out, need_bytes=2 * 1024 ** 3)
    layer = int(cfg["model"]["layer_index"])
    splits = ("train", "dev", "calibration", "test")
    pending = []
    for split in splits:
        rows = load_split_rows(cfg, split)
        if _features_match(_feature_path(out, split), rows):
            print(f"[prompt-head] reuse {split} features ({len(rows)} rows)", flush=True)
        else:
            pending.append((split, rows))
    if not pending:
        return
    import kv_safety.modeling as modeling

    model = modeling.load_model(cfg, device)
    probe = pending[0][1]
    probe_ids = [r["prompt_ids"] for r in probe]
    by_len: dict[int, list[list[int]]] = defaultdict(list)
    for seq in probe_ids:
        if len(by_len[len(seq)]) < 4:
            by_len[len(seq)].append(seq)
        if len(by_len[len(seq)]) == 4:
            break
    same = next((v for v in by_len.values() if len(v) >= 2), None)
    if same is None:
        raise RuntimeError("could not find two prompts of equal length for the batch check")
    diff = _batch_matches_single(model, layer, same[:4] if len(same) >= 4 else same[:2])
    scale = max(float(np.max(np.abs(_prompt_v(
        model, torch.tensor([same[0]], device=device), layer)[0].cpu().numpy()))), 1.0)
    rel = diff / scale
    print(f"[prompt-head] batch-vs-single max abs diff={diff:.3e} rel={rel:.3e}", flush=True)
    token_budget = 4096 if rel <= 1e-2 else 1
    if token_budget == 1:
        print("[prompt-head] batch mismatch is large; extracting one prompt at a time", flush=True)
    for split, rows in pending:
        print(f"[prompt-head] extract {split} n={len(rows)} layer={layer}", flush=True)
        arrays = extract_split(model, rows, layer, token_budget)
        if arrays["v_last"].shape[1] != model.config.num_key_value_heads * model.config.head_dim:
            raise RuntimeError(f"unexpected V width {arrays['v_last'].shape[1]}")
        _save_features(_feature_path(out, split), arrays)
        print(f"[prompt-head] wrote {split} {arrays['v_last'].shape}", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _xy(arrays: dict[str, np.ndarray], method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    key = "v_last" if method == "prompt_v_last" else "v_mean"
    groups, weights = group_weights(arrays["prompt_group"])
    return arrays[key], arrays["prompt_label"].astype(np.int64), weights, groups


def train_heads(cfg: dict, out: Path, device: str) -> None:
    train = _load_features(_feature_path(out, "train"))
    dev = _load_features(_feature_path(out, "dev"))
    tc = cfg["train"]
    hidden = int(cfg["classifier"]["hidden"])
    layer = int(cfg["model"]["layer_index"])
    rows = []
    for method in METHODS:
        Xtr, ytr, wtr, _ = _xy(train, method)
        Xdv, ydv, _, gdv = _xy(dev, method)
        for seed in [int(s) for s in tc["seeds"]]:
            dest = out / "checkpoints" / method / str(seed)
            tag = f"{method} seed={seed}"
            if (dest / "meta.json").exists() and (dest / "model.pt").exists():
                prev = json.loads((dest / "meta.json").read_text())
                print(f"[prompt-head] resume {tag} epoch={prev.get('best_epoch')}", flush=True)
                rows.append({
                    "method": method, "seed": seed, "resumed": True,
                    "best_epoch": prev.get("best_epoch"),
                    "best_dev_group_bce": prev.get("best_dev_response_bce"),
                    "n_parameters": prev.get("n_parameters"),
                    "seconds": prev.get("seconds"),
                })
                continue
            t0 = time.time()
            model, info = train_one(Xtr, ytr, wtr, Xdv, ydv, gdv, hidden, seed, tc, device, tag=tag)
            seconds = round(time.time() - t0, 1)
            meta = {
                "classifier": method,
                "method": method,
                "feature": method,
                "task": "prompt_safety_label",
                "layer_index": layer,
                "seed": seed,
                "hidden": hidden,
                "weighting": "1/count(prompt_group) on train; dev loss is the mean of per-group mean BCE",
                "prompt_tokens": "all stored prompt_ids, including the chat template and generation prompt",
                "train_rows": int(len(Xtr)),
                "train_groups": int(len(np.unique(train["prompt_group"]))),
                "dev_rows": int(len(Xdv)),
                "dev_groups": int(len(np.unique(dev["prompt_group"]))),
                "seconds": seconds,
                "best_epoch": info["best_epoch"],
                "best_dev_response_bce": info["best_dev_response_bce"],
                "model_id": cfg["model"]["id"],
                "revision": pinned_revision(cfg, "model"),
            }
            save_checkpoint(model.cpu(), dest, meta)
            rows.append({
                "method": method, "seed": seed, "resumed": False,
                "best_epoch": info["best_epoch"],
                "best_dev_group_bce": info["best_dev_response_bce"],
                "n_parameters": model.n_parameters(),
                "seconds": seconds,
            })
            print(f"[prompt-head] {tag} best_epoch={info['best_epoch']} "
                  f"dev_bce={info['best_dev_response_bce']:.4f} seconds={seconds}", flush=True)
    write_json(out / "metrics" / "train_summary.json", rows)


def _scores(model, x: np.ndarray, device: str) -> np.ndarray:
    return predict(model, x, device).astype(np.float64)


def _deduped_scores(arrays: dict[str, np.ndarray], scores: np.ndarray, method: str) -> np.ndarray:
    """One score per exact prompt_ids length+feature is not available; dedupe identical features."""
    del method
    # Identical stored vectors inside a group are the same prompt. Keep the first.
    seen = set()
    keep = []
    labels = arrays["prompt_label"]
    groups = arrays["prompt_group"].astype(str)
    for i, group in enumerate(groups):
        if labels[i] != 0:
            continue
        key = group
        if key in seen:
            continue
        seen.add(key)
        keep.append(i)
    return scores[np.array(keep, dtype=np.int64)]


def calibrate_heads(cfg: dict, out: Path, device: str) -> dict[tuple[str, int], float]:
    from .classifier import load_checkpoint

    arrays = _load_features(_feature_path(out, "calibration"))
    target = float(cfg["evaluate"]["target_fpr"])
    thresholds = {}
    rows = []
    benign = arrays["prompt_label"] == 0
    n_benign_rows = int(benign.sum())
    for method in METHODS:
        x, _, _, _ = _xy(arrays, method)
        for seed in [int(s) for s in cfg["train"]["seeds"]]:
            model, _ = load_checkpoint(out / "checkpoints" / method / str(seed), device)
            scores = _scores(model, x, device)
            # Calibration groups in this split have identical prompt ids, so the
            # first row of each benign group is the prompt, not a second vote.
            chosen = _deduped_scores(arrays, scores, method)
            thr, info = calibrate_threshold(chosen, target)
            info = {
                **info,
                "target_fpr": target,
                "split": "calibration",
                "unit": "benign_prompt_group",
                "n_benign_rows": n_benign_rows,
                "method": method,
                "seed": seed,
                "excluded": "harmful-prompt calibration rows are not treated as negatives",
            }
            write_json(out / "checkpoints" / method / str(seed) / "threshold.json", info)
            thresholds[(method, seed)] = thr
            rows.append({
                "method": method, "seed": seed, "threshold": thr,
                "n_benign_groups": info["n_safe"], "n_benign_rows": n_benign_rows,
                "n_false_alarms": info["n_false_alarms"],
                "empirical_fpr": info["empirical_fpr"],
                "wilson95_lo": info["wilson95_lo"], "wilson95_hi": info["wilson95_hi"],
                "unstable": info["unstable"], "unstable_reason": info["unstable_reason"],
                "target_fpr": target,
            })
            print(f"[prompt-head] calibrate {method} seed={seed} thr={thr:.4f} "
                  f"fpr={info['empirical_fpr']:.4f} n_groups={info['n_safe']}", flush=True)
    pd.DataFrame(rows).to_csv(out / "metrics" / "calibration.csv", index=False)
    return thresholds


def _rates(labels: np.ndarray, alarm: np.ndarray) -> dict:
    labels = labels.astype(bool)
    alarm = alarm.astype(bool)
    tp = int((labels & alarm).sum())
    fp = int((~labels & alarm).sum())
    tn = int((~labels & ~alarm).sum())
    fn = int((labels & ~alarm).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    if precision is None or recall is None or precision + recall == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    fpr = fp / (fp + tn) if fp + tn else None
    return {
        "n": int(len(labels)), "n_pos": int(labels.sum()), "n_neg": int((~labels).sum()),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1, "fpr": fpr,
    }


def _slice_rate(alarm: np.ndarray, mask: np.ndarray) -> tuple[int, float | None]:
    n = int(mask.sum())
    if n == 0:
        return 0, None
    return n, float(alarm[mask].mean())


def _eval_frame(arrays: dict[str, np.ndarray], scores: np.ndarray, thr: float,
                method: str, seed: int, split: str) -> list[dict]:
    labels = arrays["prompt_label"]
    response = arrays["response_label"]
    alarm = scores > thr
    rows = []
    for unit, idx in (
        ("row", np.arange(len(labels))),
        ("prompt_group", np.unique(group_weights(arrays["prompt_group"])[0], return_index=True)[1]),
    ):
        # prompt_group unit keeps the first row of each group. Test has a few groups
        # whose surface form differs; row is the primary unit.
        take = np.sort(idx)
        stats = _rates(labels[take], alarm[take])
        y = labels[take]
        r = response[take]
        a = alarm[take]
        hp_safe_n, hp_safe = _slice_rate(a, (y == 1) & (r == 0))
        hp_harm_n, hp_harm = _slice_rate(a, (y == 1) & (r == 1))
        bp_safe_n, bp_safe = _slice_rate(a, (y == 0) & (r == 0))
        bp_harm_n, bp_harm = _slice_rate(a, (y == 0) & (r == 1))
        fpr_w = wilson_interval(stats["fp"], stats["n_neg"]) if stats["n_neg"] else None
        rec_w = wilson_interval(stats["tp"], stats["n_pos"]) if stats["n_pos"] else None
        rows.append({
            "split": split, "unit": unit, "method": method, "seed": seed,
            "threshold": thr,
            "auroc": auroc_score(y, scores[take]),
            "average_precision": average_precision(y, scores[take]),
            **stats,
            "fpr_wilson95_lo": None if fpr_w is None else fpr_w[0],
            "fpr_wilson95_hi": None if fpr_w is None else fpr_w[1],
            "recall_wilson95_lo": None if rec_w is None else rec_w[0],
            "recall_wilson95_hi": None if rec_w is None else rec_w[1],
            "n_hp_safe_response": hp_safe_n, "alarm_rate_hp_safe_response": hp_safe,
            "n_hp_harm_response": hp_harm_n, "alarm_rate_hp_harm_response": hp_harm,
            "n_bp_safe_response": bp_safe_n, "alarm_rate_bp_safe_response": bp_safe,
            "n_bp_harm_response": bp_harm_n, "alarm_rate_bp_harm_response": bp_harm,
        })
    return rows


def evaluate_heads(cfg: dict, out: Path, device: str, thresholds: dict[tuple[str, int], float]) -> None:
    from .classifier import load_checkpoint

    frames = []
    detail = []
    for split in ("dev", "test"):
        arrays = _load_features(_feature_path(out, split))
        for method in METHODS:
            x, _, _, _ = _xy(arrays, method)
            for seed in [int(s) for s in cfg["train"]["seeds"]]:
                thr = thresholds[(method, seed)]
                model, _ = load_checkpoint(out / "checkpoints" / method / str(seed), device)
                scores = _scores(model, x, device)
                frames.extend(_eval_frame(arrays, scores, thr, method, seed, split))
                if split == "test":
                    alarm = scores > thr
                    for i in range(len(scores)):
                        detail.append({
                            "id": arrays["id"][i],
                            "prompt_group": arrays["prompt_group"][i],
                            "method": method,
                            "seed": seed,
                            "prompt_label": int(arrays["prompt_label"][i]),
                            "response_label": int(arrays["response_label"][i]),
                            "n_prompt_tokens": int(arrays["n_prompt_tokens"][i]),
                            "logit": float(scores[i]),
                            "threshold": thr,
                            "alarm": bool(alarm[i]),
                        })
                row = frames[-2]
                print(f"[prompt-head] {split} {method} seed={seed} "
                      f"auroc={row['auroc']:.4f} fpr={row['fpr']:.4f} "
                      f"recall={row['recall']:.4f} f1={row['f1']:.4f}", flush=True)
    summary = pd.DataFrame(frames)
    summary.to_csv(out / "metrics" / "summary.csv", index=False)
    pd.DataFrame(detail).to_csv(out / "metrics" / "test_per_row.csv", index=False)
    means = []
    test_rows = summary[(summary["split"] == "test") & (summary["unit"] == "row")]
    value_cols = ["auroc", "average_precision", "precision", "recall", "f1", "fpr",
                  "alarm_rate_hp_safe_response", "alarm_rate_hp_harm_response",
                  "alarm_rate_bp_safe_response", "alarm_rate_bp_harm_response"]
    for method, part in test_rows.groupby("method"):
        rec = {"split": "test", "unit": "row", "method": method, "n_seeds": int(part["seed"].nunique())}
        for col in value_cols:
            vals = part[col].astype(float).to_numpy()
            rec[f"{col}_mean"] = float(vals.mean())
            rec[f"{col}_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        means.append(rec)
    pd.DataFrame(means).to_csv(out / "metrics" / "test_mean.csv", index=False)


def write_note(cfg: dict, out: Path, disk: dict) -> None:
    layer = int(cfg["model"]["layer_index"])
    write_json(out / "note.json", {
        "task": "prompt classification from layer V",
        "layer_index": layer,
        "methods": {
            "prompt_v_last": "V of the last token in prompt_ids",
            "prompt_v_mean": "FP32 mean of V over every token in prompt_ids",
        },
        "prompt_ids": "chat-templated user turn plus generation prompt; thinking disabled. "
                      "Template tokens are included. Response tokens are not.",
        "label": "FineHarm filename _HP_ = 1, _BP_ = 0. Response harm is not the label.",
        "train": "every train row; weight 1/count(prompt_group)",
        "early_stopping": "dev group-equal BCE. Test is not used to pick the epoch.",
        "threshold": "calibration rows with prompt label 0, one row per prompt_group, "
                     "target FPR from the config. Harmful calibration prompts are excluded.",
        "alarm": "logit > threshold",
        "does_not_change": "response heads, artifacts/full/protocol_lock.json, response thresholds",
        "model_id": cfg["model"]["id"],
        "revision": pinned_revision(cfg, "model"),
        "disk": disk,
    })


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config)
    if int(cfg["model"]["layer_index"]) != 11:
        raise SystemExit(f"this run is for the locked deployment layer 11, got {cfg['model']['layer_index']}")
    out = prompt_head_dir(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    disk = ensure_disk(out, need_bytes=2 * 1024 ** 3)
    print(f"[prompt-head] disk {disk}", flush=True)
    write_note(cfg, out, disk)
    extract_all(cfg, out, device)
    train_heads(cfg, out, device)
    thresholds = calibrate_heads(cfg, out, device)
    evaluate_heads(cfg, out, device, thresholds)
    print("[prompt-head] done", flush=True)


if __name__ == "__main__":
    main()
