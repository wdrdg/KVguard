"""Score Qwen3GuardTest with the official stream protocol.

Dataset files are read from the local snapshot. Thresholds and protocol_lock are
not changed. Guard uses the official two-token decision. v_mean and v_early keep
the locked FineHarm thresholds and are reported both at the first token over the
threshold and at the first pair of tokens over the threshold.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from kv_safety.config import load_config, write_json
from kv_safety.qwen3guardtest_metrics import (
    LABEL_MAP,
    classify_response,
    consecutive_over_threshold,
    consecutive_unsafe,
    f1_pair,
    first_over_threshold,
    latency_bins,
    response_split,
)

ROOT = Path(__file__).resolve().parents[1]
QWEN3GUARDTEST_REVISION = "8a8391a583d65dc5f2be59e815404fe54289fa7c"


def dataset_dir() -> Path:
    """Local snapshot of Qwen/Qwen3GuardTest. Weights and data stay outside the repo."""
    hf_home = Path(os.environ.get("HF_HOME", ROOT / "hf_cache"))
    locked = (
        hf_home / "hub" / "datasets--Qwen--Qwen3GuardTest" / "snapshots" / QWEN3GUARDTEST_REVISION
    )
    if (locked / "thinking.jsonl").is_file():
        return locked
    raise SystemExit(
        "Qwen/Qwen3GuardTest snapshot "
        f"{QWEN3GUARDTEST_REVISION} was not found under HF_HOME={hf_home}. "
        "Download it with `python -m kv_safety.download --config configs/full.yaml`."
    )
SPLITS = {
    "thinking": {"file": "thinking.jsonl", "thinking": True, "latency": False},
    "thinking_loc": {"file": "thinking_loc.jsonl", "thinking": True, "latency": True},
    "response_loc": {"file": "response_loc.jsonl", "thinking": False, "latency": True},
}
SEEDS = (17, 42, 2026)
LAYER = 11


def load_rows(split: str) -> list[dict]:
    spec = SPLITS[split]
    rows = []
    with (dataset_dir() / spec["file"]).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def token_texts(tokenizer, input_ids: list[int]) -> list[str]:
    return [tokenizer.decode([token]) for token in input_ids]


def resolve_ids(tokenizer, row: dict) -> list[int]:
    stored = row.get("input_ids")
    if stored:
        return list(stored)
    text = tokenizer.apply_chat_template(
        row["message"], tokenize=False, add_generation_prompt=False, enable_thinking=False)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def done_ids(path: Path, seed: int | None) -> set[str]:
    found = set()
    if not path.exists():
        return found
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if seed is None or row.get("seed") == seed:
                found.add(row["unique_id"])
    return found


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def guard_levels(model, input_ids: list[int], split_idx: int) -> list[int]:
    device = next(model.parameters()).device
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        outputs = model(input_ids=ids, use_cache=False)
    logits = outputs.risk_level_logits[0, split_idx:]
    return torch.argmax(logits, dim=-1).detach().cpu().tolist()


def run_guard(out: Path, splits: list[str]) -> None:
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    from kv_safety.qwen3guard_stream import (
        _ensure_pad_token_id,
        _install_default_rope,
        _install_mask_compat,
    )

    _install_mask_compat()
    _install_default_rope()
    model_id = "Qwen/Qwen3Guard-Stream-4B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    _ensure_pad_token_id(config)
    model = AutoModel.from_pretrained(
        model_id, config=config, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True).eval()
    for split in splits:
        path = out / f"guard_{split}.jsonl"
        finished = done_ids(path, None)
        thinking = SPLITS[split]["thinking"]
        for row in load_rows(split):
            if row["unique_id"] in finished:
                continue
            ids = resolve_ids(tokenizer, row)
            split_idx = response_split(token_texts(tokenizer, ids), thinking)
            levels = guard_levels(model, ids, split_idx)
            names = [LABEL_MAP[i] for i in levels]
            strict, loose = classify_response(names)
            prediction, local_idx = consecutive_unsafe(names)
            pred_idx = None if prediction == "Safe" else split_idx + local_idx
            append_jsonl(path, {
                "unique_id": row["unique_id"],
                "split": split,
                "method": "qwen3guard_stream_4b",
                "label": row["label"],
                "unsafe_start_index": row.get("unsafe_start_index"),
                "unsafe_end_index": row.get("unsafe_end_index"),
                "split_idx": split_idx,
                "pred_strict": strict,
                "pred_loose": loose,
                "pred_idx": pred_idx,
                "n_response_tokens": len(levels),
            })
            finished.add(row["unique_id"])
            if len(finished) % 25 == 0:
                print(f"guard {split} {len(finished)}", flush=True)
    del model
    torch.cuda.empty_cache()


def v_logits(model, heads, input_ids: list[int], split_idx: int) -> dict[str, list[float]]:
    from kv_safety.features.cache_reader import check_cache, flatten_heads, layer_kv

    device = next(model.parameters()).device
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        outputs = model(input_ids=ids, use_cache=True)
        cache = outputs.past_key_values
        check_cache(cache)
        values = flatten_heads(layer_kv(cache, LAYER)[1])[0, split_idx:].float()
        denom = torch.arange(1, values.shape[0] + 1, device=device, dtype=torch.float32).view(-1, 1)
        means = torch.cumsum(values, dim=0) / denom
        scored = {}
        for name, head in heads.items():
            scored[name] = head(means).detach().cpu().tolist()
        del outputs, cache, values, means
    return scored


def run_heads(out: Path, splits: list[str]) -> None:
    from kv_safety.classifier import load_checkpoint
    from kv_safety.modeling import load_model, load_tokenizer

    cfg = load_config(str(ROOT / "configs/full.yaml"))
    tokenizer = load_tokenizer(cfg)
    model = load_model(cfg, "cuda")
    heads = {}
    thresholds = {}
    for seed in SEEDS:
        for method, folder in (
            ("v_mean", ROOT / "artifacts/full/checkpoints/11/v_mean" / str(seed)),
            ("v_early", ROOT / "artifacts/full/v_early/checkpoints" / str(seed)),
        ):
            head, _ = load_checkpoint(folder, "cuda")
            heads[f"{method}:{seed}"] = head
            thresholds[f"{method}:{seed}"] = float(json.loads((folder / "threshold.json").read_text())["threshold"])
    for split in splits:
        thinking = SPLITS[split]["thinking"]
        paths = {key: out / f"{key.split(':')[0]}_{split}.jsonl" for key in heads}
        finished = {key: done_ids(paths[key], int(key.split(":")[1])) for key in heads}
        pending = [row for row in load_rows(split) if any(row["unique_id"] not in finished[key] for key in heads)]
        for n_done, row in enumerate(pending, start=1):
            ids = resolve_ids(tokenizer, row)
            split_idx = response_split(token_texts(tokenizer, ids), thinking)
            scored = v_logits(model, heads, ids, split_idx)
            for key, logits in scored.items():
                if row["unique_id"] in finished[key]:
                    continue
                method, seed_text = key.split(":")
                thr = thresholds[key]
                any_idx = first_over_threshold(logits, thr)
                pair_idx = consecutive_over_threshold(logits, thr)
                def decision(local):
                    pred = "Unsafe" if local is not None else "Safe"
                    return pred, pred, None if local is None else split_idx + local
                strict_any, loose_any, pred_any = decision(any_idx)
                strict_pair, loose_pair, pred_pair = decision(pair_idx)
                append_jsonl(paths[key], {
                    "unique_id": row["unique_id"],
                    "split": split,
                    "method": method,
                    "seed": int(seed_text),
                    "threshold": thr,
                    "label": row["label"],
                    "unsafe_start_index": row.get("unsafe_start_index"),
                    "unsafe_end_index": row.get("unsafe_end_index"),
                    "split_idx": split_idx,
                    "pred_strict": strict_any,
                    "pred_loose": loose_any,
                    "pred_idx": pred_any,
                    "pred_strict_pair": strict_pair,
                    "pred_loose_pair": loose_pair,
                    "pred_idx_pair": pred_pair,
                    "n_response_tokens": len(logits),
                })
            if n_done % 25 == 0:
                print(f"heads {split} {n_done}/{len(pending)}", flush=True)
    del model
    torch.cuda.empty_cache()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarize(out: Path) -> dict:
    summary = {}
    for split, spec in SPLITS.items():
        summary[split] = {}
        guard_path = out / f"guard_{split}.jsonl"
        if guard_path.exists():
            rows = read_jsonl(guard_path)
            item = {"n": len(rows), "f1": f1_pair(rows)}
            if spec["latency"]:
                item["latency"] = latency_bins(rows)
            summary[split]["qwen3guard_stream_4b"] = item
        for method in ("v_mean", "v_early"):
            path = out / f"{method}_{split}.jsonl"
            if not path.exists():
                continue
            rows = read_jsonl(path)
            by_seed = []
            for seed in SEEDS:
                part = [row for row in rows if row["seed"] == seed]
                if not part:
                    continue
                pair_rows = [{
                    **row,
                    "pred_strict": row["pred_strict_pair"],
                    "pred_loose": row["pred_loose_pair"],
                    "pred_idx": row["pred_idx_pair"],
                } for row in part]
                entry = {
                    "seed": seed,
                    "n": len(part),
                    "threshold": part[0]["threshold"],
                    "any_token": {"f1": f1_pair(part)},
                    "two_token": {"f1": f1_pair(pair_rows)},
                }
                if spec["latency"]:
                    entry["any_token"]["latency"] = latency_bins(part)
                    entry["two_token"]["latency"] = latency_bins(pair_rows)
                by_seed.append(entry)
            summary[split][method] = by_seed
    write_json(out / "summary.json", summary)
    return summary


def main() -> None:
    os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "artifacts/full/qwen3guardtest"))
    parser.add_argument("--systems", nargs="+", default=["guard", "heads"], choices=["guard", "heads"])
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if "guard" in args.systems:
        run_guard(out, args.splits)
    if "heads" in args.systems:
        run_heads(out, args.splits)
    summarize(out)
    write_json(out / "note.json", {
        "dataset": f"Qwen/Qwen3GuardTest@{QWEN3GUARDTEST_REVISION}",
        "protocol": "QwenLM/Qwen3Guard eval/eval_stream.py",
        "strict_means": "Controversial counted as Safe, matching the script's printed strict line",
        "loose_means": "Controversial counted as Unsafe, matching the script's printed loose line",
        "guard_alarm": "second token of the first consecutive Unsafe pair",
        "head_alarm": "logit > the locked FineHarm threshold; any_token is the first such token, two_token matches the pair rule",
        "response_split": "thinking slices start four tokens after assistant; response_loc slices start after the empty think wrapper",
        "thresholds_not_changed": True,
    })
    print("wrote", out / "summary.json", flush=True)


if __name__ == "__main__":
    main()
