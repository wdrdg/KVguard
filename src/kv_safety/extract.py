"""Offline teacher-forced extraction of prefix-aggregated features.

python -m kv_safety.extract --config configs/pilot.yaml [--device cuda] [--splits train dev]

Per response: prefill the prompt, then feed the response in chunks (default 128) on top of the
native DynamicCache. Every configured layer is read in that same forward. After each chunk we take
the requested block outputs (hook, pre final norm) and the K/V that the chunk wrote into the cache
(K is post k-norm and post-RoPE; V is the native value, not RoPE). FP32 running sums become prefix
means. Only aggregated features at the selected positions are stored.

Train and dev store at most 32 positions (the shared supervision subset). Streaming scores are a
later replay and are not saved here. Calibration and test are not extracted.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .config import (
    BASE_FEATURES,
    REPO_ROOT,
    artifacts_dir,
    base_parser,
    load_config,
    stored_by_layer,
    write_json,
)
from .data.labels import prefix_labels, position_weights, select_train_positions
from .features.cache_reader import check_cache, new_kv
from .features.hidden_hook import BlockOutputHook
from .features.running_mean import RunningMean

SAMPLED_SPLITS = ("train", "dev")
# Previous feature stores. The finished pilot report, metrics, and checkpoints stay;
# only the recomputable feature shards and replay scores are eligible for deletion.
OLD_FEATURE_DIRS = (
    "artifacts/pilot_s42",
    "artifacts/pilot_s42_l35",
    "artifacts/smoke",
    "artifacts/pilot/features",
    "artifacts/pilot/scores",
)


def _spec_or_all(features: list[str] | None) -> list[str]:
    return list(BASE_FEATURES if features is None else features)


@torch.inference_mode()
def extract_response_features_layers(model, prompt_ids: list[int], response_ids: list[int],
                                     spec: dict[int, list[str]], chunk_size: int,
                                     positions: np.ndarray | None = None) -> dict[int, dict[str, np.ndarray]]:
    """Return {layer: {feature: [n_pos, D] float32}} at 1-based response positions."""
    from transformers import DynamicCache

    device = next(model.parameters()).device
    base = model.model if hasattr(model, "model") else model
    cfg = model.config
    T = len(response_ids)
    want = np.arange(1, T + 1) if positions is None else np.asarray(positions)
    captured: dict[int, list] = {layer: [] for layer in spec}
    hooks = []
    for layer, feats in spec.items():
        if any(f in ("h_last", "h_mean") for f in feats):
            hooks.append(BlockOutputHook(model, layer, captured[layer].append))
    try:
        for hook in hooks:
            hook.enabled = False
        cache = DynamicCache(config=cfg)
        base(input_ids=torch.tensor([prompt_ids], device=device), past_key_values=cache, use_cache=True)
        check_cache(cache)
        for hook in hooks:
            hook.enabled = True
        kv_dim = cfg.num_key_value_heads * cfg.head_dim
        state = {}
        for layer, feats in spec.items():
            state[layer] = {}
            if "h_mean" in feats or "h_last" in feats:
                state[layer]["h"] = RunningMean(1, cfg.hidden_size, device)
            if any(f in ("k_mean", "v_mean") for f in feats):
                state[layer]["k"] = RunningMean(1, kv_dim, device)
                state[layer]["v"] = RunningMean(1, kv_dim, device)
        out = {layer: {f: [] for f in feats} for layer, feats in spec.items()}
        resp = torch.tensor([response_ids], device=device)
        for start in range(0, T, chunk_size):
            ids = resp[:, start:start + chunk_size]
            n = ids.shape[1]
            base(input_ids=ids, past_key_values=cache, use_cache=True)
            sel = want[(want > start) & (want <= start + n)] - start - 1
            si = torch.as_tensor(sel, device=device) if len(sel) else None
            for layer, feats in spec.items():
                h = None
                if captured[layer]:
                    h = captured[layer].pop()
                    if captured[layer] or h.shape[1] != n:
                        raise RuntimeError(f"layer {layer} hook captured an unexpected tensor")
                k = v = None
                if "k" in state[layer]:
                    k, v = new_kv(cache, layer, n)
                hmeans = state[layer]["h"].update_prefix_means(h) if h is not None and "h" in state[layer] else None
                kmeans = state[layer]["k"].update_prefix_means(k) if k is not None else None
                vmeans = state[layer]["v"].update_prefix_means(v) if v is not None else None
                if si is None:
                    continue
                pieces = {}
                if h is not None and "h_last" in feats:
                    pieces["h_last"] = h[0].float()
                if hmeans is not None and "h_mean" in feats:
                    pieces["h_mean"] = hmeans[0]
                if kmeans is not None and "k_mean" in feats:
                    pieces["k_mean"] = kmeans[0]
                if vmeans is not None and "v_mean" in feats:
                    pieces["v_mean"] = vmeans[0]
                for name, tensor in pieces.items():
                    out[layer][name].append(tensor.index_select(0, si).cpu())
        del cache
    finally:
        for hook in hooks:
            hook.remove()
    result = {}
    for layer, cols in out.items():
        result[layer] = {}
        for name, parts in cols.items():
            if not parts:
                raise RuntimeError(f"layer {layer} feature {name} produced no rows")
            arr = torch.cat(parts).numpy()
            if not np.isfinite(arr).all():
                raise FloatingPointError(f"non-finite values in layer {layer} {name}")
            result[layer][name] = arr
    return result


@torch.inference_mode()
def extract_response_features(model, prompt_ids: list[int], response_ids: list[int], layer_idx: int,
                              chunk_size: int, positions: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Single-layer helper used by the mechanism tests. Returns all four base features."""
    return extract_response_features_layers(
        model, prompt_ids, response_ids, {layer_idx: list(BASE_FEATURES)}, chunk_size, positions)[layer_idx]


def load_prepared(out: Path, split: str) -> list[dict]:
    path = out / "data" / f"{split}.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path}; run kv_safety.data.prepare first")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split_positions(rec: dict, split: str, fc: dict) -> np.ndarray:
    if split not in SAMPLED_SPLITS:
        raise ValueError(f"split {split} is not a sampled-feature split")
    lab = prefix_labels(rec["T"], rec["a"], rec["b"])
    return select_train_positions(lab, rec["a"], rec["b"], fc["max_train_positions"], fc["priority_after_end"])


def assemble_feature(feats: dict[str, np.ndarray], name: str) -> np.ndarray:
    if name == "kv_mean":
        return np.concatenate([feats["k_mean"], feats["v_mean"]], axis=1)
    if name not in feats:
        raise KeyError(name)
    return feats[name]


def _shard_key(cfg: dict, split: str, layer: int, features: list[str], ids: list[str]) -> str:
    payload = {"model": cfg["model"]["id"], "layer": layer, "features": features,
               "chunk": cfg["features"]["chunk_size"], "split": split, "ids": ids,
               "max_pos": cfg["features"]["max_train_positions"]}
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def estimate_feature_bytes(records: list[dict], spec: dict[int, list[str]], max_pos: int,
                           hidden: int, kv_dim: int) -> int:
    dims = {"h_last": hidden, "h_mean": hidden, "k_mean": kv_dim, "v_mean": kv_dim}
    per_pos = sum(dims[f] for feats in spec.values() for f in feats) * 4
    npos = sum(min(r["T"], max_pos) for r in records)
    return int(npos * per_pos)


def ensure_disk_for_features(cfg: dict, out: Path, need_bytes: int) -> dict:
    """Delete previous pilot feature directories when the new extract would not fit."""
    usage = shutil.disk_usage(out if out.exists() else REPO_ROOT)
    margin = 3 * 1024 ** 3
    report = {"free_bytes_before": usage.free, "need_bytes": need_bytes, "deleted": []}
    if usage.free >= need_bytes + margin:
        report["action"] = "enough_space"
        return report
    for rel in OLD_FEATURE_DIRS:
        path = REPO_ROOT / rel
        if not path.exists() or path.resolve() == out.resolve():
            continue
        report["deleted"].append(str(path))
        shutil.rmtree(path)
    usage = shutil.disk_usage(out if out.exists() else REPO_ROOT)
    report["free_bytes_after"] = usage.free
    if usage.free < need_bytes + margin:
        raise SystemExit(
            f"not enough disk for features: need {need_bytes / 2**30:.1f} GiB plus margin, "
            f"free {usage.free / 2**30:.1f} GiB after deleting {report['deleted']}"
        )
    report["action"] = "deleted_old_feature_dirs"
    return report


def extract_split(model, cfg: dict, out: Path, split: str,
                  layers: set[int] | None = None) -> dict:
    if split not in SAMPLED_SPLITS:
        raise SystemExit(f"refusing to extract split={split}; only train and dev store sampled features")
    fc = cfg["features"]
    spec = stored_by_layer(cfg)
    if layers is not None:
        spec = {layer: feats for layer, feats in spec.items() if layer in layers}
        if not spec:
            raise SystemExit(f"no stored features for layers {sorted(layers)}")
    recs = load_prepared(out, split)
    ss = fc["shard_size"]
    t0 = time.time()
    layer_info = {}
    for si, s0 in enumerate(range(0, len(recs), ss)):
        chunk = recs[s0:s0 + ss]
        ids = [r["id"] for r in chunk]
        pending = []
        for layer, feats in spec.items():
            fdir = out / "features" / split / f"layer_{layer}"
            fdir.mkdir(parents=True, exist_ok=True)
            key = _shard_key(cfg, split, layer, feats, ids)
            path = fdir / f"shard_{si:05d}.npz"
            done = fdir / f"shard_{si:05d}.key"
            if path.exists() and done.exists() and done.read_text() == key:
                layer_info.setdefault(str(layer), {"shards": [], "features": feats})
                if path.name not in layer_info[str(layer)]["shards"]:
                    layer_info[str(layer)]["shards"].append(path.name)
                continue
            pending.append((layer, feats, path, done, key))
        if not pending:
            continue
        cols = {(layer, f): [] for layer, feats, *_ in pending for f in feats}
        meta = {layer: {"sample_idx": [], "position": [], "label": [], "weight": []}
                for layer, *_ in pending}
        failed_ids = []
        for j, rec in enumerate(tqdm(chunk, desc=f"{split} shard {si}", leave=False)):
            pos = split_positions(rec, split, fc)
            try:
                feats = extract_response_features_layers(
                    model, rec["prompt_ids"], rec["response_ids"],
                    {layer: features for layer, features, *_ in pending}, fc["chunk_size"], pos)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                torch.cuda.empty_cache()
                failed_ids.append({"id": rec["id"], "reason": "cuda_oom", "T": rec["T"]})
                print(f"[extract] cuda oom id={rec['id']} T={rec['T']}", flush=True)
                continue
            lab = prefix_labels(rec["T"], rec["a"], rec["b"])[pos - 1]
            w = position_weights(len(pos))
            for layer, features, *_ in pending:
                for f in features:
                    cols[(layer, f)].append(feats[layer][f].astype(np.float32, copy=False))
                meta[layer]["sample_idx"].append(np.full(len(pos), s0 + j, np.int32))
                meta[layer]["position"].append(pos.astype(np.int32))
                meta[layer]["label"].append(lab.astype(np.int16))
                meta[layer]["weight"].append(w.astype(np.float32))
        if failed_ids:
            fail_path = out / "logs" / f"extract_failures_{split}.jsonl"
            fail_path.parent.mkdir(parents=True, exist_ok=True)
            with fail_path.open("a", encoding="utf-8") as fh:
                for row in failed_ids:
                    fh.write(json.dumps(row) + "\n")
        for layer, features, path, done, key in pending:
            if not meta[layer]["sample_idx"]:
                continue
            np.savez(path, **{k: np.concatenate(v) for k, v in meta[layer].items()},
                     **{f: np.concatenate(cols[(layer, f)]) for f in features})
            done.write_text(key)
            layer_info.setdefault(str(layer), {"shards": [], "features": features})
            layer_info[str(layer)]["shards"].append(path.name)
    for layer in spec:
        info = layer_info.setdefault(str(layer), {"shards": [], "features": spec[layer]})
        info["n_responses"] = len(recs)
        info["positions"] = f"selected<={fc['max_train_positions']}"
        info["dtype"] = "float32"
    return {"n_responses": len(recs), "layers": layer_info,
            "seconds": round(time.time() - t0, 1),
            "positions": f"selected<={fc['max_train_positions']}"}


def load_split_features(out: Path, split: str, feature: str, layer: int,
                        rows: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Concatenate shards -> (X float32 [N, D], meta). kv_mean = [k_mean | v_mean]."""
    del rows
    fdir = out / "features" / split / f"layer_{layer}"
    fm = json.loads((out / "feature_manifest.json").read_text())
    names = ["k_mean", "v_mean"] if feature == "kv_mean" else [feature]
    info = fm["splits"][split]["layers"][str(layer)]
    xs, meta = [], {k: [] for k in ("sample_idx", "position", "label", "weight")}
    for sh in info["shards"]:
        with np.load(fdir / sh) as z:
            for k in meta:
                meta[k].append(z[k])
            xs.append(np.concatenate([z[n] for n in names], axis=1).astype(np.float32, copy=False))
    if not xs:
        raise RuntimeError(f"no feature shards for {split} layer {layer}")
    X = np.concatenate(xs)
    meta = {k: np.concatenate(v) for k, v in meta.items()}
    return X, meta


def feature_disk_bytes(out: Path) -> int:
    root = out / "features"
    if not root.exists():
        return 0
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--device", default=None)
    ap.add_argument("--splits", nargs="+", default=["train", "dev"])
    ap.add_argument("--layers", nargs="+", type=int, default=None,
                    help="extract only these layers; omit to extract every configured layer")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if not cfg.get("allow_test_eval", False) and "test" in args.splits:
        raise SystemExit("refusing to extract split=test because allow_test_eval is false")
    out = artifacts_dir(cfg)
    spec = stored_by_layer(cfg)
    if args.layers is not None:
        wanted = {int(x) for x in args.layers}
        spec = {layer: feats for layer, feats in spec.items() if layer in wanted}
        if not spec:
            raise SystemExit(f"no stored features for layers {sorted(wanted)}")
    hidden = int(cfg["model"]["expected_config"]["hidden_size"])
    kv_dim = int(cfg["model"]["expected_config"]["num_key_value_heads"] * cfg["model"]["expected_config"]["head_dim"])
    recs = []
    for split in args.splits:
        recs.extend(load_prepared(out, split))
    need = estimate_feature_bytes(recs, spec, cfg["features"]["max_train_positions"], hidden, kv_dim)
    disk = ensure_disk_for_features(cfg, out, need)
    print(f"[extract] estimated feature bytes={need} disk={disk['action']}", flush=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("WARNING: extracting on CPU (slow). Feature values are valid; no timing is recorded.")
    from .modeling import load_model, model_provenance

    model = load_model(cfg, device)
    fm_path = out / "feature_manifest.json"
    fm = json.loads(fm_path.read_text()) if fm_path.exists() else {"splits": {}}
    c = model.config
    kv_dim = c.num_key_value_heads * c.head_dim
    dim_of = {"h_last": c.hidden_size, "h_mean": c.hidden_size, "k_mean": kv_dim, "v_mean": kv_dim}
    feature_dims = dict(fm.get("features") or {})
    for layer, feats in spec.items():
        feature_dims[str(layer)] = {name: dim_of[name] for name in feats}
        if "k_mean" in feats and "v_mean" in feats:
            feature_dims[str(layer)]["kv_mean"] = 2 * kv_dim
    known_layers = sorted(set(fm.get("layers") or []) | set(spec))
    fm.update({
        **model_provenance(cfg), "layers": known_layers,
        "reference_layer": cfg["model"]["layer_index"],
        "dtype_backbone": cfg["model"]["dtype"], "accumulator_dtype": "float32",
        "attn_implementation": c._attn_implementation, "chunk_size": cfg["features"]["chunk_size"],
        "device": device, "torch": torch.__version__,
        "features": feature_dims,
        "disk": disk,
        "definitions": {
            "h": "output of the named decoder block (residual stream, before final norm)",
            "k": "K stored in the DynamicCache after k_norm and RoPE; 8 KV heads flattened head-major",
            "v": "V stored in the DynamicCache (no RoPE); 8 KV heads flattened head-major",
            "kv_mean": "concat(k_mean, v_mean), not stored separately",
            "positions": "1-based response-token positions; prompt, template and EOS are never averaged",
            "storage": "train and dev keep at most 32 positions; streaming scores are replayed later",
        },
    })
    layer_filter = set(spec) if args.layers is not None else None
    for split in args.splits:
        prev = fm.setdefault("splits", {}).get(split)
        new = extract_split(model, cfg, out, split, layers=layer_filter)
        if prev and "layers" in prev:
            merged = dict(prev["layers"])
            merged.update(new["layers"])
            new["layers"] = merged
        fm["splits"][split] = new
        fm["feature_disk_bytes"] = feature_disk_bytes(out)
        write_json(fm_path, fm)
        print(f"[extract] {split}: {fm['splits'][split]}", flush=True)


if __name__ == "__main__":
    main()
