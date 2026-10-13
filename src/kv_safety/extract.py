"""Offline teacher-forced extraction of prefix-aggregated features (all base features in one pass).

python -m kv_safety.extract --config configs/smoke.yaml [--device cuda] [--splits train val test]

Per response: prefill the prompt, then feed the response in chunks (default 128 tokens) on top of
the native DynamicCache. After each chunk we take the block-`layer` output (hook, pre final norm)
and the K/V that the chunk wrote into the cache (post k-norm, post-RoPE), and update FP32 running
sums -> prefix means at each position. Causality comes from the causal mask + cache: nothing from
later chunks is visible. Only aggregated features at the requested positions are stored (no raw
K/V or hidden-state sequences). Train stores the <=32 selected positions; val/test store every
response position. The base model is used (no vocabulary logits are computed).

This combined path is for feature extraction only; benchmarks run each method in its own process.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .config import BASE_FEATURES, artifacts_dir, base_parser, load_config, write_json
from .data.labels import prefix_labels, position_weights, select_train_positions
from .features.cache_reader import check_cache, new_kv
from .features.hidden_hook import BlockOutputHook
from .features.running_mean import RunningMean

DT = {"float32": np.float32, "float16": np.float16}


@torch.inference_mode()
def extract_response_features(model, prompt_ids: list[int], response_ids: list[int], layer_idx: int,
                              chunk_size: int, positions: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Return {feature: [n_pos, D] float32} at 1-based response positions (None -> all positions)."""
    from transformers import DynamicCache

    device = next(model.parameters()).device
    base = model.model if hasattr(model, "model") else model
    cfg = model.config
    T = len(response_ids)
    want = np.arange(1, T + 1) if positions is None else np.asarray(positions)
    captured: list[torch.Tensor] = []
    hook = BlockOutputHook(model, layer_idx, captured.append)
    try:
        cache = DynamicCache(config=cfg)
        hook.enabled = False
        base(input_ids=torch.tensor([prompt_ids], device=device), past_key_values=cache, use_cache=True)
        check_cache(cache)
        hook.enabled = True
        kv_dim = cfg.num_key_value_heads * cfg.head_dim
        hm = RunningMean(1, cfg.hidden_size, device)
        km, vm = RunningMean(1, kv_dim, device), RunningMean(1, kv_dim, device)
        out = {f: [] for f in BASE_FEATURES}
        resp = torch.tensor([response_ids], device=device)
        for start in range(0, T, chunk_size):
            ids = resp[:, start:start + chunk_size]
            n = ids.shape[1]
            base(input_ids=ids, past_key_values=cache, use_cache=True)
            h = captured.pop()
            assert not captured and h.shape[1] == n
            k, v = new_kv(cache, layer_idx, n)
            hmeans, kmeans, vmeans = hm.update_prefix_means(h), km.update_prefix_means(k), vm.update_prefix_means(v)
            sel = want[(want > start) & (want <= start + n)] - start - 1
            if len(sel) == 0:
                continue
            si = torch.as_tensor(sel, device=device)
            for name, t in (("h_last", h[0].float()), ("h_mean", hmeans[0]), ("k_mean", kmeans[0]),
                            ("v_mean", vmeans[0])):
                out[name].append(t.index_select(0, si).cpu())
        del cache
    finally:
        hook.remove()
    res = {f: torch.cat(v).numpy() for f, v in out.items()}
    for f, a in res.items():
        if not np.isfinite(a).all():
            raise FloatingPointError(f"non-finite values in {f}")
    return res


def load_prepared(out: Path, split: str) -> list[dict]:
    with open(out / "data" / f"{split}.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def split_positions(rec: dict, split: str, fc: dict) -> np.ndarray | None:
    if split != "train":
        return None
    lab = prefix_labels(rec["T"], rec["a"], rec["b"])
    return select_train_positions(lab, rec["a"], rec["b"], fc["max_train_positions"], fc["priority_after_end"])


def _shard_key(cfg: dict, split: str, ids: list[str]) -> str:
    h = hashlib.sha1(json.dumps([cfg["model"], cfg["features"], split, ids], sort_keys=True,
                                default=str).encode()).hexdigest()
    return h[:16]


def extract_split(model, cfg: dict, out: Path, split: str) -> dict:
    fc = cfg["features"]
    layer = cfg["model"]["layer_index"]
    recs = load_prepared(out, split)
    fdir = out / "features" / split
    fdir.mkdir(parents=True, exist_ok=True)
    dtype = np.float32 if split == "train" else DT[fc["eval_dtype"]]
    shards = []
    ss = fc["shard_size"]
    t0 = time.time()
    for si, s0 in enumerate(range(0, len(recs), ss)):
        chunk = recs[s0:s0 + ss]
        key = _shard_key(cfg, split, [r["id"] for r in chunk])
        path = fdir / f"shard_{si:05d}.npz"
        done = fdir / f"shard_{si:05d}.key"
        if path.exists() and done.exists() and done.read_text() == key:
            shards.append(path.name)
            continue
        cols: dict[str, list] = {f: [] for f in BASE_FEATURES}
        meta = {"sample_idx": [], "position": [], "label": [], "weight": []}
        for j, rec in enumerate(tqdm(chunk, desc=f"{split} shard {si}", leave=False)):
            pos = split_positions(rec, split, fc)
            feats = extract_response_features(model, rec["prompt_ids"], rec["response_ids"], layer,
                                              fc["chunk_size"], pos)
            positions = np.arange(1, rec["T"] + 1) if pos is None else pos
            lab = prefix_labels(rec["T"], rec["a"], rec["b"])[positions - 1]
            w = position_weights(len(positions)) if split == "train" else np.zeros(len(positions), np.float32)
            for f in BASE_FEATURES:
                cols[f].append(feats[f].astype(dtype))
            meta["sample_idx"].append(np.full(len(positions), s0 + j, np.int32))
            meta["position"].append(positions.astype(np.int32))
            meta["label"].append(lab.astype(np.int16))
            meta["weight"].append(w)
        np.savez(path, **{k: np.concatenate(v) for k, v in meta.items()},
                 **{f: np.concatenate(v) for f, v in cols.items()})
        done.write_text(key)
        shards.append(path.name)
    return {"n_responses": len(recs), "shards": shards, "dtype": np.dtype(dtype).name,
            "positions": "selected<=%d" % fc["max_train_positions"] if split == "train" else "all",
            "seconds": round(time.time() - t0, 1)}


def load_split_features(out: Path, split: str, feature: str, rows: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Concatenate shards -> (X float32 [N, D], meta). kv_mean = [k_mean | v_mean]."""
    fdir = out / "features" / split
    fm = json.loads((out / "feature_manifest.json").read_text())
    names = ["k_mean", "v_mean"] if feature == "kv_mean" else [feature]
    xs, meta = [], {k: [] for k in ("sample_idx", "position", "label", "weight")}
    for sh in fm["splits"][split]["shards"]:
        with np.load(fdir / sh) as z:
            for k in meta:
                meta[k].append(z[k])
            parts = [z[n] for n in names]
            xs.append(np.concatenate(parts, axis=1).astype(np.float32, copy=False))
    X = np.concatenate(xs)
    meta = {k: np.concatenate(v) for k, v in meta.items()}
    return X, meta


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--device", default=None)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = artifacts_dir(cfg)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("WARNING: extracting on CPU (slow). Feature values are valid; no timing is recorded.")
    from .modeling import load_model, model_provenance

    model = load_model(cfg, device)
    fm_path = out / "feature_manifest.json"
    fm = json.loads(fm_path.read_text()) if fm_path.exists() else {"splits": {}}
    c = model.config
    fm.update({
        **model_provenance(cfg), "layer_index": cfg["model"]["layer_index"],
        "dtype_backbone": cfg["model"]["dtype"], "accumulator_dtype": "float32",
        "attn_implementation": c._attn_implementation, "chunk_size": cfg["features"]["chunk_size"],
        "device": device, "torch": torch.__version__,
        "features": {"h_last": c.hidden_size, "h_mean": c.hidden_size, "k_mean": c.num_key_value_heads * c.head_dim,
                     "v_mean": c.num_key_value_heads * c.head_dim, "kv_mean": 2 * c.num_key_value_heads * c.head_dim},
        "definitions": {
            "h": f"output of decoder block {cfg['model']['layer_index']} (residual stream, before final norm)",
            "k/v": "the block's tensors stored in the DynamicCache (K after k_norm and RoPE), 8 KV heads "
                   "flattened head-major, no averaging over heads or dims",
            "kv_mean": "concat(k_mean, v_mean)",
            "positions": "1-based response-token positions; prompt/template tokens are never averaged",
        },
    })
    for split in args.splits:
        fm["splits"][split] = extract_split(model, cfg, out, split)
        write_json(fm_path, fm)
        print(f"[extract] {split}: {fm['splits'][split]}")


if __name__ == "__main__":
    main()
