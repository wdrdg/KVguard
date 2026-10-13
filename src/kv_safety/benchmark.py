"""Extra memory / latency of each online detector vs. a no-detector baseline (CUDA only).

python -m kv_safety.benchmark --config configs/smoke.yaml

The parent process runs every (method, batch, response length, repeat) cell in a fresh worker
process. Every method uses the same token ids, lengths, DynamicCache, attention backend, dtype,
batch, CausalLM forward (LM head included) and flush schedule: prefill the prompt, then feed the
fixed response one token per forward; the forward that consumes token t yields the score of prefix
t, so the last forward is the flush. Early termination is disabled. Detector state (running
sums, first-alarm position) lives on the GPU; there is no per-token .item() synchronisation.
Load text is a fixed benign passage: it is a resource load only, never a quality measurement.
A cell that OOMs is recorded as OOM; nothing is shortened or re-cast.
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import REPO_ROOT, artifacts_dir, base_parser, load_config, write_json
from .features.monitor import METHODS, StreamingDetector, feature_dim

LOAD_TEXT = (
    "The river town wakes slowly. Bakers open their shutters, a ferry sounds its horn, and children "
    "walk to school past the library where volunteers sort donated books by colour and size. In the "
    "afternoon the market fills with apples, bread, and cheerful conversation about the weather. "
)


def load_token_ids(tokenizer, n: int) -> list[int]:
    ids = tokenizer(LOAD_TEXT, add_special_tokens=False)["input_ids"]
    return (ids * (n // len(ids) + 1))[:n]


def make_detector(model, method: str, cfg: dict, batch: int, device) -> tuple[StreamingDetector, dict]:
    from .classifier import ScaledMLP, load_checkpoint

    info = {"classifier_source": None, "threshold_logit": 0.0}
    clf = None
    if method != "baseline_no_detector":
        out = artifacts_dir(cfg)
        seed = cfg["train"]["seeds"][0]
        ck = out / "checkpoints" / method / str(seed)
        if (ck / "meta.json").exists():
            clf, _ = load_checkpoint(ck, device)
            info["classifier_source"] = f"checkpoint:{method}/{seed}"
            qs = out / "metrics" / "quality_summary.json"
            if qs.exists():
                for s in json.loads(qs.read_text()):
                    if s["classifier"] == method and s["seed"] == seed:
                        info["threshold_logit"] = s["threshold_logit"]
        else:  # same architecture -> same memory/latency; clearly flagged
            c = model.config
            d = feature_dim(method, c.hidden_size, c.num_key_value_heads * c.head_dim)
            clf = ScaledMLP(d, cfg["classifier"]["hidden"]).to(device).eval()
            info["classifier_source"] = "random_init_same_architecture"
        for p in clf.parameters():
            p.requires_grad_(False)
    det = StreamingDetector(model, method, cfg["model"]["layer_index"], clf, batch=batch, device=device)
    return det, info


@torch.inference_mode()
def run_request(model, det: StreamingDetector, ids: torch.Tensor, P: int, T: int, thr: float,
                timing: bool = False) -> dict:
    """Prefill ids[:, :P], then replay ids[:, P:P+T] one token per forward with the detector live."""
    from transformers import DynamicCache

    cuda = ids.is_cuda
    B = ids.shape[0]
    res = {"n_forwards": 0, "n_response_tokens": T}
    if cuda and timing:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    cache = DynamicCache(config=model.config)
    det.begin_request()
    alarm_t = torch.full((B,), -1, dtype=torch.long, device=ids.device) if det.classifier is not None else None
    model(input_ids=ids[:, :P], past_key_values=cache, use_cache=True, logits_to_keep=1)
    res["n_forwards"] += 1
    if cuda and timing:
        torch.cuda.synchronize()
        res["prefill_peak_allocated"] = torch.cuda.max_memory_allocated()
        res["prefill_peak_reserved"] = torch.cuda.max_memory_reserved()
        torch.cuda.reset_peak_memory_stats()
    t1 = time.perf_counter()
    det.start_response()
    for t in range(T):
        model(input_ids=ids[:, P + t:P + t + 1], past_key_values=cache, use_cache=True)
        res["n_forwards"] += 1
        s = det.after_forward(cache, 1)  # score of prefix t+1 (1-based)
        if s is not None:
            new = (s > thr) & (alarm_t < 0)
            alarm_t = torch.where(new, torch.full_like(alarm_t, t + 1), alarm_t)
    if cuda and timing:
        torch.cuda.synchronize()
        res["decode_peak_allocated"] = torch.cuda.max_memory_allocated()
        res["decode_peak_reserved"] = torch.cuda.max_memory_reserved()
    t2 = time.perf_counter()
    det.end_request()
    res["prefill_s"], res["decode_s"] = t1 - t0, t2 - t1
    res["cache_len"] = cache.get_seq_length()
    res["persistent_state_bytes"] = det.persistent_state_bytes() + (
        alarm_t.untyped_storage().nbytes() if alarm_t is not None else 0)
    res["first_alarm"] = alarm_t.tolist() if alarm_t is not None else None  # after timing, outside loop
    del cache
    return res


def worker(cfg: dict, method: str, batch: int, T: int, rep: int) -> dict:
    from .modeling import load_model, load_tokenizer

    bc = cfg["benchmark"]
    P = bc["prompt_tokens"]
    row = {"suite": bc["suite"], "method": method, "batch": batch, "prompt_tokens": P, "response_tokens": T,
           "rep": rep, "status": "ok"}
    device = torch.device("cuda")
    torch.manual_seed(0)
    tok = load_tokenizer(cfg)
    model = load_model(cfg, device)
    det, info = make_detector(model, method, cfg, batch, device)
    row.update(info)
    if det.classifier is not None:
        row["detector_parameter_bytes"] = det.classifier.parameter_bytes()
        row["scaler_bytes"] = det.classifier.scaler_bytes()
        row["detector_n_parameters"] = det.classifier.n_parameters()
    else:
        row["detector_parameter_bytes"] = row["scaler_bytes"] = row["detector_n_parameters"] = 0
    warm = bc["warmup_response_tokens"]
    ids = torch.tensor([load_token_ids(tok, P + max(T, warm))] * batch, device=device)
    try:
        run_request(model, det, ids, P, warm, info["threshold_logit"], timing=False)  # warm-up, discarded
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        row["resident_allocated_mib"] = torch.cuda.memory_allocated() / 2**20
        r = run_request(model, det, ids, P, T, info["threshold_logit"], timing=True)
    except torch.cuda.OutOfMemoryError as e:
        row.update(status="oom", error=str(e).splitlines()[0])
        return row
    mib = 2**20
    row.update({
        "n_forwards": r["n_forwards"], "cache_len": r["cache_len"],
        "peak_allocated_mib": max(r["prefill_peak_allocated"], r["decode_peak_allocated"]) / mib,
        "peak_reserved_mib": max(r["prefill_peak_reserved"], r["decode_peak_reserved"]) / mib,
        "decode_peak_allocated_mib": r["decode_peak_allocated"] / mib,
        "decode_peak_reserved_mib": r["decode_peak_reserved"] / mib,
        "prefill_ms": r["prefill_s"] * 1e3,
        "decode_ms_per_token": r["decode_s"] * 1e3 / T,
        "tokens_per_second_total": batch * T / r["decode_s"],  # summed over the batch
        "persistent_state_bytes": r["persistent_state_bytes"],
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
    })
    return row


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    ok = raw[raw.status == "ok"].copy()
    keys = ["batch", "prompt_tokens", "response_tokens", "rep"]
    base = ok[ok.method == "baseline_no_detector"].set_index(keys)
    out = []
    for (m, b, p, t), g in raw.groupby(["method", "batch", "prompt_tokens", "response_tokens"]):
        row = {"method": m, "batch": b, "prompt_tokens": p, "response_tokens": t,
               "n_ok": int((g.status == "ok").sum()), "n_oom": int((g.status == "oom").sum()),
               "n_failed": int((~g.status.isin(["ok", "oom"])).sum())}
        g = g[g.status == "ok"]
        if len(g):
            for c in ("peak_allocated_mib", "peak_reserved_mib", "decode_peak_allocated_mib",
                      "decode_ms_per_token", "tokens_per_second_total", "prefill_ms"):
                row[c + "_median"] = g[c].median()
                row[c + "_min"], row[c + "_max"] = g[c].min(), g[c].max()
            row["persistent_state_bytes"] = int(g.persistent_state_bytes.iloc[0])
            row["detector_parameter_bytes"] = int(g.detector_parameter_bytes.iloc[0])
            row["scaler_bytes"] = int(g.scaler_bytes.iloc[0])
            row["n_forwards"] = sorted(g.n_forwards.unique().tolist())
            gi = g.set_index(keys)
            common = gi.index.intersection(base.index)
            if len(common):
                d = gi.loc[common, "peak_allocated_mib"] - base.loc[common, "peak_allocated_mib"]
                dd = gi.loc[common, "decode_peak_allocated_mib"] - base.loc[common, "decode_peak_allocated_mib"]
                lat = gi.loc[common, "decode_ms_per_token"] / base.loc[common, "decode_ms_per_token"] - 1
                row["extra_peak_mib_median"] = d.median()
                row["extra_peak_mib_min"], row["extra_peak_mib_max"] = d.min(), d.max()
                row["extra_decode_peak_mib_median"] = dd.median()
                row["latency_increase_rel_median"] = lat.median()
                bspread = base.loc[common, "peak_allocated_mib"]
                noise = max(1.0, float(bspread.max() - bspread.min()))
                row["extra_peak_reliable"] = bool(abs(d.median()) > 2 * noise)
        out.append(row)
    return pd.DataFrame(out)


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--method")
    ap.add_argument("--batch", type=int)
    ap.add_argument("--response-tokens", type=int)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--methods", nargs="+", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.worker:
        try:
            row = worker(cfg, args.method, args.batch, args.response_tokens, args.rep)
        except Exception as e:  # noqa: BLE001
            row = {"method": args.method, "batch": args.batch, "response_tokens": args.response_tokens,
                   "rep": args.rep, "status": "error", "error": f"{type(e).__name__}: {e}"}
        Path(args.out).write_text(json.dumps(row, default=str))
        return
    if not torch.cuda.is_available():
        sys.exit("benchmark requires CUDA; no CPU numbers are produced or mixed into the CUDA tables.")
    bc = cfg["benchmark"]
    methods = args.methods or bc["methods"]
    for m in methods:
        assert m in METHODS, m
    out = artifacts_dir(cfg)
    mdir = out / "metrics"
    mdir.mkdir(exist_ok=True)
    rows = []
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    cells = list(itertools.product(bc["batch_sizes"], bc["response_tokens"], range(bc["repeats"])))
    for b, T, rep in cells:
        order = methods[rep % len(methods):] + methods[:rep % len(methods)]  # rotate order per repeat
        for m in order:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
                tmp = tf.name
            cmd = [sys.executable, "-m", "kv_safety.benchmark", "--config", cfg["_config_path"], "--worker",
                   "--method", m, "--batch", str(b), "--response-tokens", str(T), "--rep", str(rep), "--out", tmp]
            p = subprocess.run(cmd, env=env, capture_output=True, text=True)
            try:
                row = json.loads(Path(tmp).read_text())
            except (OSError, json.JSONDecodeError):
                row = {"method": m, "batch": b, "response_tokens": T, "rep": rep, "status": "crashed",
                       "error": p.stderr[-2000:]}
            os.unlink(tmp) if os.path.exists(tmp) else None
            row.setdefault("prompt_tokens", bc["prompt_tokens"])
            rows.append(row)
            print(f"[benchmark] {m} B={b} T={T} rep={rep}: {row.get('status')} "
                  f"peak={row.get('peak_allocated_mib', float('nan')):.1f}MiB "
                  f"ms/tok={row.get('decode_ms_per_token', float('nan')):.2f}", flush=True)
    sfx = bc.get("output_suffix", "")
    raw = pd.DataFrame(rows)
    raw.to_csv(mdir / f"memory_latency_raw{sfx}.csv", index=False)
    summarize(raw).to_csv(mdir / f"memory_latency_summary{sfx}.csv", index=False)
    from .environment import collect

    write_json(mdir / f"benchmark_environment{sfx}.json", collect())


if __name__ == "__main__":
    main()
