"""Assemble REPORT.md and figures from whatever artifacts exist; missing stages are reported as
NOT EXECUTED rather than filled in.

python -m kv_safety.report --config configs/pilot.yaml
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import artifacts_dir, base_parser, load_config

THEORY = [("h_mean", 2560), ("k_mean", 1024), ("v_mean", 1024), ("kv_mean", 2048)]
QUALITY_COLS = ["safe_response_fpr", "timely_recall", "recall_by_end_plus_32", "recall_by_end_plus_128",
                "premature_alarm_rate", "no_alarm_rate", "nonpremature_detection_rate", "delay_median",
                "delay_p90", "final_prefix_auroc"]


def _md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    if df is None or df.empty:
        return "_(none)_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(format(v, floatfmt) if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _status(out: Path, sfx: str) -> pd.DataFrame:
    items = [("environment.json", "environment probe"), ("data_audit.json", "data audit"),
             ("split_manifest.jsonl", "split manifest"), ("feature_manifest.json", "feature extraction"),
             ("checkpoints/train_summary.json", "classifier training"),
             ("metrics/quality_summary.csv", "streaming evaluation"),
             (f"metrics/memory_latency_raw{sfx}.csv", "CUDA memory/latency benchmark")]
    return pd.DataFrame([{"stage": d, "artifact": f, "status": "EXECUTED" if (out / f).exists() else "NOT EXECUTED"}
                         for f, d in items])


def figures(out: Path, q: pd.DataFrame | None, s: pd.DataFrame | None) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fdir = out / "figures"
    fdir.mkdir(exist_ok=True)
    made = []
    if s is not None and not s.empty and "peak_allocated_mib_median" in s:
        b1 = s[s.batch == s.batch.min()]
        fig, ax = plt.subplots(figsize=(6, 4))
        for m, g in b1.groupby("method"):
            g = g.sort_values("response_tokens")
            ax.plot(g.response_tokens, g.peak_allocated_mib_median, marker="o", label=m)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("response tokens")
        ax.set_ylabel("peak allocated (MiB), median")
        ax.set_title(f"Peak CUDA memory vs length (batch={b1.batch.min()})")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fdir / "memory_vs_length.png", dpi=150)
        plt.close(fig)
        made.append("figures/memory_vs_length.png")
    if q is not None and not q.empty and s is not None and "extra_peak_mib_median" in s:
        sel = s[(s.batch == s.batch.min()) & (s.response_tokens == s.response_tokens.max())]
        qa = q.groupby("classifier")["timely_recall"].agg(["mean", "std"]).reset_index()
        merged = qa.merge(sel[["method", "extra_peak_mib_median"]], left_on="classifier", right_on="method")
        if not merged.empty:
            fig, ax = plt.subplots(figsize=(5.5, 4))
            ax.errorbar(merged.extra_peak_mib_median, merged["mean"], yerr=merged["std"].fillna(0), fmt="o")
            for _, r in merged.iterrows():
                ax.annotate(r.classifier, (r.extra_peak_mib_median, r["mean"]), fontsize=8,
                            xytext=(4, 4), textcoords="offset points")
            ax.set_xlabel(f"extra peak allocated vs baseline (MiB), T={int(sel.response_tokens.max())}")
            ax.set_ylabel("timely recall on test (mean ± sd over seeds)")
            ax.set_title("Quality vs measured overhead")
            fig.tight_layout()
            fig.savefig(fdir / "quality_vs_overhead.png", dpi=150)
            plt.close(fig)
            made.append("figures/quality_vs_overhead.png")
    return made


def main() -> None:
    ap = base_parser(__doc__)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = artifacts_dir(cfg)
    sfx = cfg["benchmark"].get("output_suffix", "")
    q = pd.read_csv(out / "metrics/quality_summary.csv") if (out / "metrics/quality_summary.csv").exists() else None
    s_path = out / f"metrics/memory_latency_summary{sfx}.csv"
    s = pd.read_csv(s_path) if s_path.exists() else None
    made = figures(out, q, s)
    L = [f"# Report: {cfg['run_name']}\n",
         "Controlled feature comparison (KV-cache running means vs hidden states) on one layer of "
         f"`{cfg['model']['id']}` (block index {cfg['model']['layer_index']}). Not a reproduction of "
         "SIREN/ShieldHead/Kelp. Labels are a sentence-level proxy (first harmful sentence), not "
         "human token-level ground truth. Conclusions apply only to this model, layer and data.\n"]
    if cfg["run_name"] == "smoke":
        L.append("> **Smoke run**: shows the pipeline executes. These numbers are NOT evidence of detection "
                 "quality, low false-positive rate or memory savings.\n")
    L += ["## Execution status\n", _md(_status(out, sfx))]
    envp = out / "environment.json"
    if envp.exists():
        e = json.loads(envp.read_text())
        L += ["## Environment\n", f"- OS: {e.get('os')}\n- Python: {e.get('python')}\n"
              f"- GPUs: {e.get('gpus')}\n- torch: {e['packages'].get('torch')}, transformers: "
              f"{e['packages'].get('transformers')}\n"]
    ap_ = out / "data_audit.json"
    if ap_.exists():
        a = json.loads(ap_.read_text())
        L += ["## Data\n", "Balanced sampling by response label: precision/F1 are not natural-traffic metrics.\n",
              "```json\n" + json.dumps({"leakage": a.get("leakage"), "splits": a.get("splits")}, indent=1) + "\n```\n"]
    L.append("## Streaming quality (test, first alarm, threshold from val safe responses at 5% FPR)\n")
    if q is not None:
        agg = q.groupby(["classifier", "layer_index", "dim", "n_parameters"])[QUALITY_COLS].agg(["mean", "std"])
        agg.columns = [f"{a}_{b}" for a, b in agg.columns]
        L.append("Mean and sd over classifier seeds:\n")
        L.append(_md(agg.reset_index()))
        L.append("\nPer seed:\n")
        L.append(_md(q[["classifier", "seed", "threshold_logit", "val_fpr", "val_n_safe", "calibration_unstable"]
                       + QUALITY_COLS]))
        L.append("`premature_alarm_rate` = alarms before the annotated harmful sentence (not counted as "
                 "success). `delay_*` are over non-premature detections only; read them with the miss rates.\n")
    else:
        L.append("NOT EXECUTED.\n")
    L.append("## Memory and latency (CUDA, separate process per method/cell)\n")
    if s is not None:
        cols = [c for c in ["method", "batch", "prompt_tokens", "response_tokens", "n_ok", "n_oom",
                            "peak_allocated_mib_median", "extra_peak_mib_median", "extra_peak_mib_min",
                            "extra_peak_mib_max", "extra_peak_reliable", "persistent_state_bytes",
                            "decode_ms_per_token_median", "latency_increase_rel_median",
                            "tokens_per_second_total_median"] if c in s]
        L.append(_md(s[cols]))
        L.append("`extra_peak_reliable=False` means no reliable difference was measured (within allocator "
                 "granularity / run-to-run spread). Throughput is summed over the batch.\n")
    else:
        L.append("NOT EXECUTED (requires a CUDA GPU).\n")
    L.append("## Theoretical accumulator sizes (sanity check only, not a result)\n")
    L.append(_md(pd.DataFrame([{"method": m, "fp32_bytes_per_sample": d * 4, "KiB": d * 4 / 1024}
                               for m, d in THEORY])))
    L.append("The model's own KV cache is common to all methods. H-mean and KV-mean accumulators differ by "
             "~2 KiB per sample, so a large peak-memory reduction is not expected a priori.\n")
    if made:
        L.append("## Figures\n" + "\n".join(f"![{m}]({m})" for m in made) + "\n")
    (out / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
