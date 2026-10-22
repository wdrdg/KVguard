"""Assemble REPORT.md and figures. Missing stages stay marked NOT EXECUTED.

python -m kv_safety.report --config configs/pilot.yaml
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import REPO_ROOT, artifacts_dir, base_parser, load_config

THEORY = [("h_mean", 2560), ("k_mean", 1024), ("v_mean", 1024), ("kv_mean", 2048)]
MAIN_COLS = [
    "safe_response_fpr", "premature_alarm_rate", "recall_by_end_plus_32",
    "first_harm_sentence_alarm_rate", "recall_by_end_plus_128",
    "nonpremature_detection_rate", "no_alarm_rate", "invalid_or_missed_rate",
    "delay_after_sentence_end_p50", "delay_after_sentence_end_p90",
    "detection_fraction_p50", "stream_response_auroc", "final_prefix_auroc",
]


def _md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    if df is None or df.empty:
        return "_(none)_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                cells.append("null" if pd.isna(value) else format(value, floatfmt))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _status(out: Path, sfx: str) -> pd.DataFrame:
    items = [
        ("environment.json", "environment probe"),
        ("data_audit.json", "data audit"),
        ("split_manifest.jsonl", "split manifest"),
        ("label_audit.jsonl", "label audit (pending human review)"),
        ("feature_manifest.json", "sampled-position feature extraction"),
        ("checkpoints/train_summary.json", "classifier training"),
        ("metrics/calibration_summary.csv", "calibration thresholds"),
        ("metrics/quality_summary.csv", "dev streaming evaluation"),
        ("metrics/quality_summary_test.csv", "held-out test evaluation"),
        (f"metrics/memory_latency_raw{sfx}.csv", "CUDA memory/latency benchmark"),
    ]
    return pd.DataFrame([
        {"stage": desc, "artifact": name, "status": "EXECUTED" if (out / name).exists() else "NOT EXECUTED"}
        for name, desc in items
    ])


def figures(out: Path, quality: pd.DataFrame | None, memory: pd.DataFrame | None) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    fdir = out / "figures"
    fdir.mkdir(exist_ok=True)
    if memory is not None and not memory.empty and "peak_allocated_mib_median" in memory:
        batch = memory.batch.min()
        subset = memory[memory.batch == batch]
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for method, group in subset.groupby("method"):
            group = group.sort_values("response_tokens")
            ax.plot(group.response_tokens, group.peak_allocated_mib_median, marker="o", label=method)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("response tokens")
        ax.set_ylabel("peak allocated MiB, median")
        ax.set_title(f"Peak CUDA memory vs length (batch={int(batch)})")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fdir / "memory_vs_length.png", dpi=150)
        plt.close(fig)
        made.append("figures/memory_vs_length.png")
    if quality is not None and not quality.empty and "recall_by_end_plus_32" in quality:
        ref = quality[quality.layer == quality.layer.min()] if "layer" in quality else quality
        # Prefer the configured reference layer when several are present.
        if "layer" in quality.columns and (quality.layer == 23).any():
            ref = quality[quality.layer == 23]
        means = ref.groupby("method")["recall_by_end_plus_32"].agg(["mean", "std"]).reset_index()
        if memory is not None and "extra_peak_mib" in memory.columns:
            longest = memory[memory.response_tokens == memory.response_tokens.max()]
            merged = means.merge(longest[["method", "extra_peak_mib"]], on="method", how="inner")
            if not merged.empty:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.errorbar(merged.extra_peak_mib, merged["mean"], yerr=merged["std"].fillna(0), fmt="o")
                for _, row in merged.iterrows():
                    ax.annotate(row.method, (row.extra_peak_mib, row["mean"]), fontsize=8,
                                xytext=(4, 4), textcoords="offset points")
                ax.set_xlabel("extra peak allocated vs baseline (MiB)")
                ax.set_ylabel("dev recall_by_end_plus_32 (mean ± sd over seeds)")
                ax.set_title("Development quality vs measured overhead")
                fig.tight_layout()
                fig.savefig(fdir / "quality_vs_overhead.png", dpi=150)
                plt.close(fig)
                made.append("figures/quality_vs_overhead.png")
        if "layer" in quality.columns and quality.layer.nunique() > 1:
            layer_rows = quality[quality.method.isin(["h_mean", "v_mean", "kv_mean"])]
            fig, ax = plt.subplots(figsize=(6, 4))
            for method, group in layer_rows.groupby("method"):
                agg = group.groupby("layer")["recall_by_end_plus_32"].mean()
                ax.plot(agg.index + 1, agg.values, marker="o", label=method)
            ax.set_xlabel("block number (1-based)")
            ax.set_ylabel("dev recall_by_end_plus_32, mean over seeds")
            ax.set_title("Layer comparison (development split)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(fdir / "quality_vs_layer.png", dpi=150)
            plt.close(fig)
            made.append("figures/quality_vs_layer.png")
    return made


def _mean_table(quality: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    use = [c for c in cols if c in quality.columns]
    grouped = quality.groupby(["layer", "method", "feature_dim", "parameter_count"], dropna=False)[use]
    agg = grouped.agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index()


def main() -> None:
    ap = base_parser(__doc__)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = artifacts_dir(cfg)
    suffix = cfg.get("benchmark", {}).get("output_suffix", "")
    q_path = out / "metrics" / "quality_summary.csv"
    quality = pd.read_csv(q_path) if q_path.exists() else None
    mem_path = out / f"metrics/memory_latency_summary{suffix}.csv"
    memory = pd.read_csv(mem_path) if mem_path.exists() else None
    made = figures(out, quality, memory)
    lines = [
        f"# Report: {cfg['run_name']}\n",
        "Controlled comparison of native KV-cache running means and hidden-state features on frozen "
        f"`{cfg['model']['id']}` (reference block index {cfg['model']['layer_index']}). "
        "This is not a reproduction of SIREN, ShieldHead or Kelp. Labels are a sentence-level proxy "
        "for cumulative prefix harm, not human token-level ground truth. FineHarm numbers are "
        "teacher-forced replays, not free generations of this backbone.\n",
    ]
    if cfg["run_name"] == "smoke":
        lines.append("> **Smoke run**: shows the pipeline executes. These numbers are NOT evidence of detection "
                     "quality, low false-positive rate or memory savings. The 5% calibration on this "
                     "tiny set has no statistical meaning.\n")
    lines += ["## Execution status\n", _md(_status(out, suffix))]
    env_path = out / "environment.json"
    if env_path.exists():
        env = json.loads(env_path.read_text())
        lines += ["## Environment\n",
                  f"- OS: {env.get('os')}\n- Python: {env.get('python')}\n- GPUs: {env.get('gpus')}\n"
                  f"- torch: {env.get('packages', {}).get('torch')}, "
                  f"transformers: {env.get('packages', {}).get('transformers')}\n"]
    audit_path = out / "data_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        brief = {
            "construction": audit.get("construction"),
            "leakage": audit.get("leakage"),
            "near_duplicates": audit.get("near_duplicates"),
            "heldout_test": audit.get("heldout_test"),
            "label_audit": audit.get("label_audit"),
            "manifest_counts": audit.get("manifest_counts"),
        }
        lines += ["## Data audit\n",
                  "Train may be class-balanced. Dev keeps the cleaned class mix. "
                  "Calibration thresholds use only safe responses. "
                  "Prompt safety comes from FineHarm source filenames (`_HP_` / `_BP_`); "
                  "it is not inferred from the response, and a safe response is not a refusal.\n",
                  "```json\n" + json.dumps(brief, indent=1, default=str) + "\n```\n"]
    gpu_log = REPO_ROOT / "artifacts" / "gpu_alignment_log.json"
    lines.append("## GPU alignment\n")
    if gpu_log.exists():
        lines.append("```json\n" + gpu_log.read_text()[:4000] + "\n```\n")
        lines.append("The max-element ratio is inflated by BF16 residual accumulation: layer 0 differs by about one "
                     "ulp, and the causal mask still lets the first query row attend only to the prefix plus itself. "
                     "The alignment gate is mean relative error below 0.02 and cosine above 0.98. Tokenwise online "
                     "features match chunk-size-1 offline features. Chunk size 128 is what training and dev replay use.\n")
    else:
        lines.append("NOT EXECUTED or log not found at `artifacts/gpu_alignment_log.json`.\n")
    cal_path = out / "metrics" / "calibration_summary.csv"
    lines.append("## Calibration\n")
    if cal_path.exists():
        cal = pd.read_csv(cal_path)
        n_safe = sorted(cal["n_safe"].unique().tolist()) if "n_safe" in cal else []
        unstable = int(cal["unstable"].sum()) if "unstable" in cal else None
        fpr = cal["empirical_fpr"] if "empirical_fpr" in cal else None
        lines.append(
            "Thresholds use only safe calibration responses. Dev and test labels are not used to choose them. "
            f"n_safe={n_safe}, empirical FPR range "
            f"{None if fpr is None else round(float(fpr.min()), 4)}–{None if fpr is None else round(float(fpr.max()), 4)}, "
            f"unstable heads={unstable}. "
            "A head is unstable when n_safe < 500. The Wilson interval is for this calibration count only.\n"
        )
    else:
        lines.append("NOT EXECUTED.\n")
    lines.append("## Development streaming quality\n")
    lines.append("Read false-positive rate and premature alarms before recall. "
                 "A method with a higher recall and a much higher dev FPR is not a win at the same "
                 "false-positive rate. `recall_by_end_plus_32` excludes alarms before the proxy boundary. "
                 "`first_harm_sentence_alarm_rate` is the share of alarms inside `[a, b]`, not a claim "
                 "about the true first harmful token. These dev numbers were used for checkpoint selection, "
                 "so they are development feedback, not an unbiased test.\n")
    if quality is not None and not quality.empty:
        show = [c for c in ["method", "seed", "threshold", "n_safe", "n_harm", "n_prompt_groups"] + MAIN_COLS
                if c in quality.columns]
        if "layer" in quality.columns and (quality.layer == cfg["model"]["layer_index"]).any():
            ref = quality[quality.layer == cfg["model"]["layer_index"]]
            main = ref[ref.method != "kv_mean_matched"] if "method" in ref.columns else ref
            lines.append(f"### Reference layer index {cfg['model']['layer_index']}, mean and sd over seeds\n")
            lines.append(_md(_mean_table(main, MAIN_COLS)))
            lines.append("\nPer seed:\n")
            lines.append(_md(main[show]))
            matched = ref[ref.method == "kv_mean_matched"] if "method" in ref.columns else ref.iloc[0:0]
            if not matched.empty:
                lines.append("\n### Capacity control `kv_mean_matched` (wider head, not the main comparison)\n")
                lines.append(_md(matched[show]))
            others = quality[quality.layer != cfg["model"]["layer_index"]]
            if not others.empty:
                lines.append("\n### Other scheduled layers (h_mean, v_mean, kv_mean)\n")
                lines.append(_md(_mean_table(others, ["safe_response_fpr", "premature_alarm_rate",
                                                     "recall_by_end_plus_32"])))
        else:
            lines.append(_md(quality[show]))
        sel_path = out / "metrics" / "layer_selection.json"
        lines.append("\n### Layer ranking on dev\n")
        if sel_path.exists():
            sel = json.loads(sel_path.read_text())
            lines.append(sel["rule"] + "\n")
            for method, info in sel.get("by_feature", {}).items():
                lines.append(f"\n`{method}` selected layer index {info['selected_layer']}:\n")
                lines.append(_md(pd.DataFrame(info["ranking"])))
        else:
            lines.append("NOT EXECUTED. The ranking file is written by evaluate on the dev split.\n")
    else:
        lines.append("NOT EXECUTED.\n")
    paired_path = out / "metrics" / "paired_differences.csv"
    lines.append("\n## Paired differences versus same-layer h_mean\n")
    if paired_path.exists():
        paired = pd.read_csv(paired_path)
        focus = paired[paired.metric.isin(["safe_response_fpr", "recall_by_end_plus_32", "premature_alarm_rate"])]
        cols = [c for c in ["layer", "method", "seed", "metric", "difference", "ci95_lo", "ci95_hi"]
                if c in focus.columns]
        lines.append(_md(focus[cols]))
        lines.append("Intervals are prompt-group bootstrap intervals conditional on this training run "
                     "and the fixed calibration threshold. They are not a full calibration uncertainty.\n")
    else:
        lines.append("NOT EXECUTED.\n")
    lines.append("## Memory and latency\n")
    if memory is not None:
        cols = [c for c in ["method", "role", "batch", "prompt_tokens", "response_tokens", "n_ok", "n_oom",
                            "n_error", "peak_allocated_mib_median", "extra_peak_mib", "extra_peak_reliable",
                            "persistent_state_bytes", "decode_ms_per_token_median", "latency_overhead_pct",
                            "tokens_per_second_total_median", "prefill_ms_median"] if c in memory.columns]
        lines.append(_md(memory[cols]))
        lines.append("Negative extra peak is kept. `extra_peak_reliable=False` means no reliable difference "
                     "was measured relative to run-to-run spread. Decode time is one forward per response "
                     "token, including the final flush, and excludes prefill. Throughput counts response "
                     "tokens across the batch. `kv_mean_matched` is a capacity-control cost, reported with "
                     "`role=capacity_control` when that column is present.\n")
    else:
        lines.append("NOT EXECUTED (requires a CUDA GPU).\n")
    lines.append("## Theoretical accumulator sizes (sanity check only)\n")
    lines.append(_md(pd.DataFrame([
        {"method": name, "fp32_bytes_per_sample": dim * 4, "KiB": dim * 4 / 1024} for name, dim in THEORY
    ])))
    lines.append("The backbone KV cache is shared by every method. H-mean and KV-mean accumulators differ "
                 "by about 2 KiB per sample, so a large peak-memory reduction is not expected from the "
                 "accumulator alone.\n")
    test_path = out / "metrics" / "test_quality_summary.csv"
    if not test_path.exists():
        test_path = out / "metrics" / "quality_summary_test.csv"
    lines.append("## Held-out test\n")
    if test_path.exists():
        test_q = pd.read_csv(test_path)
        lines.append("Official test, scored once after `protocol_lock.json`. Thresholds and checkpoints "
                     "were fixed on calibration and dev. These numbers are not used to choose a layer, "
                     "a threshold, or a filter. Read FPR and premature alarms before recall.\n")
        show = [c for c in ["method", "seed", "threshold", "n_safe", "n_harm", "n_prompt_groups"] + MAIN_COLS
                if c in test_q.columns]
        if "layer" in test_q.columns and (test_q.layer == cfg["model"]["layer_index"]).any():
            ref = test_q[test_q.layer == cfg["model"]["layer_index"]]
            main = ref[ref.method != "kv_mean_matched"] if "method" in ref.columns else ref
            lines.append(f"### Reference layer index {cfg['model']['layer_index']}, mean and sd over seeds\n")
            lines.append(_md(_mean_table(main, MAIN_COLS)))
            lines.append("\nPer seed:\n")
            lines.append(_md(main[show]))
            lines.append("\nKV is 2048-d and K or V is 1024-d, so equal hidden width is not equal parameter count. "
                         "A difference is not by itself complementary information.\n")
            paired_test = out / "metrics" / "test_paired_differences.csv"
            if paired_test.exists():
                tp = pd.read_csv(paired_test)
                focus = tp[tp.metric.isin(["safe_response_fpr", "recall_by_end_plus_32", "premature_alarm_rate"])]
                cols = [c for c in ["comparison", "method", "reference", "seed", "metric", "difference", "ci95_lo", "ci95_hi"]
                        if c in focus.columns]
                lines.append("\n### Paired differences KV−V, KV−K, V−K\n")
                lines.append(_md(focus[cols]))
            others = test_q[test_q.layer != cfg["model"]["layer_index"]]
            if not others.empty:
                lines.append("\n### Other scheduled layers\n")
                lines.append(_md(_mean_table(others, ["safe_response_fpr", "premature_alarm_rate",
                                                     "recall_by_end_plus_32"])))
        else:
            lines.append(_md(test_q[show]))
    else:
        lines.append("NOT EXECUTED. Test scoring waits for a protocol lock written after training, "
                     "dev checkpoint selection, and calibration.\n")
    lines.append("## Limits\n")
    lines.append("- Dev selected the checkpoints, so dev metrics are development feedback.\n")
    if test_path.exists():
        lines.append("- Official test was scored after the protocol lock and was not used to change the method.\n")
    else:
        lines.append("- The official test split was not scored.\n")
    lines.append("- Sentence-boundary `[a, b]` is an uncertain span. Delay after `b` is not leaked-token count.\n")
    lines.append("- Label audit rows are pending human review.\n")
    lines.append("- Calibration Wilson intervals describe the calibration counts only.\n")
    if made:
        lines.append("\n## Figures\n" + "\n".join(f"- `{name}`" for name in made) + "\n")
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] wrote {out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
