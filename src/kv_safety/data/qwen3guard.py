"""Qwen3GuardTest: download audit only in this phase (response_loc is used after the main study).

response_loc samples are all harmful, so they can only measure detection timing, never FPR/AUROC.
Its unsafe_start_index/unsafe_end_index refer to the original input_ids/template and must be
mapped token -> char -> target token before use; that mapping is intentionally not applied here.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from .splits import prompt_group


def _load_any(path: Path):
    import pandas as pd

    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in (".json", ".jsonl"):
        try:
            return pd.read_json(path, lines=path.suffix == ".jsonl")
        except ValueError:
            return pd.read_json(path, lines=True)
    return None


def audit(root: Path, fineharm_groups: set[str]) -> dict:
    out: dict = {"root": str(root), "files": {}}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        df = _load_any(p)
        if df is None:
            continue
        rec: dict = {"n_rows": len(df), "columns": {c: str(t) for c, t in df.dtypes.items()}}
        for c in ("source", "subset", "split", "label", "category"):
            if c in df.columns and df[c].map(lambda x: isinstance(x, (str, int, bool))).all():
                rec[f"{c}_counts"] = dict(Counter(df[c].astype(str)).most_common(20))
        for c in ("unsafe_start_index", "unsafe_end_index"):
            if c in df.columns:
                s = df[c].dropna()
                rec[f"{c}_summary"] = {"n": int(len(s)), "min": float(s.min()) if len(s) else None,
                                       "max": float(s.max()) if len(s) else None}
        pcol = next((c for c in ("prompt", "query", "question") if c in df.columns), None)
        if pcol is None and "message" in df.columns:
            pcol = None
        if pcol is not None:
            groups = {prompt_group(str(x)) for x in df[pcol]}
            rec["prompt_overlap_with_fineharm"] = len(groups & fineharm_groups)
        else:
            rec["prompt_overlap_with_fineharm"] = "not computed (no flat prompt column; inspect message format)"
        out["files"][str(p.relative_to(root))] = rec
    return out
