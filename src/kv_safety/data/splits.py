"""Leak-free split resolution and stratified sampling of whole responses.

Normalised prompts are used only to detect duplicates; model inputs keep the original text.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np

_WS = re.compile(r"\s+")


def normalise_prompt(p: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFKC", p)).strip().lower()


def prompt_group(p: str) -> str:
    return hashlib.sha1(normalise_prompt(p).encode("utf-8")).hexdigest()[:16]


def resolve_leakage(splits: dict[str, list[dict]], priority: list[str]) -> tuple[dict[str, list[dict]], dict]:
    """Remove exact prompt-response duplicates and cross-split prompt groups.

    A prompt group present in several splits is kept only in the highest-priority split
    (priority[0] first, default test > val > train) and dropped from the others.
    """
    stats: dict = {"exact_duplicates_removed": Counter(), "leaked_group_records_removed": Counter(),
                   "leaked_groups": 0}
    group_owner: dict[str, str] = {}
    for split in priority:
        for r in splits.get(split, []):
            g = prompt_group(r["prompt"])
            group_owner.setdefault(g, split)
    stats["leaked_groups"] = count_leaked_groups(splits)
    seen_pairs: set = set()
    out: dict[str, list[dict]] = {}
    for split in priority:
        kept = []
        for r in splits.get(split, []):
            g = prompt_group(r["prompt"])
            r = dict(r, prompt_group=g)
            if group_owner[g] != split:
                stats["leaked_group_records_removed"][split] += 1
                continue
            pair = (g, r["response"])
            if pair in seen_pairs:
                stats["exact_duplicates_removed"][split] += 1
                continue
            seen_pairs.add(pair)
            kept.append(r)
        out[split] = kept
    stats["exact_duplicates_removed"] = dict(stats["exact_duplicates_removed"])
    stats["leaked_group_records_removed"] = dict(stats["leaked_group_records_removed"])
    return out, stats


def count_leaked_groups(splits: dict[str, list[dict]]) -> int:
    where = defaultdict(set)
    for sp, rs in splits.items():
        for r in rs:
            where[prompt_group(r["prompt"])].add(sp)
    return sum(1 for v in where.values() if len(v) > 1)


def stratified_sample(records: list[dict], budget: int, seed: int) -> tuple[list[dict], dict]:
    """Sample up to `budget` responses, balancing response_label as far as possible."""
    rng = np.random.default_rng(seed)
    by = {0: [r for r in records if r["response_label"] == 0], 1: [r for r in records if r["response_label"] == 1]}
    n_h = min(len(by[1]), budget // 2)
    n_s = min(len(by[0]), budget - n_h)
    n_h = min(len(by[1]), budget - n_s)
    chosen = []
    for lab, n in ((0, n_s), (1, n_h)):
        idx = rng.permutation(len(by[lab]))[:n]
        chosen.extend(by[lab][i] for i in sorted(idx))
    order = rng.permutation(len(chosen))
    chosen = [chosen[i] for i in order]
    info = {
        "available": {"safe": len(by[0]), "harmful": len(by[1])},
        "selected": {"safe": n_s, "harmful": n_h},
        "sampling_ratio": {"safe": n_s / len(by[0]) if by[0] else None,
                           "harmful": n_h / len(by[1]) if by[1] else None},
    }
    return chosen, info
