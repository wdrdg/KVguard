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


def tag_prompt_groups(splits: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Copy records and attach prompt_group. Does not drop anything."""
    return {
        split: [dict(r, prompt_group=prompt_group(r["prompt"])) for r in rows]
        for split, rows in splits.items()
    }


def audit_official_overlap(splits: dict[str, list[dict]]) -> dict:
    """Count cross-split prompt groups and within-split exact duplicates without removing them."""
    where: dict[str, set[str]] = defaultdict(set)
    for split, rows in splits.items():
        for r in rows:
            where[r["prompt_group"]].add(split)
    pair_counts = {}
    for a, b in (("train", "test"), ("train", "val"), ("val", "test")):
        groups = {g for g, owners in where.items() if a in owners and b in owners}
        pair_counts[f"{a}_and_{b}"] = {
            "prompt_groups": len(groups),
            a: sum(r["prompt_group"] in groups for r in splits.get(a, [])),
            b: sum(r["prompt_group"] in groups for r in splits.get(b, [])),
        }
    exact = {}
    for split, rows in splits.items():
        seen: set[tuple[str, str]] = set()
        extra = 0
        for r in rows:
            key = (r["prompt_group"], r["response"])
            if key in seen:
                extra += 1
            seen.add(key)
        exact[split] = extra
    return {
        "action": "audited_not_removed",
        "cross_split_prompt_groups": pair_counts,
        "exact_duplicate_extra_records": exact,
        "note": "Official train and test rows are kept even when a prompt group also appears in another split.",
    }


def count_leaked_groups(splits: dict[str, list[dict]]) -> int:
    where = defaultdict(set)
    for sp, rs in splits.items():
        for r in rs:
            where[prompt_group(r["prompt"])].add(sp)
    return sum(1 for v in where.values() if len(v) > 1)


def _take(rng: np.random.Generator, rows: list[dict], n: int) -> list[dict]:
    if n <= 0 or not rows:
        return []
    idx = rng.permutation(len(rows))[:n]
    return [rows[int(i)] for i in idx]


def _sample_safe_keeping_prompt_mix(rows: list[dict], n: int, rng: np.random.Generator) -> list[dict]:
    """Within safe responses, keep the harmful-prompt vs other-prompt mix when the tag exists."""
    if n >= len(rows):
        return list(rows)
    tagged = [r for r in rows if r.get("prompt_safety_label") is not None]
    if len(tagged) != len(rows):
        return _take(rng, rows, n)
    harm_prompt = [r for r in rows if r.get("prompt_safety_label") == 1]
    other = [r for r in rows if r.get("prompt_safety_label") != 1]
    n_hp = int(round(n * len(harm_prompt) / len(rows))) if rows else 0
    n_hp = min(len(harm_prompt), n_hp)
    n_other = min(len(other), n - n_hp)
    n_hp = min(len(harm_prompt), n - n_other)
    return _take(rng, harm_prompt, n_hp) + _take(rng, other, n_other)


def stratified_sample(records: list[dict], budget: int, seed: int) -> tuple[list[dict], dict]:
    """Sample up to `budget` responses, balancing response_label as far as possible."""
    rng = np.random.default_rng(seed)
    by = {0: [r for r in records if r["response_label"] == 0], 1: [r for r in records if r["response_label"] == 1]}
    n_h = min(len(by[1]), budget // 2)
    n_s = min(len(by[0]), budget - n_h)
    n_h = min(len(by[1]), budget - n_s)
    chosen = _sample_safe_keeping_prompt_mix(by[0], n_s, rng) + _take(rng, by[1], n_h)
    order = rng.permutation(len(chosen))
    chosen = [chosen[i] for i in order]
    hp_safe = [r for r in by[0] if r.get("prompt_safety_label") == 1]
    hp_safe_kept = [r for r in chosen if r["response_label"] == 0 and r.get("prompt_safety_label") == 1]
    info = {
        "available": {"safe": len(by[0]), "harmful": len(by[1])},
        "selected": {"safe": n_s, "harmful": n_h},
        "sampling_ratio": {"safe": n_s / len(by[0]) if by[0] else None,
                           "harmful": n_h / len(by[1]) if by[1] else None},
        "harmful_prompt_safe_available": len(hp_safe),
        "harmful_prompt_safe_selected": len(hp_safe_kept),
    }
    return chosen, info


def _group(records: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[r["prompt_group"]].append(r)
    return groups


def partition_prompt_groups(records: list[dict], seed: int, fraction: float = 0.5
                            ) -> tuple[list[dict], list[dict], dict]:
    """Split prompt groups into two disjoint pools, stratified by the group's response labels.

    ``fraction`` is the share of groups assigned to the first pool. Response class mix is kept
    close to the source because safe-only, harmful-only and mixed groups are split separately.
    """
    groups = _group(records)
    rng = np.random.default_rng(seed)
    strata: dict[str, list[str]] = defaultdict(list)
    for gid, recs in groups.items():
        labs = {r["response_label"] for r in recs}
        key = "safe" if labs == {0} else "harmful" if labs == {1} else "mixed"
        strata[key].append(gid)
    first_ids: set[str] = set()
    stratum_sizes = {}
    for key, gids in strata.items():
        gids = list(gids)
        rng.shuffle(gids)
        n_first = int(round(len(gids) * fraction))
        first_ids.update(gids[:n_first])
        stratum_sizes[key] = {"groups": len(gids), "first": n_first, "second": len(gids) - n_first}
    first, second = [], []
    for gid, recs in groups.items():
        (first if gid in first_ids else second).extend(recs)
    info = {"stratum_groups": stratum_sizes, "n_first": len(first), "n_second": len(second),
            "n_groups_first": len({r["prompt_group"] for r in first}),
            "n_groups_second": len({r["prompt_group"] for r in second})}
    return first, second, info


def sample_by_prompt_group(records: list[dict], budget: int, seed: int) -> tuple[list[dict], dict]:
    """Sample whole prompt groups up to ``budget`` responses, targeting the original class ratio.

    Groups that would break the safe/harmful targets are deferred, then used to fill leftover
    slots without exceeding the budget. The original class ratio is therefore approximate.
    """
    if budget >= len(records):
        return list(records), {"capped": False, "selected_responses": len(records),
                               "selected_groups": len({r["prompt_group"] for r in records})}
    groups = _group(records)
    n_safe = sum(r["response_label"] == 0 for r in records)
    target_safe = int(round(budget * n_safe / len(records))) if records else 0
    target_harm = budget - target_safe
    rng = np.random.default_rng(seed)
    gids = list(groups)
    rng.shuffle(gids)
    chosen: list[dict] = []
    safe = harm = 0
    deferred: list[str] = []
    for gid in gids:
        recs = groups[gid]
        ds = sum(r["response_label"] == 0 for r in recs)
        dh = len(recs) - ds
        if safe + ds <= target_safe and harm + dh <= target_harm:
            chosen.extend(recs)
            safe += ds
            harm += dh
        else:
            deferred.append(gid)
    for gid in deferred:
        recs = groups[gid]
        if len(chosen) + len(recs) <= budget:
            chosen.extend(recs)
    info = {
        "capped": True,
        "target_safe": target_safe,
        "target_harmful": target_harm,
        "selected_responses": len(chosen),
        "selected_safe": sum(r["response_label"] == 0 for r in chosen),
        "selected_harmful": sum(r["response_label"] == 1 for r in chosen),
        "selected_groups": len({r["prompt_group"] for r in chosen}),
        "source_responses": len(records),
        "source_groups": len(groups),
    }
    return chosen, info


def audit_near_duplicates(records: list[dict]) -> dict:
    """Count residual near-duplicates after exact (prompt group, response text) removal.

    Rule: NFKC + casefold + collapsed whitespace on the response, compared inside a prompt group.
    Near-duplicates are recorded and kept. They are not dropped.
    """
    buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    for r in records:
        buckets[(r["prompt_group"], normalise_prompt(r["response"]))].append(r["response"])
    groups = extra = 0
    for originals in buckets.values():
        if len(originals) > 1 and len(set(originals)) > 1:
            groups += 1
            extra += len(originals) - 1
    return {
        "rule": "within a prompt group, responses equal after NFKC + casefold + whitespace collapse, "
                "but not exact string duplicates (those are already removed)",
        "near_duplicate_groups": groups,
        "near_duplicate_extra_records": extra,
        "action": "audited_not_removed",
    }


def class_counts(records: list[dict]) -> dict:
    safe = sum(r["response_label"] == 0 for r in records)
    harm = sum(r["response_label"] == 1 for r in records)
    n = len(records)
    return {"n": n, "safe": safe, "harmful": harm,
            "harmful_ratio": (harm / n) if n else None,
            "n_prompt_groups": len({r["prompt_group"] for r in records})}
