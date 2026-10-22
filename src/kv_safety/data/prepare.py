"""Build leak-free splits with token-level alignment.

python -m kv_safety.data.prepare --config configs/pilot.yaml

Pilot keeps official test unread for scoring: it is used only to drop leaked prompt groups and to
audit counts. Cleaned official val is split into dev and calibration. Train/dev store the rows
extraction will read; calibration stores every safe response in that pool.

Outputs (artifacts_dir):
  data_audit.json
  split_manifest.jsonl          train / dev / calibration only (no raw text)
  alignment_errors.jsonl
  label_audit.jsonl             pending human review; no raw text
  data/<split>.jsonl            token ids used by extraction (local, gitignored)
  local_checks/                 raw-text spot checks, local only
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from ..config import REPO_ROOT, artifacts_dir, base_parser, load_config, read_manifest, write_json
from ..modeling import load_tokenizer
from .alignment import align_record
from .fineharm import load_fineharm
from .labels import IGNORE, prefix_labels
from .splits import (
    audit_near_duplicates,
    class_counts,
    count_leaked_groups,
    audit_official_overlap,
    partition_prompt_groups,
    prompt_group,
    resolve_leakage,
    sample_by_prompt_group,
    stratified_sample,
    tag_prompt_groups,
)

PILOT_SOURCES = ("train", "val", "test")


def fineharm_paths(cfg: dict) -> dict[str, Path]:
    man = read_manifest(cfg).get("fineharm")
    if not man:
        raise SystemExit("FineHarm not in resource_manifest.json; run kv_safety.download first")
    root = Path(man["local_path"])
    files = cfg["data"]["fineharm"]["files"]
    return {split: root / name for split, name in files.items()}


def _quantiles(values: list[int]) -> dict:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    qs = [0, 10, 25, 50, 75, 90, 99, 100]
    got = np.percentile(arr, qs)
    return {f"p{q}": float(v) for q, v in zip(qs, got)} | {"mean": float(arr.mean())}


def _supervision(rows: list[tuple[dict, object]]) -> dict:
    if not rows:
        return {"n": 0}
    valid_frac, ignore_frac = [], []
    for rec, al in rows:
        lab = prefix_labels(al.T, al.a, al.b)
        n_ign = int((lab == IGNORE).sum())
        valid_frac.append((al.T - n_ign) / al.T)
        ignore_frac.append(n_ign / al.T)
    return {
        "n": len(rows),
        "mean_valid_position_fraction": float(np.mean(valid_frac)),
        "mean_ignored_position_fraction": float(np.mean(ignore_frac)),
    }


def _prompt_safety_coverage(records: list[dict]) -> dict:
    n = len(records)
    known = sum(r.get("prompt_safety_label") is not None for r in records)
    return {
        "n": n,
        "n_known": known,
        "coverage": (known / n) if n else None,
        "rule": "FineHarm source filename _HP_/_BP_ is the prompt tag; _HR_/_BR_ is not used. "
                "Missing or unparsed source is unavailable and is not inferred from the response.",
        "status": "available" if known else "unavailable",
    }


def _public_row(rec: dict, al, split: str) -> dict:
    return {
        "id": rec["id"], "split": split, "source_split": rec["source_split"],
        "source_index": rec["source_index"], "prompt_group": rec["prompt_group"],
        "response_label": rec["response_label"], "prompt_safety_label": rec.get("prompt_safety_label"),
        "dataset_source": rec.get("dataset_source"), "harm_sentence_index": rec.get("harm_sentence_index"),
        "n_sentences": rec.get("n_sentences"), "T": al.T, "a": al.a, "b": al.b,
        "n_prompt_tokens": len(al.prompt_ids), "response_source": rec["response_source"],
    }


def _align_pool(records: list[dict], tokenizer, cfg: dict, split_name: str, errors_f) -> tuple[list, dict]:
    dc = cfg["data"]
    aligned, err_counts, too_long, context_excluded, boundary, missing_sentence = [], Counter(), 0, 0, 0, 0
    for r in records:
        if not r.get("sentence_labels"):
            missing_sentence += 1
        al, err = align_record(tokenizer, r, cfg["model"].get("enable_thinking", False))
        if err:
            err_counts[err] += 1
            errors_f.write(json.dumps({"id": r["id"], "split": split_name, "reason": err}) + "\n")
            continue
        limit = dc.get("max_length")
        if limit is not None and al.total_len > int(limit):
            too_long += 1
            errors_f.write(json.dumps({
                "id": r["id"], "split": split_name, "reason": "max_length",
                "total_len": al.total_len,
            }) + "\n")
            continue
        context_limit = dc.get("context_limit")
        if context_limit is not None and al.total_len > int(context_limit):
            context_excluded += 1
            errors_f.write(json.dumps({
                "id": r["id"], "split": split_name, "reason": "context_limit",
                "total_len": al.total_len, "context_limit": int(context_limit),
            }) + "\n")
            continue
        boundary += al.n_boundary_tokens > 0
        aligned.append((r, al))
    info = {
        "after_leakage": len(records),
        "alignment_errors": dict(err_counts),
        "alignment_failures": int(sum(err_counts.values())),
        "missing_sentence_labels": missing_sentence,
        "excluded_too_long": too_long,
        "excluded_context_limit": context_excluded,
        "eligible": len(aligned),
        "harmful_sentence_with_boundary_tokens": boundary,
        "class_eligible": class_counts([r for r, _ in aligned]),
        "length_quantiles_eligible": _quantiles([al.T for _, al in aligned]),
        "supervision_eligible": _supervision(aligned),
        "prompt_safety": _prompt_safety_coverage([r for r, _ in aligned]),
    }
    return aligned, info


def _write_split(path: Path, rows: list[tuple[dict, object]], split: str, manifest: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec, al in rows:
            row = _public_row(rec, al, split)
            manifest.append(row)
            f.write(json.dumps({**row, "prompt_ids": al.prompt_ids, "response_ids": al.response_ids}) + "\n")


def _select_label_audit(rows: list[tuple[dict, object]], n: int, seed: int) -> list[dict]:
    """Preselect review rows from train/dev. Pending until a person reviews them."""
    pools = {
        "first_sentence_harmful": [],
        "later_harmful": [],
        "harmful_prompt_safe_response": [],
        "safe_response": [],
    }
    for rec, al in rows:
        item = _public_row(rec, al, rec.get("_audit_split", rec["source_split"]))
        if rec["response_label"] == 1 and rec.get("harm_sentence_index") == 0:
            pools["first_sentence_harmful"].append(item)
        elif rec["response_label"] == 1 and (rec.get("harm_sentence_index") or 0) > 0:
            pools["later_harmful"].append(item)
        elif rec["response_label"] == 0 and rec.get("prompt_safety_label") == 1:
            pools["harmful_prompt_safe_response"].append(item)
        elif rec["response_label"] == 0:
            pools["safe_response"].append(item)
    rng = random.Random(seed)
    quota = max(1, n // 4)
    picked = []
    seen = set()
    for name in ("first_sentence_harmful", "later_harmful", "harmful_prompt_safe_response", "safe_response"):
        pool = [x for x in pools[name] if x["id"] not in seen]
        rng.shuffle(pool)
        for item in pool[:quota]:
            picked.append({**item, "audit_category": name, "review_status": "pending_human_review"})
            seen.add(item["id"])
    if len(picked) < n:
        rest = [x for pool in pools.values() for x in pool if x["id"] not in seen]
        rng.shuffle(rest)
        for item in rest:
            if len(picked) >= n:
                break
            picked.append({**item, "audit_category": "fill", "review_status": "pending_human_review"})
            seen.add(item["id"])
    return picked[:n]


def _smoke_assign(aligned: list[tuple[dict, object]], budgets: dict, seed: int) -> dict[str, list]:
    """Disjoint prompt groups drawn only from the train source."""
    rng = np.random.default_rng(seed)
    groups: dict[str, list] = {}
    for pair in aligned:
        groups.setdefault(pair[0]["prompt_group"], []).append(pair)
    gids = list(groups)
    rng.shuffle(gids)

    def labs(gid: str) -> set[int]:
        return {p[0]["response_label"] for p in groups[gid]}

    unused = list(gids)

    def take(budget: int, prefer_label: int | None, require_safe: bool) -> list:
        chosen: list = []
        chosen_ids: list[str] = []
        deferred: list[str] = []
        order = list(unused)
        if prefer_label is not None:
            order.sort(key=lambda g: 0 if prefer_label in labs(g) else 1)
        if require_safe:
            order.sort(key=lambda g: 0 if 0 in labs(g) else 1)
        for gid in order:
            recs = groups[gid]
            if len(chosen) + len(recs) > budget:
                deferred.append(gid)
                continue
            chosen.extend(recs)
            chosen_ids.append(gid)
            if len(chosen) >= budget:
                break
        for gid in chosen_ids:
            unused.remove(gid)
        if require_safe and not any(p[0]["response_label"] == 0 for p in chosen):
            for gid in list(unused):
                if 0 in labs(gid) and len(chosen) + len(groups[gid]) <= max(budget, len(groups[gid])):
                    chosen.extend(groups[gid])
                    unused.remove(gid)
                    break
        return chosen

    # Fill calibration and dev before train so a small pool cannot starve the safe-calibration set.
    calibration = take(budgets["calibration"], 0, True)
    dev = take(budgets["dev"], None, False)
    if unused and dev and len({p[0]["response_label"] for p in dev}) < 2:
        missing = 0 if not any(p[0]["response_label"] == 0 for p in dev) else 1
        extra = next((g for g in list(unused) if missing in labs(g)
                      and len(dev) + len(groups[g]) <= budgets["dev"]), None)
        if extra is not None:
            dev.extend(groups[extra])
            unused.remove(extra)
    train: list = []
    want = 0
    guard = 0
    while len(train) < budgets["train"] and unused and guard < len(gids) * 2:
        guard += 1
        pick = next((g for g in unused if want in labs(g) and len(train) + len(groups[g]) <= budgets["train"]), None)
        if pick is None:
            pick = next((g for g in unused if len(train) + len(groups[g]) <= budgets["train"]), None)
        if pick is None:
            break
        train.extend(groups[pick])
        unused.remove(pick)
        want = 1 - want
    return {"train": train, "dev": dev, "calibration": calibration}


def prepare(cfg: dict, tokenizer, paths: dict[str, Path], out: Path) -> dict:
    dc = cfg["data"]
    construction = dc.get("construction", "pilot")
    raw, audit_norm = load_fineharm(paths, dc["sentence_joiner"])
    if construction == "full_official":
        resolved = tag_prompt_groups(raw)
        _, leak_if_dropped = resolve_leakage(
            {k: [dict(r) for r in v] for k, v in raw.items()}, dc["split_priority"])
        leak = {
            "action": "not_removed",
            "would_have_removed_if_pilot_rule": leak_if_dropped,
            "official_overlap": audit_official_overlap(resolved),
            "note": "Train and test keep every official row. Overlap is disclosed, not used to delete rows.",
        }
    else:
        resolved, leak = resolve_leakage(raw, dc["split_priority"])
        assert count_leaked_groups(resolved) == 0
    if construction == "smoke_from_train":
        leak = dict(leak)
        leak["note"] = ("smoke rows are disjoint prompt groups from the cleaned train source; "
                        "val/test are read only to drop leaked prompt groups and exact duplicates")
    near = audit_near_duplicates([r for rows in resolved.values() for r in rows])
    audit: dict = {
        "construction": construction,
        "normalise": audit_norm,
        "leakage": leak,
        "near_duplicates": near,
        "prompt_safety_label": {
            "status_if_unparsed": "unavailable",
            "not_inferred_from_response": True,
        },
        "splits": {},
        "heldout_test": None,
    }
    errors_f = open(out / "alignment_errors.jsonl", "w", encoding="utf-8")
    (out / "data").mkdir(exist_ok=True)
    manifest: list[dict] = []
    kept_for_check: list[tuple[dict, object]] = []

    if construction == "smoke_from_train":
        aligned, info = _align_pool(resolved.get("train", []), tokenizer, cfg, "train_source", errors_f)
        assigned = _smoke_assign(aligned, dc["budgets"], dc["sample_seed"])
        audit["splits"]["train_source"] = info
        for split, rows in assigned.items():
            _write_split(out / "data" / f"{split}.jsonl", rows, split, manifest)
            audit["splits"][split] = {
                "retained": class_counts([r for r, _ in rows]),
                "length_quantiles": _quantiles([al.T for _, al in rows]),
                "supervision": _supervision(rows),
                "prompt_safety": _prompt_safety_coverage([r for r, _ in rows]),
                "note": "smoke subset; not a statistical calibration set" if split == "calibration" else "",
            }
            if split in ("train", "dev"):
                for rec, al in rows:
                    rec["_audit_split"] = split
                    kept_for_check.append((rec, al))
        audit["heldout_test"] = {"status": "not_read", "reason": "smoke is built from the train source only"}
    elif construction == "full_official":
        aligned_by = {}
        for split in PILOT_SOURCES:
            aligned, info = _align_pool(resolved.get(split, []), tokenizer, cfg, split, errors_f)
            aligned_by[split] = aligned
            audit["splits"][split] = info
            info["official_n"] = len(resolved.get(split, []))
            info["processed_n"] = len(aligned)
            missing = {r["id"] for r in resolved.get(split, [])} - {r["id"] for r, _ in aligned}
            info["unprocessed_ids"] = sorted(missing)
            info["id_coverage"] = (len(aligned) / info["official_n"]) if info["official_n"] else None
        cleaned_val, _ = resolve_leakage(
            {k: [dict(r) for r in v] for k, v in raw.items()}, dc["split_priority"])
        cleaned_groups = {r["prompt_group"] for r in cleaned_val["val"]}
        # Same groups the pilot partitioned: leakage-cleaned val, and the 2048 cap it used.
        pilot_like = [r for r, al in aligned_by["val"]
                      if r["prompt_group"] in cleaned_groups and al.total_len <= 2048]
        first, second, part_info = partition_prompt_groups(pilot_like, dc["sample_seed"])
        dev_groups = {r["prompt_group"] for r in first}
        cal_groups = {r["prompt_group"] for r in second}
        extra = [r for r, al in aligned_by["val"]
                 if r["prompt_group"] in cleaned_groups and r["prompt_group"] not in dev_groups
                 and r["prompt_group"] not in cal_groups]
        extra_first, extra_second, extra_info = partition_prompt_groups(extra, dc["sample_seed"])
        dev_groups |= {r["prompt_group"] for r in extra_first}
        cal_groups |= {r["prompt_group"] for r in extra_second}
        part_info["length_excluded_groups_added"] = extra_info
        pilot_rel = dc.get("pilot_split_manifest", "artifacts/pilot/split_manifest.jsonl")
        pilot_path = REPO_ROOT / pilot_rel
        pilot_dev, pilot_cal = set(), set()
        if pilot_path.exists():
            for line in pilot_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["split"] == "dev":
                    pilot_dev.add(row["prompt_group"])
                elif row["split"] == "calibration":
                    pilot_cal.add(row["prompt_group"])
            if not pilot_dev <= dev_groups or not pilot_cal <= cal_groups:
                raise RuntimeError("pilot dev/calibration groups are not a subset of the recovered partition")
        part_info["pilot_dev_groups_verified"] = len(pilot_dev)
        part_info["pilot_calibration_groups_verified"] = len(pilot_cal)
        part_info["rule"] = "pilot leakage-cleaned val, partition_prompt_groups seed=sample_seed; all rows in each pool kept"
        audit["val_partition"] = part_info
        dev_rows = [(r, al) for r, al in aligned_by["val"] if r["prompt_group"] in dev_groups]
        cal_pool = [(r, al) for r, al in aligned_by["val"] if r["prompt_group"] in cal_groups]
        withheld = [(r, al) for r, al in aligned_by["val"]
                    if r["prompt_group"] not in dev_groups and r["prompt_group"] not in cal_groups]
        cal_rows = [(r, al) for r, al in cal_pool if r["response_label"] == 0]
        train_rows = aligned_by["train"]
        test_rows = aligned_by["test"]
        audit["splits"]["dev"] = {
            "pool": class_counts([r for r, _ in dev_rows]),
            "retained": class_counts([r for r, _ in dev_rows]),
            "length_quantiles": _quantiles([al.T for _, al in dev_rows]),
            "supervision": _supervision(dev_rows),
            "prompt_safety": _prompt_safety_coverage([r for r, _ in dev_rows]),
            "note": "every official val response whose prompt group is in the recovered dev pool",
        }
        audit["splits"]["calibration"] = {
            "pool": class_counts([r for r, _ in cal_pool]),
            "retained_safe": class_counts([r for r, _ in cal_rows]),
            "excluded_harmful_from_calibration_file": sum(r["response_label"] == 1 for r, _ in cal_pool),
            "length_quantiles": _quantiles([al.T for _, al in cal_rows]),
            "supervision": _supervision(cal_rows),
            "prompt_safety": _prompt_safety_coverage([r for r, _ in cal_rows]),
            "note": "every safe response in the recovered calibration pool",
        }
        audit["splits"]["train"]["retained"] = class_counts([r for r, _ in train_rows])
        audit["splits"]["test"]["retained"] = class_counts([r for r, _ in test_rows])
        audit["val_withheld_prompt_overlap"] = {
            "n": len(withheld),
            "ids": [r["id"] for r, _ in withheld],
            "reason": "prompt group was kept in official test by the pilot leakage rule and is not reused as dev or calibration",
        }
        audit["heldout_test"] = {
            "status": "prepared_not_scored",
            "scored": False,
            "processed_n": len(test_rows),
            "official_n": audit["splits"]["test"]["official_n"],
            "unprocessed_ids": audit["splits"]["test"]["unprocessed_ids"],
            "reason": "test rows are written for the locked evaluation; scores are not computed in prepare",
        }
        for split, rows in (("train", train_rows), ("dev", dev_rows), ("calibration", cal_rows), ("test", test_rows)):
            _write_split(out / "data" / f"{split}.jsonl", rows, split, manifest)
            if split in ("train", "dev"):
                for rec, al in rows:
                    rec["_audit_split"] = split
                    kept_for_check.append((rec, al))
        dev_g = {r["prompt_group"] for r, _ in dev_rows}
        cal_g = {r["prompt_group"] for r, _ in cal_rows}
        if dev_g & cal_g:
            raise RuntimeError("dev and calibration share a prompt group")
        write_json(out / "test_exposure.json", {
            "pilot_scored_test": False,
            "evidence": "artifacts/pilot has no data/test.jsonl and evaluate was never run with split=test",
            "this_run_scored_test": False,
            "written_before_any_test_score": True,
        })
    else:
        aligned_by = {}
        for split in PILOT_SOURCES:
            aligned, info = _align_pool(resolved.get(split, []), tokenizer, cfg, split, errors_f)
            aligned_by[split] = aligned
            audit["splits"][split] = info
        dev_pool_pairs, cal_pool_pairs, part_info = _split_pairs(
            aligned_by["val"], dc["sample_seed"])
        audit["val_partition"] = part_info
        dev_pairs, dev_samp = sample_by_prompt_group(
            _as_records(dev_pool_pairs), dc["budgets"]["dev"], dc["sample_seed"] + 1)
        dev_by_id = {r["id"]: (r, al) for r, al in dev_pool_pairs}
        dev_rows = [dev_by_id[r["id"]] for r in dev_pairs]
        cal_rows = [(r, al) for r, al in cal_pool_pairs if r["response_label"] == 0]
        train_recs, train_samp = stratified_sample(
            [r for r, _ in aligned_by["train"]], dc["budgets"]["train"], dc["sample_seed"])
        train_by_id = {r["id"]: (r, al) for r, al in aligned_by["train"]}
        train_rows = [train_by_id[r["id"]] for r in train_recs]
        audit["splits"]["train"]["sampling"] = train_samp
        audit["splits"]["dev"] = {
            "pool": class_counts([r for r, _ in dev_pool_pairs]),
            "sampling": dev_samp,
            "retained": class_counts([r for r, _ in dev_rows]),
            "length_quantiles": _quantiles([al.T for _, al in dev_rows]),
            "supervision": _supervision(dev_rows),
            "prompt_safety": _prompt_safety_coverage([r for r, _ in dev_rows]),
        }
        audit["splits"]["calibration"] = {
            "pool": class_counts([r for r, _ in cal_pool_pairs]),
            "retained_safe": class_counts([r for r, _ in cal_rows]),
            "excluded_harmful_from_calibration_file": sum(r["response_label"] == 1 for r, _ in cal_pool_pairs),
            "length_quantiles": _quantiles([al.T for _, al in cal_rows]),
            "supervision": _supervision(cal_rows),
            "prompt_safety": _prompt_safety_coverage([r for r, _ in cal_rows]),
            "note": "threshold calibration uses every safe response in this pool; harmful pool rows are not scored",
        }
        audit["splits"]["train"]["retained"] = class_counts([r for r, _ in train_rows])
        audit["splits"]["train"]["length_quantiles"] = _quantiles([al.T for _, al in train_rows])
        audit["splits"]["train"]["supervision"] = _supervision(train_rows)
        audit["heldout_test"] = {
            "status": "held_out",
            "eligible": audit["splits"]["test"]["eligible"],
            "class_eligible": audit["splits"]["test"]["class_eligible"],
            "excluded_too_long": audit["splits"]["test"]["excluded_too_long"],
            "alignment_failures": audit["splits"]["test"]["alignment_failures"],
            "scored": False,
            "reason": "pilot does not extract or score the official test split",
        }
        for split, rows in (("train", train_rows), ("dev", dev_rows), ("calibration", cal_rows)):
            _write_split(out / "data" / f"{split}.jsonl", rows, split, manifest)
            if split in ("train", "dev"):
                for rec, al in rows:
                    rec["_audit_split"] = split
                    kept_for_check.append((rec, al))
        dev_groups = {r["prompt_group"] for r, _ in dev_rows}
        cal_groups = {r["prompt_group"] for r, _ in cal_rows}
        if dev_groups & cal_groups:
            raise RuntimeError("dev and calibration share a prompt group")
    errors_f.close()
    with open(out / "split_manifest.jsonl", "w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")
    audit["manifest_counts"] = dict(Counter(r["split"] for r in manifest))
    audit["manifest_prompt_group_overlap"] = _overlap(manifest)
    label_rows = _select_label_audit(kept_for_check, n=100, seed=dc["sample_seed"])
    with open(out / "label_audit.jsonl", "w", encoding="utf-8") as f:
        for row in label_rows:
            f.write(json.dumps(row) + "\n")
    audit["label_audit"] = {
        "n": len(label_rows),
        "review_status": "pending_human_review",
        "categories": dict(Counter(r["audit_category"] for r in label_rows)),
        "note": "[a, b] is an uncertain span inside the first harmful sentence, not a confirmed first "
                "harmful token. Cumulative labels are not the current-sentence label. "
                "No model self-check is treated as human review.",
    }
    note = out / "local_checks" / "label_audit_note.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        "# Label audit\n\n"
        f"Status: **pending human review** ({len(label_rows)} rows in `label_audit.jsonl`).\n\n"
        "A person still needs to check sentence labels and sentence boundaries. "
        "Model scores are not a substitute.\n\n"
        "`[a, b]` (1-based, inclusive) is the token span overlapping the first harmful sentence. "
        "Tokens inside the sentence before `b` are ignored by the training loss (`-100`); "
        "they are not a confirmed harm onset. After `b`, the cumulative prefix label stays harmful "
        "even if later sentences are safe.\n\n"
        f"Category counts: `{audit['label_audit']['categories']}`\n",
        encoding="utf-8",
    )
    write_alignment_check(tokenizer, kept_for_check, out / "local_checks" / "alignment_check.md",
                          n=20, seed=dc["sample_seed"])
    write_json(out / "data_audit.json", audit)
    return audit


def _as_records(pairs: list[tuple[dict, object]]) -> list[dict]:
    return [r for r, _ in pairs]


def _split_pairs(pairs: list[tuple[dict, object]], seed: int):
    records = [r for r, _ in pairs]
    first, second, info = partition_prompt_groups(records, seed)
    by_id = {r["id"]: (r, al) for r, al in pairs}
    return [by_id[r["id"]] for r in first], [by_id[r["id"]] for r in second], info


def _overlap(manifest: list[dict]) -> dict:
    groups: dict[str, set[str]] = {}
    for row in manifest:
        groups.setdefault(row["split"], set()).add(row["prompt_group"])
    names = list(groups)
    bad = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = groups[a] & groups[b]
            if shared:
                bad.append({"a": a, "b": b, "n": len(shared)})
    return {"overlaps": bad}


def write_alignment_check(tokenizer, examples, path: Path, n: int, seed: int) -> None:
    """Human spot check of token boundaries. Contains raw (possibly harmful) text: local only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    harmful = [e for e in examples if e[1].a is not None]
    safe = [e for e in examples if e[1].a is None]
    pick = rng.sample(harmful, min(len(harmful), n // 2 + n % 2)) + rng.sample(safe, min(len(safe), n // 2))
    lines = ["# Alignment spot check (LOCAL ONLY - contains raw dataset text)\n"]
    for r, al in pick:
        lines.append(f"## {r['id']}  label={r['response_label']}  T={al.T}  a={al.a}  b={al.b}  "
                     f"boundary_tokens={al.n_boundary_tokens}\n")
        if al.a is not None:
            j = r["sentence_labels"].index(1)
            s, e = r["sentence_spans"][j]
            lines.append("first harmful sentence (chars):\n```\n" + r["response"][s:e] + "\n```")
            lines.append("tokens [a-1 | a..b | b+1] decoded:\n```\n"
                         + repr(tokenizer.decode(al.response_ids[max(0, al.a - 2):al.a - 1])) + " | "
                         + repr(tokenizer.decode(al.response_ids[al.a - 1:al.b])) + " | "
                         + repr(tokenizer.decode(al.response_ids[al.b:al.b + 1])) + "\n```\n")
        else:
            lines.append("safe response; first 20 tokens decoded:\n```\n"
                         + repr(tokenizer.decode(al.response_ids[:20])) + "\n```\n")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--skip-qwen3guardtest-audit", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = artifacts_dir(cfg)
    tok = load_tokenizer(cfg)
    audit = prepare(cfg, tok, fineharm_paths(cfg), out)
    if not args.skip_qwen3guardtest_audit and cfg["data"].get("construction", "pilot") != "smoke_from_train":
        qg = read_manifest(cfg).get("qwen3guardtest")
        if qg:
            from .qwen3guard import audit as qg_audit

            groups = {json.loads(l)["prompt_group"] for l in open(out / "split_manifest.jsonl")}
            audit["qwen3guardtest"] = qg_audit(Path(qg["local_path"]), groups)
            audit["qwen3guardtest"]["scored"] = False
        else:
            audit["qwen3guardtest"] = "not downloaded"
        write_json(out / "data_audit.json", audit)
    summary = {s: d.get("retained", d.get("retained_safe", d.get("class_eligible")))
               for s, d in audit["splits"].items()}
    print(json.dumps({"construction": audit["construction"], "leakage": audit["leakage"],
                      "near_duplicates": audit["near_duplicates"], "splits": summary,
                      "manifest": audit["manifest_counts"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
