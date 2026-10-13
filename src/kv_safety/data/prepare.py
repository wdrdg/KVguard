"""Build leak-free, length-filtered, label-balanced splits with token-level alignment.

python -m kv_safety.data.prepare --config configs/pilot.yaml

Outputs (artifacts_dir):
  data_audit.json            schema, counts, removals, sampling ratios (no raw text)
  split_manifest.jsonl       one row per selected response (ids, lengths, [a,b], no raw text)
  alignment_errors.jsonl     id + reason for every record that could not be aligned
  data/<split>.jsonl         token ids + labels used by extraction (local, gitignored)
  local_checks/alignment_check.md   >=20 random alignment examples with raw text (local only)
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from ..config import artifacts_dir, base_parser, load_config, read_manifest, write_json
from ..modeling import load_tokenizer
from .alignment import align_record
from .fineharm import load_fineharm
from .splits import count_leaked_groups, resolve_leakage, stratified_sample

SPLITS = ("train", "val", "test")


def fineharm_paths(cfg: dict) -> dict[str, Path]:
    man = read_manifest(cfg).get("fineharm")
    if not man:
        raise SystemExit("FineHarm not in resource_manifest.json; run kv_safety.download first")
    root = Path(man["local_path"])
    return {s: root / f for s, f in cfg["data"]["fineharm"]["files"].items()}


def prepare(cfg: dict, tokenizer, paths: dict[str, Path], out: Path) -> dict:
    dc = cfg["data"]
    raw, audit_norm = load_fineharm(paths, dc["sentence_joiner"])
    resolved, leak = resolve_leakage(raw, dc["split_priority"])
    assert count_leaked_groups(resolved) == 0
    audit: dict = {"normalise": audit_norm, "leakage": leak, "splits": {}}
    errors_f = open(out / "alignment_errors.jsonl", "w", encoding="utf-8")
    (out / "data").mkdir(exist_ok=True)
    manifest_rows, kept_examples = [], []
    for split in SPLITS:
        recs = resolved.get(split, [])
        aligned, err_counts, too_long, boundary = [], Counter(), 0, 0
        for r in recs:
            al, err = align_record(tokenizer, r, cfg["model"].get("enable_thinking", False))
            if err:
                err_counts[err] += 1
                errors_f.write(json.dumps({"id": r["id"], "split": split, "reason": err}) + "\n")
                continue
            if al.total_len > dc["max_length"]:
                too_long += 1
                continue
            boundary += al.n_boundary_tokens > 0
            aligned.append((r, al))
        by_id = {r["id"]: (r, al) for r, al in aligned}
        chosen, samp = stratified_sample([r for r, _ in aligned], dc["budgets"][split],
                                         dc["sample_seed"] + SPLITS.index(split))
        audit["splits"][split] = {
            "after_leakage": len(recs), "alignment_errors": dict(err_counts),
            "excluded_too_long": too_long, "eligible": len(aligned),
            "harmful_sentence_with_boundary_tokens": boundary, **samp,
        }
        with open(out / "data" / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in chosen:
                _, al = by_id[r["id"]]
                row = {"id": r["id"], "split": split, "source_split": r["source_split"],
                       "source_index": r["source_index"], "prompt_group": r["prompt_group"],
                       "response_label": r["response_label"], "T": al.T, "a": al.a, "b": al.b,
                       "n_prompt_tokens": len(al.prompt_ids), "response_source": r["response_source"]}
                manifest_rows.append(row)
                f.write(json.dumps({**row, "prompt_ids": al.prompt_ids, "response_ids": al.response_ids}) + "\n")
                kept_examples.append((r, al))
    errors_f.close()
    with open(out / "split_manifest.jsonl", "w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row) + "\n")
    write_alignment_check(tokenizer, kept_examples, out / "local_checks" / "alignment_check.md",
                          n=20, seed=dc["sample_seed"])
    write_json(out / "data_audit.json", audit)
    return audit


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
    if not args.skip_qwen3guardtest_audit:
        qg = read_manifest(cfg).get("qwen3guardtest")
        if qg:
            from .qwen3guard import audit as qg_audit

            groups = {json.loads(l)["prompt_group"] for l in open(out / "split_manifest.jsonl")}
            audit["qwen3guardtest"] = qg_audit(Path(qg["local_path"]), groups)
        else:
            audit["qwen3guardtest"] = "not downloaded"
        write_json(out / "data_audit.json", audit)
    summary = {s: {k: v for k, v in d.items() if k in ("eligible", "selected", "excluded_too_long")}
               for s, d in audit["splits"].items()}
    print(json.dumps({"leakage": audit["leakage"], "splits": summary}, indent=2))


if __name__ == "__main__":
    main()
