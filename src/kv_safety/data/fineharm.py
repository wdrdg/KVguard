"""FineHarm adapter: schema audit and normalisation into response records.

The actual JSON schema is inspected, not assumed. A record becomes
    {id, source_split, source_index, prompt, response, sentence_spans, sentence_labels,
     response_label, response_source}
where sentence_spans are character [start, end) offsets into `response`.
Records that cannot be normalised unambiguously are returned as errors, never guessed.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

PROMPT_KEYS = ("prompt", "query", "instruction", "question")
RESPONSE_KEYS = ("response", "answer", "output", "completion")
RESPONSE_LABEL_KEYS = ("response_label", "label", "harmful", "is_harmful")

_POS = {"1", "true", "harmful", "unsafe", "yes"}
_NEG = {"0", "false", "harmless", "safe", "no", "benign"}


def load_json_records(path: Path) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]  # JSONL fallback
    if isinstance(data, dict):
        for k in ("data", "records", "items"):
            if isinstance(data.get(k), list):
                return data[k]
        raise ValueError(f"{path}: top-level dict without a record list (keys={list(data)[:10]})")
    return data


def parse_binary(v: Any) -> int | None:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)) and v in (0, 1):
        return int(v)
    if isinstance(v, str) and v.strip().lower() in _POS | _NEG:
        return 1 if v.strip().lower() in _POS else 0
    return None


def audit_schema(records: list[dict], max_examples: int = 3) -> dict:
    """Field names, types and label value counts. Never includes raw text."""
    key_counts: Counter = Counter()
    type_counts: dict[str, Counter] = {}
    label_values: dict[str, Counter] = {}
    for r in records:
        for k, v in r.items():
            key_counts[k] += 1
            type_counts.setdefault(k, Counter())[type(v).__name__] += 1
            if "label" in k or k in RESPONSE_LABEL_KEYS:
                vals = v if isinstance(v, list) else [v]
                for x in vals:
                    if not isinstance(x, (list, dict)):
                        label_values.setdefault(k, Counter())[repr(x)] += 1
    return {
        "n_records": len(records),
        "keys": dict(key_counts),
        "types": {k: dict(v) for k, v in type_counts.items()},
        "label_values": {k: dict(v.most_common(20)) for k, v in label_values.items()},
    }


def _first_key(r: dict, keys: Iterable[str]) -> str | None:
    for k in keys:
        if k in r and r[k] is not None:
            return k
    return None


def locate_sentences(response: str, sentences: list[str]) -> list[tuple[int, int]] | None:
    """Find each sentence sequentially in `response`; None if any is missing (ambiguous)."""
    spans, cur = [], 0
    for s in sentences:
        idx = response.find(s, cur)
        if idx < 0:
            return None
        spans.append((idx, idx + len(s)))
        cur = idx + len(s)
    return spans


def join_sentences(sentences: list[str], joiner: str) -> tuple[str, list[tuple[int, int]]]:
    spans, parts, cur = [], [], 0
    for i, s in enumerate(sentences):
        if i:
            parts.append(joiner)
            cur += len(joiner)
        spans.append((cur, cur + len(s)))
        parts.append(s)
        cur += len(s)
    return "".join(parts), spans


def prompt_safety_from_source(source: str | None) -> tuple[int | None, str]:
    """FineHarm `source` filenames encode the prompt category as ``_BP_`` / ``_HP_``.

    The response half of those names (``_BR_`` / ``_HR_``) is not consulted. This is the dataset's
    own prompt tag, not a label inferred from the response. Unrecognised values stay unavailable.
    """
    if not isinstance(source, str) or not source:
        return None, "missing_source"
    if "_HP_" in source:
        return 1, "source_filename_HP"
    if "_BP_" in source:
        return 0, "source_filename_BP"
    return None, "source_filename_unparsed"


def normalise_record(r: dict, source_split: str, index: int, joiner: str) -> tuple[dict | None, str | None]:
    pk = _first_key(r, PROMPT_KEYS)
    if pk is None or not isinstance(r[pk], str):
        return None, "missing_prompt"
    sentences = r.get("sentences")
    labels_raw = r.get("sentence_labels")
    if not isinstance(sentences, list) or not isinstance(labels_raw, list):
        return None, "missing_sentences_or_labels"
    if len(sentences) != len(labels_raw) or not sentences:
        return None, "sentence_label_length_mismatch"
    if not all(isinstance(s, str) for s in sentences):
        return None, "non_string_sentence"
    labels = [parse_binary(x) for x in labels_raw]
    if any(x is None for x in labels):
        return None, "unparseable_sentence_label"

    rk = _first_key(r, RESPONSE_KEYS)
    response, spans, source = None, None, None
    if rk is not None and isinstance(r[rk], str):
        spans = locate_sentences(r[rk], sentences)
        if spans is not None:
            response, source = r[rk], f"field:{rk}"
    if response is None:
        response, spans = join_sentences(sentences, joiner)
        source = "joined_sentences"
    if any(e <= s for s, e in spans):
        return None, "empty_sentence"

    response_label = int(any(labels))
    lk = _first_key(r, RESPONSE_LABEL_KEYS)
    if lk is not None:
        given = parse_binary(r[lk])
        if given is not None and given != response_label:
            return None, f"response_label_conflict:{lk}"
    rid = r.get("id", r.get("idx"))
    rid = f"{source_split}-{rid if rid is not None else index}"
    dataset_source = r.get("source") if isinstance(r.get("source"), str) else None
    prompt_safety, prompt_safety_rule = prompt_safety_from_source(dataset_source)
    harm_at = labels.index(1) if response_label else None
    return {
        "id": rid,
        "source_split": source_split,
        "source_index": index,
        "prompt": r[pk],
        "response": response,
        "sentence_spans": spans,
        "sentence_labels": labels,
        "response_label": response_label,
        "response_source": source,
        "dataset_source": dataset_source,
        "prompt_safety_label": prompt_safety,
        "prompt_safety_rule": prompt_safety_rule,
        "n_sentences": len(labels),
        "harm_sentence_index": harm_at,
    }, None


def load_fineharm(paths: dict[str, Path], joiner: str) -> tuple[dict[str, list[dict]], dict]:
    out, audit = {}, {}
    for split, path in paths.items():
        raw = load_json_records(path)
        recs, errors = [], Counter()
        sources = Counter()
        for i, r in enumerate(raw):
            rec, err = normalise_record(r, split, i, joiner)
            if err:
                errors[err] += 1
                continue
            sources[rec["response_source"]] += 1
            recs.append(rec)
        ids = [r["id"] for r in recs]
        if len(set(ids)) != len(ids):  # make ids unique if the source ids collide
            for i, r in enumerate(recs):
                r["id"] = f"{split}-{r['source_index']}"
        out[split] = recs
        audit[split] = {
            "file": str(path),
            "schema": audit_schema(raw),
            "n_normalised": len(recs),
            "normalise_errors": dict(errors),
            "response_source": dict(sources),
            "response_label_counts": dict(Counter(r["response_label"] for r in recs)),
        }
    return out, audit
