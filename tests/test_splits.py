from kv_safety.data.fineharm import join_sentences, locate_sentences, normalise_record
from kv_safety.data.splits import count_leaked_groups, prompt_group, resolve_leakage, stratified_sample


def rec(split, i, prompt, response, label):
    return {"id": f"{split}-{i}", "source_split": split, "source_index": i, "prompt": prompt,
            "response": response, "response_label": label}


def test_same_prompt_never_crosses_splits_priority_test_val_train():
    splits = {
        "train": [rec("train", 0, "How do I bake?", "r1", 0), rec("train", 1, "Other", "r2", 1),
                  rec("train", 2, "  how do I   BAKE? ", "r3", 1)],
        "val": [rec("val", 0, "Other", "r4", 0)],
        "test": [rec("test", 0, "how do i bake?", "r5", 0)],
    }
    out, st = resolve_leakage(splits, ["test", "val", "train"])
    assert count_leaked_groups(out) == 0
    assert [r["id"] for r in out["test"]] == ["test-0"]
    assert [r["id"] for r in out["val"]] == ["val-0"]
    assert out["train"] == []
    assert st["leaked_group_records_removed"] == {"train": 3}
    # ids preserved and traceable
    assert all(r["prompt_group"] == prompt_group(r["prompt"]) for s in out.values() for r in s)


def test_exact_duplicates_removed():
    splits = {"train": [rec("train", 0, "p", "same", 0), rec("train", 1, "p", "same", 0),
                        rec("train", 2, "p", "diff", 1)], "val": [], "test": []}
    out, st = resolve_leakage(splits, ["test", "val", "train"])
    assert [r["id"] for r in out["train"]] == ["train-0", "train-2"]
    assert st["exact_duplicates_removed"] == {"train": 1}


def test_stratified_sample_balances_and_is_deterministic():
    recs = [rec("train", i, f"p{i}", "r", int(i < 30)) for i in range(100)]
    a, info = stratified_sample(recs, 40, seed=3)
    b, _ = stratified_sample(recs, 40, seed=3)
    assert [r["id"] for r in a] == [r["id"] for r in b]
    assert info["selected"] == {"safe": 20, "harmful": 20}
    c, info = stratified_sample(recs, 80, seed=3)  # harmful scarce -> take all 30, fill with safe
    assert info["selected"] == {"safe": 50, "harmful": 30}


def test_sentence_location_and_join():
    assert locate_sentences("Hi there. Bad bit.", ["Hi there.", "Bad bit."]) == [(0, 9), (10, 18)]
    assert locate_sentences("Hi there.", ["missing"]) is None
    text, spans = join_sentences(["A b.", "C d."], "\n")
    assert text == "A b.\nC d." and spans == [(0, 4), (5, 9)]


def test_normalise_record_prefers_trustworthy_response_and_checks_labels():
    r = {"prompt": "q", "response": "One. Two.", "sentences": ["One.", "Two."], "sentence_labels": [0, 1]}
    n, err = normalise_record(r, "train", 0, "\n")
    assert err is None and n["response_source"] == "field:response" and n["response_label"] == 1
    r2 = dict(r, response="something else")
    n2, _ = normalise_record(r2, "train", 1, "\n")
    assert n2["response_source"] == "joined_sentences" and n2["response"] == "One.\nTwo."
    r3 = dict(r, label=0)
    assert normalise_record(r3, "train", 2, "\n")[1].startswith("response_label_conflict")
    r4 = dict(r, sentence_labels=[0, "maybe"])
    assert normalise_record(r4, "train", 3, "\n")[1] == "unparseable_sentence_label"
