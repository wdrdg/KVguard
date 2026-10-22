"""End-to-end glue test on synthetic data + the tiny random Qwen3 (CPU). Proves the stages
connect (prepare -> extract -> train -> evaluate -> report); numbers are meaningless."""

import json
import random
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from conftest import TINY_LAYER, ToyFastTokenizer
from kv_safety.config import load_config

WORDS = "alpha beta gamma delta river stone light cloud paper green swift calm".split()


def sentence(rng, bad=False):
    w = [rng.choice(WORDS) for _ in range(rng.randint(3, 9))]
    if bad:
        w.insert(1, "harm")
    return " ".join(w).capitalize() + "."


def make_split(rng, n, tag):
    out = []
    for i in range(n):
        harmful = i % 2 == 0
        k = rng.randint(2, 5)
        bad_at = rng.randint(0, k - 1) if harmful else -1
        sents = [sentence(rng, j == bad_at) for j in range(k)]
        out.append({"prompt": f"{tag} question {i} {rng.choice(WORDS)}?", "sentences": sents,
                    "sentence_labels": [int(j >= bad_at and harmful and j == bad_at) for j in range(k)]})
    return out


@pytest.fixture()
def synthetic_run(tmp_path, tiny_qwen3):
    rng = random.Random(0)
    data = tmp_path / "fineharm"
    data.mkdir()
    splits = {s: make_split(rng, n, s) for s, n in (("train", 40), ("val", 24), ("test", 24))}
    splits["test"].append(dict(splits["test"][0]))  # exact duplicate -> removed
    leaked = "  " + splits["test"][3]["prompt"].upper()  # same prompt after normalisation -> dropped from train
    splits["train"].append({"prompt": leaked, "sentences": ["Calm river."], "sentence_labels": [0]})
    for split, recs in splits.items():
        (data / f"FineHarm-{split}.json").write_text(json.dumps(recs))
    cfg = yaml.safe_load(open("configs/smoke.yaml"))
    cfg["artifacts_dir"] = str(tmp_path / "art")
    cfg["resource_manifest"] = str(tmp_path / "resource_manifest.json")
    cfg["model"]["layer_index"] = TINY_LAYER
    cfg["offline"] = False
    cfg["model"]["expected_config"] = None
    cfg["data"]["construction"] = "pilot"
    cfg["data"]["budgets"] = {"train": 30, "dev": 20}
    cfg["allow_test_eval"] = False
    cfg["features"]["matrix"] = [{"layer": TINY_LAYER, "features": ["h_last", "h_mean", "k_mean", "v_mean", "kv_mean"]}]
    cfg["features"].pop("types", None)
    cfg["features"]["chunk_size"] = 16
    cfg["features"]["shard_size"] = 7
    cfg["classifier"]["matched"] = {"enabled": True, "name": "kv_mean_matched", "base_feature": "kv_mean",
                                    "layer": TINY_LAYER, "hidden": 12}
    cfg["classifier"]["hidden"] = 8
    cfg["train"]["seeds"] = [17, 42]
    cfg["evaluate"]["bootstrap_samples"] = 20
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    (tmp_path / "resource_manifest.json").write_text(json.dumps({"fineharm": {"local_path": str(data)}}))
    return load_config(p), p, tiny_qwen3


def test_full_pipeline(synthetic_run, monkeypatch):
    cfg, cfg_path, model = synthetic_run
    from kv_safety.config import artifacts_dir
    from kv_safety.data.prepare import fineharm_paths, prepare
    from kv_safety.extract import extract_split

    out = artifacts_dir(cfg)
    audit = prepare(cfg, ToyFastTokenizer(), fineharm_paths(cfg), out)
    assert audit["leakage"]["exact_duplicates_removed"] == {"test": 1}
    assert audit["leakage"]["leaked_group_records_removed"] == {"train": 1}
    man = [json.loads(l) for l in open(out / "split_manifest.jsonl")]
    assert all("prompt" not in r and "response" not in r for r in man)  # no raw text
    groups = {s: {r["prompt_group"] for r in man if r["split"] == s} for s in ("train", "dev", "calibration")}
    assert not (groups["train"] & groups["dev"]) and not (groups["dev"] & groups["calibration"])
    assert "test" not in {r["split"] for r in man}
    assert (out / "local_checks" / "alignment_check.md").exists()
    assert (out / "label_audit.jsonl").exists()

    fm = {"splits": {}, "layer_index": TINY_LAYER}
    for split in ("train", "dev"):
        fm["splits"][split] = extract_split(model, cfg, out, split)
        (out / "feature_manifest.json").write_text(json.dumps(fm))
    again = extract_split(model, cfg, out, "train")
    layer_key = str(TINY_LAYER)
    assert again["layers"][layer_key]["shards"] == fm["splits"]["train"]["layers"][layer_key]["shards"]
    from kv_safety.extract import load_split_features

    X, m = load_split_features(out, "train", "kv_mean", TINY_LAYER)
    assert X.shape[1] == 64 and X.dtype == np.float32
    per_resp = pd.Series(m["weight"]).groupby(m["sample_idx"]).sum()
    assert np.allclose(per_resp, 1.0, atol=1e-5)
    assert (pd.Series(m["position"]).groupby(m["sample_idx"]).size() <= 32).all()
    Xd, md = load_split_features(out, "dev", "h_mean", TINY_LAYER)
    assert (pd.Series(md["position"]).groupby(md["sample_idx"]).size() <= 32).all()
    assert len(Xd) < sum(r["T"] for r in man if r["split"] == "dev") or all(
        r["T"] <= 32 for r in man if r["split"] == "dev")

    from kv_safety import calibrate, evaluate, report, train
    from kv_safety.replay import assert_split_allowed

    with pytest.raises(SystemExit):
        assert_split_allowed(cfg, "test", None)
    monkeypatch.setattr("kv_safety.modeling.load_model", lambda cfg, device: model)
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(cfg_path), "--all-features"])
    train.main()
    assert (out / "checkpoints" / str(TINY_LAYER) / "kv_mean_matched" / "42" / "model.pt").exists()
    monkeypatch.setattr(sys, "argv", ["calibrate", "--config", str(cfg_path), "--split", "calibration",
                                      "--all-features"])
    calibrate.main()
    cal = pd.read_csv(out / "metrics" / "calibration_summary.csv")
    assert (cal.empirical_fpr <= 0.05 + 1e-9).all()
    monkeypatch.setattr(sys, "argv", ["evaluate", "--config", str(cfg_path), "--split", "dev", "--all-features"])
    evaluate.main()
    q = pd.read_csv(out / "metrics" / "quality_summary.csv")
    assert set(q.method) == {"h_last", "h_mean", "k_mean", "v_mean", "kv_mean", "kv_mean_matched"}
    assert len(q) == 12
    assert set(q.split) == {"dev"} and set(q.evaluation_role) == {"development"}
    pr = pd.read_parquet(out / "metrics" / "per_response.parquet")
    assert set(pr.split) == {"dev"}
    monkeypatch.setattr(sys, "argv", ["report", "--config", str(cfg_path)])
    report.main()
    rep = (out / "REPORT.md").read_text()
    assert "NOT EXECUTED (requires a CUDA GPU)" in rep and "Smoke" in rep
