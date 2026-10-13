"""Real Qwen/Qwen3-4B alignment tests (CUDA + locally downloaded pinned checkpoint).

pytest -q -m gpu        # writes artifacts/gpu_alignment_log.json with the measured max errors

Inputs are short benign synthetic texts; no dataset text is used. Tolerances are fixed up front
for BF16 (relative to the feature's mean magnitude) and must not be loosened to make a run pass:
investigate masks, position ids, template, cache length and accumulation order instead.
"""

import json

import numpy as np
import pytest
import torch

from kv_safety.config import REPO_ROOT, load_config

pytestmark = pytest.mark.gpu
REL_TOL = 2e-2
LOG: dict = {}

TEXTS = [
    "The weather today is mild with light wind from the west.",
    "Bread rises because yeast produces carbon dioxide during fermentation.",
    "A library card lets you borrow books, films and music for free.",
    "Rivers carry sediment downstream and deposit it where the water slows.",
    "To boil an egg, place it in simmering water for about nine minutes.",
    "The violin has four strings tuned in perfect fifths.",
    "Photosynthesis converts light energy into chemical energy in plants.",
    "A good night's sleep helps memory consolidation.",
    "Tide tables predict high and low water for coastal towns.",
    "The museum opens at ten and closes at six on weekdays.",
    "Bees communicate the location of flowers with a waggle dance.",
    "Compound interest grows savings faster over long periods.",
    "Snow forms when water vapour freezes directly into ice crystals.",
    "Most maps place north at the top by convention.",
    "Stretching before running may reduce stiffness for some people.",
    "The recipe needs two cups of flour and one egg.",
    "Trains in the region run every fifteen minutes during peak hours.",
    "Glass is made by melting sand with soda and lime.",
    "Owls can rotate their heads much further than humans can.",
    "A haiku traditionally has three lines of five, seven and five syllables.",
]


@pytest.fixture(scope="module")
def real():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cfg = load_config(REPO_ROOT / "configs" / "smoke.yaml")
    cfg["offline"] = True
    from kv_safety.modeling import load_model, load_tokenizer

    try:
        tok = load_tokenizer(cfg)
        model = load_model(cfg, "cuda")
    except (OSError, SystemExit, ValueError) as e:
        pytest.skip(f"pinned Qwen3-4B not available locally: {e}")
    yield cfg, tok, model
    out = REPO_ROOT / "artifacts" / "gpu_alignment_log.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(LOG, indent=2, default=float))


def ids_for(tok, text):
    from kv_safety.data.alignment import render_prompt

    p = tok(render_prompt(tok, "Tell me one fact."), add_special_tokens=False)["input_ids"]
    r = tok(text, add_special_tokens=False)["input_ids"]
    return p, r


def rel_err(a, b):
    return float(np.abs(a - b).max() / (np.abs(b).mean() + 1e-12))


def test_config_layer_dims_and_frozen(real):
    cfg, _, m = real
    c = m.config
    assert (c.num_hidden_layers, c.hidden_size, c.num_key_value_heads, c.head_dim) == (36, 2560, 8, 128)
    assert c._attn_implementation == "sdpa" and next(m.parameters()).dtype == torch.bfloat16
    assert all(not p.requires_grad for p in m.parameters())
    assert cfg["model"]["layer_index"] == 23


def test_cache_is_post_rope_k_of_layer_23(real):
    from transformers import DynamicCache
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    from kv_safety.features.cache_reader import layer_kv

    cfg, tok, m = real
    L = cfg["model"]["layer_index"]
    attn = m.model.layers[L].self_attn
    cap = {}
    h = attn.register_forward_pre_hook(lambda mod, a, kw: cap.update(x=kw["hidden_states"],
                                                                     pe=kw["position_embeddings"]),
                                       with_kwargs=True)
    p, r = ids_for(tok, TEXTS[0])
    cache = DynamicCache(config=m.config)
    with torch.inference_mode():
        m(input_ids=torch.tensor([p + r], device="cuda"), past_key_values=cache, use_cache=True)
    h.remove()
    x = cap["x"]
    shape = (*x.shape[:-1], -1, attn.head_dim)
    with torch.inference_mode():
        k = attn.k_norm(attn.k_proj(x).view(shape)).transpose(1, 2)
        v = attn.v_proj(x).view(shape).transpose(1, 2)
        _, kr = apply_rotary_pos_emb(k, k, *cap["pe"])
    ck, cv = layer_kv(cache, L)
    assert torch.equal(ck, kr) and torch.equal(cv, v)
    assert not torch.equal(ck, layer_kv(cache, L - 1)[0])
    LOG["cache_k_equals_recomputed_post_rope"] = True


def test_chunked_vs_tokenwise_20_samples(real):
    from kv_safety.extract import extract_response_features

    cfg, tok, m = real
    L = cfg["model"]["layer_index"]
    worst = {}
    for text in TEXTS:
        p, r = ids_for(tok, text)
        a = extract_response_features(m, p, r, L, 128)
        b = extract_response_features(m, p, r, L, 1)
        for f in a:
            assert np.isfinite(a[f]).all()
            worst[f] = max(worst.get(f, 0.0), rel_err(a[f], b[f]))
    LOG["chunk128_vs_tokenwise_max_rel_err"] = worst
    assert all(v < REL_TOL for v in worst.values()), worst


def test_prefix_invariant_to_suffix(real):
    from kv_safety.extract import extract_response_features

    cfg, tok, m = real
    L = cfg["model"]["layer_index"]
    p, pre = ids_for(tok, "Here is a calm sentence about gardens and")
    _, s1 = ids_for(tok, " tomatoes growing in summer sun.")
    _, s2 = ids_for(tok, " a completely different ending about trains at night.")
    a = extract_response_features(m, p, pre + s1, L, 128)
    b = extract_response_features(m, p, pre + s2, L, 128)
    errs = {f: rel_err(a[f][:len(pre)], b[f][:len(pre)]) for f in a}
    LOG["prefix_invariance_max_rel_err"] = errs
    assert all(v < REL_TOL for v in errs.values()), errs


@pytest.mark.parametrize("method", ["h_last", "h_mean", "k_mean", "v_mean", "kv_mean"])
def test_online_alignment_prefill_first_decode_last(real, method):
    from transformers import DynamicCache

    from kv_safety.extract import extract_response_features
    from kv_safety.features.monitor import StreamingDetector

    cfg, tok, m = real
    L = cfg["model"]["layer_index"]
    p, r = ids_for(tok, TEXTS[1])
    off = extract_response_features(m, p, r, L, 1)
    off["kv_mean"] = np.concatenate([off["k_mean"], off["v_mean"]], 1)
    det = StreamingDetector(m, method, L, None)
    try:
        cache = DynamicCache(config=m.config)
        det.begin_request()
        with torch.inference_mode():
            m(input_ids=torch.tensor([p], device="cuda"), past_key_values=cache, use_cache=True)
            assert cache.get_seq_length() == len(p)  # prefill: no response token in cache yet
            det.start_response()
            errs = []
            for t, tk in enumerate(r, 1):  # t=1 first token, ..., t=T last token (flush)
                m(input_ids=torch.tensor([[tk]], device="cuda"), past_key_values=cache, use_cache=True)
                det.after_forward(cache, 1)
                assert cache.get_seq_length() == len(p) + t
                f = det.current_feature()
                assert f.dtype == torch.float32
                errs.append(rel_err(f[0].cpu().numpy(), off[method][t - 1]))
        LOG.setdefault("online_vs_offline_max_rel_err", {})[method] = max(errs)
        assert max(errs) < REL_TOL
    finally:
        det.remove()


def test_monitor_does_not_modify_logits_and_state_constant(real):
    from kv_safety.benchmark import run_request
    from kv_safety.classifier import ScaledMLP
    from kv_safety.features.monitor import METHODS, StreamingDetector, feature_dim

    cfg, tok, m = real
    L = cfg["model"]["layer_index"]
    p, r = ids_for(tok, " ".join(TEXTS[:4]))
    ids = torch.tensor([p + r], device="cuda")
    with torch.inference_mode():
        ref = m(input_ids=ids).logits
    forwards = set()
    for method in METHODS:
        clf = None if method == "baseline_no_detector" else ScaledMLP(feature_dim(method, 2560, 1024), 128).cuda().eval()
        det = StreamingDetector(m, method, L, clf)
        try:
            with torch.inference_mode():
                ref2 = m(input_ids=ids).logits
            assert torch.equal(ref, ref2), method  # hook leaves full-sequence logits bitwise unchanged
            r1 = run_request(m, det, ids, len(p), len(r) // 2, thr=0.0)
            r2 = run_request(m, det, ids, len(p), len(r), thr=0.0)
            assert r1["persistent_state_bytes"] == r2["persistent_state_bytes"]
            forwards.add((r2["n_forwards"], r2["cache_len"]))
        finally:
            det.remove()
    assert len(forwards) == 1
