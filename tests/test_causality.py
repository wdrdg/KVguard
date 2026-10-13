"""Mechanism tests on a randomly initialised miniature Qwen3 (CPU, FP32).

They exercise the same transformers Qwen3 code path and cache API as Qwen3-4B, so they verify
the extraction/online logic (cache = post-RoPE K/V of the right layer, causality, token
alignment, no logit changes, O(1) state). Real-model numerics are covered by the gpu tests.
"""

import numpy as np
import pytest
import torch
from transformers import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from conftest import TINY_LAYER
from kv_safety.benchmark import run_request
from kv_safety.classifier import ScaledMLP
from kv_safety.extract import extract_response_features
from kv_safety.features.cache_reader import check_cache, layer_kv, new_kv
from kv_safety.features.hidden_hook import BlockOutputHook
from kv_safety.features.monitor import METHODS, StreamingDetector

PROMPT = [5, 17, 33, 2, 9, 41, 7]


def resp(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(10, 250, (n,), generator=g).tolist()


def test_cache_holds_post_rope_knorm_k_of_the_selected_layer(tiny_qwen3):
    m = tiny_qwen3
    attn = m.model.layers[TINY_LAYER].self_attn
    cap = {}

    def pre(mod, args, kwargs):
        cap["x"] = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        cap["pe"] = kwargs["position_embeddings"]

    h = attn.register_forward_pre_hook(pre, with_kwargs=True)
    cache = DynamicCache(config=m.config)
    ids = torch.tensor([PROMPT + resp(9)])
    with torch.inference_mode():
        m(input_ids=ids, past_key_values=cache, use_cache=True)
    h.remove()
    x = cap["x"]
    shape = (*x.shape[:-1], -1, attn.head_dim)
    k = attn.k_norm(attn.k_proj(x).view(shape)).transpose(1, 2)
    v = attn.v_proj(x).view(shape).transpose(1, 2)
    cos, sin = cap["pe"]
    _, k_rope = apply_rotary_pos_emb(k, k, cos, sin)
    ck, cv = layer_kv(cache, TINY_LAYER)
    assert torch.allclose(ck, k_rope, atol=1e-5)
    assert not torch.allclose(ck, k, atol=1e-3)  # it is NOT the pre-RoPE projection
    assert torch.allclose(cv, v, atol=1e-5)
    other_k, _ = layer_kv(cache, TINY_LAYER + 1)
    assert not torch.allclose(other_k, ck)
    check_cache(cache)


def test_chunked_equals_tokenwise_extraction(tiny_qwen3):
    r = resp(45, 1)
    a = extract_response_features(tiny_qwen3, PROMPT, r, TINY_LAYER, chunk_size=128)
    b = extract_response_features(tiny_qwen3, PROMPT, r, TINY_LAYER, chunk_size=1)
    c = extract_response_features(tiny_qwen3, PROMPT, r, TINY_LAYER, chunk_size=7)
    for f in a:
        assert a[f].shape[0] == 45 and a[f].dtype == np.float32
        assert np.abs(a[f] - b[f]).max() < 1e-4, f
        assert np.abs(a[f] - c[f]).max() < 1e-4, f


def test_selected_positions_match_full(tiny_qwen3):
    r = resp(30, 2)
    full = extract_response_features(tiny_qwen3, PROMPT, r, TINY_LAYER, 8)
    pos = np.array([1, 8, 9, 30])
    sel = extract_response_features(tiny_qwen3, PROMPT, r, TINY_LAYER, 8, pos)
    for f in full:
        assert np.allclose(sel[f], full[f][pos - 1], atol=1e-6)


def test_prefix_features_do_not_see_the_future(tiny_qwen3):
    prefix = resp(20, 3)
    a = extract_response_features(tiny_qwen3, PROMPT, prefix + resp(15, 4), TINY_LAYER, 16)
    b = extract_response_features(tiny_qwen3, PROMPT, prefix + resp(15, 5), TINY_LAYER, 16)
    for f in a:
        assert np.abs(a[f][:20] - b[f][:20]).max() < 1e-5, f
        assert np.abs(a[f][20:] - b[f][20:]).max() > 1e-4, f


def test_prompt_tokens_are_not_averaged(tiny_qwen3):
    r = resp(5, 6)
    a = extract_response_features(tiny_qwen3, PROMPT, r, TINY_LAYER, 4)
    # h_mean at position 1 equals h_last at position 1 -> only the response token is in the mean
    assert np.allclose(a["h_mean"][0], a["h_last"][0], atol=1e-6)
    assert np.allclose(a["h_mean"][2], a["h_last"][:3].mean(0), atol=1e-5)


@pytest.mark.parametrize("method", ["h_last", "h_mean", "k_mean", "v_mean", "kv_mean"])
def test_online_detector_matches_offline_features_and_prefix_index(tiny_qwen3, method):
    """Score after feeding response token t belongs to prefix t (1-based)."""
    m = tiny_qwen3
    r = resp(12, 7)
    off = extract_response_features(m, PROMPT, r, TINY_LAYER, 128)
    off["kv_mean"] = np.concatenate([off["k_mean"], off["v_mean"]], 1)
    det = StreamingDetector(m, method, TINY_LAYER, classifier=None)
    cache = DynamicCache(config=m.config)
    det.begin_request()
    with torch.inference_mode():
        m(input_ids=torch.tensor([PROMPT]), past_key_values=cache, use_cache=True)
        det.start_response()
        # nothing counted at prefill: the cache holds only the prompt
        assert cache.get_seq_length() == len(PROMPT)
        for t, tok in enumerate(r, start=1):
            m(input_ids=torch.tensor([[tok]]), past_key_values=cache, use_cache=True)
            det.after_forward(cache, 1)
            assert cache.get_seq_length() == len(PROMPT) + t
            feat = det.current_feature()[0].numpy()
            assert np.abs(feat - off[method][t - 1]).max() < 1e-4
    det.remove()


def test_monitor_does_not_change_logits_and_state_is_constant(tiny_qwen3, request):
    m = tiny_qwen3
    ids = torch.tensor([PROMPT + resp(40, 8)])
    P, T = len(PROMPT), 40
    with torch.inference_mode():
        ref = m(input_ids=ids).logits
    for method in ["h_last", "h_mean", "k_mean", "v_mean", "kv_mean"]:
        from kv_safety.features.monitor import feature_dim

        d = feature_dim(method, 64, 32)
        det = StreamingDetector(m, method, TINY_LAYER, ScaledMLP(d, 8).eval())
        request.addfinalizer(det.remove)
        cache = DynamicCache(config=m.config)
        det.begin_request()
        logits = []
        sizes = []
        with torch.inference_mode():
            logits.append(m(input_ids=ids[:, :P], past_key_values=cache, use_cache=True).logits)
            det.start_response()
            for t in range(T - 1):
                logits.append(m(input_ids=ids[:, P + t:P + t + 1], past_key_values=cache, use_cache=True).logits)
                s = det.after_forward(cache, 1)
                assert s.shape == (1,)
                sizes.append(det.persistent_state_bytes())
        got = torch.cat(logits, 1)
        assert torch.allclose(got, ref[:, :P + T - 1], atol=1e-4), method
        assert len(set(sizes)) == 1, method
        assert det.hook is None or len(det.hook.layer._forward_hooks) == 1
        # hidden-state methods hold no history list; only the fixed accumulators
        assert sum(v.numel() for v in det.state.values()) + sum(x.sum.numel() for x in det.means.values()) <= 64
        for x in det.means.values():
            assert x.sum.dtype == torch.float32
        det.remove()


def test_reset_between_requests(tiny_qwen3):
    m = tiny_qwen3
    det = StreamingDetector(m, "kv_mean", TINY_LAYER, None)

    def run(r):
        cache = DynamicCache(config=m.config)
        det.begin_request()
        with torch.inference_mode():
            m(input_ids=torch.tensor([PROMPT]), past_key_values=cache, use_cache=True)
            det.start_response()
            for tok in r:
                m(input_ids=torch.tensor([[tok]]), past_key_values=cache, use_cache=True)
                det.after_forward(cache, 1)
        return det.current_feature().clone()

    a1 = run(resp(10, 9))
    run(resp(25, 10))
    a2 = run(resp(10, 9))
    assert torch.allclose(a1, a2, atol=1e-6)


def test_new_kv_reads_only_new_positions_without_copying_full_cache(tiny_qwen3):
    m = tiny_qwen3
    cache = DynamicCache(config=m.config)
    with torch.inference_mode():
        m(input_ids=torch.tensor([PROMPT]), past_key_values=cache, use_cache=True)
        k_full, _ = layer_kv(cache, TINY_LAYER)
        k_new, _ = new_kv(cache, TINY_LAYER, 1)
    assert k_new.shape == (1, 1, 32)
    # a view into the cache storage, not a clone of the whole cache
    assert k_full.untyped_storage().data_ptr() == layer_kv(cache, TINY_LAYER)[0].untyped_storage().data_ptr()


def test_benchmark_schedule_identical_across_methods(tiny_qwen3):
    m = tiny_qwen3
    ids = torch.tensor([PROMPT + resp(16, 11)] * 2)
    seen = set()
    for method in METHODS:
        from kv_safety.features.monitor import feature_dim

        clf = None if method == "baseline_no_detector" else ScaledMLP(feature_dim(method, 64, 32), 8).eval()
        det = StreamingDetector(m, method, TINY_LAYER, clf, batch=2)
        r = run_request(m, det, ids, len(PROMPT), 16, thr=0.0)
        seen.add((r["n_forwards"], r["cache_len"], r["n_response_tokens"]))
        det.remove()
    assert seen == {(17, len(PROMPT) + 16, 16)}


def test_no_gradients(tiny_qwen3):
    assert all(not p.requires_grad for p in tiny_qwen3.parameters())
    r = extract_response_features(tiny_qwen3, PROMPT, resp(4), TINY_LAYER, 2)
    assert all(np.isfinite(v).all() for v in r.values())


def test_hook_is_removed(tiny_qwen3):
    layer = tiny_qwen3.model.layers[TINY_LAYER]
    n0 = len(layer._forward_hooks)
    h = BlockOutputHook(tiny_qwen3, TINY_LAYER, lambda x: None)
    assert len(layer._forward_hooks) == n0 + 1
    h.remove()
    assert len(layer._forward_hooks) == n0
