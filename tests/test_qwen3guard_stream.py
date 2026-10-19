import torch

from kv_safety.qwen3guard_stream import (
    _default_rope_parameters,
    _ensure_pad_token_id,
    _install_default_rope,
    _install_mask_compat,
)


class _MissingPad:
    """Matches transformers 5: a missing pad id raises until it is assigned."""

    def __getattribute__(self, key):
        if key == "pad_token_id" and "pad_token_id" not in object.__getattribute__(self, "__dict__"):
            raise AttributeError(key)
        return object.__getattribute__(self, key)


def test_missing_pad_token_id_defaults_to_none():
    config = _MissingPad()
    _ensure_pad_token_id(config)
    assert config.pad_token_id is None


def test_existing_pad_token_id_is_left_alone():
    class Config:
        pad_token_id = 151643

    config = Config()
    _ensure_pad_token_id(config)
    assert config.pad_token_id == 151643


def test_default_rope_matches_qwen3_frequencies_and_registers():
    class Config:
        rope_theta = 1000000.0
        head_dim = 128
        partial_rotary_factor = 1.0

    inv_freq, scale = _default_rope_parameters(Config())
    assert scale == 1.0
    assert inv_freq.shape == (64,)
    assert torch.isfinite(inv_freq).all()
    _install_default_rope()
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    assert ROPE_INIT_FUNCTIONS["default"] is _default_rope_parameters


def test_mask_compat_renames_old_embed_kwarg():
    import transformers.masking_utils as masking

    original_causal = masking.create_causal_mask
    original_sliding = masking.create_sliding_window_causal_mask
    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return "mask"

    try:
        masking.create_causal_mask = fake
        masking.create_sliding_window_causal_mask = fake
        _install_mask_compat()
        out = masking.create_causal_mask(input_embeds="emb", cache_position="pos", config="cfg")
        assert out == "mask"
        assert seen == {"inputs_embeds": "emb", "config": "cfg"}
        _install_mask_compat()
        assert masking.create_causal_mask._kvguard_mask_compat
    finally:
        masking.create_causal_mask = original_causal
        masking.create_sliding_window_causal_mask = original_sliding
