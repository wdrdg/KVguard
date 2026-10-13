import pytest
import torch


@pytest.fixture(scope="session")
def tiny_qwen3():
    """Randomly initialised miniature Qwen3 (same code path as Qwen3-4B) for CPU mechanism tests.
    Synthetic weights: validates causality/cache/alignment logic, says nothing about the real model."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=1024,
                      rms_norm_eps=1e-6, tie_word_embeddings=False)
    cfg._attn_implementation = "sdpa"
    m = Qwen3ForCausalLM(cfg).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


TINY_LAYER = 2


class ToyFastTokenizer:
    """Word/space/newline tokenizer with offsets and hashed ids < 256; '<|im_end|>' is special (id 0)."""
    is_fast = True
    all_special_ids = [0]

    def _id(self, s):
        import zlib

        return 0 if s == "<|im_end|>" else zlib.crc32(s.encode()) % 250 + 5

    def apply_chat_template(self, msgs, tokenize, add_generation_prompt, enable_thinking):
        assert enable_thinking is False and add_generation_prompt and not tokenize
        return f"<user>{msgs[0]['content']}</user><assistant>\n<think>\n\n</think>\n\n"

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        import re

        ids, offs = [], []
        for m in re.finditer(r"<\|im_end\|>| ?[A-Za-z']+|[.,!?]|\n|\s+|<[^>]*>|.", text):
            ids.append(self._id(m.group()))
            offs.append((m.start(), m.end()))
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offs
        return out

    def decode(self, ids):
        return " ".join(f"<{i}>" for i in ids)
