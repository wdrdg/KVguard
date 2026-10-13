"""Chat-template construction and sentence -> response-token alignment via offset mapping.

Model input = token ids of the chat-templated user turn with the generation prompt
(thinking disabled), followed by the token ids of the response tokenised on its own. This
is exactly what the model sees when it generates the response itself, so the prompt/response
boundary never falls inside a token. The trailing <|im_end|>/EOS is not fed and never
counted or supervised.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Aligned:
    prompt_ids: list[int]
    response_ids: list[int]
    a: int | None             # 1-based first token of the first harmful sentence
    b: int | None             # 1-based last token of the first harmful sentence
    n_boundary_tokens: int    # tokens of that sentence that also cover chars outside it

    @property
    def T(self) -> int:
        return len(self.response_ids)

    @property
    def total_len(self) -> int:
        return len(self.prompt_ids) + len(self.response_ids)


def render_prompt(tokenizer, prompt: str, enable_thinking: bool = False) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def _strip_span(text: str, s: int, e: int) -> tuple[int, int]:
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def align_record(tokenizer, rec: dict, enable_thinking: bool = False) -> tuple[Aligned | None, str | None]:
    prompt_text = render_prompt(tokenizer, rec["prompt"], enable_thinking)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    enc = tokenizer(rec["response"], add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]
    if not ids:
        return None, "empty_response_tokens"
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    if any(i in special for i in ids):
        return None, "special_token_in_response"
    if any(e <= s for s, e in offs):
        return None, "empty_offset_in_response"
    a = b = None
    n_boundary = 0
    labels = rec["sentence_labels"]
    if any(labels):
        j = labels.index(1)
        cs, ce = _strip_span(rec["response"], *rec["sentence_spans"][j])
        if ce <= cs:
            return None, "harmful_sentence_whitespace_only"
        hit = [i for i, (s, e) in enumerate(offs) if s < ce and e > cs]
        if not hit:
            return None, "harmful_sentence_no_tokens"
        if hit != list(range(hit[0], hit[-1] + 1)):
            return None, "harmful_sentence_noncontiguous_tokens"
        a, b = hit[0] + 1, hit[-1] + 1
        n_boundary = sum(1 for i in hit if offs[i][0] < cs or offs[i][1] > ce)
    return Aligned(prompt_ids, ids, a, b, n_boundary), None
