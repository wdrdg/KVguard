from conftest import ToyFastTokenizer
from kv_safety.data.alignment import align_record


def mk(response, spans, labels):
    return {"prompt": "q", "response": response, "sentence_spans": spans, "sentence_labels": labels}


def test_harmful_span_maps_to_tokens_1_based():
    tok = ToyFastTokenizer()
    resp = "Sure thing.\nDo bad stuff.\nOk."
    al, err = align_record(tok, mk(resp, [(0, 11), (12, 25), (26, 29)], [0, 1, 0]))
    assert err is None
    toks = [resp[s:e] for s, e in tok(resp, return_offsets_mapping=True)["offset_mapping"]]
    assert toks[al.a - 1] == "Do" and toks[al.b - 1] == "."
    assert (al.a, al.b) == (5, 8)
    assert al.T == len(toks)
    assert al.prompt_ids and al.n_boundary_tokens == 0


def test_boundary_crossing_token_is_included_and_counted():
    tok = ToyFastTokenizer()
    resp = "ok bad"            # second sentence chars [3,6) but token ' bad' covers [2,6)
    al, err = align_record(tok, mk(resp, [(0, 2), (3, 6)], [0, 1]))
    assert err is None and (al.a, al.b) == (2, 2) and al.n_boundary_tokens == 1


def test_safe_response_has_no_span():
    al, err = align_record(ToyFastTokenizer(), mk("All good.", [(0, 9)], [0]))
    assert err is None and al.a is None and al.b is None


def test_special_token_in_response_is_an_error_not_silently_dropped():
    al, err = align_record(ToyFastTokenizer(), mk("hi<|im_end|>", [(0, 12)], [1]))
    assert al is None and err == "special_token_in_response"
