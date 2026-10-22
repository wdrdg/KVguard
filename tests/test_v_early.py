from kv_safety.data.labels import IGNORE, prefix_labels
from kv_safety.v_early import early_labels


def test_harmful_prompt_and_harmful_response_is_positive_from_the_first_token():
    assert early_labels(6, 3, 5, 1, 1).tolist() == [1, 1, 1, 1, 1, 1]


def test_safe_response_is_negative_even_when_the_prompt_is_harmful():
    assert early_labels(4, None, None, 1, 0).tolist() == [0, 0, 0, 0]
    assert early_labels(4, None, None, 0, 0).tolist() == [0, 0, 0, 0]


def test_benign_prompt_keeps_the_harm_span():
    got = early_labels(10, 4, 6, 0, 1)
    assert got.tolist() == prefix_labels(10, 4, 6).tolist()
    assert got[2] == 0 and got[3] == IGNORE and got[5] == 1
