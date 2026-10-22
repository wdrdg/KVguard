from kv_safety.data.labels import IGNORE, prefix_labels
from kv_safety.v_fuse import fuse_labels


def test_harmful_prompt_marks_every_generated_token():
    # The old span would be 0, ignore, ignore, 1. A harmful prompt overrides that.
    assert fuse_labels(6, 2, 4, 1).tolist() == [1, 1, 1, 1, 1, 1]


def test_harmful_prompt_safe_response_is_positive():
    assert fuse_labels(4, None, None, 1).tolist() == [1, 1, 1, 1]


def test_benign_prompt_keeps_the_span_rule():
    got = fuse_labels(10, 4, 6, 0)
    assert got.tolist() == prefix_labels(10, 4, 6).tolist()
    assert got[3] == IGNORE and got[5] == 1 and got[2] == 0


def test_benign_safe_response_stays_negative():
    assert fuse_labels(3, None, None, 0).tolist() == [0, 0, 0]


def test_prompt_label_must_be_binary():
    try:
        fuse_labels(3, None, None, None)
    except (TypeError, ValueError):
        return
    raise AssertionError("missing prompt label was accepted")
