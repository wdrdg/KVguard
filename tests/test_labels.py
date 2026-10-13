import numpy as np
import pytest

from kv_safety.data.labels import IGNORE, position_weights, prefix_labels, select_train_positions


def test_boundaries():
    lab = prefix_labels(10, 4, 6)
    # t: 1 2 3 | 4 5 | 6 7 8 9 10
    assert lab.tolist() == [0, 0, 0, IGNORE, IGNORE, 1, 1, 1, 1, 1]


def test_single_token_sentence():
    lab = prefix_labels(5, 3, 3)
    assert lab.tolist() == [0, 0, 1, 1, 1]


def test_first_sentence_harmful():
    lab = prefix_labels(6, 1, 3)
    assert lab.tolist() == [IGNORE, IGNORE, 1, 1, 1, 1]


def test_all_safe():
    assert prefix_labels(4, None, None).tolist() == [0, 0, 0, 0]


def test_invalid_span():
    with pytest.raises(ValueError):
        prefix_labels(5, 4, 6)


def test_selection_keeps_priorities_and_limits():
    T, a, b = 300, 100, 120
    lab = prefix_labels(T, a, b)
    pos = select_train_positions(lab, a, b, 32, 32)
    assert len(pos) <= 32 and len(set(pos)) == len(pos)
    assert {a - 1, b, b + 32} <= set(pos.tolist())
    assert all(lab[p - 1] != IGNORE for p in pos)
    assert pos.min() >= 1 and pos.max() <= T
    assert pos.min() <= 10 and pos.max() >= T - 10  # spread over the prefix range


def test_selection_short_response_takes_all_valid():
    lab = prefix_labels(8, 3, 5)
    pos = select_train_positions(lab, 3, 5)
    assert pos.tolist() == [1, 2, 5, 6, 7, 8]


def test_selection_out_of_range_priorities_dropped():
    T, a, b = 100, 1, 90
    lab = prefix_labels(T, a, b)
    pos = select_train_positions(lab, a, b)
    assert pos.min() >= 1 and pos.max() <= T  # a-1=0 and b+32>T are skipped


def test_equal_weight_per_response():
    for n in (3, 32):
        assert np.isclose(position_weights(n).sum(), 1.0)
