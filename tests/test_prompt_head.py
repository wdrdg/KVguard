import numpy as np

from kv_safety.prompt_head import _batches, _rates, group_weights


def test_group_weights_sum_to_one_per_group():
    groups = np.array(["a", "b", "a", "a", "b"])
    codes, weights = group_weights(groups)
    assert codes.tolist() == [0, 1, 0, 0, 1]
    assert np.isclose(weights[codes == 0].sum(), 1.0)
    assert np.isclose(weights[codes == 1].sum(), 1.0)
    assert np.allclose(weights[codes == 0], 1 / 3)


def test_batches_keep_equal_lengths_inside_the_token_budget():
    lengths = np.array([4, 4, 4, 8, 8])
    batches = _batches(lengths, token_budget=8)
    assert [int(lengths[idx[0]]) for _, idx in batches] == [4, 4, 8, 8]
    assert [len(idx) for _, idx in batches] == [2, 1, 1, 1]
    assert sorted(np.concatenate([idx for _, idx in batches]).tolist()) == [0, 1, 2, 3, 4]


def test_rates_count_prompt_positives():
    stats = _rates(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]))
    assert stats["tp"] == 1 and stats["fn"] == 1 and stats["fp"] == 1 and stats["tn"] == 1
    assert stats["precision"] == 0.5 and stats["recall"] == 0.5 and stats["fpr"] == 0.5
