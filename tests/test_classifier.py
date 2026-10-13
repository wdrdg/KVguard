import numpy as np
import torch

from kv_safety.classifier import MIN_SCALE, ScaledMLP, fit_weighted_scaler, load_checkpoint, save_checkpoint
from kv_safety.data.labels import IGNORE
from kv_safety.train import response_equal_bce, train_one, weighted_bce


def test_weighted_scaler_gives_each_response_equal_weight_and_handles_zero_variance():
    # response A: 1 row of value 0; response B: 3 rows of value 4 -> weighted mean 2, not 3
    x = np.array([[0.0, 7.0], [4.0, 7.0], [4.0, 7.0], [4.0, 7.0]], np.float32)
    w = np.array([1.0, 1 / 3, 1 / 3, 1 / 3], np.float32)
    mean, scale = fit_weighted_scaler(x, w)
    assert np.allclose(mean, [2.0, 7.0])
    assert np.isclose(scale[0], 2.0)
    assert scale[1] == 1.0  # zero variance -> safe denominator
    m = ScaledMLP(2, 4, mean, scale)
    assert torch.isfinite(m(torch.tensor(x))).all()
    assert MIN_SCALE > 0


def test_scaler_only_sees_training_rows(monkeypatch):
    seen = {}
    import kv_safety.train as tr

    real = tr.fit_weighted_scaler

    def spy(x, w):
        seen["n"] = len(x)
        return real(x, w)

    monkeypatch.setattr(tr, "fit_weighted_scaler", spy)
    rng = np.random.default_rng(0)
    Xtr = rng.normal(size=(40, 3)).astype(np.float32)
    ytr = (Xtr[:, 0] > 0).astype(np.int64)
    wtr = np.full(40, 0.25, np.float32)
    Xva = rng.normal(size=(1000, 3)).astype(np.float32) + 100  # would shift the mean if used
    yva = (Xva[:, 0] > 100).astype(np.int64)
    sva = np.repeat(np.arange(100), 10)
    tc = dict(lr=1e-2, weight_decay=1e-4, batch_size=16, max_epochs=3, patience=3)
    model, _ = train_one(Xtr, ytr, wtr, Xva, yva, sva, 8, 0, tc, "cpu")
    assert seen["n"] == 40
    assert abs(float(model.mean[0]) - Xtr[:, 0].mean()) < 1e-4


def test_ignored_rows_contribute_no_loss_and_long_responses_do_not_dominate():
    logits = torch.tensor([5.0, -5.0, 0.0, 0.0])
    labels = torch.tensor([IGNORE, 0, 1, 1])
    w = torch.tensor([1.0, 0.5, 0.25, 0.25])
    l1 = weighted_bce(logits, labels, w)
    l2 = weighted_bce(torch.tensor([-99.0, -5.0, 0.0, 0.0]), labels, w)
    assert torch.isclose(l1, l2)
    # one short response (1 row, w=1) and one long (4 rows, w=1/4 each) weigh equally
    lg = torch.zeros(5)
    lab = torch.tensor([1, 0, 0, 0, 0])
    ww = torch.tensor([1.0, 0.25, 0.25, 0.25, 0.25])
    per_short = torch.nn.functional.binary_cross_entropy_with_logits(lg[:1], lab[:1].float())
    assert torch.isclose(weighted_bce(lg, lab, ww), per_short)  # all BCE equal at logit 0
    # validation BCE is a mean over responses, not rows
    v = response_equal_bce(np.array([0.0, 10.0, 10.0, 10.0]), np.array([1, 1, 1, 1]), np.array([0, 1, 1, 1]))
    assert np.isclose(v, (np.log(2) + np.log1p(np.exp(-10))) / 2)


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    m = ScaledMLP(6, 5, np.arange(6, dtype=np.float32), np.ones(6, np.float32) * 2)
    save_checkpoint(m, tmp_path / "ck", {"feature": "k_mean", "layer_index": 23})
    m2, meta = load_checkpoint(tmp_path / "ck")
    x = torch.randn(3, 6)
    assert torch.equal(m(x), m2(x))
    assert meta["n_parameters"] == 6 * 5 + 5 + 5 + 1
    assert m2.mean.dtype == torch.float32 and m2.fc1.weight.dtype == torch.float32
    s = np.load(tmp_path / "ck" / "scaler.npz")
    assert np.array_equal(s["mean"], np.arange(6))


def test_matched_width_parameter_count_is_close_to_h_mean():
    h = ScaledMLP(2560, 128).fc1.weight.numel()
    kv = ScaledMLP(2048, 160).fc1.weight.numel()
    assert h == kv == 327680
