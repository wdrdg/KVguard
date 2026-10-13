import torch

from kv_safety.features.cache_reader import flatten_heads
from kv_safety.features.running_mean import RunningMean


def test_running_mean_equals_explicit_prefix_mean():
    torch.manual_seed(0)
    x = torch.randn(2, 37, 8, dtype=torch.bfloat16)
    rm = RunningMean(2, 8)
    for t in range(37):
        rm.update(x[:, t:t + 1])
        ref = x[:, :t + 1].float().mean(1)
        assert torch.allclose(rm.mean(), ref, atol=1e-6)
    assert rm.sum.dtype == torch.float32


def test_tokenwise_and_chunked_agree():
    torch.manual_seed(1)
    x = torch.randn(1, 50, 16)
    a, b = RunningMean(1, 16), RunningMean(1, 16)
    tok = []
    for t in range(50):
        a.update(x[:, t:t + 1])
        tok.append(a.mean().clone())
    tok = torch.stack(tok, 1)
    ch = torch.cat([b.update_prefix_means(x[:, s:s + 13]) for s in range(0, 50, 13)], 1)
    assert torch.allclose(tok, ch, atol=1e-5)
    assert b.count == 50


def test_state_size_independent_of_T():
    rm = RunningMean(3, 32)
    before = rm.state_bytes()
    for _ in range(500):
        rm.update(torch.randn(3, 1, 32))
    assert rm.state_bytes() == before == 3 * 32 * 4


def test_reset_clears_previous_request():
    rm = RunningMean(1, 4)
    rm.update(torch.full((1, 5, 4), 100.0))
    rm.reset()
    rm.update(torch.ones(1, 2, 4))
    assert torch.allclose(rm.mean(), torch.ones(1, 4))


def test_flatten_heads_order_is_head_major():
    x = torch.arange(2 * 3 * 4).float().view(1, 2, 3, 4)  # [B, H, n, d]
    f = flatten_heads(x)
    assert f.shape == (1, 3, 8)
    # token 1 = head0 dims (x[0,0,1]) then head1 dims (x[0,1,1])
    assert torch.equal(f[0, 1], torch.cat([x[0, 0, 1], x[0, 1, 1]]))
