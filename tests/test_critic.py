"""The InfoNCE critic: that it recovers known answers, and that its two failure modes bite."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mechinfo.critic import infonce_bits
from mechinfo.estimators import gaussian_mutual_information


def _channel(n, d_y, snr, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, 1))
    M = np.zeros((d_y, 1)); M[0, 0] = snr
    return a, a @ M.T + rng.normal(size=(n, d_y)), gaussian_mutual_information(M, 1.0, 1.0)


def test_it_recovers_a_known_gaussian_channel():
    """The bound should be close and below. On GPU at d_y=34 this lands within 2-7%."""
    a, y, exact = _channel(20_000, 8, 2.0)
    r = infonce_bits(a, y, batch=256, steps=600, device="cpu", seed=0)
    assert r["bits"] < exact + 0.05, "InfoNCE is a LOWER bound; above the truth means a bug"
    assert r["bits"] > 0.5 * exact, "and it must actually find most of it"


def test_the_bound_cannot_exceed_log2_batch():
    """The ceiling is structural, and reporting a value near it means the truth is above."""
    a, y, exact = _channel(20_000, 4, 60.0)
    batch = 16
    r = infonce_bits(a, y, batch=batch, steps=1500, device="cpu", seed=0)
    assert exact > np.log2(batch), "test needs a channel richer than the ceiling"
    assert r["bits"] <= np.log2(batch) + 1e-6, "the bound is structural and must hold"
    assert r["saturation"] > 0.6, "and saturation must make the pinning visible"


def test_independent_variables_give_about_zero():
    """No relationship, no bits. A critic that finds structure here is fitting noise."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=(20_000, 1))
    y = rng.normal(size=(20_000, 6))
    r = infonce_bits(a, y, batch=256, steps=600, device="cpu", seed=0)
    assert abs(r["bits"]) < 0.15


def test_the_estimate_is_measured_on_held_out_data():
    """The guard against memorisation, checked by removing it."""
    a, y, exact = _channel(600, 6, 3.0)
    r = infonce_bits(a, y, batch=64, steps=1500, device="cpu", seed=0, holdout=0.5)
    assert r["bits"] < exact + 0.10,\
        "a tiny training set inflating the held-out estimate means the split leaked"
