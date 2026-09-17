"""The nonparametric estimator, checked against answers we know exactly."""
import numpy as np
import pytest

from mechinfo.estimators import gaussian_mutual_information, ksg_mutual_information


def test_it_recovers_a_gaussian_channel_it_was_not_told_about():
    rng = np.random.default_rng(0)
    a = rng.normal(size=200_000)
    z = 10.0 * a + rng.normal(size=200_000)
    truth = 0.5 * np.log2(1 + 100.0)
    assert ksg_mutual_information(a, z) == pytest.approx(truth, rel=0.05)


def test_normalising_is_not_optional():
    """The joint search uses a Chebyshev metric, so without it the largest-scale variable
    sets every distance. Skipping this once reported -91% error where the truth was -21%."""
    rng = np.random.default_rng(1)
    a = rng.normal(size=(40_000, 4))
    z = 10.0 * a + rng.normal(size=(40_000, 4))
    truth = 4 * 0.5 * np.log2(101.0)
    good = ksg_mutual_information(a, z, normalise=True)
    bad = ksg_mutual_information(a, z, normalise=False)
    assert good > 0.7 * truth
    assert bad < 0.3 * truth


def test_a_deterministic_map_gives_a_number_that_is_only_the_sample_count():
    """The estimator does NOT rescue us from declaring a state resolution. On y = 3x the
    true information is infinite; KSG reports log2(N), which looks like an answer."""
    rng = np.random.default_rng(2)
    got = []
    for n in (4_000, 64_000):
        a = rng.normal(size=n)
        got.append(ksg_mutual_information(a, 3.0 * a))
    assert got[1] - got[0] == pytest.approx(np.log2(64_000 / 4_000), abs=0.6)


def test_differences_between_designs_survive_better_than_absolute_values():
    """What makes it usable for comparing mechanisms: the bias is largely common-mode."""
    rng = np.random.default_rng(3)
    N, d = 200_000, 3
    a = rng.normal(size=(N, d))
    out = []
    for snr in (100.0, 25.0):
        M = np.sqrt(snr) * np.eye(d)
        z = a @ M.T + rng.normal(size=(N, d))
        out.append((gaussian_mutual_information(M, 1.0, 1.0),
                    ksg_mutual_information(a, z)))
    (t_hi, e_hi), (t_lo, e_lo) = out
    assert e_hi < t_hi and e_lo < t_lo                       # biased low, as documented
    assert (e_hi - e_lo) == pytest.approx(t_hi - t_lo, rel=0.25)   # the gap survives
