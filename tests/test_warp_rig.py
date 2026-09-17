"""The Warp rig must be the CPU rig. Run on Warp's CPU backend, so no GPU is needed."""
import numpy as np
import pytest

# Skipped only when warp is absent, not when a GPU is. The point of this file, per the
# docstring above, is that it runs on Warp's CPU backend so it pins something on an ordinary
# machine -- that is unchanged. What this guard buys is collection: `passive_info.warp_rig`
# does `import warp` at module level (it must; `@wp.kernel` runs at import), so without this
# the whole `pytest tests/` run died at collection on any checkout that has not installed the
# `sim` extra, and every invocation needed an `--ignore` flag.
pytest.importorskip("warp", reason="install the 'sim' extra: uv sync --extra sim")

from passive_info.embodied import simulate
from passive_info.information import n_modes
from passive_info.warp_rig import rollout


@pytest.mark.parametrize("vertical", ["fixed", True], ids=["clamped", "free"])
@pytest.mark.parametrize("ks,label", [(None, "proprioceptive"), (1e4, "sea")])
def test_batched_rollouts_match_the_cpu_trajectory(ks, label, vertical):
    """Gate 1: same inputs, same trajectory, to float32 over 1700 steps. Both benches."""
    K = n_modes()
    amps = np.zeros((3, K))
    amps[1, 0] = 5.0
    amps[2] = np.random.default_rng(0).normal(0.0, 3.0, K)

    X, ok = rollout(ks, amps, device="cpu", vertical=vertical)
    assert ok.all()
    for w in range(len(amps)):
        Xc, okc = simulate(ks, amps[w], vertical=vertical)
        assert okc
        Xc = np.asarray(Xc)
        assert X[w].shape == Xc.shape
        assert np.abs(X[w] - Xc).max() / np.abs(Xc).max() < 2e-3


@pytest.mark.parametrize("vertical", ["fixed", True], ids=["clamped", "free"])
@pytest.mark.parametrize("ks", [None, 70.0, 1e4])
def test_gate_1_at_the_resolution_the_estimator_declares(ks, vertical):
    """Gate 1 in the units that decide whether a bit count is affected."""
    from passive_info.information import SIGMA_A, SIGMA_ENERGY
    from passive_info.sim_info import _energy_half

    K = n_modes()
    a = np.random.default_rng(0).normal(0.0, SIGMA_A, size=(3, K))
    E_half = _energy_half(ks, 5.0e-5, 0.1, "menagerie", "spring", True, vertical=vertical)

    X, ok = rollout(ks, a, device="cpu", vertical=vertical)
    assert ok.all()
    for w in range(len(a)):
        Xc, okc = simulate(ks, a[w], vertical=vertical)
        assert okc
        err = np.abs((X[w] - np.asarray(Xc)) @ E_half.T).max()
        assert err < 0.05 * SIGMA_ENERGY


def test_the_models_are_identical_at_step_one():
    """The sharp version of gate 1: a dynamics difference would show immediately."""
    import mujoco
    import mujoco_warp as mjw
    import warp as wp

    from passive_info.embodied import SIM_DT, _sim_rig

    m, d, _ = _sim_rig(None, SIM_DT, ground="spring")
    m2, d2, _ = _sim_rig(None, SIM_DT, ground="spring")
    with wp.ScopedDevice("cpu"):
        wm, wd = mjw.put_model(m), mjw.put_data(m, d, nworld=1)
        mjw.step(wm, wd)
        mujoco.mj_step(m2, d2)
        qw = np.concatenate([wd.qpos.numpy()[0], wd.qvel.numpy()[0]])
    qc = np.concatenate([d2.qpos, d2.qvel])
    assert np.abs(qc - qw).max() / np.abs(qc).max() < 1e-6


def test_finite_differences_are_the_operation_float32_cannot_do():
    """Why gate 2 is tabled, pinned so it is not rediscovered as a bug."""
    K = n_modes()
    rel = {}
    for eps in (5.0, 500.0):
        amps = np.zeros((2, K))
        amps[1, 0] = eps
        Xg, _ = rollout(None, amps, device="cpu")
        X0, _ = simulate(None, np.zeros(K))
        X1, _ = simulate(None, amps[1])
        dc = (np.asarray(X1) - np.asarray(X0)) / eps
        dg = (Xg[1] - Xg[0]) / eps
        rel[eps] = float(np.abs(dc - dg).max() / np.abs(dc).max())

    assert rel[5.0] > 0.2, "the small-eps column should be dominated by float32 error"
    # a fixed absolute error over a 100x larger perturbation -> ~100x smaller relative error
    assert rel[5.0] / rel[500.0] > 20.0,\
        "error must scale as 1/eps; if it does not, this is no longer just precision"


def test_sample_rollouts_draws_from_the_prior_and_reports_losses():
    """The production path's shapes and its finite flag, which callers must not ignore."""
    from passive_info.warp_rig import sample_rollouts

    a, X, ok = sample_rollouts(None, n=4, seed=0, stride=100, device="cpu")
    assert a.shape == (4, n_modes())
    assert X.shape[0] == 4 and X.shape[2] == 6      # clamped: 3 qpos + 3 qvel
    assert ok.shape == (4,) and ok.all()

    _, Xf, okf = sample_rollouts(None, n=4, seed=0, stride=100, device="cpu", vertical=True)
    assert Xf.shape[2] == 8 and okf.all()           # free: the trunk slide is back


def test_the_observation_declares_a_resolution_and_a_metric():
    """The deterministic-map trap, pinned. Both halves of `_observe` are load-bearing."""
    import torch

    from passive_info.embodied import SIM_DT
    from passive_info.sim_info import _energy_half, _observe

    E_half = _energy_half(None, SIM_DT, 0.1, "menagerie", "spring", True, device="cpu")
    assert E_half.shape[0] == E_half.shape[1]
    assert torch.allclose(E_half, E_half.T, atol=1e-6), "E^(1/2) must be symmetric"

    X = torch.randn(6, 5, E_half.shape[0], generator=torch.Generator().manual_seed(0))
    quiet = _observe(X, E_half, 0.0, torch.Generator().manual_seed(1))
    noisy = _observe(X, E_half, 1e-3, torch.Generator().manual_seed(1))
    assert not torch.allclose(quiet, noisy), "sigma must actually perturb the observation"
    assert torch.allclose(quiet, X @ E_half.T, atol=1e-5), "sigma=0 must be the pure whitening"
    # the whitening is not the identity, or the metric is doing nothing
    assert not torch.allclose(E_half, torch.eye(E_half.shape[0]), atol=1e-6)
