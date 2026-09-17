"""MC_W: what the mechanism's own dynamics carry forward, and why it is not our measure."""
from __future__ import annotations

import numpy as np
import pytest

from passive_info.channel import morphological_bits
from passive_info.information import profile_bits
from mechinfo.mechanics import energy_metric, zoh
from passive_info.systems import proprioceptive, series_elastic

KS = (70.0, 1e3, 1e4, 1e6)


def _designs():
    yield "rigid", proprioceptive(clamped=True, readout="energy")
    for k in KS:
        yield f"ks={k:.0e}", series_elastic(k, clamped=True, readout="energy")


def _joint_route(sys, dt):
    """MI from the joint covariance of (x_k, x_{k+1}). Shares no line with the implementation."""
    P = np.linalg.inv(energy_metric(sys.K, sys.H))
    A_d, _ = zoh(sys.A, sys.B_F, dt)
    n = P.shape[0]
    J = np.zeros((2 * n, 2 * n))
    J[:n, :n] = J[n:, n:] = P
    J[:n, n:], J[n:, :n] = P @ A_d.T, A_d @ P
    sld = lambda M: np.linalg.slogdet(M)[1]
    return float((2.0 * sld(P) - sld(J)) / (2.0 * np.log(2.0)))


@pytest.mark.parametrize("name,sys", list(_designs()))
def test_the_process_noise_route_agrees_with_the_joint_covariance_route(name, sys):
    assert morphological_bits(sys, dt=1 / 4000.0) == pytest.approx(
        _joint_route(sys, 1 / 4000.0), rel=1e-9)


def test_a_singular_energy_metric_raises_rather_than_returning_a_number():
    """The free bench has a rigid-body mode storing no energy, so E^-1 does not exist."""
    free = proprioceptive(clamped=False, readout="energy")
    assert free.n_zero_modes() >= 1
    with pytest.raises(np.linalg.LinAlgError, match="singular"):
        morphological_bits(free)


def test_it_diverges_as_the_step_shrinks_and_so_is_never_quoted_bare():
    """Consecutive states of a continuous process are perfectly correlated."""
    sys = series_elastic(1e3, clamped=True, readout="energy")
    vals = [morphological_bits(sys, dt=1.0 / fs) for fs in (1e3, 1e4, 1e5, 1e6)]
    assert all(b > a for a, b in zip(vals, vals[1:]))


def test_the_stiff_limit_control_does_NOT_return_to_the_rigid_leg():
    """MC_W carries a dimension floor, and this is the caveat on every comparison below."""
    rigid = proprioceptive(clamped=True, readout="energy")
    stiff = series_elastic(1e6, clamped=True, readout="energy")
    soft = series_elastic(70.0, clamped=True, readout="energy")

    floors, effects = [], []
    for fs in (4000.0, 20000.0):
        dt = 1.0 / fs
        floors.append(morphological_bits(stiff, dt) - morphological_bits(rigid, dt))
        effects.append(morphological_bits(soft, dt) - morphological_bits(rigid, dt))

    assert all(f > 0.0 for f in floors), floors        # never returns to the rigid leg
    assert floors[1] > floors[0], floors               # and the floor grows with a finer step
    assert effects[0] / floors[0] > 2.0 * (effects[1] / floors[1]), (effects, floors)


def test_mc_w_falls_monotonically_with_series_stiffness():
    """The robust half of the story: a softer body carries more of its own dynamics."""
    for fs in (1e3, 4e3, 2e4, 1e5):
        v = [morphological_bits(series_elastic(k, clamped=True, readout="energy"), 1.0 / fs)
             for k in KS]
        assert all(a > b for a, b in zip(v, v[1:])), (fs, v)


def test_whether_the_two_measures_disagree_DEPENDS_ON_THE_BANDWIDTH():
    """The honest finding, and it is narrower than "compliance trades computation for sense"."""
    sea = [series_elastic(k, clamped=True, readout="energy") for k in KS]
    mc = np.argsort([morphological_bits(s) for s in sea])

    def rank_corr(band):
        inf = np.argsort([profile_bits(s, band_hz=band)["bits"] for s in sea])
        return float(np.corrcoef(mc, inf)[0, 1])

    assert rank_corr(100.0) < -0.5, rank_corr(100.0)
    assert rank_corr(200.0) > -0.5, rank_corr(200.0)


def test_mc_w_is_not_available_on_the_articulated_leg():
    """Pins the gap named above: the Go2 plants have a singular energy metric."""
    from passive_info.embodied import go2_proprioceptive, go2_series_elastic
    for sys in (go2_proprioceptive(), go2_series_elastic(1e3)):
        assert sys.n_zero_modes() > 0
        with pytest.raises(np.linalg.LinAlgError, match="singular"):
            morphological_bits(sys)


# ---------------------------------------------------------------------------------------
# Under control, which is where MC_W is actually MC_W: the conditioning is not vacuous.
# ---------------------------------------------------------------------------------------

def test_an_unstable_loop_raises_instead_of_reporting_the_lyapunov_solution():
    """The silent-wrong this guards. Above ~1e5 at 4 kHz the sampled torque loop diverges."""
    from passive_info.control import closed_loop_spectral_radius, morphological_information, sea
    for k in (1e5, 1e6):
        assert closed_loop_spectral_radius(sea(k), 4000.0) > 1.0
        with pytest.raises(np.linalg.LinAlgError, match="unstable"):
            morphological_information(sea(k), fs=4000.0)


def test_the_stationary_covariance_matches_a_power_iteration_where_the_loop_is_stable():
    """Independent numerical route: iterate `S <- F S F^T + Q` from zero rather than solve."""
    import scipy.linalg as sla
    from passive_info.channel import cond_mi_bits, gibbs_pair
    from passive_info.control import closed_loop_matrix, morphological_information, sea
    from mechinfo.mechanics import zoh

    for k in (70.0, 1e3, 1e4):
        rig = sea(k)
        F, L, n = closed_loop_matrix(rig, 4000.0)
        _, Q = gibbs_pair(rig.K, rig.H, zoh(rig.A, rig.B_u, 1 / 4000.0)[0], rig.label)
        m = F.shape[0]
        Qa = np.zeros((m, m))
        Qa[:n, :n] = Q
        S_xi = np.zeros((m, m))
        for _ in range(500_000):
            nxt = F @ S_xi @ F.T + Qa
            if np.max(np.abs(nxt - S_xi)) <= 1e-14 * max(1.0, np.max(np.abs(nxt))):
                S_xi = nxt
                break
            S_xi = nxt
        S_xi = 0.5 * (S_xi + S_xi.T)
        Px = np.zeros((n, m))
        Px[:, :n] = np.eye(n)
        M = np.vstack([Px @ F, Px, L])
        load = np.zeros((2 * n + 1, m))
        load[:n, :] = Px
        S = M @ S_xi @ M.T + load @ Qa @ load.T
        got = cond_mi_bits(0.5 * (S + S.T), np.arange(n), np.arange(n, 2 * n),
                           np.arange(2 * n, 2 * n + 1))
        assert morphological_information(rig, fs=4000.0)["mc_w"] == pytest.approx(got, rel=1e-7)


def test_conditioning_on_the_command_always_removes_information():
    """The gap between the curves of `plate_morphological` is positive and design-dependent."""
    from passive_info.control import morphological_information, sea
    costs = [morphological_information(sea(k), fs=4000.0)["command_cost"]
             for k in (70.0, 1e3, 1e4)]
    assert all(c > 0.0 for c in costs), costs


def test_the_rigid_leg_runs_open_loop_and_agrees_with_the_passive_route():
    """Two code paths, one number: `gains_for` gives the rigid leg no loop to condition on."""
    from passive_info.control import morphological_information, rigid
    d = morphological_information(rigid(), fs=4000.0)
    assert d["open_loop"] and d["command_cost"] == 0.0
    assert d["mc_w"] == pytest.approx(
        morphological_bits(proprioceptive(clamped=True, readout="energy"), 1 / 4000.0),
        rel=1e-9)


def test_a_faster_loop_stays_stable_further_and_reaches_the_stiff_designs():
    """The control rate decides whether the sweep has a control on it at all."""
    from passive_info.control import F_CUT, closed_loop_spectral_radius
    from passive_info.control import morphological_information, sea
    grid = np.logspace(np.log10(70.0), 6.0, 60)
    stable = {fs: [k for k in grid if closed_loop_spectral_radius(sea(k), fs, F_CUT) <= 1.0]
              for fs in (4000.0, 20000.0)}
    assert len(stable[4000.0]) < len(grid)                      # the gate bites at 4 kHz
    assert len(stable[20000.0]) > len(stable[4000.0])           # and bites less at 20 kHz
    assert stable[4000.0][-1] < 1e5 < stable[20000.0][-1]       # only 20 kHz reaches stiff
    for k in stable[20000.0][-3:]:
        assert morphological_information(sea(k), fs=20000.0)["mc_w"] > 0.0
