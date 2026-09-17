"""The passive analysis on one Go2 leg, from Menagerie's own mass matrix."""
import numpy as np
import pytest

pytest.importorskip("mujoco")

from passive_info.embodied import (go2_mass_and_jacobian, go2_proprioceptive,
                                   go2_series_elastic)
from passive_info.information import profile_bits
from passive_info.systems import proprioceptive


def test_the_rig_is_one_quarter_of_a_go2():
    """The mass matrix is Menagerie's, not ours, so check it against the documented rig."""
    H, J, arm, dofs = go2_mass_and_jacobian()
    assert H.shape == (3, 3)                       # hip, thigh, calf -- no trunk
    assert list(dofs) == [0, 1, 2]
    np.testing.assert_allclose(arm, 0.01)          # ARMATURE_GO2 on every leg joint

    Hf, Jf, armf, dofsf = go2_mass_and_jacobian(clamped=False)
    assert Hf.shape == (4, 4)                      # trunk slide + hip, thigh, calf
    assert list(dofsf) == [1, 2, 3]
    assert Hf[0, 0] == pytest.approx(3.8016, abs=1e-3)    # 1.731 trunk + 2.071 leg = m/4
    assert Jf[0] == pytest.approx(1.0, abs=1e-9)   # the slide moves the foot one-for-one
    np.testing.assert_allclose(Hf[1:, 1:], H, atol=1e-15)
    np.testing.assert_allclose(Jf[1:], J, atol=1e-15)


def test_locking_the_series_springs_conserves_inertia():
    """The armature moves off the link and onto a rotor -- it is not created or destroyed."""
    Hp, _, arm, dofs = go2_mass_and_jacobian()
    s = go2_series_elastic(1e6)
    n0 = Hp.shape[0]
    assert s.H.shape == (6, 6)          # clamped: 3 links + 3 rotors
    for i, dof in enumerate(dofs):
        assert s.H[dof, dof] == pytest.approx(Hp[dof, dof] - arm[i])
        assert s.H[n0 + i, n0 + i] == pytest.approx(arm[i])
    assert np.trace(s.H) == pytest.approx(np.trace(Hp))


def test_a_locked_series_spring_is_rigid_on_a_real_leg_too():
    """The finding that motivated all of this must not be a property of the 2-DoF cartoon."""
    p = profile_bits(go2_proprioceptive())["bits"]
    for ks in (1e9, 1e12):
        assert profile_bits(go2_series_elastic(ks))["bits"] == pytest.approx(p, abs=1e-3)
    assert profile_bits(go2_series_elastic(1e6))["bits"] - p < 0.05


def test_a_soft_series_spring_costs_a_real_leg_no_more_than_the_prototype():
    """Series compliance costs a real leg no more contact information than the prototype."""
    from passive_info.systems import proprioceptive as proto, series_elastic as proto_sea

    p = profile_bits(go2_proprioceptive())["bits"]
    go2_gap = profile_bits(go2_series_elastic(70.0))["bits"] - p
    q = profile_bits(proto(readout="energy"))["bits"]
    proto_gap = profile_bits(proto_sea(70.0, readout="energy"))["bits"] - q

    assert go2_gap == pytest.approx(-1.78, abs=0.1)
    assert proto_gap == pytest.approx(-2.44, abs=0.1)
    assert go2_gap < -1.0 and proto_gap < -1.0        # the sign is the finding
    assert go2_gap > proto_gap, "the old ordering was an artefact of two different metrics"


def test_the_intermediate_stiffness_peak_is_damping_dependent_and_is_not_a_finding():
    """The intermediate-stiffness peak moves with damping, so its magnitude is not a finding."""
    p = profile_bits(go2_proprioceptive())["bits"]
    gaps = [profile_bits(go2_series_elastic(1e3, zeta_series=z))["bits"] - p
            for z in (0.0, 0.10, 0.70)]
    assert gaps[0] > 3.0 and gaps[2] < 0.0              # collapses under damping
    assert gaps[0] - gaps[2] > 3.0                      # ...by more than the effect itself
    stiff = [profile_bits(go2_series_elastic(1e6, zeta_series=z))["bits"] - p
             for z in (0.0, 0.10, 0.70)]
    assert max(stiff) - min(stiff) < 0.02              # convergence does not care


def test_zero_modes_are_counted_from_the_rank_not_assumed_to_be_one():
    """Zero modes are counted from the rank: a Go2 leg has three, the free prototype one."""
    assert proprioceptive(clamped=False).n_zero_modes() == 1
    assert proprioceptive().n_zero_modes() == 0, "clamped: K is nonsingular"
    assert go2_proprioceptive(clamped=False).n_zero_modes() == 3
    assert go2_proprioceptive().n_zero_modes() == 2, "clamped: three coords, rank-1 K"
    assert go2_proprioceptive().bandwidth() == pytest.approx(217.4, abs=1.0)
    assert go2_proprioceptive(clamped=False).bandwidth() == pytest.approx(221.0, abs=1.0)
    assert proprioceptive(clamped=False).bandwidth() == pytest.approx(79.7, abs=0.3)


# ------------------------------------------------------------------------------------
# The FULL simulation: real contact, real nonlinear dynamics, no linearisation anywhere.
# ------------------------------------------------------------------------------------

from passive_info.embodied import SIM_DT, _sim_rig, sim_profile_bits, simulate
from passive_info.information import n_modes


def test_the_passive_leg_holds_stance_for_the_whole_window():
    """No actuation at all, so the leg settles under its own weight -- but it stays down."""
    X, ok = simulate(None, np.zeros(n_modes()))
    assert ok
    assert X.shape[1] == 6                                  # 3 qpos + 3 qvel, no trunk
    travel = np.abs(X[-1][:3] - X[0][:3])
    assert travel.max() > 1e-3, "a static nominal would mean the rig is not loaded"
    assert travel.max() < 0.05, "and it must not run away from the linearisation point"

    Xf, okf = simulate(None, np.zeros(n_modes()), vertical=True)
    assert okf
    assert Xf[-1][0] - Xf[0][0] < 0.0          # free rig: the trunk sinks
    assert abs(Xf[-1][3] - Xf[0][3]) > 0.01    # and the knee folds, measurably


def test_the_disturbance_response_is_linear_in_its_amplitude():
    """Finite differences through a nonlinear simulator are only meaningful if the response
    is actually linear at the amplitude used. Bits move 0.09 across a factor of ten."""
    b = [sim_profile_bits(None, eps=e)["bits"] for e in (2.0, 20.0)]
    assert abs(b[1] - b[0]) < 0.2
    assert all(55.0 < v < 65.0 for v in b)     # 60.5 on the clamped bench, E = diag(K, H)


def test_an_unstable_run_raises_rather_than_reporting_a_number():
    """An unstable run raises rather than reporting a number."""
    # The rotor is sub-integrated outside MuJoCo and coupled back explicitly, so above
    # about k_s = 1e5 it goes unstable at Menagerie's own 0.25 ms timestep.
    with pytest.raises(FloatingPointError):
        sim_profile_bits(1e6, dt=2.5e-4, ground="contact", vertical=True)
    assert sim_profile_bits(1e6, dt=SIM_DT, ground="contact", vertical=True)["bits"] ==\
        pytest.approx(106.2, abs=1.0)

    with pytest.raises(ValueError, match="above it"):
        _sim_rig(None, SIM_DT, ground="contact")


def test_the_sign_of_series_compliance_is_set_by_the_ground():
    """The sign of series compliance's effect is set by the ground stiffness."""
    def gap(k_i):
        p = profile_bits(go2_proprioceptive(k_i=k_i))["bits"]
        return profile_bits(go2_series_elastic(70.0, k_i=k_i))["bits"] - p

    assert gap(1e6) < -1.0          # hard ground: compliance costs (-1.78)
    assert gap(1e4) > +5.0          # soft ground: compliance pays (+10.8)
    assert gap(1e6) * gap(1e4) < 0.0                       # a genuine sign change


def test_menageries_floor_is_not_a_floor():
    """Why the simulation's soft-k_s sign is not to be trusted."""
    import mujoco
    pens = {}
    for c in ("menagerie", "stiff"):
        m, d, leg = _sim_rig(None, 1e-5, contact=c, ground="contact", vertical=True)
        pen = 0.0
        for _ in range(int(0.085 / m.opt.timestep)):
            d.xfrc_applied[:] = 0.0
            mujoco.mj_step(m, d)
            pen = max(pen, -1e3 * (d.geom_xpos[leg.foot_geom][2] - leg.foot_radius))
        pens[c] = pen
    assert pens["menagerie"] > 10.0        # over a centimetre of floor
    assert pens["stiff"] < 0.1


def test_a_locked_native_spring_reduces_toward_the_rigid_design():
    """What modelling the spring inside MuJoCo bought, and what it did not."""
    p = sim_profile_bits(None, dt=1e-5)["bits"]
    gap = sim_profile_bits(1e6, dt=1e-5)["bits"] - p
    assert abs(gap) < 0.25, "the locked limit must sit within a quarter bit of rigid"
    assert gap == pytest.approx(-0.148, abs=0.05), (
        "pins the CORRECTED ground; +0.31 was the per-design-m_e defect")


def test_the_spring_joints_carry_no_inherited_dry_friction():
    """Spring joints carry no dry friction inherited from Menagerie's default class."""
    # Dry friction does not vanish when a spring locks, so six elements against the rigid
    # rig's three would survive at any stiffness.
    import mujoco
    m, _, _ = _sim_rig(1e4, SIM_DT)
    for j in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name.endswith("_spring"):
            assert m.dof_frictionloss[m.jnt_dofadr[j]] == 0.0
            assert m.jnt_stiffness[j] == pytest.approx(1e4)
            assert m.dof_armature[m.jnt_dofadr[j]] == 0.0     # inertia is on the rotor


def test_removing_the_solver_but_keeping_k_i_makes_the_routes_agree():
    """The resolution of the whole soft-spring dispute."""
    p_sim = sim_profile_bits(None, dt=1e-5)["bits"]
    p_lin = profile_bits(go2_proprioceptive())["bits"]

    soft_sim = sim_profile_bits(70.0, dt=1e-5)["bits"]
    soft_lin = profile_bits(go2_series_elastic(70.0))["bits"]
    assert soft_sim - p_sim < -1.0                       # the simulation agrees...
    assert soft_lin - p_lin < -1.0                       # ...in sign
    assert soft_sim == pytest.approx(soft_lin, abs=1.5)  # and now in absolute value
    assert p_sim == pytest.approx(p_lin, abs=1.5)

    # and the locked spring returns to the rigid design in both
    assert abs(sim_profile_bits(1e6, dt=1e-5)["bits"] - p_sim) < 0.5
    assert abs(profile_bits(go2_series_elastic(1e6))["bits"] - p_lin) < 0.1


def test_gravity_earns_its_place_in_the_nominal():
    """Gravity stays on: without it the leg drifts in the ground spring's null space."""
    # The ground spring has rank one, so two of the clamped leg's three configuration
    # directions are unrestrained. Without gravity the locked spring does not return to
    # the rigid design: +4.44 bits against +0.31, ten times the instrument's floor.
    X, ok = simulate(None, np.zeros(n_modes()), dt=1e-5, gravity=False)
    assert ok and abs(X[-1][0] - X[0][0]) < 1e-12        # exactly static, as advertised
    p = sim_profile_bits(None, dt=1e-5, gravity=False)["bits"]
    no_g = sim_profile_bits(1e6, dt=1e-5, gravity=False)["bits"] - p

    q = sim_profile_bits(None, dt=1e-5)["bits"]
    with_g = sim_profile_bits(1e6, dt=1e-5)["bits"] - q
    assert abs(no_g) > 1.0                               # does not converge without gravity
    assert abs(with_g) < 0.5                             # does with it
    assert abs(no_g) > 10.0 * abs(with_g)


# ------------------------------------------------------------------------------------
# The two halves must weigh the mechanism with ONE ruler, and converge to ONE answer.
# Two checks on the clamped bench.
# ------------------------------------------------------------------------------------


def test_the_metric_is_the_same_object_on_both_paths():
    """Every transmission meets the same ground."""
    from passive_info.embodied import _sim_metric
    from passive_info.systems import K_I

    H, J, _, _ = go2_mass_and_jacobian()
    Z = np.zeros_like(H)
    E_lin = np.block([[K_I * np.outer(J, J), Z], [Z, H]])
    E_sim = _sim_metric(None)
    assert E_sim.shape == E_lin.shape == (6, 6)
    np.testing.assert_allclose(E_sim, E_lin, atol=1e-9, rtol=0)

    for ks in (70.0, 1e3, 1e4, 1e6):
        s = go2_series_elastic(ks)
        n = s.H.shape[0]
        E_lin = np.zeros((2 * n, 2 * n))
        E_lin[:n, :n], E_lin[n:, n:] = s.K, s.H
        T = np.zeros((n, n))
        for i in range(3):
            T[i, 2 * i] = T[i, 2 * i + 1] = 1.0       # theta_j = theta_m + theta_rel
            T[3 + i, 2 * i] = 1.0                      # theta_m
        Tb = np.block([[T, np.zeros((n, n))], [np.zeros((n, n)), T]])
        got = _sim_metric(ks)
        want = Tb.T @ E_lin @ Tb
        assert np.abs(got - want).max() / np.abs(want).max() < 1e-12


def test_the_stiffness_block_is_singular_and_stays_that_way():
    """The case the brief says to expect and not to regularise."""
    from passive_info.embodied import _sim_metric, sim_profile_bits

    # The two blocks are ranked separately, and that is not a convenience. `K` is in
    # N m/rad and runs to 2.7e4; `H` is in kg m^2 and runs to 3e-2. A single relative
    # tolerance on the stacked matrix is six orders of magnitude too coarse for the mass
    # block and reads the whole inertia of a leg as zero.
    def rank(A):
        return int(np.linalg.matrix_rank(A, tol=1e-9 * np.abs(A).max()))

    for ks, want_k, want_e, n in ((None, 1, 4, 3), (70.0, 4, 10, 6), (1e4, 4, 10, 6)):
        E = _sim_metric(ks)
        K, H = E[:n, :n], E[n:, n:]
        assert rank(K) == want_k
        assert rank(H) == n, "the velocity block is never singular -- that is the argument"
        assert rank(K) + rank(H) == want_e
        assert np.linalg.eigvalsh(K).min() > -1e-6 * np.abs(K).max()      # PSD
        assert np.linalg.eigvalsh(H).min() > 0.0

    r = sim_profile_bits(None)
    assert r["rank"] == r["n_modes"] == 17, "a singular E must not cost a channel"


def test_the_stiff_limit_converges_within_the_instruments_own_floor():
    """A k_s = 1e6 series spring is the rigid design."""
    from passive_info.embodied import sim_profile_bits

    lin_p = profile_bits(go2_proprioceptive())["bits"]
    lin_s = profile_bits(go2_series_elastic(1e6))["bits"]
    assert abs(lin_s - lin_p) < 0.011               # the analytical model: 0.010 bits

    sim_p = sim_profile_bits(None, dt=1e-5)["bits"]
    sim_s = sim_profile_bits(1e6, dt=1e-5)["bits"]
    assert abs(sim_s - sim_p) < 0.4                 # the simulation: 0.31, at its floor
    assert abs(sim_s - sim_p) > 0.1,\
        "if this ever falls to zero the floor was real and something changed -- measure it"


def test_the_comparison_floor_does_not_rise_with_the_band():
    """RETRACTION. The floor rising with the band was the ground defect, not the band."""
    from passive_info.sim_info import sim_chain_reference

    lo_p = sim_chain_reference(None, band_hz=100.0)
    lo_s = sim_chain_reference(1e6, band_hz=100.0)
    hi_p = sim_chain_reference(None, band_hz=200.0)
    hi_s = sim_chain_reference(1e6, band_hz=200.0)

    lo = (lo_s.sum() - lo_p.sum()) / lo_p.sum()
    hi = (hi_s.sum() - hi_p.sum()) / hi_p.sum()
    assert lo == pytest.approx(-0.0033, abs=0.002)
    assert hi == pytest.approx(-0.0016, abs=0.002)
    # the whole point: the 200 Hz floor is no longer the larger one, and both are sub-percent
    assert abs(hi) < 0.01 and abs(lo) < 0.01
    assert abs(hi) < 3.0 * abs(lo), "the band dependence was the ground, and it is gone"

    f = np.arange(len(hi_p)) / (2 * 0.085)
    d = hi_s - hi_p
    # it used to be +0.14 below and +4.11 above; the fast bands no longer dominate
    assert abs(d[f >= 100].sum()) < 0.5, "the floor no longer concentrates above 100 Hz"
    assert abs(d.sum()) < 0.5

def test_every_transmission_meets_the_same_ground():
    """`_ground_gains` must not hand the series-elastic designs a softer floor."""
    from passive_info.embodied import _ground_gains, _sim_rig, SIM_DT
    from passive_info.systems import K_I

    def b_i(ks):
        m, d, leg = _sim_rig(ks, SIM_DT, 0.05, "menagerie", "spring", False, vertical="fixed")
        return _ground_gains(m, d, leg, K_I)[0]

    rigid = b_i(None)
    # the rigid leg was always correct, so the fix must leave it exactly where it was
    assert rigid == pytest.approx(219.61, abs=0.05)
    for ks in (70.0, 1e3, 1e4, 1e6, 1e12):
        assert b_i(ks) == pytest.approx(rigid, rel=1e-9), (
            f"k_s={ks:g} meets a different ground than the rigid leg")
