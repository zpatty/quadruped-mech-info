"""Tests for the passive rebuild. Every one pins something the model must not lose."""
import numpy as np
import pytest

from passive_info.information import (FS, SIGMA_THETA, bits, bits_vs_time, profile_bits,
                                      sensitivity, time_to_resolve)
from passive_info.systems import proprioceptive, series_elastic


def test_the_mechanics_reproduce_the_published_frequencies():
    """The paper gives 79.7 Hz rigid and 13.5 Hz at k_s = 70 for this prototype."""
    assert abs(proprioceptive(clamped=False).bandwidth() - 79.7) / 79.7 < 0.01
    assert abs(series_elastic(70.0, clamped=False).bandwidth() - 13.5) / 13.5 < 0.01
    # and the clamped default, pinned so the divergence cannot drift unnoticed
    assert proprioceptive().bandwidth() == pytest.approx(79.58, abs=0.01)
    assert series_elastic(70.0).bandwidth() == pytest.approx(13.27, abs=0.01)


def test_a_rigid_series_spring_is_the_proprioceptive_design():
    """As k_s -> infinity the two models must become the same machine, in every quantity."""
    p, rigid = proprioceptive(), series_elastic(1e6)
    assert abs(p.bandwidth() - rigid.bandwidth()) / p.bandwidth() < 1e-3
    assert abs(bits(p) - bits(rigid)) < 0.02
    assert time_to_resolve(p) == pytest.approx(time_to_resolve(rigid), abs=0.26)


def test_there_is_no_actuation_anywhere():
    """The point of the rebuild: one sensor, no controller, nothing driving the mechanism."""
    for s in (proprioceptive(), series_elastic(70.0)):
        assert s.B_F.shape[1] == 1, "the only input is the unknown contact force"
        assert s.C.shape[0] == 1, "the only sensor is the motor-side encoder"
        assert s.C.sum() == 1.0 and s.C[0, s.n - 1] == 1.0, "and it reads the MOTOR"


def test_series_compliance_costs_latency_not_eventual_information():
    """The result. Over a whole contact the designs are equal; they differ in when."""
    ds = [proprioceptive()] + [series_elastic(k) for k in (70.0, 300.0, 1e3, 1e4)]
    full = [bits(s) for s in ds]
    assert max(full) - min(full) < 0.5, f"eventual bits are nearly equal: {full}"

    t = [time_to_resolve(s) for s in ds]
    assert t[1] / t[0] > 4.0, f"but a soft spring is far slower to resolve: {t}"
    # monotone in k_s: softer is always slower
    soft_to_stiff = t[1:]
    assert soft_to_stiff == sorted(soft_to_stiff, reverse=True), (
        f"time to resolve must fall monotonically with k_s: {soft_to_stiff}")


def test_the_early_sensitivity_is_where_the_designs_differ():
    """Resolution-free version of the same claim, with no sensor model in it at all."""
    ds = [proprioceptive()] + [series_elastic(k) for k in (70.0, 300.0, 1e3, 1e4)]
    peak = [np.abs(sensitivity(s)).max() for s in ds]
    early = [abs(sensitivity(s)[int(round(1e-3 * FS))]) for s in ds]
    assert max(peak) / min(peak) < 2.0, f"peak sensitivity barely differs: {peak}"
    assert max(early) / min(early) > 100.0, f"the 1 ms sensitivity hugely does: {early}"


def test_bits_rise_monotonically_with_elapsed_contact():
    """More of the contact cannot mean less information about it."""
    for s in (proprioceptive(), series_elastic(70.0), series_elastic(1e4)):
        b = [r["bits"] for r in bits_vs_time(s)]
        assert b == sorted(b), f"{s.label}: {b}"


def test_resolution_shifts_the_scale_but_not_the_ordering():
    """sigma_theta is the only instrument parameter, and it must not decide the answer."""
    ds = [proprioceptive()] + [series_elastic(k) for k in (70.0, 300.0, 1e3)]
    order = None
    for sigma in (SIGMA_THETA, SIGMA_THETA / 4, SIGMA_THETA / 16):
        t = [time_to_resolve(s, sigma_theta=sigma) for s in ds]
        assert all(np.isfinite(t)), f"sigma={sigma}: {t}"
        rank = list(np.argsort(t))
        if order is None:
            order = rank
        assert rank == order, f"ordering changed at sigma={sigma}: {t}"


# -- an unknown load PROFILE, not a step amplitude -------------------------------------

def test_a_soft_spring_destroys_modes_not_just_amplitude():
    """The result a step response cannot show: rank, not gain."""
    from passive_info.information import n_modes, profile_bits
    from passive_info.systems import proprioceptive, series_elastic

    k = n_modes()
    prop = profile_bits(proprioceptive(readout="motor"))
    soft = profile_bits(series_elastic(70.0, readout="motor"))
    assert prop["rank"] == k, f"a rigid drivetrain keeps every mode: {prop['rank']}/{k}"
    assert soft["rank"] < k / 2, f"a soft one loses most of them: {soft['rank']}/{k}"
    assert prop["bits"] / soft["bits"] > 2.0


def test_rank_rises_monotonically_with_series_stiffness():
    from passive_info.information import profile_bits
    from passive_info.systems import series_elastic

    ranks = [profile_bits(series_elastic(k, readout="motor"))["rank"]
             for k in (70.0, 150.0, 300.0, 1e3, 1e4)]
    assert ranks == sorted(ranks), f"stiffer must keep more modes: {ranks}"


def test_the_rigid_limit_matches_the_proprioceptive_design_on_the_profile_too():
    """The same correctness check as for the step, now for the vector estimand."""
    from passive_info.information import profile_bits
    from passive_info.systems import proprioceptive, series_elastic

    p = profile_bits(proprioceptive(readout="motor"))
    r = profile_bits(series_elastic(1e6, readout="motor"))
    assert p["rank"] == r["rank"]
    assert abs(p["bits"] - r["bits"]) / p["bits"] < 0.05


def test_the_state_readouts_channel_count_is_a_fact_not_an_unfairness():
    """The SEA's state readout has 4 channels against the proprioceptive design's 2."""
    from passive_info.systems import proprioceptive, series_elastic

    p, s = proprioceptive(readout="state"), series_elastic(70.0, readout="state")
    assert p.C.shape[0] == 2 and s.C.shape[0] == 4, "the SEA has one more coordinate"
    assert proprioceptive(readout="motor").C.shape[0] == 1
    assert series_elastic(70.0, readout="motor").C.shape[0] == 1


def test_the_motor_channels_interior_peak_does_not_survive_reading_the_full_state():
    """The "impedance-matched sweet spot" in mode coupling was specific to theta_m."""
    import numpy as np

    from passive_info.systems import mode_coupling, series_elastic

    ks_grid = np.logspace(2, 6, 20)
    motor = np.array([mode_coupling(series_elastic(k, readout="motor")) for k in ks_grid])
    state = np.array([mode_coupling(series_elastic(k, readout="state")) for k in ks_grid])

    j = int(np.argmax(motor))
    assert 0 < j < len(ks_grid) - 1, "the motor channel keeps its interior peak"
    assert motor[0] < motor[j] and motor[-1] < motor[j]

    assert state[0] >= state[-1] * 0.5, "the state channel does not collapse at high k_s"
    # monotone-ish: no point should exceed the softest design by more than a small margin
    assert state.max() < state[0] * 1.15, (
        f"the state channel's best-of-6-channel coupling should not show a sharp interior "
        f"peak the way the motor-only channel does: {state}")


def test_every_series_elastic_design_beats_rigid_when_the_whole_state_is_read():
    """Under state readout, ANY series compliance beats none, in mean AND worst case."""
    from passive_info.information import ensemble_summary
    from passive_info.systems import proprioceptive, series_elastic

    rigid = ensemble_summary(proprioceptive(readout="state"))
    seas = [ensemble_summary(series_elastic(k, readout="state"))
            for k in (70.0, 300.0, 1000.0, 2728.0, 1e4, 1e5, 1e6)]
    assert all(e["mean"] > rigid["mean"] for e in seas)
    assert all(e["worst"] > rigid["worst"] for e in seas)
    means = [e["mean"] for e in seas]
    assert max(means) / min(means) < 1.2, (
        f"the choice of k_s among series-elastic designs should matter far less than "
        f"having series elasticity at all: {means}")


def test_non_dimensionalising_uses_the_mechanisms_own_scales():
    """Lengths divide by the pinion radius, rates by one shared reference frequency."""
    import numpy as np
    from passive_info.systems import R, omega_ref, proprioceptive

    w = omega_ref()
    assert 400.0 < w < 600.0, f"the rigid contact mode, ~500 rad/s: {w:.1f}"

    # Clamped, every configuration coordinate is already an angle and there is nothing to
    # convert; only the rates are scaled.
    C = proprioceptive(readout="state").C
    assert np.isclose(C[0, 0], 1.0), "clamped: theta is already dimensionless"
    assert np.isclose(C[1, 1], 1.0 / w)

    # Free, coordinate 0 is the body translation in metres and r is not a free parameter.
    Cf = proprioceptive(clamped=False, readout="state").C
    assert np.isclose(Cf[0, 0], 1.0 / R), "body translation scaled by the pinion radius"
    assert np.isclose(Cf[1, 1], 1.0), "angles are already dimensionless"
    wf = omega_ref(clamped=False)
    assert np.isclose(Cf[2, 2], 1.0 / (R * wf)) and np.isclose(Cf[3, 3], 1.0 / wf)


# -- the series spring must be damped, and a residual is understood, not fixed away ----

def test_an_undamped_series_spring_rings_and_it_is_a_known_bug():
    """The exact bug this package already caused once ("resonance trap")."""
    from passive_info.information import profile_bits
    from passive_info.systems import proprioceptive, series_elastic

    rigid = profile_bits(proprioceptive(readout="motor"))["bits"]
    undamped = profile_bits(series_elastic(1e3, readout="motor", zeta_series=0.0))["bits"]
    damped = profile_bits(series_elastic(1e3, readout="motor"))["bits"]  # default zeta
    # 2.3% on the clamped bench (68.40 against 66.84); it was larger with the body free.
    assert undamped > rigid * 1.02, (
        f"undamped k_s=1e3 should visibly exceed rigid ({undamped:.1f} vs {rigid:.1f}); "
        "if this stops reproducing, the resonance may have moved off the mode grid")
    assert damped < undamped, "damping the spring must shrink the excess"


def test_series_spring_damping_scales_with_critical_ratio():
    """`ZETA_SERIES` is a fraction of critical damping at every k_s, not a fixed number."""
    import numpy as np
    from passive_info.systems import I_ROT, ZETA_SERIES, series_elastic

    for ks in (70.0, 1000.0, 1e4):
        s = series_elastic(ks)
        expected = 2.0 * ZETA_SERIES * np.sqrt(ks * I_ROT)
        # D_struct enters via the assembled A; recover b_s from the (theta_m, theta_m) term
        # of the damping block rather than re-deriving it, so the test checks the assembled
        # system rather than repeating the formula under test.
        Hinv = np.linalg.inv(s.H)
        D_eff = -(s.A[s.n:, s.n:]) 
        # D = H^-1 D_phys, so D_phys = H @ D_eff
        D_phys = s.H @ D_eff
        assert D_phys[-1, -1] == pytest.approx(expected, rel=1e-6)


def test_the_second_mode_leakage_is_real_not_a_band_hz_convenience():
    """A residual excess at intermediate k_s survives heavy damping, and that is expected."""
    from passive_info.information import profile_bits
    from passive_info.systems import series_elastic

    rigid = profile_bits(series_elastic(1e6, readout="motor"))["bits"]
    mid = profile_bits(series_elastic(1e4, readout="motor"))["bits"]
    assert mid > rigid, "the residual should still be present at the default damping"
    for zeta in (0.2, 0.5):
        heavy = profile_bits(series_elastic(1e4, readout="motor", zeta_series=zeta))["bits"]
        assert heavy > rigid, (
            f"and should survive heavier spring damping (zeta={zeta}), unlike the ringing "
            "bug above -- if it vanishes, the excess was mis-diagnosed")

    # restricting the band below BOTH second modes should remove the effect: at that point
    # neither design's second mode can leak in, and the ordering should follow bandwidth.
    low_band = dict(band_hz=15.0)
    rigid_lo = profile_bits(series_elastic(1e6, readout="motor"), **low_band)["bits"]
    mid_lo = profile_bits(series_elastic(1e4, readout="motor"), **low_band)["bits"]
    assert mid_lo <= rigid_lo * 1.02, (
        f"clear of both second modes the two should be nearly equal: {mid_lo:.2f} vs "
        f"{rigid_lo:.2f}")


def test_bandwidth_drops_the_zero_mode_even_when_it_is_not_exactly_zero():
    """A threshold-based filter can keep the zero mode instead of dropping it."""
    from passive_info.systems import series_elastic

    # The clamped bench has no zero mode at all -- K is nonsingular for both designs -- so
    # this is checked where the behaviour exists. It still matters: the Go2 rig in
    # `embodied.py` has three zero modes, which is how the multiplicity was found.
    s = series_elastic(1060.8, clamped=False)
    assert s.n_zero_modes() == 1 and series_elastic(1060.8).n_zero_modes() == 0
    assert s.modes_hz()[0] < 1e-5, "the zero mode should still be tiny"
    assert abs(s.bandwidth() - 49.12) < 0.01, (
        f"bandwidth() must be the SECOND mode, not the near-zero first one: "
        f"{s.bandwidth():.4f}")


# -- an ensemble of disturbance bandwidths, and what actually drives the k_s=1e4 result ---

def test_a_bandwidth_ensemble_still_favours_an_intermediate_stiffness_on_average():
    """Averaging over bandwidths does not erase the effect -- it makes it more precise."""
    from passive_info.information import ensemble_summary
    from passive_info.systems import proprioceptive, series_elastic

    mid = ensemble_summary(series_elastic(2728.0, readout="motor"))
    rigid = ensemble_summary(proprioceptive(readout="motor"))
    soft = ensemble_summary(series_elastic(70.0, readout="motor"))
    assert mid["mean"] > rigid["mean"] > soft["mean"]


def test_the_ensembles_worst_case_tells_a_different_story_than_its_mean():
    """The ensemble's worst case ranks designs differently from its mean."""
    from passive_info.information import BAND_ENSEMBLE_HZ, ensemble_summary
    from passive_info.systems import proprioceptive, series_elastic

    ks_vals = (70.0, 300.0, 1000.0, 2728.0, 1e4)
    designs = [series_elastic(k, readout="motor") for k in ks_vals]
    summaries = [ensemble_summary(s) for s in designs]
    worsts = [e["worst"] for e in summaries] + [
        ensemble_summary(proprioceptive(readout="motor"))["worst"]]

    assert all(worsts[i] >= worsts[i + 1] for i in range(len(worsts) - 1)), (
        f"worst-case bits should fall monotonically as k_s rises: {worsts}")
    assert worsts[0] > worsts[-1], "softest design has the highest worst-case, rigid the lowest"
    assert all(e["worst_band_hz"] == min(BAND_ENSEMBLE_HZ) for e in summaries), (
        "every design's worst band should be the lowest one in the ensemble")


def test_the_second_mode_coupling_is_non_monotonic_and_explains_the_ensemble_result():
    """Coupling to the second mode peaks at an intermediate k_s rather than merely moving."""
    import numpy as np

    from passive_info.information import profile_bits
    from passive_info.systems import mode_coupling, series_elastic

    ks_grid = np.logspace(2, 6, 40)
    coupling = np.array([mode_coupling(series_elastic(k)) for k in ks_grid])
    bits = np.array([profile_bits(series_elastic(k, readout="motor"))["bits"]
                     for k in ks_grid])

    j_couple, j_bits = int(np.argmax(coupling)), int(np.argmax(bits))
    assert 0 < j_couple < len(ks_grid) - 1, "coupling must peak in the interior, not an edge"
    assert coupling[0] < coupling[j_couple] and coupling[-1] < coupling[j_couple], (
        "coupling must fall away on BOTH sides of the peak")
    assert abs(np.log10(ks_grid[j_couple]) - np.log10(ks_grid[j_bits])) < 0.15, (
        f"the coupling peak (k_s={ks_grid[j_couple]:.0f}) and the profile-bits peak "
        f"(k_s={ks_grid[j_bits]:.0f}) should coincide")


def test_mode_coupling_is_not_the_impact_mitigation_factor():
    """A tempting but wrong link, checked and rejected rather than assumed."""
    import numpy as np

    from passive_info.systems import mode_coupling, series_elastic

    def imf(sys, n_base=1):
        Lam = 1.0 / (sys.J @ np.linalg.solve(sys.H, sys.J.T)).item()
        Hb, Jb = sys.H[:n_base, :n_base], sys.J[:, :n_base]
        Lam_L = 1.0 / (Jb @ np.linalg.solve(Hb, Jb.T)).item()
        return 1.0 - Lam / Lam_L

    ks_vals = (300.0, 1000.0, 2728.0, 1e4, 1e5, 1e6)
    imfs = [imf(series_elastic(k)) for k in ks_vals]
    couplings = [mode_coupling(series_elastic(k)) for k in ks_vals]
    assert max(imfs) - min(imfs) < 1e-6, f"IMF must stay flat: {imfs}"
    assert max(couplings) / min(couplings) > 100.0, f"coupling must not: {couplings}"


# ------------------------------------------------------------------------------------
# The energy metric, and the redundant-coordinate artifact it exists to remove.
# ------------------------------------------------------------------------------------

def test_one_ground_for_every_design():
    """`b_i` must not depend on what is bolted above the foot."""
    from passive_info.systems import reference_m_e
    # zeta_series=0 removes the spring's OWN damper, which on the clamped layout lands on
    # the same coordinate as the ground (theta_j) and would otherwise mask it. The point is
    # the ground term, not the spring's.
    p, s = proprioceptive(), series_elastic(1e12, zeta_series=0.0)
    naive = lambda sy: 1.0 / float((sy.J @ np.linalg.solve(sy.H, sy.J.T)).item())
    assert naive(p) == pytest.approx(4.0000, abs=1e-3)
    assert naive(s) == pytest.approx(3.0000, abs=1e-3)      # the defect, still visible in H
    assert reference_m_e() == pytest.approx(naive(p), rel=1e-12)
    # what matters: the assembled ground damping is identical. D = -H A[n:, n:], and the
    # ground contributes J^T b_i J, whose (0, 0) entry is b_i r^2 in this layout.
    d_p = -p.H @ p.A[p.n:, p.n:]
    d_s = -s.H @ s.A[s.n:, s.n:]
    assert d_p[0, 0] == pytest.approx(d_s[0, 0], rel=1e-9)
    assert d_p[0, 0] == pytest.approx(6.0, abs=0.05)        # b_i r^2, b_i = 600.0


def test_a_duplicated_readout_buys_bits_it_has_not_earned():
    """Why `state` was wrong: independent per-channel noise pays for redundancy."""
    import dataclasses
    prop = proprioceptive(readout="state")
    base = profile_bits(prop)
    dup = profile_bits(dataclasses.replace(prop, C=np.vstack([prop.C, prop.C])))
    gain = dup["bits"] - base["bits"]
    assert gain == pytest.approx(0.5 * base["rank"], abs=0.1)     # pure SNR arithmetic
    assert gain > 8.0


@pytest.mark.parametrize("readout", ["energy", "kinetic"])
def test_a_locked_series_spring_is_informationally_a_rigid_transmission(readout):
    """The point of the whole exercise: bits must converge as k_s -> inf."""
    p = profile_bits(proprioceptive(readout=readout))["bits"]
    assert profile_bits(series_elastic(1e8, readout=readout))["bits"] == pytest.approx(p, abs=1e-3)
    assert profile_bits(series_elastic(1e12, readout=readout))["bits"] == pytest.approx(p, abs=1e-3)
    stale = (profile_bits(series_elastic(1e12, readout="state"))["bits"]
             - profile_bits(proprioceptive(readout="state"))["bits"])
    assert stale == pytest.approx(8.46, abs=0.1)      # the artifact `energy` removes


def test_a_soft_series_spring_costs_contact_information():
    """The headline, in bits: at the prototype's published k_s the SEA knows measurably less."""
    p = profile_bits(proprioceptive(readout="energy"))["bits"]
    assert profile_bits(series_elastic(70.0, readout="energy"))["bits"] - p < -2.4
    assert profile_bits(series_elastic(1e3, readout="energy"))["bits"] - p < 0.0
    # and the shallow interior optimum survives -- shallower on the pure energy metric,
    # +0.05 where the regularised one read +0.13, but present and stable under sigma
    assert profile_bits(series_elastic(1e4, readout="energy"))["bits"] - p > 0.04


def test_the_regulariser_is_gone_and_adding_it_back_changes_nothing():
    """The `w^2 H` regulariser is gone from `E`, and adding it back changes nothing."""
    import dataclasses
    from passive_info.systems import omega_ref
    for mult in (0.0, 0.01, 1.0, 100.0):
        w = omega_ref() * mult
        def build(sys):
            n = sys.n
            E = np.zeros((2 * n, 2 * n))
            E[:n, :n], E[n:, n:] = sys.K + w ** 2 * sys.H, sys.H
            lam, V = np.linalg.eigh(E)
            return dataclasses.replace(sys, C=(V * np.sqrt(np.clip(lam, 0, None))) @ V.T)
        p = profile_bits(build(proprioceptive(readout="energy")))["bits"]
        s = profile_bits(build(series_elastic(1e12, readout="energy")))["bits"]
        assert s == pytest.approx(p, abs=1e-3)


def test_the_comparison_does_not_ride_on_the_precision_anchor():
    """Absolute bits move 191 -> 1 over four decades of sigma; the gap does not move."""
    from passive_info.information import SIGMA_ENERGY
    for mult in (0.01, 1.0, 10.0):
        kw = dict(sigma=SIGMA_ENERGY * mult)
        p = profile_bits(proprioceptive(readout="energy"), **kw)["bits"]
        assert profile_bits(series_elastic(1e12, readout="energy"), **kw)["bits"] ==\
            pytest.approx(p, abs=1e-3)
        assert profile_bits(series_elastic(1e4, readout="energy"), **kw)["bits"] - p > 0.04


def test_the_convergence_does_not_depend_on_the_ground_at_all():
    """`k_i = 0` removes the ground spring entirely, leaving the transmission alone."""
    for r in ("energy", "kinetic"):
        p = profile_bits(proprioceptive(readout=r, k_i=0.0))["bits"]
        assert profile_bits(series_elastic(1e9, readout=r, k_i=0.0))["bits"] ==\
            pytest.approx(p, abs=1e-3)


def test_the_ground_decides_whether_series_compliance_helps_or_hurts():
    """The single variable that controls the sign, on the prototype as on the Go2 leg."""
    def gap(k_i):
        kw = dict(readout="kinetic", k_i=k_i)
        return (profile_bits(series_elastic(70.0, **kw))["bits"]
                - profile_bits(proprioceptive(**kw))["bits"])

    assert gap(1e6) < -2.0          # stiff ground: compliance costs
    assert gap(0.0) > +3.0          # no ground: compliance pays


# ------------------------------------------------------------------------------------
# The prior over contact force, and where in frequency the information sits.
# ------------------------------------------------------------------------------------

def test_the_coefficients_are_bookkeeping_and_the_span_is_not():
    """`profile_bits` expands the force in DCT modes, which looks like a basis dependence."""
    from passive_info.information import (SIGMA_A, SIGMA_ENERGY, n_modes, response_matrix)
    K = n_modes()
    sys = series_elastic(70.0, readout="energy")
    M = response_matrix(sys)
    ref = profile_bits(sys)["bits"]
    for seed in (1, 2, 3):
        Q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(K, K)))
        G = (SIGMA_A / SIGMA_ENERGY) ** 2 * ((M @ Q).T @ (M @ Q))
        rotated = 0.5 * np.linalg.slogdet(np.eye(K) + G)[1] / np.log(2.0)
        assert rotated == pytest.approx(ref, abs=1e-7)


def test_the_kernel_form_reproduces_the_coefficient_form_exactly():
    """Same quantity with no coefficients at all: I(F;z) = 1/2 logdet(I + L K L^T/sigma^2)."""
    from passive_info.spectrum import k_bandlimited, kernel_bits, sample_times
    t = sample_times()
    for sys in (proprioceptive(readout="energy"), series_elastic(70.0, readout="energy")):
        assert kernel_bits(sys, k_bandlimited(t)) == pytest.approx(
            profile_bits(sys)["bits"], abs=1e-6)


def test_the_eigenchannels_account_for_every_bit():
    from passive_info.spectrum import channel_spectrum
    sys = series_elastic(1e4, readout="energy")
    assert sum(b for _, b in channel_spectrum(sys)) == pytest.approx(
        profile_bits(sys)["bits"], abs=1e-6)


def test_a_mode_outside_the_band_ADDS_nothing_though_it_may_carry_plenty():
    """The mechanism, and a correction to how it was first stated."""
    from passive_info.spectrum import marginal_modal_bits, modal_bits
    sea = series_elastic(1e4, readout="energy")
    assert sea.modes_hz()[-1] == pytest.approx(192.5, abs=2.0)

    near = lambda rows, f0: min(rows, key=lambda r: abs(r[0] - f0))[1]
    assert near(modal_bits(sea, band_hz=100.0), 192.5) > 20.0        # carries plenty alone
    assert near(marginal_modal_bits(sea, band_hz=100.0), 192.5) < 1.0    # but adds nothing
    assert near(marginal_modal_bits(sea, band_hz=200.0), 192.5) > 5.0    # until band arrives


def test_the_low_mode_of_a_soft_spring_is_nearly_redundant():
    """Why a k_s = 70 SEA does not look special at low frequency despite a 13.5 Hz mode."""
    from passive_info.spectrum import marginal_modal_bits, modal_bits
    soft = series_elastic(70.0, readout="energy")
    near = lambda rows, f0: min(rows, key=lambda r: abs(r[0] - f0))[1]
    assert near(modal_bits(soft), 13.5) > 4.0
    assert near(marginal_modal_bits(soft), 13.5) < 0.5
    assert near(marginal_modal_bits(soft), 92.2) > 40.0


def test_the_soft_spring_result_holds_only_below_the_crossover():
    """A caveat on the headline, pinned so it cannot be quoted without its condition."""
    rigid = proprioceptive(readout="energy")
    soft = series_elastic(70.0, readout="energy")
    gap = lambda b: (profile_bits(soft, band_hz=b)["bits"]
                     - profile_bits(rigid, band_hz=b)["bits"])
    assert gap(100.0) < -2.0                      # the headline
    assert gap(200.0) > 2.0                       # and its reversal
    assert abs(gap(30.0)) < 1.0                   # near nothing where running lives


def test_the_chain_rule_decomposition_is_exact():
    """A frequency decomposition that is rigorously information theoretic AND adds up."""
    from passive_info.spectrum import chain_bits
    for band in (100.0, 200.0):
        for group in (1, 3):
            for sys in (proprioceptive(readout="energy"),
                        series_elastic(70.0, readout="energy"),
                        series_elastic(1e4, readout="energy")):
                rows = chain_bits(sys, band_hz=band, group=group)
                assert sum(b for _, b in rows) == pytest.approx(
                    profile_bits(sys, band_hz=band)["bits"], abs=1e-6)


def test_the_finest_frequency_band_is_set_by_the_window_not_by_us():
    """1/2T = 5.9 Hz over an 85 ms window. Time-frequency uncertainty, not a method limit."""
    from passive_info.spectrum import chain_bits
    rows = chain_bits(series_elastic(1e4, readout="energy"), band_hz=200.0, group=1)
    assert len(rows) == 34
    assert rows[1][0] - rows[0][0] == pytest.approx(1.0 / (2 * 0.085), rel=1e-9)


def test_the_sub_resonant_window_must_be_held_fixed_across_designs():
    """The middle panel of `plate_information_spectrum`, and why it uses a fixed cutoff."""
    from passive_info.spectrum import chain_bits
    cut = proprioceptive(readout="energy").bandwidth()

    def below(ks, fixed):
        sys = series_elastic(ks, readout="energy")
        edge = cut if fixed else sys.bandwidth()
        vals = [b for f, b in chain_bits(sys, band_hz=200.0) if f < edge]
        return len(vals), sum(vals)

    # the confound, if the window is allowed to move
    n_soft, b_soft = below(70.0, fixed=False)
    n_stiff, b_stiff = below(1e4, fixed=False)
    assert b_stiff / b_soft > 4.0 and n_stiff / n_soft > 4.0
    assert (b_stiff / n_stiff) / (b_soft / n_soft) < 1.25

    # held fixed, every design gets the same bands and an interior peak survives
    counts = {below(k, fixed=True)[0] for k in (70.0, 1e3, 3e3, 1e4, 1e6)}
    assert len(counts) == 1
    rigid = below(1e6, fixed=True)[1]
    assert below(3e3, fixed=True)[1] - rigid > 1.4      # +1.44 on the pure energy metric
    assert below(70.0, fixed=True)[1] - rigid < -2.0


# The reference table below is what the sampled route must be checked against before a
# new estimator is trusted. Generated by `spectrum.chain_bits` itself, so it is a
# Regression pin on the analytic path and not a claim that anything else must match it.
CHAIN_REFERENCE = {
    ("proprioceptive", 100.0): [
        3.327992, 3.347902, 3.380116, 3.444611, 3.520029, 3.630608, 3.751805, 3.914185,
        4.087117, 4.311243, 4.542773, 4.828983, 5.083588, 5.296876, 5.304933, 5.119270,
        4.830785],
    ("series_elastic_1e4", 200.0): [
        3.296166, 3.286701, 3.331622, 3.393003, 3.477568, 3.581908, 3.713106, 3.868694,
        4.057298, 4.277508, 4.537625, 4.827570, 5.118583, 5.301722, 5.240184, 4.967458,
        4.648575, 4.363738, 4.122777, 3.924801, 3.763829, 3.637540, 3.542940, 3.479444,
        3.446531, 3.442974, 3.469328, 3.519885, 3.594332, 3.677041, 3.763457, 3.824773,
        3.854580, 3.829337],
    ("series_elastic_70", 100.0): [
        3.335400, 3.363735, 3.398362, 3.433353, 3.478829, 3.553530, 3.644138, 3.757886,
        3.885016, 4.039222, 4.205663, 4.404044, 4.606493, 4.833771, 5.020539, 5.161732,
        5.157866],
}


def _reference_system(name):
    return {"proprioceptive": lambda: proprioceptive(readout="energy"),
            "series_elastic_70": lambda: series_elastic(70.0, readout="energy"),
            "series_elastic_1e4": lambda: series_elastic(1e4, readout="energy")}[name]()


@pytest.mark.parametrize("key", sorted(CHAIN_REFERENCE))
def test_the_chain_terms_themselves_are_pinned_not_merely_their_sum(key):
    """The per-band values, not just the total. Nothing else in this file pins them."""
    from passive_info.spectrum import chain_bits

    name, band = key
    rows = chain_bits(_reference_system(name), band_hz=band)
    got = np.array([b for _, b in rows])
    want = np.array(CHAIN_REFERENCE[key])
    assert got.shape == want.shape
    assert got == pytest.approx(want, abs=1e-5)


def test_the_streaming_reuse_is_four_wherever_the_buffer_goes():
    """Reuse, not the buffer, is the calibrated quantity -- so it is what gets pinned."""
    from passive_info.sim_info import STREAM_BUFFER, STREAM_REFRESH

    assert STREAM_REFRESH * 1024 / STREAM_BUFFER == pytest.approx(4.0)


def test_both_halves_read_the_streaming_settings_from_one_place():
    """The passive runner hardcoded buffer=16384/refresh=64 instead of importing them."""
    import inspect

    from passive_info import run_controlled, run_passive

    for mod in (run_passive, run_controlled):
        text = inspect.getsource(mod)
        assert "STREAM_BUFFER" in text and "STREAM_REFRESH" in text, mod.__name__
        assert "buffer=16_384" not in text and "refresh_every=64" not in text, mod.__name__


def test_the_marginal_of_a_lone_mode_is_the_whole_total():
    """A one-coordinate mechanism has one mode, and removing it must leave nothing."""
    from passive_info.spectrum import marginal_modal_bits

    sys = proprioceptive(readout="energy")
    assert sys.n == 1
    rows = marginal_modal_bits(sys, band_hz=100.0)
    assert len(rows) == 1
    assert rows[0][1] == profile_bits(sys, band_hz=100.0)["bits"]


def test_a_readout_of_nothing_is_worth_nothing():
    """The property the fix above rests on, asserted directly rather than inferred."""
    import dataclasses

    sys = proprioceptive(readout="energy")
    blind = dataclasses.replace(sys, C=np.zeros((0, 2 * sys.n)))
    assert profile_bits(blind, band_hz=100.0)["bits"] == 0.0


def test_the_figure_grid_is_the_analytic_answer():
    """`figures._chain_grid` is what the heatmaps draw, so it is worth checking directly."""
    from passive_info import figures as F

    ks_grid = np.logspace(np.log10(70.0), 6.0, 5)
    freqs, Z = F._chain_grid(ks_grid, 200.0)
    assert Z.shape == (len(freqs), len(ks_grid))
    for j, k in enumerate(ks_grid):
        total = profile_bits(series_elastic(k, readout="energy"), band_hz=200.0)["bits"]
        assert Z[:, j].sum() == pytest.approx(total, abs=1e-9)

    modes = F._mode_curves(ks_grid)
    assert modes.shape == (len(ks_grid), 2)          # clamped SEA: two non-zero modes
    assert np.all(np.diff(modes[:, -1]) > 0), "the internal mode rises with stiffness"
