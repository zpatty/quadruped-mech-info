"""Commanding a force through each transmission."""
import numpy as np
import pytest

from passive_info.control import (gains_for, rigid, sea, sigma_tau_for, simulate,
                                  static_gain, tracking_error)
from passive_info.systems import I_ROT, R


def _cmd(t, rise=0.005, t0=0.005, amp=100.0):
    """A 100 N command with a 5 ms raised-cosine rise. A step is not commandable: its
    infinite slope drives the derivative term to 269 N m to deliver 10 N m of steady
    torque, which no actuator would survive."""
    if t < t0:
        return 0.0
    u = min((t - t0) / rise, 1.0)
    return amp * (0.5 - 0.5 * np.cos(np.pi * u))


def test_every_transmission_has_the_same_static_force_gain():
    """Wensing eq. (44), tau_d = r f_d, holds for all of them -- the transmissions differ in
    how the force arrives, not in how much arrives eventually."""
    for rg in (rigid(), sea(70.0), sea(1e3), sea(1e6)):
        assert static_gain(rg) == pytest.approx(1.0 / R, rel=1e-9)
        assert rg.tau_for(100.0) == pytest.approx(R * 100.0)


def test_a_soft_series_spring_cannot_be_commanded_open_loop():
    """The premise of closing the loop at all. The rigid leg only rings at its own mode."""
    assert tracking_error(simulate(rigid(), _cmd))["rms_frac"] < 0.03
    assert tracking_error(simulate(sea(70.0), _cmd))["rms_frac"] > 0.4
    assert tracking_error(simulate(sea(70.0), _cmd, **gains_for(sea(70.0))))["rms_frac"] < 0.03


def test_torque_loop_gains_must_scale_with_the_spring():
    """A fixed gain set does not survive the sweep. k_d = 0.02 rescues a k_s = 70 leg and makes a
    k_s = 1e4 leg diverge, because the spring's mode moves from 13 Hz to 190 Hz.
    """
    soft = tracking_error(simulate(sea(70.0), _cmd, k_p=1.0, k_d=0.02))["rms_frac"]
    stiff = tracking_error(simulate(sea(1e4), _cmd, k_p=1.0, k_d=0.02))["rms_frac"]
    assert soft < 0.03
    assert not np.isfinite(stiff) or stiff > 10.0

    g = gains_for(sea(1e4))
    assert g["k_d"] == pytest.approx(2 * 0.7 * np.sqrt(I_ROT / 1e4))
    assert tracking_error(simulate(sea(1e4), _cmd, **g))["rms_frac"] < 0.1


def test_a_stiffer_spring_is_a_coarser_torque_sensor():
    """The SEA's sensor IS its spring, read to a fixed angular resolution, so torque
    resolution degrades in proportion to stiffness."""
    assert sigma_tau_for(rigid()) == 0.0
    assert sigma_tau_for(sea(70.0)) == pytest.approx(7e-4)
    assert sigma_tau_for(sea(1e4)) == pytest.approx(0.1)


def test_the_high_stiffness_limit_is_bandwidth_and_not_sensing():
    """A retraction, pinned so it cannot come back."""
    err = lambda ks, closed, st=None: tracking_error(simulate(
        sea(ks), _cmd, sigma_tau=st, **(gains_for(sea(ks)) if closed else {})))["rms_frac"]

    assert err(70.0, True) < 0.1 * err(70.0, False)        # soft: the loop rescues it
    assert err(1e4, True) < err(1e4, False)                # and it still helps at 1e4

    perfect = err(1e5, True, st=0.0)                       # noiseless, and still hopeless
    assert not np.isfinite(perfect) or perfect > 10 * err(1e5, False)


def test_the_derivative_filter_is_part_of_the_limit_and_not_only_the_sample_rate():
    """A second retraction. The high-stiffness limit was first credited entirely to the sampling
    rate, on evidence that had varied the rate and the derivative filter TOGETHER.
    """
    def ok(ks, fs, fc):
        e = tracking_error(simulate(sea(ks), _cmd, fs=fs, f_cut=fc,
                                    **gains_for(sea(ks))))["rms_frac"]
        return np.isfinite(e) and e < 1.0

    assert not ok(1e5, 4000.0, 4000.0)      # k_s=1e5: fails at 4 kHz whatever the filter
    assert ok(1e5, 8000.0, 400.0)           #          works at 8 kHz whatever the filter
    assert not ok(1e6, 20000.0, 1000.0)     # k_s=1e6: 20 kHz is not sufficient on its own
    assert ok(1e6, 20000.0, 2000.0)         #          the cut-off must clear the mode


def test_conditioning_on_an_action_that_carries_nothing_changes_nothing():
    """The structural check that the cmi computes what it claims."""
    from passive_info.control import conditional_information

    r = conditional_information(rigid())
    assert r["i_action"] == pytest.approx(0.0, abs=1e-9)
    assert r["cmi"] == pytest.approx(r["i_state"], abs=1e-9)

    for ks in (70.0, 1e4):
        o = conditional_information(sea(ks), closed=False)
        assert o["i_action"] == pytest.approx(0.0, abs=1e-9)
        assert o["cmi"] == pytest.approx(o["i_state"], abs=1e-9)


def test_the_action_absorbs_what_the_spring_already_sensed():
    """The result: where a mechanism's knowledge of the world actually sits."""
    from passive_info.control import conditional_information

    rig = conditional_information(rigid())
    soft = conditional_information(sea(70.0))
    stiff = conditional_information(sea(1e4))

    assert rig["cmi"] > 70.0                       # all of it is in the state
    assert soft["cmi"] < 5.0                       # almost none of it is
    assert soft["cmi"] < stiff["cmi"]              # a better sensor absorbs more
    assert soft["i_joint"] > rig["i_joint"]        # and the SEA system knows more overall


def test_the_torque_loop_is_what_limits_stiffness_not_the_spring():
    """Past a stiffness the sampled loop diverges, and raising the rate moves the boundary."""
    from passive_info.control import closed_loop_spectral_radius, sea

    for ks in (70.0, 1e3, 1e4, 3e4):
        assert closed_loop_spectral_radius(sea(ks), fs=4000.0) < 1.0
    assert closed_loop_spectral_radius(sea(1e5), fs=4000.0) > 1.0
    # The limit belongs to the controller: the same spring is fine four times faster.
    assert closed_loop_spectral_radius(sea(1e5), fs=16000.0) < 1.0


def test_what_the_robot_is_asked_to_do_cannot_change_what_it_can_learn():
    """The information is invariant to the commanded force -- exactly, not approximately."""
    import numpy as np

    from mechinfo.mechanics import zoh
    from passive_info.control import (F_CUT, _disturbed_plant, gains_for, sea,
                                      sigma_tau_for)
    from passive_info.information import SIGMA_A, SIGMA_ENERGY, T_WINDOW, mode_basis, n_modes

    def bits(rig, f_ref, fs=500.0, band=30.0, amp=100.0):
        dt, n_t = 1.0 / fs, int(round(T_WINDOW * fs)) + 1
        Phi = mode_basis(n_t, n_modes(T_WINDOW, band))
        B_f, C = _disturbed_plant(rig)
        Ad, Bd_f = zoh(rig.A, B_f, dt)
        _, Bd_u = zoh(rig.A, rig.B_u, dt)
        g = gains_for(rig)
        alpha = dt / (dt + 1.0 / (2.0 * np.pi * F_CUT))
        tau_d = rig.tau_for(amp * np.sin(2.0 * np.pi * f_ref * np.arange(n_t) * dt))

        def run(f_seq, n_seq):
            x = np.zeros((rig.A.shape[0], 1))
            deriv, prev, xs, us = 0.0, 0.0, [], []
            for k in range(n_t - 1):
                err = tau_d[k] - (float((rig.C_tau @ x).item()) + n_seq[k])
                deriv += alpha * ((err - prev) / dt - deriv)
                prev = err
                u = tau_d[k] + g["k_p"] * err + g["k_d"] * deriv
                us.append(u)
                x = Ad @ x + Bd_f * f_seq[k] + Bd_u * u
                xs.append((C @ x).ravel().copy())
            return np.concatenate(xs), np.asarray(us)

        z = np.zeros(n_t)
        b_x, b_u = run(z, z)                            # the reference acting alone
        ax, au = zip(*(run(Phi[:, j], z) for j in range(Phi.shape[1])))
        nx, nu = zip(*(run(z, np.eye(n_t)[k]) for k in range(n_t - 1)))
        Ga_x = np.column_stack([c - b_x for c in ax])
        Ga_u = np.column_stack([c - b_u for c in au])
        Gn_x = np.column_stack([c - b_x for c in nx])
        Gn_u = np.column_stack([c - b_u for c in nu])

        sg = sigma_tau_for(rig)
        Sx, Su = SIGMA_A ** 2 * (Ga_x @ Ga_x.T), SIGMA_A ** 2 * (Ga_u @ Ga_u.T)
        Sxu = SIGMA_A ** 2 * (Ga_x @ Ga_u.T)
        Nx = sg ** 2 * (Gn_x @ Gn_x.T) + SIGMA_ENERGY ** 2 * np.eye(Ga_x.shape[0])
        Nu = sg ** 2 * (Gn_u @ Gn_u.T) + 1e-18 * np.eye(Ga_u.shape[0])
        Nxu = sg ** 2 * (Gn_x @ Gn_u.T)
        ld = lambda M: np.linalg.slogdet(M)[1]
        top = np.block([[Sx + Nx, Sxu + Nxu], [(Sxu + Nxu).T, Su + Nu]])
        bot = np.block([[Nx, Nxu], [Nxu.T, Nu]])
        return (0.5 * (ld(Sx + Nx) - ld(Nx)) / np.log(2.0),
                0.5 * (ld(top) - ld(bot) - ld(Su + Nu) + ld(Nu)) / np.log(2.0))

    rig = sea(70.0)
    quiet, slow, fast = bits(rig, 0.0), bits(rig, 5.0), bits(rig, 150.0)
    for other in (slow, fast):
        assert other[0] == pytest.approx(quiet[0], abs=1e-9)
        assert other[1] == pytest.approx(quiet[1], abs=1e-9)


def test_a_faster_controller_does_not_grant_a_finer_observer():
    """The control rate must not leak into what the observer sees."""
    from passive_info.control import conditional_information, rigid

    ref = [conditional_information(rigid(), fs=f, f_cut=2000.0)["i_joint"]
           for f in (4000.0, 8000.0, 20000.0)]
    assert max(ref) - min(ref) < 1e-9, ref

    # The series elastic leg is NOT invariant, and should not be: a loop that reads its
    # spring five times as often at the same per-sample resolution really does learn more.
    # This is a caveat to record, not a defect to suppress.
    sea_cmi = [conditional_information(sea(70.0), fs=f, f_cut=2000.0)["cmi"]
               for f in (4000.0, 20000.0)]
    assert sea_cmi[0] > 2.0 * sea_cmi[1]


def test_the_toeplitz_noise_jacobian_equals_one_simulation_per_noise_sample():
    """The fast build must agree with the literal one, to machine precision."""
    import numpy as np

    from mechinfo.mechanics import zoh
    from passive_info.control import (F_CUT, _closed_loop_responses, _disturbed_plant,
                                      gains_for, sea)
    from passive_info.information import T_WINDOW, mode_basis, n_modes

    rig, fs, band = sea(1e3), 1000.0, 30.0
    _, _, Gn_x, Gn_u = _closed_loop_responses(rig, True, fs, band, F_CUT, fs_obs=fs)

    dt, n_t = 1.0 / fs, int(round(T_WINDOW * fs)) + 1
    B_f, C = _disturbed_plant(rig)
    Ad, _ = zoh(rig.A, B_f, dt)
    _, Bd_u = zoh(rig.A, rig.B_u, dt)
    g = gains_for(rig)
    alpha = dt / (dt + 1.0 / (2.0 * np.pi * F_CUT))

    def literal(k_impulse):
        x = np.zeros((rig.A.shape[0], 1))
        deriv, prev, xs, us = 0.0, 0.0, [], []
        for k in range(n_t - 1):
            err = -(float((rig.C_tau @ x).item()) + (1.0 if k == k_impulse else 0.0))
            deriv += alpha * ((err - prev) / dt - deriv)
            prev = err
            u = g["k_p"] * err + g["k_d"] * deriv
            us.append(u)
            x = Ad @ x + Bd_u * u
            xs.append((C @ x).ravel().copy())
        return np.concatenate(xs), np.asarray(us)

    for k in (0, 7, n_t - 2):
        want_x, want_u = literal(k)
        assert np.allclose(Gn_x[:, k], want_x, atol=1e-12), k
        assert np.allclose(Gn_u[:, k], want_u, atol=1e-12), k
