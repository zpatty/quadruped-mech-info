"""A mechanism with more than one series spring, which `Rig` could not express at all."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from passive_info.control import (F_CUT, _closed_loop_responses, closed_loop_matrix,
                                  conditional_information, conditional_spectrum, gains_for,
                                  rigid, sea, sigma_tau_for, simulate, static_gain)

FS, BAND = 4000.0, 100.0


def two_spring():
    """One `sea` rig widened to two sensors and two command channels."""
    r = sea(1e3)
    return dataclasses.replace(
        r, ks=np.array([1e3, 2e3]),
        C_tau=np.vstack([r.C_tau, r.C_tau * 0.5]),
        B_u=np.hstack([r.B_u, r.B_u * 0.5]),
        tau_map=np.array([0.1, 0.05]))


def test_a_rig_reports_its_own_capabilities_rather_than_them_being_read_off_ks():
    assert (rigid().n_springs, rigid().has_sensor, rigid().has_loop) == (0, False, False)
    assert (sea(1e3).n_springs, sea(1e3).has_sensor) == (1, True)
    assert two_spring().n_springs == 2


def test_sensor_resolution_and_gains_are_per_spring():
    """A stiffer spring is a coarser torque sensor, so the resolutions cannot be shared."""
    assert np.allclose(sigma_tau_for(two_spring()), [0.01, 0.02])
    k_d = gains_for(two_spring(), i_rot=np.array([0.01, 0.04]))["k_d"]
    assert np.allclose(k_d, 2 * 0.7 * np.sqrt(np.array([0.01, 0.04]) / np.array([1e3, 2e3])))
    assert k_d[0] != k_d[1]          # genuinely per channel, not broadcast


def test_the_augmented_state_grows_by_two_per_sensor():
    """[plant | m filtered derivatives | m previous errors], and L is (m_u, n + 2m)."""
    n_plant = two_spring().A.shape[0]
    F, L, n = closed_loop_matrix(two_spring(), FS)
    assert F.shape == (n_plant + 4, n_plant + 4) and L.shape == (2, n_plant + 4)
    assert n == n_plant
    F1, L1, _ = closed_loop_matrix(sea(1e3), FS)
    assert F1.shape == (n_plant + 2, n_plant + 2) and L1.shape == (1, n_plant + 2)


def test_tau_for_refuses_to_spread_one_scalar_over_several_motors():
    """`r * f_d` on two channels would tell both motors to do the whole job."""
    no_map = dataclasses.replace(two_spring(), tau_map=None)
    with pytest.raises(ValueError, match="tau_map"):
        no_map.tau_for(40.0)
    assert np.allclose(two_spring().tau_for(40.0), [4.0, 2.0])


def test_the_time_domain_loop_runs_with_vector_commands():
    cmd = lambda t: 40.0 * np.sin(2.0 * np.pi * 20.0 * t)
    run = simulate(two_spring(), cmd, **gains_for(two_spring(), i_rot=np.array([0.01, 0.02])))
    assert run["tau_cmd"].shape[1] == 2 and run["tau_meas"].shape[1] == 2
    assert run["f_delivered"].ndim == 1          # still one foot, so still one force
    assert np.all(np.isfinite(run["tau_cmd"]))
    # one channel stays 1-D, because figures.plate_force_control plots it against t
    assert simulate(sea(1e3), cmd, **gains_for(sea(1e3)))["tau_meas"].ndim == 1


def test_the_noise_jacobians_get_a_block_per_sensor():
    """Columns are ordered (step, channel), so each spring's resolution weights its own."""
    Ga_x, Ga_u, Gn_x, Gn_u = _closed_loop_responses(two_spring(), True, FS, BAND, F_CUT)
    n_steps = Gn_u.shape[0] // 2
    assert Gn_x.shape[1] == n_steps * 2 and Gn_u.shape == (n_steps * 2, n_steps * 2)
    assert Ga_u.shape[0] == n_steps * 2           # one command row per channel per step


def test_the_information_identities_hold_on_a_two_spring_rig():
    """I(a;x,u) >= I(a;u) exactly, and the per-band chain rule telescopes to the scalar."""
    d = conditional_information(two_spring(), fs=FS, band=BAND)
    assert d["i_joint"] >= d["i_action"] - 1e-9
    assert d["cmi"] == pytest.approx(d["i_joint"] - d["i_action"], rel=1e-12)
    _, cmi, _, _ = conditional_spectrum(two_spring(), fs=FS, band=BAND)
    assert float(np.sum(cmi)) == pytest.approx(d["cmi"], rel=1e-6)


def test_static_gain_reports_one_number_per_command_channel():
    assert static_gain(sea(1e3)) == pytest.approx(1.0 / sea(1e3).r)
    assert np.asarray(static_gain(two_spring())).shape == (2,)
