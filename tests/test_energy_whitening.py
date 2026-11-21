"""The numerical consequences of every result in docs/whitening/energy_whitening.tex.

Each test names the result it pins. The checks themselves live in
docs/whitening/checks.py so that the note, its figures and these tests share one source.
"""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "docs" / "whitening"))
import checks  # noqa: E402


def test_energy_identity():
    """Lemma (energy identity): |E^{1/2} x_q|^2 = q'Kq + qdot'H qdot."""
    assert checks.check_energy_identity()["max_rel_err"] < 1e-12


def test_every_square_root_and_the_bare_gram_agree():
    """Lemma (every square root gives the same bits)."""
    for v in checks.check_square_roots().values():
        assert v["chol"] == pytest.approx(v["sym"], abs=1e-9)
        assert v["gram"] == pytest.approx(v["sym"], abs=1e-9)


def test_whitened_noise_is_physical_noise_with_covariance_E_inverse():
    """Lemma (only the noise model matters), part (c)."""
    r = checks.check_noise_equivalence()
    assert r["physical_noise"] == pytest.approx(r["whitened"], abs=1e-9)


def test_energy_bits_are_coordinate_invariant_and_per_coordinate_bits_are_not():
    """Theorem (coordinate invariance) and its counterexample."""
    r = checks.check_invariance()
    assert r["energy_spread"] < 1e-9
    assert r["state_spread"] > 100.0


def test_degrees_relabelling_matches_the_exact_snr_formula():
    """Example (radians versus degrees)."""
    r = checks.check_degrees_prediction()
    assert r["observed"] == pytest.approx(r["exact"], abs=1e-6)
    assert r["observed"] == pytest.approx(r["high_snr"], abs=0.1)


def test_gibbs_noise_maximizes_entropy_at_fixed_mean_energy():
    """Theorem (maximum entropy), sampled over Gaussian competitors."""
    r = checks.check_maxent(trials=500)
    assert r["min_gap"] > 0.0
    assert r["effective_kT"] / r["room_kT"] > 1e15


def test_E_inverse_is_the_fdt_stationary_covariance():
    """Proposition (fluctuation-dissipation balance) and gibbs_pair's Q >= 0."""
    for v in checks.check_fdt().values():
        assert v["lyap_resid"] < 1e-12
        assert v["Q_min_eig"] > 0.0


def test_whitened_dynamics_are_port_hamiltonian_with_modal_singular_values():
    """Theorem (port-Hamiltonian form): sym part = -R <= 0, svd(Omega) = omega."""
    for row in checks.check_port_hamiltonian()["rows"]:
        assert row["sym_err"] < 1e-12
        assert row["max_eig_sym"] < 1e-10
        np.testing.assert_allclose(row["sv"], row["omega"], rtol=1e-9)


def test_duplicate_readout_gain_is_the_exact_formula():
    """Proposition (a free second look)."""
    r = checks.check_duplicate()
    assert r["gain"] == pytest.approx(r["predicted"], abs=1e-9)
    assert r["gain"] == pytest.approx(r["half_rank"], abs=0.1)


def test_energy_form_restricts_to_the_rigid_leg_on_the_locked_subspace():
    """Lemma (the energy form restricts to the locked mechanism)."""
    r = checks.check_locked_restriction()
    assert r["K_V"] == pytest.approx(r["K_rigid"], rel=1e-12)
    assert r["H_V"] == pytest.approx(r["H_rigid"], rel=1e-12)
    assert r["w_K_w"] == pytest.approx(r["ks"], rel=1e-12)


def test_stiff_limit_rates_and_the_size_of_the_artifact():
    """Theorem (constraint limit), Proposition (rates for the SEA)."""
    r = checks.check_stiff_limit(np.logspace(2, 12, 11))
    assert abs(r["gap_energy"][-1]) < 1e-5
    dup = checks.check_duplicate()["gain"]
    assert r["gap_state"][-1] == pytest.approx(dup, abs=1e-3)
    assert r["slope_spring"] == pytest.approx(-0.5, abs=0.02)
    assert r["slope_rotor"] <= -0.5 + 0.02            # the proven rate is a bound


def test_snr_asymptotics_and_sigma_independent_gap():
    """Proposition (low- and high-SNR asymptotics)."""
    r = checks.check_snr(np.array([1e-3, 1e-2, 1e-1, 1.0, 1e3]))
    np.testing.assert_allclose(r["gap"][:4], r["det_gap"], atol=2e-3)
    for v in r["sys"].values():
        assert np.all(v["exact"] <= v["low"] + 1e-9)            # (a) is an upper bound
        assert v["exact"][-1] == pytest.approx(v["low"][-1], rel=0.01)
        assert v["exact"][0] == pytest.approx(v["high"][0], abs=1e-3)


def test_repository_sampling_convention_is_fs_consistent():
    """Remark (sampling rate)."""
    for b in checks.check_sampling().values():
        assert abs(b[(8000.0, "repo convention")] - b[(4000.0, "repo convention")]) < 0.2
        assert b[(8000.0, "fixed amplitude")] - b[(4000.0, "fixed amplitude")] > 8.0


def test_subset_game_consistency_and_the_two_kinds_of_redundancy():
    """Proposition (properties of the subset game) on the stiff SEA."""
    r = checks.check_subsets()
    for v in r["totals"].values():
        assert v["all_coords"] == pytest.approx(v["whitened"], abs=1e-8)
    f = r["fusion"]
    assert f["qboth"] == pytest.approx(f["q_rigid"], abs=1e-3)          # exact duplicates
    assert f["vboth"] == pytest.approx(f["v_rigid"], abs=1e-3)          # precision fusion
    phi = r["shapley"][1e8]
    assert phi[0] == pytest.approx(phi[1], abs=0.01)                    # equal positions
    assert phi[2] - phi[3] > 4.0                                        # unequal velocities
    for ks, p in r["shapley"].items():
        assert np.all(p >= -1e-9)                                       # monotone game
    assert r["position_noise_corr"][1e8] > 0.9999


def test_mcw_closed_form():
    """Proposition (closed form for MC_W)."""
    for v in checks.check_mcw_closed_form().values():
        assert v["closed"] == pytest.approx(v["code"], abs=1e-6)
        assert v["sv_max"] < 1.0


def test_mesh_independence_of_energy_bits():
    """Theorem (mesh independence) on the finite-element rod."""
    r = checks.check_rod(Ns=(8, 16, 32, 64))
    assert abs(r["energy"][-1] - r["energy"][-2]) < 0.05
    assert r["per_node_bits_per_doubling"] == pytest.approx(r["predicted_per_doubling"],
                                                            rel=0.05)


def test_floating_base_loses_nothing_when_force_and_ground_share_J():
    """Proposition (no loss when force and ground share the Jacobian) and counterexample."""
    r = checks.check_floating()
    assert abs(r["J_phi0"]) < 1e-12
    assert r["foot_energy"] == pytest.approx(r["foot_kinetic"], abs=1e-9)
    assert r["trunk_kinetic"] - r["trunk_energy"] > 10.0


# ---------------------------------------------------------------------------------------
# Part II: is energy the right choice?
# ---------------------------------------------------------------------------------------

def test_uniqueness_step_four_identity():
    """Theorem (uniqueness), step 4: d/dt(a q'Kq + b qd'Hqd) = 2(a-b) qd'Kq."""
    assert checks.check_uniqueness_flow()["max_rel_err"] < 1e-10


def test_only_energy_and_modal_weightings_are_flow_invariant():
    """(U5): energy and W_p are preserved by free motion; per-coordinate and kinetic are not."""
    r = checks.check_flow_invariance()
    for name in ("energy", "action (p=-1)", "p=+1"):
        assert r[name] < 1e-10
    assert r["per-coordinate"] > 1.0 and r["kinetic"] > 0.5
    assert r["det_Phi"] == pytest.approx(1.0, abs=1e-10)        # Liouville


def test_modal_weightings_are_not_local():
    """(U2): the ground stiffness moves the action metric's motor-velocity entry."""
    r = checks.check_locality()
    e = [v["energy"] for v in r.values()]
    a = [v["action"] for v in r.values()]
    assert max(e) - min(e) < 1e-12
    assert max(a) / min(a) > 2.0


def test_free_motion_creates_information_only_without_energy_noise():
    """Proposition (information is created only by work), all three parts."""
    r = checks.check_free_motion()
    und, damp = r["undamped"], r["damped"]
    for name in ("energy", "action"):
        assert np.ptp(und[name]) < 1e-9                                    # (i)
    assert np.all(np.diff(damp["energy"]) <= 1e-12)                        # (ii)
    assert und["per-coordinate"].max() > 3.0                               # (iii)
    assert und["kinetic"].max() > 3.0
    assert damp["per-coordinate"].max() > 2.5                              # even damped


def test_frequency_weighted_stiff_limit_threshold_is_p_equals_two():
    """The weighted fast-mode contribution scales like k_s^(p/2 - 1)."""
    r = checks.check_alternatives_sea()
    g = r["gaps"]
    for p in ("p=-2", "p=-1", "p=0"):
        assert abs(g[p][-1]) < 1e-5
    assert g["p=1"][-2] / g["p=1"][-1] == pytest.approx(10.0, rel=0.05)    # k_s^-1/2
    assert g["p=2"][-1] == pytest.approx(g["p=2"][-2], abs=0.01) and g["p=2"][-1] > 1.0
    assert g["p=3"][-1] > 50.0
    assert g["per-coordinate"][-1] == pytest.approx(8.4716, abs=1e-3)
    for name, spread in r["spread"].items():
        if name != "per-coordinate":
            assert spread < 1e-9
    assert r["spread"]["per-coordinate"] > 100.0


def test_frequency_weighted_mesh_behaviour():
    """Rod: p <= 0 and kinetic converge; p = +1 keeps growing."""
    b = checks.check_alternatives_rod()["bits"]
    for name in ("p=-1", "p=0", "kinetic"):
        assert abs(b[name][-1] - b[name][-2]) < 0.05
    assert b["p=1"][-1] - b["p=1"][-2] > 2.0


def test_data_driven_whitening_erases_the_body():
    """Proposition (data-driven whitening erases the body)."""
    r = checks.check_data_driven_whitening()
    for name in ("rigid", "SEA 70", "SEA 1e4"):
        assert r[name] == pytest.approx(r["predicted"], abs=1e-8)


def test_langevin_innovations_formula_matches_the_stacked_covariance():
    """Proposition (full observation: an exact formula)."""
    for v in checks.check_langevin_innovations().values():
        assert v["innovations"] == pytest.approx(v["stacked"], abs=1e-8)


def test_langevin_full_state_information_depends_only_on_contact_damping():
    """Proposition (only the dissipation at the contact matters)."""
    r = checks.check_langevin_dt_limit(fs_list=(4e3, 16e3, 64e3))
    JDJ = [v["JDJ"] for v in r["rows"].values()]
    assert max(JDJ) - min(JDJ) < 1e-12                                      # = 1/b_i
    for v in r["rows"].values():
        np.testing.assert_allclose(v["ratio"], 0.5, atol=1e-4)


def test_langevin_routing_reproduces_the_proprioceptive_argument():
    """Langevin body: equal ceiling, rigid at the ceiling, SEA climbing with stiffness."""
    r = checks.check_langevin_designs(ks_list=(70.0, 1e3, 1e4, 1e6))
    rows = r["rows"]
    full = [v["full"] for v in rows.values()]
    assert max(full) - min(full) < 1e-4
    assert rows["rigid"]["motor"] / rows["rigid"]["full"] > 0.998
    motor = [rows[k]["motor"] for k in r["ks"]]
    assert all(np.diff(motor) > 0)
    for k in r["ks"]:
        assert rows[k]["motor"] <= rows["rigid"]["full"] + 1e-9                 # the ceiling
    # the readout-only model lets a resonant SEA beat the rigid leg through the motor
    assert rows[1e4]["readout_motor"] > rows["rigid"]["readout_motor"]
