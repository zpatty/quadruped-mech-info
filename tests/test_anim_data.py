"""Checks that the animation data agree with the models of the paper."""
from __future__ import annotations

import numpy as np
import pytest

from passive_info.anim import data
from passive_info.information import profile_bits


@pytest.fixture(scope="module")
def go2():
    return data.scenario("go2")


@pytest.fixture(scope="module")
def proto():
    return data.scenario("proto")


def test_body_inertias_rebuild_mujocos_mass_matrix():
    from passive_info.embodied import go2_mass_and_jacobian
    Hb, arm = data.go2_body_inertias()
    H, _, _, _ = go2_mass_and_jacobian()
    np.testing.assert_allclose(sum(Hb.values()) + np.diag(arm), H, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("kind", ["go2", "proto"])
def test_energy_partition_sums_to_the_stored_energy(kind, go2, proto):
    sc = go2 if kind == "go2" else proto
    for d in sc.designs:
        np.testing.assert_allclose(d.energy.sum(1), d.total_energy,
                                   rtol=1e-8, atol=1e-12 * d.total_energy.max())
        assert d.energy.min() > -1e-12 * d.total_energy.max()


def test_per_mode_energy_is_the_diagonal_of_the_decomposition(go2):
    # The hue weights leave out the cross terms between modes. With independent amplitudes
    # these cancel on average, so the diagonal should have about the right size.
    for d in go2.designs:
        ratio = d.hue_w.sum((1, 2))[d.n_contact // 2:] / d.total_energy[d.n_contact // 2:]
        assert 0.05 < np.median(ratio) < 20.0


@pytest.mark.parametrize("kind", ["go2", "proto"])
def test_final_bits_are_the_papers_profile_bits(kind, go2, proto):
    from passive_info.embodied import go2_proprioceptive, go2_series_elastic
    from passive_info.systems import proprioceptive, series_elastic
    sc = go2 if kind == "go2" else proto
    for d in sc.designs:
        if kind == "go2":
            sys = go2_proprioceptive() if d.ks is None else go2_series_elastic(d.ks)
        else:
            sys = (proprioceptive(readout="energy") if d.ks is None
                   else series_elastic(d.ks, readout="energy"))
        want = profile_bits(sys, band_hz=sc.band_hz)["bits"]
        assert d.bits_t[d.n_contact - 1] == pytest.approx(want, rel=1e-9)
        np.testing.assert_allclose(d.spec_t.sum(1), d.bits_t, rtol=1e-9)
        assert np.all(np.diff(d.bits_t) >= -1e-9)


def test_every_design_sees_the_same_force(go2):
    f0 = go2.designs[0].force
    for d in go2.designs[1:]:
        np.testing.assert_array_equal(d.force, f0)
    assert np.all(go2.designs[0].force[go2.designs[0].n_contact:] == 0.0)


def test_contact_rings_down_in_the_tail(go2):
    # The state does not return to zero, because the clamped leg has directions with zero
    # stiffness. The renderer moves the leg back to stance. The stored energy must decay.
    for d in go2.designs:
        peak = d.total_energy[:d.n_contact].max()
        assert d.total_energy[-1] < 0.2 * peak


def test_parts_together_give_the_total_bits():
    # The element forms sum to E, so stacking every element's readout is the energy readout.
    from passive_info.embodied import go2_proprioceptive
    from passive_info.information import FS, T_WINDOW
    d = data.go2_design(None, data.contact_amplitudes())
    sys = go2_proprioceptive()
    Xj, n_w = data._mode_responses(sys, data.BAND_HZ, 0.0)
    gamma = data._gamma(sys)
    from mechinfo.mechanics import energy_metric
    G = gamma * np.einsum("tki,ij,tlj->kl", Xj[:n_w], energy_metric(sys.K, sys.H), Xj[:n_w])
    total = 0.5 * np.linalg.slogdet(np.eye(len(G)) + G)[1] / np.log(2)
    assert d.bits_t[d.n_contact - 1] == pytest.approx(total, rel=1e-9)
    assert np.all(d.part_bits[d.n_contact - 1] <= total + 1e-9)
    assert np.all(np.diff(d.part_bits, axis=0) >= -1e-9)


def test_band_of_consistent_forces_starts_at_the_prior_and_narrows():
    from passive_info.information import mode_basis
    d = data.go2_design(None, data.contact_amplitudes())
    K = len(d.mode_freqs)
    Phi = mode_basis(d.post_sd.shape[1], K)
    prior_sd = data.SIGMA_A * np.sqrt((Phi ** 2).sum(1))
    np.testing.assert_allclose(d.post_sd[0], prior_sd, rtol=1e-9)
    assert np.all(d.post_sd[d.n_contact - 1] < 0.2 * prior_sd)
    # with this much information the centre of the band is close to the applied force
    err = np.abs(d.post_mean[d.n_contact - 1] - d.force[:d.n_contact])
    assert err.max() < 0.1 * np.abs(d.force).max()
