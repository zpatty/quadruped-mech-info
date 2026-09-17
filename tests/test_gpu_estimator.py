"""The GPU estimator against the CPU one: how closely they agree, and where they do not."""
import numpy as np
import pytest

# As in `test_warp_rig.py`: this skips when warp is missing, not when a GPU is. The CPU-backend
# intent stated above is untouched -- the guard exists so that `mechinfo.gpu_estimator`'s
# module-level `import warp` cannot kill collection for the whole suite.
pytest.importorskip("warp", reason="install the 'sim' extra: uv sync --extra sim")

from mechinfo.estimators import ksg_mutual_information
from mechinfo.gpu_estimator import ksg_mutual_information_gpu


def _gaussian_pair(n, d_y, snr, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, 1))
    M = np.zeros((d_y, 1)); M[0, 0] = snr
    return a, a @ M.T + rng.normal(size=(n, d_y))


@pytest.mark.parametrize("d_y,snr", [(3, 2.0), (8, 4.0), (16, 1.0)])
def test_the_gpu_path_matches_the_cpu_estimator_on_continuous_data(d_y, snr):
    """Agreement to about a millionth of a bit, which is what float32 costs and no more."""
    a, y = _gaussian_pair(4000, d_y, snr)
    cpu = ksg_mutual_information(a, y, k=4)
    gpu = ksg_mutual_information_gpu(a, y, k=4, device="cpu")
    assert gpu == pytest.approx(cpu, abs=1e-5)


def test_ties_are_where_single_precision_stops_being_free():
    """A limitation, pinned rather than fixed, because it decides what this may be used on."""
    rng = np.random.default_rng(3)
    a = np.round(rng.normal(size=(2000, 1)), 1)
    y = np.round(a @ np.ones((1, 4)) * 2.0 + rng.normal(size=(2000, 4)), 1)
    cpu = ksg_mutual_information(a, y, k=4)
    gpu = ksg_mutual_information_gpu(a, y, k=4, device="cpu")
    assert abs(gpu - cpu) > 1e-4, "ties used to disagree; if they no longer do, say why"
    assert abs(gpu - cpu) < 0.2, "and the disagreement must stay small enough to be a caveat"


def test_the_chain_spectrum_matches_the_cpu_one_draw_for_draw():
    """Same seed must mean the same samples, or the two spectra are not comparable."""
    from mechinfo.estimators import ksg_chain_bits
    from mechinfo.gpu_estimator import ksg_chain_bits_gpu
    from passive_info.information import FS, SIGMA_A, SIGMA_ENERGY, T_WINDOW
    from passive_info.spectrum import response_matrix
    from passive_info.systems import series_elastic

    M = response_matrix(series_elastic(1e4, readout="energy"), T_WINDOW, FS, 40.0)
    cpu = ksg_chain_bits(M, SIGMA_ENERGY, SIGMA_A, n_samples=1500, n_blocks=1, seed=7)
    gpu = ksg_chain_bits_gpu(M, SIGMA_ENERGY, SIGMA_A, n_samples=1500, n_blocks=1, seed=7,
                             device="cpu")
    assert gpu.shape == cpu.shape
    assert gpu == pytest.approx(cpu, abs=1e-5)
