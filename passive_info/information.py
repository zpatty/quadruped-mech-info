"""How much a single motor-side encoder learns about an unknown step contact force."""
from __future__ import annotations

import numpy as np

from mechinfo.mechanics import bits_from_gram, zoh
from passive_info.systems import K_I, R, Passive, proprioceptive, series_elastic

FS = 4000.0              #: Hz, sampling rate of the encoder
T_WINDOW = 0.085         #: s, the reference paper's own contact duration
SIGMA_THETA = 1.0e-4     #: rad, ~16-bit over one revolution. Stated, and swept in `main`.
SIGMA_F = 100.0          #: N, prior width on the unknown load


def sensitivity(sys: Passive, t_window: float = T_WINDOW, fs: float = FS) -> np.ndarray:
    """d(readout)/dF_0 at each sample. Shape (n_times, n_readout)."""
    dt = 1.0 / fs
    Ad, Bd = zoh(sys.A, sys.B_F, dt)
    G = np.zeros_like(sys.B_F)
    out = [(sys.C @ G).ravel().copy()]
    for _ in range(int(round(t_window * fs))):
        G = Bd + Ad @ G
        out.append((sys.C @ G).ravel().copy())
    return np.asarray(out)


def bits(sys: Passive, t_window: float = T_WINDOW, fs: float = FS,
         sigma_theta: float = SIGMA_THETA, sigma_f: float = SIGMA_F) -> float:
    """Mutual information about the step amplitude, in bits, over `t_window`."""
    S = sensitivity(sys, t_window, fs)
    fisher = float(np.sum(S ** 2) / sigma_theta ** 2)
    return 0.5 * np.log2(1.0 + sigma_f ** 2 * fisher)


def readout_gap(ks, t_window: float = T_WINDOW, **kw) -> dict:
    """What the mechanism responds to, against what the actuator can see."""
    build = (lambda **k: proprioceptive(**k)) if ks is None else (
        lambda **k: series_elastic(ks, **k))
    out = dict(ks=ks)
    for r in ("motor", "q", "state"):
        out[r] = bits(build(readout=r), t_window, **kw)
    out["gap"] = out["state"] - out["motor"]
    return out


def bits_vs_time(sys: Passive, taus_ms=(1, 2, 3, 5, 8, 12, 18, 26, 38, 55, 85),
                 fs: float = FS, **kw) -> list[dict]:
    """Bits against how much of the contact has elapsed -- the latency of knowing."""
    longest = max(taus_ms) * 1e-3
    S = sensitivity(sys, longest, fs)
    sigma_theta = kw.get("sigma_theta", SIGMA_THETA)
    sigma_f = kw.get("sigma_f", SIGMA_F)
    rows = []
    for tau in taus_ms:
        n = int(round(tau * 1e-3 * fs)) + 1
        fisher = float(np.sum(S[:n] ** 2) / sigma_theta ** 2)
        rows.append(dict(tau_ms=float(tau),
                         bits=0.5 * np.log2(1.0 + sigma_f ** 2 * fisher),
                         peak_rad_per_n=float(np.abs(S[:n]).max())))
    return rows


def time_to_resolve(sys: Passive, sigma_theta: float = SIGMA_THETA, f_ref: float = SIGMA_F,
                    t_window: float = T_WINDOW, fs: float = FS) -> float:
    """ms until a load of `f_ref` newtons moves the encoder by one resolution unit."""
    S = np.abs(sensitivity(sys, t_window, fs)).max(axis=1) * f_ref
    hit = np.flatnonzero(S >= sigma_theta)
    return float(hit[0]) / fs * 1e3 if hit.size else float("nan")


# ---------------------------------------------------------------------------------------
# An unknown force PROFILE rather than a step amplitude.
#
# A step has one unknown number, so its Fisher information is a signal-to-noise ratio and
# says nothing about computation. Expanding the load in a mode basis makes the estimand a
# vector and the information a matrix, and what is counted becomes how many independent
# things about the contact the mechanism retains. Rank rather than gain.
# ---------------------------------------------------------------------------------------

BAND_HZ = 100.0          #: bandwidth of the unknown load; sets the mode budget as 2BT
SIGMA_A = 100.0          #: N, prior width per mode coefficient

#: Noise scale for the energy-whitened readouts, in sqrt(joules). Anchored to a physical
#: event rather than picked: the potential energy stored in the ground spring by one encoder
#: unit of foot travel. It sets the zero of the bit scale, not the comparison.
SIGMA_ENERGY = np.sqrt(K_I) * R * SIGMA_THETA


def natural_sigma(readout: str) -> float:
    """The noise scale whose units match a readout. `energy`/`kinetic` are not radians."""
    return SIGMA_ENERGY if readout in ("energy", "kinetic") else SIGMA_THETA


def mode_basis(n: int, k: int) -> np.ndarray:
    """First `k` orthonormal DCT-II modes on `n` samples. Shape (n, k)."""
    t = (np.arange(n) + 0.5) / n
    B = np.cos(np.pi * np.outer(t, np.arange(k)))
    return B / np.linalg.norm(B, axis=0, keepdims=True)


def n_modes(t_window: float = T_WINDOW, band_hz: float = BAND_HZ) -> int:
    """2BT: the degrees of freedom of a band-limited load, not a number chosen by hand."""
    return max(1, int(round(2.0 * band_hz * t_window)))


def response_matrix(sys: Passive, t_window: float = T_WINDOW, fs: float = FS,
                    band_hz: float = BAND_HZ) -> np.ndarray:
    """Readout response to each mode of the unknown load. Shape (n_times * n_out, K)."""
    dt = 1.0 / fs
    n_t = int(round(t_window * fs)) + 1
    Phi = mode_basis(n_t, n_modes(t_window, band_hz))
    Ad, Bd = zoh(sys.A, sys.B_F, dt)

    cols = []
    for j in range(Phi.shape[1]):
        x = np.zeros((sys.A.shape[0], 1))
        out = [(sys.C @ x).ravel().copy()]
        for k in range(n_t - 1):
            x = Ad @ x + Bd * Phi[k, j]
            out.append((sys.C @ x).ravel().copy())
        cols.append(np.asarray(out).ravel())
    return np.column_stack(cols)


def profile_bits(sys: Passive, t_window: float = T_WINDOW, fs: float = FS,
                 sigma: float | None = None, sigma_a: float = SIGMA_A,
                 band_hz: float = BAND_HZ) -> dict:
    """I(a ; z) for an unknown load profile: 1/2 logdet(I + gamma M^T M)."""
    if sigma is None:
        sigma = natural_sigma(sys.readout)
    M = response_matrix(sys, t_window, fs, band_hz)
    gram = (sigma_a / sigma) ** 2 * (M.T @ M)
    eig = np.linalg.eigvalsh(gram)[::-1]
    per = 0.5 * np.log2(1.0 + np.clip(eig, 0.0, None))
    return dict(bits=bits_from_gram(gram), eigbits=per,
                n_modes=int(M.shape[1]),
                rank=int(np.sum(per > 1.0)))


# ---------------------------------------------------------------------------------------
# An ensemble of disturbance bandwidths, rather than one chosen value.
#
# `profile_bits` at a single `band_hz` rewards whichever design happens to have a resonance
# just above that cutoff, which is a property of the analysis choice and not the mechanism.
# Averaging over a spread of bandwidths fixes it: a resonance can align with at most a narrow
# slice of the ensemble.
# ---------------------------------------------------------------------------------------

#: A log-uniform spread of plausible disturbance bandwidths. Not fitted to any design's
#: modes -- chosen before looking at the outcome, spanning "slow" (20 Hz) to "fast enough
#: that no relevant SEA mode is still off the edge" (320 Hz).
BAND_ENSEMBLE_HZ = tuple(20.0 * 2.0 ** k for k in range(5))   # 20, 40, 80, 160, 320


def bits_over_bandwidths(sys: Passive, bands_hz=BAND_ENSEMBLE_HZ,
                         t_window: float = T_WINDOW, **kw) -> list[dict]:
    """`profile_bits` repeated over an ensemble of assumed disturbance bandwidths."""
    out = []
    for b in bands_hz:
        r = profile_bits(sys, t_window, band_hz=b, **kw)
        out.append(dict(band_hz=b, bits=r["bits"], rank=r["rank"], n_modes=r["n_modes"]))
    return out


def ensemble_summary(sys: Passive, bands_hz=BAND_ENSEMBLE_HZ, **kw) -> dict:
    """Mean and worst-case bits over the bandwidth ensemble."""
    rows = bits_over_bandwidths(sys, bands_hz, **kw)
    b = np.array([r["bits"] for r in rows])
    return dict(mean=float(b.mean()), worst=float(b.min()), best=float(b.max()),
                worst_band_hz=float(bands_hz[int(np.argmin(b))]), rows=rows)
