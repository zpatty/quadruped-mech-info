"""Where the information sits: in frequency, and in coordinates."""
from __future__ import annotations

import dataclasses
from itertools import combinations
from math import factorial

import numpy as np
import scipy.linalg as sla

from passive_info.channel import ModeChannel, chain, shapley
from mechinfo.mechanics import bits_from_gram, energy_metric, zoh
from passive_info.information import (BAND_HZ, FS, SIGMA_A, T_WINDOW, mode_basis,
                                      n_modes, natural_sigma, profile_bits,
                                      response_matrix)
from passive_info.systems import Passive


def response_operator(sys: Passive, t_window: float = T_WINDOW, fs: float = FS) -> np.ndarray:
    """`L`, shape (n_t * n_out, n_t): the readout's response to a unit input at each sample."""
    n_t = int(round(t_window * fs)) + 1
    Ad, Bd = zoh(sys.A, sys.B_F, 1.0 / fs)
    h, x = [np.zeros(sys.C.shape[0])], Bd.copy()
    for _ in range(n_t - 1):
        h.append((sys.C @ x).ravel().copy())
        x = Ad @ x
    n_out = sys.C.shape[0]
    L = np.zeros((n_t * n_out, n_t))
    for k in range(n_t):
        for i in range(k + 1):
            L[k * n_out:(k + 1) * n_out, i] = h[k - i]
    return L


def kernel_bits(sys: Passive, K: np.ndarray, sigma: float | None = None,
                t_window: float = T_WINDOW, fs: float = FS) -> float:
    """`I(F; z)` for a Gaussian-process prior on the force. No basis, no coefficients."""
    if sigma is None:
        sigma = natural_sigma(sys.readout)
    L = response_operator(sys, t_window, fs)
    S = L @ K @ L.T / sigma ** 2
    return bits_from_gram(S)


def sample_times(t_window: float = T_WINDOW, fs: float = FS) -> np.ndarray:
    return np.arange(int(round(t_window * fs)) + 1) / fs


def k_bandlimited(t: np.ndarray, sigma_a: float = SIGMA_A, band_hz: float = BAND_HZ,
                  t_window: float = T_WINDOW) -> np.ndarray:
    """The prior `profile_bits` already uses, written as a kernel: flat in band, zero above."""
    Phi = mode_basis(len(t), n_modes(t_window, band_hz))
    return sigma_a ** 2 * (Phi @ Phi.T)


def k_matern32(t: np.ndarray, var: float, ell: float) -> np.ndarray:
    """Once differentiable, spectrum falling as w^-4. The realistic middle ground."""
    r = np.sqrt(3.0) * np.abs(t[:, None] - t[None, :]) / ell
    return var * (1.0 + r) * np.exp(-r)


def k_sqexp(t: np.ndarray, var: float, ell: float) -> np.ndarray:
    """Analytic sample paths, Gaussian spectrum. Very smooth forces."""
    return var * np.exp(-0.5 * ((t[:, None] - t[None, :]) / ell) ** 2)


def k_ou(t: np.ndarray, var: float, ell: float) -> np.ndarray:
    """Ornstein-Uhlenbeck: Lorentzian spectrum, the roughest of the three."""
    return var * np.exp(-np.abs(t[:, None] - t[None, :]) / ell)


def channel_spectrum(sys: Passive, band_hz: float = BAND_HZ, sigma: float | None = None,
                     sigma_a: float = SIGMA_A,
                     t_window: float = T_WINDOW, fs: float = FS) -> list[tuple[float, float]]:
    """Per-eigenchannel bits and the frequency each channel is centred on."""
    if sigma is None:
        sigma = natural_sigma(sys.readout)
    n_t = int(round(t_window * fs)) + 1
    Phi = mode_basis(n_t, n_modes(t_window, band_hz))
    M = response_matrix(sys, t_window, fs, band_hz)
    lam, V = np.linalg.eigh((sigma_a / sigma) ** 2 * (M.T @ M))
    bits = 0.5 * np.log2(1.0 + np.clip(lam, 0.0, None))
    freqs = np.fft.rfftfreq(n_t, 1.0 / fs)
    out = []
    for i in range(V.shape[1]):
        p = np.abs(np.fft.rfft(Phi @ V[:, i])) ** 2
        out.append((float(freqs @ p / p.sum()), float(bits[i])))
    return sorted(out)


def bits_by_frequency(sys: Passive, edges: np.ndarray, band_hz: float = BAND_HZ,
                      **kw) -> np.ndarray:
    """Eigenchannel bits binned by the frequency each channel listens at."""
    out = np.zeros(len(edges) - 1)
    for f, b in channel_spectrum(sys, band_hz, **kw):
        j = int(np.searchsorted(edges, f, side="right")) - 1
        if 0 <= j < len(out):
            out[j] += b
    return out


def modal_bits(sys: Passive, band_hz: float = BAND_HZ, sigma: float | None = None,
               **kw) -> list[tuple[float, float]]:
    """(frequency, bits) per mode, each mode read on its own."""
    if sigma is None:
        sigma = natural_sigma(sys.readout)
    n = sys.n
    w2, Phi_m = sla.eigh(sys.K, sys.H)                 # Phi^T H Phi = I
    inv = Phi_m.T @ sys.H                              # eta = Phi^-1 q = Phi^T H q

    out = []
    for m in range(n):
        C = np.zeros((2, 2 * n))
        C[0, :n] = np.sqrt(max(w2[m], 0.0)) * inv[m]
        C[1, n:] = inv[m]
        one = dataclasses.replace(sys, C=C)
        out.append((float(np.sqrt(max(w2[m], 0.0)) / (2 * np.pi)),
                    profile_bits(one, sigma=sigma, band_hz=band_hz, **kw)["bits"]))
    return out


def marginal_modal_bits(sys: Passive, band_hz: float = BAND_HZ, sigma: float | None = None,
                        **kw) -> list[tuple[float, float]]:
    """(mode frequency, marginal bits) -- bits with every mode, minus bits without this one."""


    if sigma is None:
        sigma = natural_sigma(sys.readout)
    n = sys.n
    w2, Phi_m = sla.eigh(sys.K, sys.H)
    inv = Phi_m.T @ sys.H

    def readout(modes):
        rows = []
        for m in modes:
            r1 = np.zeros(2 * n)
            r1[:n] = np.sqrt(max(w2[m], 0.0)) * inv[m]
            r2 = np.zeros(2 * n)
            r2[n:] = inv[m]
            rows += [r1, r2]
    # An empty mode set is a readout of nothing, worth zero bits, and on a one-coordinate
    # mechanism it is the only case `rest` can take. `np.vstack([])` raised there instead.
        C = np.vstack(rows) if rows else np.zeros((0, 2 * n))
        return dataclasses.replace(sys, C=C)

    every = list(range(n))
    total = profile_bits(readout(every), sigma=sigma, band_hz=band_hz, **kw)["bits"]
    out = []
    for m in every:
        rest = [k for k in every if k != m]
        without = profile_bits(readout(rest), sigma=sigma, band_hz=band_hz, **kw)["bits"]
        out.append((float(np.sqrt(max(w2[m], 0.0)) / (2 * np.pi)), float(total - without)))
    return out


def chain_bits(sys: Passive, band_hz: float = BAND_HZ, sigma: float | None = None,
               t_window: float = T_WINDOW, fs: float = FS,
               group: int = 1, sigma_a: float = SIGMA_A) -> list[tuple[float, float]]:
    """(frequency, bits) per frequency band, by the chain rule -- exact and additive."""
    ch = ModeChannel.from_system(sys, band_hz, sigma, t_window, fs, sigma_a)
    return chain(lambda start: ch.bits(range(start, ch.n_modes)),
                 ch.n_modes, group, ch.freqs)


def fisher_bits(sys: Passive, band_hz: float = BAND_HZ, sigma: float | None = None,
                t_window: float = T_WINDOW, fs: float = FS,
                group: int = 1, sigma_a: float = SIGMA_A) -> list[tuple[float, float]]:
    """(frequency, bits) per band from Fisher information, for comparison with `chain_bits`."""
    # Goes badly negative once a band is under-excited, and cannot tell you which regime
    # you are in. Safe only while every band is well excited.
    ch = ModeChannel.from_system(sys, band_hz, sigma, t_window, fs, sigma_a)
    return chain(lambda start: ch.log_volume(range(start, ch.n_modes)),
                 ch.n_modes, group, ch.freqs)


def coordinate_bits(sys: Passive, coords, band_hz: float = BAND_HZ,
                    sigma: float | None = None, t_window: float = T_WINDOW,
                    fs: float = FS, sigma_a: float = SIGMA_A) -> float:
    """Bits from observing only `coords` of the state, with the metric supplying the noise."""
    if sigma is None:
        sigma = natural_sigma(sys.readout)


    n = sys.n
    E = np.zeros((2 * n, 2 * n))
    E[:n, :n] = sys.K
    E[n:, n:] = sys.H
    Einv = np.linalg.inv(E)

    coords = np.asarray(coords, int)
    N_S = Einv[np.ix_(coords, coords)]                  # marginal noise on the subset
    # Pseudo-inverse, because a redundant observation makes this singular and that is the
    # point: two readings of one coordinate have perfectly correlated noise, so the
    # covariance is rank deficient and the duplicate gets exactly zero weight. `inv` would
    # raise instead of returning the right answer.
    W = np.linalg.pinv(N_S, rcond=1e-12, hermitian=True)

    # Response of the raw state to each input mode, then restricted to `coords`.
    dt = 1.0 / fs
    n_t = int(round(t_window * fs)) + 1
    K = n_modes(t_window, band_hz)
    Phi = mode_basis(n_t, K)
    Ad, Bd = zoh(sys.A, sys.B_F, dt)
    cols = []
    for j in range(K):
        x = np.zeros((sys.A.shape[0], 1))
        out = [x[coords, 0].copy()]
        for k in range(n_t - 1):
            x = Ad @ x + Bd * Phi[k, j]
            out.append(x[coords, 0].copy())
        cols.append(np.asarray(out))
    M = np.stack(cols, axis=-1)                          # (n_t, |S|, K)

    gram = sum(M[t].T @ W @ M[t] for t in range(M.shape[0])) * (sigma_a / sigma) ** 2
    return bits_from_gram(gram)


def marginal_coordinate_bits(sys: Passive, **kw) -> list[tuple[int, float, float]]:
    """(coordinate, standalone bits, marginal bits) for every state coordinate."""
    n2 = 2 * sys.n
    allc = list(range(n2))
    total = coordinate_bits(sys, allc, **kw)
    out = []
    for i in range(n2):
        alone = coordinate_bits(sys, [i], **kw)
        without = coordinate_bits(sys, [c for c in allc if c != i], **kw)
        out.append((i, alone, total - without))
    return out


def _coordinate_pieces(sys: Passive, band_hz: float, sigma: float | None,
                       t_window: float, fs: float, sigma_a: float = SIGMA_A):
    """(X, Einv, gamma, K): everything a coordinate decomposition needs, computed once."""
    if sigma is None:
        sigma = natural_sigma(sys.readout)
    n2 = 2 * sys.n
    Einv = np.linalg.inv(energy_metric(sys.K, sys.H))

    dt, n_t = 1.0 / fs, int(round(t_window * fs)) + 1
    K = n_modes(t_window, band_hz)
    Phi = mode_basis(n_t, K)
    Ad, Bd = zoh(sys.A, sys.B_F, dt)
    cols = []
    for j in range(K):
        x, out = np.zeros((n2, 1)), [np.zeros(n2)]
        for k in range(n_t - 1):
            x = Ad @ x + Bd * Phi[k, j]
            out.append(x[:, 0].copy())
        cols.append(np.asarray(out))
    X = np.stack(cols, axis=-1)                       # (n_t, 2n, K)
    return X, Einv, (sigma_a / sigma) ** 2, K


def _subset_weights(Einv, n2):
    """pinv of the metric's marginal noise for every coordinate subset, computed once."""
    W = {}
    for r in range(n2 + 1):
        for S in combinations(range(n2), r):
            if S:
                idx = np.asarray(S, int)
                W[S] = np.linalg.pinv(Einv[np.ix_(idx, idx)], rcond=1e-12, hermitian=True)
    return W


def shapley_coordinate_bits(sys: Passive, band_hz: float = BAND_HZ,
                            sigma: float | None = None, t_window: float = T_WINDOW,
                            fs: float = FS, sigma_a: float = SIGMA_A) -> np.ndarray:
    """Shapley value of each state coordinate, in bits. Sums to the total exactly."""
    # 2^(2n) subset evaluations: cheap at four coordinates, a hard ceiling by six.
    X, Einv, gamma, K = _coordinate_pieces(sys, band_hz, sigma, t_window, fs, sigma_a)
    n2 = X.shape[1]
    W = _subset_weights(Einv, n2)

    values = {(): 0.0}
    for r in range(1, n2 + 1):
        for S in combinations(range(n2), r):
            idx = np.asarray(S, int)
            Xs = X[:, idx, :]
            G = gamma * np.einsum("tik,ij,tjl->kl", Xs, W[S], Xs)
            values[S] = bits_from_gram(G)
    return shapley(values, n2)


def temporal_coordinate_shapley(sys: Passive, band_hz: float = BAND_HZ,
                                sigma: float | None = None, t_window: float = T_WINDOW,
                                fs: float = FS, sigma_a: float = SIGMA_A):
    """(times, phi) with `phi[t, i]` the bits coordinate `i` contributes AT sample `t`."""
    X, Einv, gamma, K = _coordinate_pieces(sys, band_hz, sigma, t_window, fs, sigma_a)
    n_t, n2 = X.shape[0], X.shape[1]
    W = _subset_weights(Einv, n2)

    subsets = [S for r in range(1, n2 + 1) for S in combinations(range(n2), r)]
    ld = lambda G: bits_from_gram(G)

    base = np.zeros((K, K))
    phi = np.zeros((n_t, n2))
    for t in range(n_t):
        prev = ld(base)
        values = {(): 0.0}
        step = {}
        for S in subsets:
            idx = np.asarray(S, int)
            xs = X[t, idx, :]
            step[S] = gamma * (xs.T @ W[S] @ xs)
            values[S] = ld(base + step[S]) - prev
        phi[t] = shapley(values, n2)
        base = base + step[tuple(range(n2))]
    return np.arange(n_t) / fs, phi
