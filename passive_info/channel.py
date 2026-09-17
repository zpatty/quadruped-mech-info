"""The linear Gaussian channel, and the two ways this package takes it apart."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import factorial

import numpy as np

from passive_info.information import (FS, SIGMA_A, T_WINDOW, natural_sigma,
                                      response_matrix)
from mechinfo.mechanics import bits_from_gram, energy_metric, sym_sqrt, zoh


@dataclass(frozen=True)
class ModeChannel:
    """`z = M a + n` with `a ~ N(0, sigma_a^2 I)` and white noise of variance `sigma^2`."""
    M: np.ndarray               #: (n_obs, K) -- response of the readout to each input mode
    gamma: float                #: (sigma_a / sigma)^2
    t_window: float

    @property
    def n_modes(self) -> int:
        return self.M.shape[1]

    @property
    def freqs(self) -> np.ndarray:
        """Mode j completes j/2 cycles across the window, so it sits at j/(2T) Hz."""
        return np.arange(self.n_modes) / (2.0 * self.t_window)

    def gram(self, cols=None) -> np.ndarray:
        A = self.M if cols is None else self.M[:, np.asarray(cols, int)]
        return self.gamma * (A.T @ A)

    def bits(self, cols=None) -> float:
        """Mutual information in bits: `1/2 logdet(I + G)`."""
        if cols is not None and not len(cols):
            return 0.0
        return bits_from_gram(self.gram(cols))

    def log_volume(self, cols=None) -> float:
        """`1/2 logdet(G)` -- the Fisher count, the high-SNR limit of `bits`."""
        if cols is not None and not len(cols):
            return 0.0
        return float(0.5 * np.linalg.slogdet(self.gram(cols))[1] / np.log(2.0))

    @classmethod
    def from_system(cls, sys, band_hz: float, sigma: float | None = None,
                    t_window: float | None = None, fs: float | None = None,
                    sigma_a: float = SIGMA_A):
        """`sigma_a` is last because both callers pass the others positionally."""
        t_window = T_WINDOW if t_window is None else t_window
        fs = FS if fs is None else fs
        sigma = natural_sigma(sys.readout) if sigma is None else sigma
        return cls(M=response_matrix(sys, t_window, fs, band_hz),
                   gamma=(sigma_a / sigma) ** 2, t_window=t_window)


# ---------------------------------------------------------------------------------------
# The two decompositions
# ---------------------------------------------------------------------------------------

def chain(value, n: int, group: int = 1, labels=None) -> list[tuple[float, float]]:
    """Chain rule as telescoping suffix differences: `value(a..n) - value(b..n)`."""
    edges = list(range(0, n, group)) + [n]
    labels = np.arange(n) if labels is None else np.asarray(labels)
    return [(float(labels[a:b].mean()), value(a) - value(b))
            for a, b in zip(edges[:-1], edges[1:])]


def shapley(worth, n: int) -> np.ndarray:
    """Shapley value of each of `n` players, from a dict-like `worth[tuple(sorted(S))]`."""
    phi = np.zeros(n)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for r in range(n):
            for S in combinations(others, r):
                w = factorial(len(S)) * factorial(n - len(S) - 1) / factorial(n)
                phi[i] += w * (worth[tuple(sorted(S + (i,)))] - worth[tuple(sorted(S))])
    return phi


# ---------------------------------------------------------------------------------------
# Morphological computation
# ---------------------------------------------------------------------------------------

def gibbs_pair(K: np.ndarray, H: np.ndarray, A_d: np.ndarray, label: str = "system"):
    """`(P, Q)`: the equilibrium covariance `E^-1`, and the process noise that sustains it."""
    E = energy_metric(K, H)
    lam = np.linalg.eigvalsh(E)
    if lam[0] <= lam[-1] * 1e-12:
        raise np.linalg.LinAlgError(
            f"{label}: the energy metric is singular, so there is no equilibrium covariance "
            "E^-1 to condition on. Defined on the clamped bench only.")
    P = np.linalg.inv(E)
    Q = 0.5 * (P - A_d @ P @ A_d.T + (P - A_d @ P @ A_d.T).T)
    if np.linalg.eigvalsh(Q)[0] <= 0.0:
        raise np.linalg.LinAlgError(
            f"{label}: E^-1 is not a stationary covariance at this step -- some direction of "
            "this mechanism is undamped and has no stationary law.")
    return P, Q


def cond_mi_bits(S: np.ndarray, a, b, c) -> float:
    """`I(A;B|C)` in bits from one joint covariance `S`, with `a`, `b`, `c` index blocks."""
    a, b, c = (np.asarray(i, int) for i in (a, b, c))

    def h(keep):
        if not len(c):                # nothing to condition on: plain differential entropy
            return np.linalg.slogdet(S[np.ix_(keep, keep)])[1]
        Scc = S[np.ix_(c, c)]
        schur = (S[np.ix_(keep, keep)]
                 - S[np.ix_(keep, c)] @ np.linalg.solve(Scc, S[np.ix_(c, keep)]))
        return np.linalg.slogdet(schur)[1]

    return float((h(a) + h(b) - h(np.concatenate([a, b]))) / (2.0 * np.log(2.0)))


def _conditioned(P: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Covariance of `x` given the linear readout `action @ x`, by Schur complement."""
    L = np.atleast_2d(action)
    PLt = P @ L.T
    return P - PLt @ np.linalg.solve(L @ PLt, PLt.T)


def morphological_bits(sys, dt: float | None = None, action=None,
                       fs_enc: float = FS) -> float:
    """`MC_W` of Zahedi and Ay (2013): `I(x_{k+1} ; x_k | a_k)` in bits, over one step `dt`."""
    # Diverges as dt -> 0, so quote it at a stated dt. Raises on a singular E or an
    # indefinite Q rather than returning a plausible number.
    dt = 1.0 / fs_enc if dt is None else dt
    A_d, _ = zoh(sys.A, sys.B_F, dt)
    P, Q = gibbs_pair(sys.K, sys.H, A_d, label=sys.label)
    Sigma = P if action is None else _conditioned(P, action)
    half = sym_sqrt(np.linalg.inv(Q))
    return bits_from_gram(half @ (A_d @ Sigma @ A_d.T) @ half)
