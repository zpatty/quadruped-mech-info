"""Mutual information from samples, by Kraskov-Stogbauer-Grassberger."""
from __future__ import annotations

import numpy as np


def ksg_mutual_information(x: np.ndarray, y: np.ndarray, k: int = 4,
                           normalise: bool = True) -> float:
    """I(x;y) in bits, by KSG algorithm 1."""
    # Cannot much exceed log2(N) bits, and returns log2(N) on a deterministic map rather
    # than infinity. Declare a state resolution.
    from scipy.spatial import cKDTree
    from scipy.special import digamma

    x = np.atleast_2d(np.asarray(x, float).T).T
    y = np.atleast_2d(np.asarray(y, float).T).T
    if len(x) != len(y):
        raise ValueError(f"x has {len(x)} samples, y has {len(y)}")
    if normalise:
        x = (x - x.mean(0)) / np.where(x.std(0) > 0, x.std(0), 1.0)
        y = (y - y.mean(0)) / np.where(y.std(0) > 0, y.std(0), 1.0)

    n = len(x)
    eps = cKDTree(np.hstack([x, y])).query(np.hstack([x, y]), k + 1, p=np.inf,
                                           workers=-1)[0][:, k]
    nx = cKDTree(x).query_ball_point(x, eps - 1e-12, p=np.inf, return_length=True,
                                     workers=-1) - 1
    ny = cKDTree(y).query_ball_point(y, eps - 1e-12, p=np.inf, return_length=True,
                                     workers=-1) - 1
    return float((digamma(k) - np.mean(digamma(nx + 1) + digamma(ny + 1))
                  + digamma(n)) / np.log(2.0))


def gaussian_mutual_information(M: np.ndarray, sigma_a: float, sigma: float) -> float:
    """Exact bits for `y = M a + noise`. For validating the estimator."""
    G = (sigma_a / sigma) ** 2 * (M.T @ M)
    return float(0.5 * np.linalg.slogdet(np.eye(G.shape[0]) + G)[1] / np.log(2.0))


def ksg_chain_bits(M: np.ndarray, sigma: float, sigma_a: float, group: int = 1,
                   n_samples: int = 20_000, n_blocks: int = 2, k: int = 4,
                   seed: int = 0) -> np.ndarray:
    """The chain-rule spectrum `I(a_S ; z | a_<S)` per band, from samples."""
    G = M.T @ M
    d = len(G)
    L = np.linalg.cholesky(G + 1e-12 * np.eye(d))
    rng = np.random.default_rng(seed)
    edges = list(range(0, d, group)) + [d]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        acc = []
        for _ in range(n_blocks):
            A = rng.normal(0.0, sigma_a, size=(n_samples, d))
            A[:, :lo] = rng.normal(0.0, sigma_a, size=lo)      # held across the block
            S = A @ G.T + sigma * rng.normal(size=(n_samples, d)) @ L.T
            acc.append(ksg_mutual_information(A[:, lo:hi], S, k=k))
        out.append(float(np.mean(acc)))
    return np.array(out)
