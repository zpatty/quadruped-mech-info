"""The KSG estimator on Warp, where sample size is the binding constraint."""
from __future__ import annotations

import numpy as np
import warp as wp

#: Largest `k` the kernels support. `k` is a neighbour count, always small.
MAX_K = 16


@wp.kernel
def _knn_eps(Z: wp.array2d(dtype=wp.float32), k: int,
             top: wp.array2d(dtype=wp.float32), eps: wp.array(dtype=wp.float32)):
    """`eps[i]` = Chebyshev distance from point i to its k-th neighbour, self excluded."""
    i = wp.tid()
    n = Z.shape[0]
    d = Z.shape[1]
    for a in range(k):
        top[i, a] = 1.0e30
    for j in range(n):
        if j != i:
            dist = float(0.0)
            for c in range(d):
                v = wp.abs(Z[i, c] - Z[j, c])
                if v > dist:
                    dist = v
            if dist < top[i, k - 1]:
                p = k - 1
                while p > 0:
                    if top[i, p - 1] > dist:
                        top[i, p] = top[i, p - 1]
                        p = p - 1
                    else:
                        break                      # p is the insertion point
                top[i, p] = dist
    eps[i] = top[i, k - 1]


@wp.kernel
def _count_within(X: wp.array2d(dtype=wp.float32), eps: wp.array(dtype=wp.float32),
                  out: wp.array(dtype=wp.int32)):
    """`out[i]` = #{j != i : ||X_i - X_j||_inf < eps[i]}. The KSG marginal count."""
    i = wp.tid()
    n = X.shape[0]
    d = X.shape[1]
    # Strictly <. The CPU path spells this `<= eps - 1e-12`, a no-op in float32 that
    # biases the estimate down about 0.25 bits.
    r = eps[i]
    c = int(0)
    for j in range(n):
        if j != i:
            dist = float(0.0)
            for cc in range(d):
                v = wp.abs(X[i, cc] - X[j, cc])
                if v > dist:
                    dist = v
            if dist < r:
                c = c + 1
    out[i] = c


def _as2d(a) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a.reshape(len(a), -1) if a.ndim > 1 else a.reshape(-1, 1)


def _normalise(a: np.ndarray) -> np.ndarray:
    s = a.std(0)
    return (a - a.mean(0)) / np.where(s > 0, s, 1.0)


def ksg_mutual_information_gpu(x, y, k: int = 4, normalise: bool = True,
                               device: str | None = None) -> float:
    """I(x;y) in bits, by KSG algorithm 1, on the GPU."""
    from scipy.special import digamma

    if not 1 <= k <= MAX_K:
        raise ValueError(f"k must be in [1, {MAX_K}], got {k}")
    x, y = _as2d(x), _as2d(y)
    if len(x) != len(y):
        raise ValueError(f"x has {len(x)} samples, y has {len(y)}")
    if normalise:
        x, y = _normalise(x), _normalise(y)

    n = len(x)
    device = device or ("cuda:0" if wp.get_cuda_device_count() else "cpu")
    Z = wp.array(np.ascontiguousarray(np.hstack([x, y]), dtype=np.float32), device=device)
    Xa = wp.array(np.ascontiguousarray(x, dtype=np.float32), device=device)
    Ya = wp.array(np.ascontiguousarray(y, dtype=np.float32), device=device)
    eps = wp.zeros(n, dtype=wp.float32, device=device)
    top = wp.zeros((n, MAX_K), dtype=wp.float32, device=device)
    nx = wp.zeros(n, dtype=wp.int32, device=device)
    ny = wp.zeros(n, dtype=wp.int32, device=device)

    wp.launch(_knn_eps, dim=n, inputs=[Z, k, top, eps], device=device)
    wp.launch(_count_within, dim=n, inputs=[Xa, eps, nx], device=device)
    wp.launch(_count_within, dim=n, inputs=[Ya, eps, ny], device=device)
    wp.synchronize_device(device)

    nx_h = nx.numpy().astype(np.int64)
    ny_h = ny.numpy().astype(np.int64)
    return float((digamma(k) - np.mean(digamma(nx_h + 1) + digamma(ny_h + 1))
                  + digamma(n)) / np.log(2.0))


def ksg_chain_bits_gpu(M: np.ndarray, sigma: float, sigma_a: float, group: int = 1,
                       n_samples: int = 20_000, n_blocks: int = 2, k: int = 4,
                       seed: int = 0, device: str | None = None) -> np.ndarray:
    """`estimators.ksg_chain_bits` on the GPU. Identical sampling and conditioning."""
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
            A[:, :lo] = rng.normal(0.0, sigma_a, size=lo)
            S = A @ G.T + sigma * rng.normal(size=(n_samples, d)) @ L.T
            acc.append(ksg_mutual_information_gpu(A[:, lo:hi], S, k=k, device=device))
        out.append(float(np.mean(acc)))
    return np.array(out)


def validate(n_grid=(2_000, 5_000), k: int = 4, seed: int = 0, device: str | None = None):
    """GPU against CPU on identical samples, against the exact value, and on speed."""
    import time

    from mechinfo.estimators import gaussian_mutual_information, ksg_mutual_information

    rng = np.random.default_rng(seed)
    d_y = 8
    Mtrue = rng.normal(size=(d_y, 3))
    print(f"{'N':>8} {'analytic':>10} {'cpu':>10} {'gpu':>10} {'gpu-cpu':>9} "
          f"{'t_cpu':>8} {'t_gpu':>8} {'speedup':>8}")
    rows = []
    for n in n_grid:
        a = rng.normal(size=(n, 3))
        y = a @ Mtrue.T + 0.5 * rng.normal(size=(n, d_y))
        exact = gaussian_mutual_information(Mtrue.T / 0.5, 1.0, 1.0)

        t0 = time.time(); c = ksg_mutual_information(a, y, k=k); t_cpu = time.time() - t0
        t0 = time.time(); g = ksg_mutual_information_gpu(a, y, k=k, device=device)
        t_gpu = time.time() - t0

        print(f"{n:>8} {exact:>10.4f} {c:>10.4f} {g:>10.4f} {g - c:>+9.5f} "
              f"{t_cpu:>8.2f} {t_gpu:>8.2f} {t_cpu / max(t_gpu, 1e-9):>7.1f}x")
        rows.append(dict(n=n, exact=exact, cpu=c, gpu=g, dev=g - c,
                         t_cpu=t_cpu, t_gpu=t_gpu))
    worst = max(abs(r["dev"]) for r in rows)
    print(f"\nworst |gpu - cpu| = {worst:.6f} bits")
    return rows
