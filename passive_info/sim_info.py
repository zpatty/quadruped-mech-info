"""Information through the simulation rather than through a linearisation of it."""
from __future__ import annotations

import numpy as np

from passive_info.embodied import SIM_DT
from mechinfo.mechanics import bits_from_gram
from passive_info.systems import ZETA_SERIES


def _observe(X, E_half, sigma, gen):
    """Whiten by the energy metric and add the declared resolution -- on the device."""
    # The noise is not optional: the map from a to x is deterministic, so without a declared
    # resolution the information is infinite or zero, never a property of the mechanism.
    import torch
    Y = X @ E_half.T
    Y += sigma * torch.randn(Y.shape, generator=gen, device=Y.device, dtype=Y.dtype)
    return Y


def _energy_half(ks, dt, zeta_series, contact, ground, gravity, device=None,
                 vertical="fixed"):
    """`E^(1/2)` for the simulated rig, from the `diag(K, H)` of `embodied._sim_metric`."""
    from passive_info.embodied import _sim_metric

    from mechinfo.mechanics import sym_sqrt

    H = sym_sqrt(_sim_metric(ks, dt, zeta_series, contact, ground, gravity,
                             vertical=vertical))
    if device is None:
        return H
    import torch
    return torch.tensor(H, dtype=torch.float32, device=device)


def _flat_trajectories(ks, a, batch_worlds, stride, dt, device, E_half, sigma, gen, **kw):
    """Run `a` in chunks; return whitened, noised, flattened observations on the device."""
    import torch

    from passive_info.warp_rig import rollout

    X, ok = [], []
    for s in range(0, len(a), batch_worlds):
        chunk = a[s:s + batch_worlds]
        Xc, _ = rollout(ks, chunk, dt=dt, device=device, as_torch=True, **kw)
        Y = _observe(Xc[:, ::stride], E_half, sigma, gen)
        okc = torch.isfinite(Y).all(dim=2).all(dim=1)
        X.append(Y.reshape(len(chunk), -1))
        ok.append(okc)
    return torch.cat(X), torch.cat(ok)


#: One definition of the streaming settings, shared by every measurement here. Reuse is
#: `STREAM_REFRESH * batch / STREAM_BUFFER` = 4 at a batch of 1024, and the ratio is what
#: matters: reuse 16 does not converge on the real rig. A refill's cost barely depends on its
#: width, so wide refills are nearly free.
STREAM_BUFFER = 32_768
STREAM_REFRESH = 128


def sim_chain_bits_streaming(ks: float | None, band_hz: float = 100.0,
                             group: int = 1, steps: int = 4000, batch: int = 1024,
                             buffer: int = STREAM_BUFFER,
                             refresh_every: int = STREAM_REFRESH,
                             sigma_a: float | None = None, sigma: float | None = None,
                             seed: int = 0, dt: float = SIM_DT,
                             zeta_series: float = ZETA_SERIES, stride: int = 1,
                             device: str | None = None, verbose: bool = True,
                             trace_every: int | None = None, vertical="fixed",
                             **critic_kw) -> np.ndarray:
    """`sim_chain_bits`, with rollouts generated during training instead of held in memory."""
    from mechinfo.critic import infonce_bits_streaming
    from passive_info.information import SIGMA_A, SIGMA_ENERGY, n_modes

    sigma_a = SIGMA_A if sigma_a is None else sigma_a
    sigma = SIGMA_ENERGY if sigma is None else sigma
    import torch
    E_half = _energy_half(ks, dt, zeta_series, "menagerie", "spring", True, device=device,
                          vertical=vertical)
    K = n_modes(band_hz=band_hz)
    edges = list(range(0, K, group)) + [K]
    out, dropped = [], []

    for lo, hi in zip(edges[:-1], edges[1:]):
        rng = np.random.default_rng(seed + 1000 * lo)
        gen = torch.Generator(device=device).manual_seed(seed + 1000 * lo)
        held = rng.normal(0.0, sigma_a, size=lo)        # drawn once, held across the band
        state = {"lost": 0, "total": 0}

        def refill(n, lo=lo, hi=hi, held=held, rng=rng, gen=gen, state=state):
            a = rng.normal(0.0, sigma_a, size=(n, K))
            if lo:
                a[:, :lo] = held
            X, ok = _flat_trajectories(ks, a, buffer, stride, dt, device, E_half, sigma,
                                       gen, zeta_series=zeta_series, vertical=vertical)
            state["total"] += n
            state["lost"] += int((~ok).sum().item())
            if not bool(ok.any()):
                raise FloatingPointError(f"every rollout non-finite at k_s={ks}")
            at = torch.tensor(a[:, lo:hi], dtype=torch.float32, device=X.device)
            return at[ok], X[ok]

        # verbose is passed through, not defaulted. Omitting it silently disabled the
        # per-refill progress this loop was instrumented with, which is the whole reason a
        # four-hour run was able to produce no output at all.
        r = infonce_bits_streaming(refill, batch=batch, steps=steps, buffer=buffer,
                                   refresh_every=refresh_every, device=device,
                                   seed=seed, verbose=verbose,
                                   trace_every=trace_every, **critic_kw)
        frac = state["lost"] / max(state["total"], 1)
        dropped.append(frac)
        out.append(r["bits"])
        if verbose:
            print(f"  band {lo:>2}-{hi:<2}  {r['bits']:>8.4f} bits  train {r['train_last']:>7.3f}"
                  f"  sat {r['saturation']:.3f}  gen {r['generated']:>9,}  "
                  f"dropped {frac:.2%}", flush=True)
    if dropped and max(dropped) > 0.01:
        print(f"WARNING: up to {max(dropped):.1%} of rollouts non-finite and dropped")
    return np.array(out)


def amplitude_scan(ks: float | None = None, sigmas=(5.0, 20.0, 50.0, 100.0, 200.0),
                   n: int = 512, band_hz: float = 100.0, dt: float = SIM_DT,
                   device: str | None = None, vertical="fixed"):
    """Is the mechanism linear between the finite-difference amplitude and the prior's?"""
    from passive_info.information import n_modes
    from passive_info.warp_rig import rollout

    K = n_modes(band_hz=band_hz)
    print(f"{'sigma_a':>9} {'finite':>8} {'resp/sig':>12} {'vs 5 N':>9} {'max|resp|':>10}")
    ref = None
    rows = []
    for s in sigmas:
        a = np.random.default_rng(0).normal(0.0, s, size=(n, K))
        X, ok = rollout(ks, a, dt=dt, device=device, vertical=vertical)
        Xi = X[ok]
        resp = Xi - Xi.mean(axis=0, keepdims=True)      # drop the common nominal trajectory
        r = float(np.std(resp) / s)
        ref = r if ref is None else ref
        print(f"{s:>9.1f} {ok.mean():>7.1%} {r:>12.5e} {r / ref:>9.4f} "
              f"{np.abs(resp).max():>10.3e}")
        rows.append(dict(sigma_a=s, finite=float(ok.mean()), per_unit=r, ratio=r / ref))
    return rows


def dimension_scan(ks: float | None = None, strides=(1, 10, 50, 200), n: int = 20_000,
                   band: int = 8, band_hz: float = 100.0, dt: float = SIM_DT,
                   device: str | None = None, vertical="fixed", **critic_kw):
    """Is the critic learning the channel, or memorising the batch? And what does stride cost?"""
    from mechinfo.critic import infonce_bits
    from passive_info.information import SIGMA_A, SIGMA_ENERGY, n_modes

    K = n_modes(band_hz=band_hz)
    E_half = _energy_half(ks, dt, ZETA_SERIES, "menagerie", "spring", True,
                          vertical=vertical)
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, SIGMA_A, size=(n, K))
    a[:, :band] = rng.normal(0.0, SIGMA_A, size=band)          # held, as the chain rule needs

    print(f"{'stride':>7} {'dims':>8} {'train':>9} {'holdout':>9} {'gap':>8} {'sat':>7}")
    rows = []
    for st in strides:
        X, ok = _flat_trajectories(ks, a, 16384, st, dt, device, E_half, SIGMA_ENERGY,
                                   np.random.default_rng(1), vertical=vertical)
        r = infonce_bits(a[ok, band:band + 1], X[ok], device=device, seed=0, **critic_kw)
        train = r["curve"][-1]
        print(f"{st:>7} {X.shape[1]:>8} {train:>9.3f} {r['bits']:>9.3f} "
              f"{train - r['bits']:>8.3f} {r['saturation']:>7.3f}", flush=True)
        rows.append(dict(stride=st, dims=int(X.shape[1]), train=train, holdout=r["bits"]))
    return rows


if __name__ == "__main__":
    import argparse
    import time

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--amplitude", action="store_true")
    p.add_argument("--chain", action="store_true")
    p.add_argument("--dimscan", action="store_true")
    p.add_argument("--stream", action="store_true",
                   help="the chain spectrum with rollouts generated during training")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--refresh", type=int, default=256)
    p.add_argument("--ks", type=float, default=None)
    p.add_argument("--band", type=float, default=100.0)
    p.add_argument("--n", type=int, default=20_000)
    p.add_argument("--worlds", type=int, default=4096)
    p.add_argument("--stride", type=int, default=1)
    args = p.parse_args()

    import warp as wp
    wp.init()
    dev = "cuda" if wp.get_cuda_device_count() else "cpu"
    print(f"device: {dev}\n")

    if args.amplitude:
        amplitude_scan(args.ks, band_hz=args.band, device=dev)

    if args.dimscan:
        dimension_scan(args.ks, n=args.n, band_hz=args.band, device=dev)

    if args.stream:
        from passive_info.embodied import sim_profile_bits
        lin = sim_profile_bits(args.ks)
        print(f"linearised simulation (CPU finite differences): {lin['bits']:.3f} bits, "
              f"rank {lin['rank']}/{lin['n_modes']}")
        print(f"streamed simulation, band {args.band:g} Hz, {args.steps} steps, "
              f"stride {args.stride}:")
        t0 = time.time()
        got = sim_chain_bits_streaming(args.ks, band_hz=args.band, steps=args.steps,
                                       buffer=args.worlds, refresh_every=args.refresh,
                                       stride=args.stride, device=dev)
        print(f"\ntotal {got.sum():.3f} bits over {len(got)} bands vs "
              f"linearised {lin['bits']:.3f}  ({time.time() - t0:.0f} s)")

    if args.chain:
        from passive_info.embodied import sim_profile_bits
        lin = sim_profile_bits(args.ks)
        print(f"\nlinearised simulation (CPU finite differences): {lin['bits']:.3f} bits, "
              f"rank {lin['rank']}/{lin['n_modes']}")
        print(f"sampled simulation, band {args.band:g} Hz, N = {args.n}, "
              f"stride {args.stride}:")
        t0 = time.time()
        got = sim_chain_bits(args.ks, band_hz=args.band, n_samples=args.n,
                             batch_worlds=args.worlds, stride=args.stride, device=dev)
        print(f"\ntotal {got.sum():.3f} bits over {len(got)} bands "
              f"({time.time() - t0:.0f} s)")


def sim_chain_reference(ks: float | None, band_hz: float = 100.0, group: int = 1,
                        eps: float | None = None, dt: float = SIM_DT,
                        sigma: float | None = None, zeta_series: float = ZETA_SERIES,
                        contact: str = "menagerie", ground: str = "spring",
                        gravity: bool = True, vertical="fixed") -> np.ndarray:
    """Per-band linearised reference for the simulated spectrum. The shape check."""
    from passive_info.embodied import SIM_EPS, _sim_metric, simulate
    from passive_info.information import SIGMA_A, SIGMA_ENERGY, n_modes

    eps = SIM_EPS if eps is None else eps
    sigma = SIGMA_ENERGY if sigma is None else sigma
    K = n_modes(band_hz=band_hz)

    base, ok = simulate(ks, np.zeros(K), dt, zeta_series, contact=contact, ground=ground,
                        gravity=gravity, vertical=vertical)
    cols = []
    for j in range(K):
        a = np.zeros(K); a[j] = eps
        xj, okj = simulate(ks, a, dt, zeta_series, contact=contact, ground=ground,
                           gravity=gravity, vertical=vertical)
        ok &= okj
        cols.append((np.asarray(xj) - np.asarray(base)) / eps)
    if not ok:
        raise FloatingPointError(f"unusable finite-difference run at k_s={ks}")

    M = np.stack(cols, axis=-1)
    E = _sim_metric(ks, dt, zeta_series, contact, ground, gravity, vertical=vertical)
    gram = sum(M[t].T @ E @ M[t] for t in range(M.shape[0])) * (SIGMA_A / sigma) ** 2

    def ld(cols_):
        if not len(cols_):
            return 0.0
        G = gram[np.ix_(cols_, cols_)]
        return bits_from_gram(G)

    edges = list(range(0, K, group)) + [K]
    return np.array([ld(list(range(a, K))) - ld(list(range(b, K)))
                     for a, b in zip(edges[:-1], edges[1:])])
