"""Command-line drivers for `mechinfo.critic`, run against the Wensing prototype."""
from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np

from mechinfo.critic import infonce_chain_bits, validate
from passive_info.information import FS, SIGMA_A, SIGMA_ENERGY, T_WINDOW
from passive_info.spectrum import chain_bits, response_matrix
from passive_info.systems import series_elastic

RESULTS = pathlib.Path(__file__).parent / "results"


def _chain(args, dev):
    """One stiffness, term by term, against the analytic chain."""
    sysk = series_elastic(args.ks, readout="energy")
    exact = np.array([b for _, b in chain_bits(sysk, band_hz=args.band)])
    M = response_matrix(sysk, T_WINDOW, FS, args.band)
    print(f"\nk_s = {args.ks:g}, band = {args.band:g} Hz, {len(exact)} terms, "
          f"analytic total {exact.sum():.3f} bits")
    t0 = time.time()
    got = infonce_chain_bits(M, SIGMA_ENERGY, SIGMA_A, n_samples=args.n, n_blocks=1,
                             seed=0, device=dev, batch=args.batch, steps=args.steps)
    dt = time.time() - t0
    corr = float(np.corrcoef(exact, got)[0, 1])
    print(f"\n{'band':>5} {'exact':>9} {'infonce':>9} {'err':>8}")
    for j, (e, g) in enumerate(zip(exact, got)):
        print(f"{j:>5} {e:>9.4f} {g:>9.4f} {g - e:>+8.4f}")
    print(f"\ntotal   exact {exact.sum():.3f}   infonce {got.sum():.3f}   "
          f"ratio {got.sum()/exact.sum():.4f}")
    print(f"corr {corr:.4f} | ridge exact {int(np.argmax(exact))} "
          f"infonce {int(np.argmax(got))} | {dt:.0f} s")
    print("\nKSG on the same grid: ratio 0.1054-0.1425, corr 0.2611-0.3145, ridge 17-18")


def _grid(args, dev):
    """The whole stiffness grid, cached so the two `plate_method_comparison` panels compare."""
    cache = np.load(RESULTS / "method_comparison.npz")
    ks_grid = cache["key"][:-3]
    ex_all, nce_all = [], []
    print(f"{'k_s':>10} {'exact':>9} {'infonce':>9} {'ratio':>7} {'corr':>7} "
          f"{'ridge_ex':>9} {'ridge_n':>8} {'secs':>7}")
    for i, k in enumerate(ks_grid):
        sysk = series_elastic(float(k), readout="energy")
        exact = np.array([b for _, b in chain_bits(sysk, band_hz=args.band)])
        M = response_matrix(sysk, T_WINDOW, FS, args.band)
        t0 = time.time()
        got = infonce_chain_bits(M, SIGMA_ENERGY, SIGMA_A, n_samples=args.n,
                                 n_blocks=1, seed=i, device=dev,
                                 batch=args.batch, steps=args.steps)
        dt = time.time() - t0
        print(f"{k:>10.4g} {exact.sum():>9.3f} {got.sum():>9.3f} "
              f"{got.sum()/exact.sum():>7.4f} "
              f"{np.corrcoef(exact, got)[0, 1]:>7.4f} "
              f"{int(np.argmax(exact)):>9} {int(np.argmax(got)):>8} {dt:>7.0f}",
              flush=True)
        ex_all.append(exact); nce_all.append(got)
    ex_all, nce_all = np.array(ex_all).T, np.array(nce_all).T
    RESULTS.mkdir(exist_ok=True)
    np.savez(RESULTS / "critic_grid.npz", ks=ks_grid, band=args.band,
             exact=ex_all, infonce=nce_all)
    cor = [np.corrcoef(ex_all[:, j], nce_all[:, j])[0, 1] for j in range(len(ks_grid))]
    rat = nce_all.sum(0) / ex_all.sum(0)
    print(f"\nOVER THE GRID   ratio median {np.median(rat):.4f} "
          f"[{rat.min():.4f}, {rat.max():.4f}]")
    print(f"                corr  median {np.median(cor):.4f} "
          f"[{min(cor):.4f}, {max(cor):.4f}]")
    print(f"published KSG   ratio median 0.1100, corr median 0.4558 "
          f"[0.2976, 0.5781]")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--validate", action="store_true",
                   help="the critic against exact linear-Gaussian answers")
    p.add_argument("--chain", action="store_true", help="the spectrum vs the analytic table")
    p.add_argument("--grid", action="store_true",
                   help="the full 12-stiffness grid, as in plate_method_comparison")
    p.add_argument("--ks", type=float, default=1e4)
    p.add_argument("--band", type=float, default=200.0)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--n", type=int, default=200_000)
    args = p.parse_args()

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__} | device {dev}"
          f"{' | ' + torch.cuda.get_device_name(0) if dev == 'cuda' else ''}\n")

    if args.validate:
        validate(device=dev, batch=args.batch, steps=args.steps, n=args.n)
    if args.chain:
        _chain(args, dev)
    if args.grid:
        _grid(args, dev)


if __name__ == "__main__":
    main()
