"""Command-line drivers for `mechinfo.gpu_estimator`, run against the Wensing prototype."""
from __future__ import annotations

import argparse
import time

import numpy as np
import warp as wp

from mechinfo.gpu_estimator import ksg_chain_bits_gpu, validate
from passive_info.information import FS, SIGMA_A, SIGMA_ENERGY, T_WINDOW
from passive_info.spectrum import chain_bits, response_matrix
from passive_info.systems import series_elastic


def _scaling(args, dev):
    sysk = series_elastic(args.ks, readout="energy")
    exact = np.array([b for _, b in chain_bits(sysk, band_hz=args.band)])
    M = response_matrix(sysk, T_WINDOW, FS, args.band)
    print(f"\nk_s = {args.ks:g}, band = {args.band:g} Hz, {len(exact)} chain terms, "
          f"analytic total {exact.sum():.3f} bits")
    print(f"{'N':>9} {'ksg total':>10} {'ratio':>7} {'corr':>7} "
          f"{'ridge_ex':>9} {'ridge_ksg':>10} {'secs':>8}")
    for n in args.n:
        t0 = time.time()
        got = ksg_chain_bits_gpu(M, SIGMA_ENERGY, SIGMA_A, n_samples=n,
                                 n_blocks=1, seed=0, device=dev)
        dt = time.time() - t0
        corr = float(np.corrcoef(exact, got)[0, 1])
        print(f"{n:>9} {got.sum():>10.3f} {got.sum() / exact.sum():>7.4f} "
              f"{corr:>7.4f} {int(np.argmax(exact)):>9} {int(np.argmax(got)):>10} "
              f"{dt:>8.1f}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--validate", action="store_true", help="GPU vs CPU vs analytic")
    p.add_argument("--scaling", action="store_true",
                   help="time the chain spectrum against N on one stiffness")
    p.add_argument("--ks", type=float, default=1e4)
    p.add_argument("--band", type=float, default=200.0)
    p.add_argument("--n", type=int, nargs="+", default=[10_000, 40_000, 160_000])
    args = p.parse_args()

    wp.init()
    dev = "cuda:0" if wp.get_cuda_device_count() else "cpu"
    print(f"device: {dev}\n")

    if args.validate:
        validate(device=dev)
    if args.scaling:
        _scaling(args, dev)


if __name__ == "__main__":
    main()
