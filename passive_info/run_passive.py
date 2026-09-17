"""One design's passive spectrum, with its own convergence trace. Submitted once per stiffness."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

RESULTS = pathlib.Path(__file__).parent / "results"


#: Called inside `main`, after argument parsing. The rollout modules pull in torch, warp and
#: MuJoCo from the `sim` extra, so importing at module scope makes `--help` and a mistyped
#: argument fail with a `ModuleNotFoundError` instead of a useful message.
def _require_sim():
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError as e:
        raise SystemExit(
            f"{e.name} is missing: the rollout stack lives behind the `sim` extra.\n"
            f"  uv sync --extra sim        (Linux + NVIDIA; see pyproject.toml)\n"
            f"Reproducing the paper's FIGURES needs none of this -- they read the committed "
            f"JSON in passive_info/results/ and run on the CPU core alone.") from e


def _parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("design", help="a series stiffness in N m/rad, or `prop` for the rigid leg")
    p.add_argument("steps", nargs="?", type=int, default=800)
    p.add_argument("band_hz", nargs="?", type=float, default=100.0)
    return p.parse_args(argv)


def _unused_path(base: pathlib.Path) -> pathlib.Path:
    """Append-only: a colliding name gets a suffix, and says so."""
    out, n = base.with_suffix(".json"), 1
    while out.exists():
        out = base.with_suffix(f".{n}.json")
        n += 1
    if out != base.with_suffix(".json"):
        print(f"NOTE {base.name}.json exists; writing {out.name} instead (nothing overwritten)")
    return out


def main(argv=None) -> None:
    args = _parse(argv)                 # before the heavy imports, so --help always works
    _require_sim()
    import torch

    from passive_info.embodied import SIM_DT, _ground_gains, _sim_rig
    from passive_info.sim_info import (STREAM_BUFFER, STREAM_REFRESH,
                                       sim_chain_bits_streaming, sim_chain_reference)
    from passive_info.systems import K_I

    ks = None if args.design in ("prop", "none", "None") else float(args.design)
    steps, band = args.steps, args.band_hz
    label = "proprioceptive" if ks is None else f"ks{ks:.6g}"
    if band != 100.0:
        label = f"{label}_b{band:.0f}"      # band is part of the identity, not a footnote

    # The ground is recorded for the same reason the band is: runs are append-only, so a
    # corrected run lands beside the defective one and this field is the only thing that
    # tells them apart. 219.61 is the locked ground, 132.72 the pre-fix one.
    m, d, leg = _sim_rig(ks, SIM_DT, 0.05, "menagerie", "spring", False, vertical="fixed")
    ground_b_i = float(_ground_gains(m, d, leg, K_I)[0])
    print(f"  ground: b_i = {ground_b_i:.2f}  (locked transmission; all designs equal)",
          flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # The reference must use the same band as the measurement. `sim_profile_bits` has no band
    # argument -- it calls n_modes() with defaults -- so measuring at 200 Hz compared 34 bands
    # against a 17-band reference and reported a ratio of 1.99. A lower bound above its own
    # reference is impossible, which is what gave it away.
    ref_bands = sim_chain_reference(ks, band_hz=band)
    lin = {"bits": float(ref_bands.sum()), "rank": len(ref_bands), "n_modes": len(ref_bands)}
    print(f"design {label}  band {band:.0f} Hz   "
          f"linearised reference {lin['bits']:.3f} bits, "
          f"rank {lin['rank']}/{lin['n_modes']}", flush=True)

    t0 = time.time()
    # The streaming settings come from `sim_info`, not from here. They were hardcoded in this
    # file, so raising them at their definition moved the controlled half and left the passive
    # half behind -- two measurements on different reuse, looking like a result.
    bits = sim_chain_bits_streaming(ks, band_hz=band, steps=steps, buffer=STREAM_BUFFER,
                                    refresh_every=STREAM_REFRESH, device=dev, verbose=True,
                                    trace_every=200)
    el = time.time() - t0
    tot = float(bits.sum())
    print(f"\n{label}: total {tot:.3f} bits over {len(bits)} bands  "
          f"vs linearised {lin['bits']:.3f}  ratio {tot / lin['bits']:.4f}  ({el:.0f} s)")

    RESULTS.mkdir(exist_ok=True)
    out = _unused_path(RESULTS / f"grid_{label}")
    out.write_text(json.dumps(
        dict(ks=ks, label=label, steps=steps, band_hz=band, ground_b_i=ground_b_i,
             reference_bands=ref_bands.tolist(), bands=bits.tolist(), total=tot,
             linearised=lin["bits"], ratio=tot / lin["bits"], seconds=el,
             job=os.environ.get("SLURM_JOB_ID"), node=os.environ.get("SLURMD_NODENAME")),
        indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
