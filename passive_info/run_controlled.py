"""One design's controlled spectrum: I(a;x,u), I(a;u) and the conditional I(a;x|u)."""
from __future__ import annotations

import argparse
import json
import os
import time

from passive_info.run_passive import RESULTS, _require_sim, _unused_path


def _parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("design", help="a series stiffness in N m/rad, or `prop` for the rigid leg")
    p.add_argument("steps", nargs="?", type=int, default=800)
    p.add_argument("band_hz", nargs="?", type=float, default=100.0)
    p.add_argument("batch", nargs="?", type=int, default=1024)
    a = p.parse_args(argv)
    if a.band_hz > 1000.0:
        raise SystemExit(
            f"band_hz={a.band_hz:g} is not a contact band -- the third positional is the BAND "
            f"and the fourth the batch. Did you mean `... {a.steps} 200 {int(a.band_hz)}`?")
    return a


def main(argv=None) -> None:
    args = _parse(argv)                 # before the heavy imports, so --help always works
    _require_sim()
    import numpy as np

    from passive_info.controlled_sim import sim_chain_bits_controlled
    from passive_info.embodied import SIM_DT, _ground_gains, _sim_rig
    from passive_info.sim_info import STREAM_BUFFER, STREAM_REFRESH
    from passive_info.systems import K_I

    ks = None if args.design in ("prop", "none", "None") else float(args.design)
    steps, band, batch = args.steps, args.band_hz, args.batch
    label = "proprioceptive" if ks is None else f"ks{ks:.6g}"
    if band != 100.0:
        label = f"{label}_b{band:.0f}"      # band is part of the identity, not a footnote

    # The ground is recorded for the same reason the band is; `run_passive.main` carries the
    # full note. 219.61 is the locked ground, 132.72 the pre-fix series-elastic one.
    m, d, leg = _sim_rig(ks, SIM_DT, 0.05, "menagerie", "spring", False, vertical="fixed")
    ground_b_i = float(_ground_gains(m, d, leg, K_I)[0])
    print(f"  ground: b_i = {ground_b_i:.2f}  (locked transmission; all designs equal)",
          flush=True)
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"controlled: design {label}, band {band:.0f} Hz, {steps} steps, batch {batch} "
          f"(ceiling {np.log2(batch):.1f} bits/band), device {dev}", flush=True)

    t0 = time.time()
    r = sim_chain_bits_controlled(ks, band_hz=band, steps=steps, batch=batch,
                                  device=dev, verbose=True)
    el = time.time() - t0

    j, a, c = r["joint"].sum(), r["action"].sum(), r["cmi"].sum()
    print(f"\n{label}: joint {j:.3f}  action {a:.3f}  conditional {c:.3f} bits "
          f"({len(r['joint'])} bands, loop {'CLOSED' if r['closed_loop'] else 'OPEN'}, "
          f"{el:.0f}s)")
    if not r["closed_loop"]:
        print("  loop open -> I(a;u)=0 structurally, so conditional == joint by construction")

    RESULTS.mkdir(exist_ok=True)
    out = _unused_path(RESULTS / f"ctrl_{label}")
    out.write_text(json.dumps(
        dict(ks=ks, label=label, steps=steps, batch=batch, band_hz=band,
             ground_b_i=ground_b_i,
             # Recorded because they changed mid-campaign, and a result that does not state
             # them cannot be told from one measured at the old settings. `reuse` is the
             # calibrated quantity; keep it at 4.
             buffer=STREAM_BUFFER, refresh_every=STREAM_REFRESH,
             reuse=STREAM_REFRESH * batch / STREAM_BUFFER,
             closed_loop=bool(r["closed_loop"]),
             joint=r["joint"].tolist(), action=r["action"].tolist(), cmi=r["cmi"].tolist(),
             joint_total=float(j), action_total=float(a), cmi_total=float(c),
             dropped=r["dropped"], seconds=el,
             job=os.environ.get("SLURM_JOB_ID"), node=os.environ.get("SLURMD_NODENAME")),
        indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
