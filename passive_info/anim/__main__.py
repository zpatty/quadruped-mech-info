"""Render the animations.

    uv run python -m passive_info.anim go2 --cut loop        # rigid Go2, information glow
    uv run python -m passive_info.anim go2 --cut compare     # rigid and two SEAs in turn
    uv run python -m passive_info.anim go2 --cut talk
    uv run python -m passive_info.anim flow --cut loop --preview
    uv run python -m passive_info.anim go2 --stills 0.55       # PNGs at 55% of each contact

Output goes to passive_info/anim/out/.
"""
from __future__ import annotations

import argparse
import pathlib
import time

from passive_info.anim import compose, data


def _ks(s: str):
    return None if s.lower() in ("rigid", "none", "prop") else float(s)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m passive_info.anim",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("style", choices=("go2", "flow"))
    ap.add_argument("--cut", choices=("loop", "compare", "talk"), default="loop")
    ap.add_argument("--ks", type=_ks, nargs="+", default=None,
                    help="designs: 'rigid' and/or stiffnesses. Loop default: rigid 1e4 1e3; "
                         "talk compares rigid against each listed stiffness (default 1e4 1e3)")
    ap.add_argument("--preview", action="store_true", help="960x540 for a quick look")
    ap.add_argument("--size", type=int, nargs=2, default=(1920, 1080), metavar=("W", "H"),
                    help="frame size (default 1920x1080; 1080x1080 for --cut loop)")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--slow", type=float, default=None,
                    help="slow-motion factor (default 55 loop, 90 talk)")
    ap.add_argument("--gain", type=float, default=250.0, help="display motion gain")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--no-webm", action="store_true")
    ap.add_argument("--no-text", action="store_true", help="compare cut without any labels")
    ap.add_argument("--stills", type=float, nargs="*", default=None,
                    help="render PNGs at these fractions of each contact instead of a video")
    ap.add_argument("--out", type=pathlib.Path, default=compose.OUT)
    a = ap.parse_args(argv)

    kind = "go2" if a.style == "go2" else "proto"
    w, h = (960, 540) if a.preview else a.size
    if a.cut == "loop":
        if a.style != "go2":
            ap.error("the single-design loop exists for the go2 style; use --cut compare")
        # square by default, for the round frame on the website
        w, h = (540, 540) if a.preview else (a.size if a.size != (1920, 1080) else (1080, 1080))
        job = compose.build_info_loop(w, h, a.fps, a.slow or 80.0, a.gain)
    elif a.cut == "compare":
        ks = tuple(a.ks) if a.ks else data.DEFAULT_KS
        job = compose.build_loop(kind, ks, w, h, a.fps, a.slow or 55.0, a.gain,
                                 text=not a.no_text)
    else:
        pair = tuple(k for k in (a.ks or (1.0e4, 1.0e3)) if k is not None)
        job = compose.build_talk(kind, pair, w, h, a.fps, a.slow or 90.0, a.gain)
    name = (f"{a.style}_{a.cut}" + ("_notext" if a.cut == "compare" and a.no_text else "")
            + ("_preview" if a.preview else ""))

    t0 = time.time()
    if a.stills is not None:
        fr = a.stills or [0.55]
        idx = []
        if a.cut in ("loop", "compare"):
            for i in range(len(job.ks)):
                idx += compose.contact_frames(job, i, fr)
        else:
            n = len(job.shots)
            idx = [int(f * (n - 1)) for f in fr]
        paths = compose.render(job, name, a.workers, frames=idx, out_dir=a.out)
    else:
        print(f"{name}: {len(job.shots)} frames at {w}x{h}, {len(job.shots) / a.fps:.1f} s")
        paths = compose.render(job, name, a.workers, webm=not a.no_webm, out_dir=a.out)
    print(f"done in {time.time() - t0:.0f} s")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
