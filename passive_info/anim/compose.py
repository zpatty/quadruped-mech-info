"""Cuts, rendering workers and encoding.

Each style has two cuts:

  loop   no text, for a website. The designs play in turn, starting with the rigid one, and
         the display returns to rest between them, so the last frame leads into the first.
  talk   for a slide. The rigid design and a series-elastic design play side by side under the
         same force. Labels give the design, the time into the contact, the bits so far and
         the frequency axis of the spectrum, and state both scale factors.

Worker processes render the frames, each with its own MuJoCo context, and send raw RGB to
ffmpeg, so no per-frame files are written.
"""
from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
from dataclasses import dataclass, field

import numpy as np

from passive_info.anim import data, style
from passive_info.anim.timeline import (Shot, energy_ref, info_track, particle_track, sample,
                                        segment)
from passive_info.information import T_WINDOW

OUT = pathlib.Path(__file__).resolve().parent / "out"


@dataclass
class Job:
    """Everything a worker needs to draw any frame of one cut."""
    kind: str                        #: "go2" or "proto"
    cut: str                         #: "loop" or "talk"
    width: int
    height: int
    fps: float
    slow: float
    gain: float
    ks: tuple
    shots: list = field(default_factory=list)            # loop: [Shot]; talk: [(Shot, Shot)]
    tracks: list = field(default_factory=list)           # per panel: [FrameParticles]
    cards: list = field(default_factory=list)            # talk: per frame title-card alpha
    text: bool = True                                    # loop: draw the short labels
    force: str = "prior"                                 # data.scenario(force=...)


# ---------------------------------------------------------------------------------------
# Building cuts
# ---------------------------------------------------------------------------------------

def build_loop(kind: str, ks=data.DEFAULT_KS, width=1920, height=1080, fps=30.0,
               slow=55.0, gain=250.0, text: bool = True) -> Job:
    sc = data.scenario(kind, ks)
    shots = []
    for i in range(len(sc.designs)):
        shots += segment(i, fps, slow, pre=0.4)
    # run the particle update over two laps and keep the second one, so that the particle
    # state at the start of the loop matches the state at the end
    track = particle_track(sc, shots + shots, fps)[len(shots):]
    return Job(kind, "loop", width, height, fps, slow, gain, tuple(ks), shots, [track],
               text=text)


#: Camera for the square information loop: close on the front-right leg.
INFO_VIEW = dict(lookat=(0.15, -0.105, 0.105), distance=0.72, azimuth=132.0, elevation=-12.0)


def build_info_loop(width=1080, height=1080, fps=30.0, slow=80.0, gain=250.0,
                    start_frac: float = 0.9) -> Job:
    """The single-design loop: the rigid Go2 leg, with glow and dots for information.

    Frame 0 is the poster image that the website shows before the video plays, so the loop
    is rotated to start at `start_frac` of the contact, when most of the information is in.
    The rotation does not break the loop, because the frames are drawn cyclically.
    """
    sc = data.scenario("go2", (None,), force="contact")
    shots = segment(0, fps, slow, pre=0.6)
    track = info_track(sc, shots + shots, fps)[len(shots):]
    t0 = start_frac * T_WINDOW
    k = min((i for i, s in enumerate(shots) if s.t >= 0), key=lambda i: abs(shots[i].t - t0))
    shots, track = shots[k:] + shots[:k], track[k:] + track[:k]
    return Job("go2", "info", width, height, fps, slow, gain, (None,), shots, [track],
               force="contact")


def build_talk(kind: str, ks_pair=(1.0e4, 1.0e3), width=1920, height=1080, fps=30.0,
               slow=90.0, gain=250.0, title_s=3.5) -> Job:
    """Title card, then the rigid design beside each SEA in turn, with a pause on the final spectra."""
    ks = (None,) + tuple(ks_pair)
    sc = data.scenario(kind, ks)
    pairs, cards = [], []
    n_title = int(round(title_s * fps))
    calm = Shot(0, -1.0, 1.0, 0.0, 0.0)
    for k in range(n_title):
        pairs.append((calm, Shot(1, -1.0, 1.0, 0.0, 0.0)))
        x = k / n_title
        cards.append(float(min(1.0, 4 * x, 4 * (1 - x))))
    for j in range(1, len(ks)):
        left = segment(0, fps, slow, pre=0.6, hold=2.5)
        right = segment(j, fps, slow, pre=0.6, hold=2.5)
        pairs += list(zip(left, right))
        cards += [0.0] * len(left)
    tracks = []
    for side in (0, 1):
        shots = [p[side] for p in pairs]
        tracks.append(particle_track(sc, shots, fps, seed=side))
    return Job(kind, "talk", width, height, fps, slow, gain, ks, pairs, tracks, cards)


# ---------------------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------------------

_W: dict = {}


def _init(job: Job):
    from passive_info.anim.particles import graph_distances
    _W["job"] = job
    sc = data.scenario(job.kind, job.ks, force=job.force)
    _W["sc"] = sc
    _W["e_ref"] = energy_ref(sc)
    _W["b_ref"] = float(max(d.part_bits.max() for d in sc.designs))
    d0 = sc.designs[0]
    idx = {e: i for i, e in enumerate(d0.elements)}
    _W["nxt"] = graph_distances(len(d0.elements), [(idx[a], idx[b]) for a, b in d0.edges])[1]
    pw = job.width if job.cut in ("loop", "info") else job.width // 2
    if job.kind == "go2":
        from passive_info.anim.go2 import Go2Stage, View
        view = (View(**INFO_VIEW) if job.cut == "info" else View() if job.cut == "loop" else View(lookat=(0.09, -0.10, 0.07), distance=1.12,
                                                     azimuth=128.0, elevation=-15.0))
        _W["stage"] = Go2Stage(pw, job.height, job.gain, view)
    else:
        from passive_info.anim.flow import FlowStage
        _W["stage"] = FlowStage(pw, job.height, job.gain)


def _panel(i: int, side: int, shot: Shot, prev: Shot) -> np.ndarray:
    job, sc = _W["job"], _W["sc"]
    d = sc.designs[shot.design]
    smp, smp_prev = sample(d, shot), sample(sc.designs[prev.design], prev)
    track = job.tracks[side]
    parts, parts_prev = track[i], track[i - 1]
    t_anim = i / job.fps
    if job.cut == "info":
        from passive_info.anim.go2 import draw_info
        return draw_info(_W["stage"], d, smp, parts, parts_prev, smp_prev, _W["b_ref"], t_anim,
                         _W["nxt"])
    if job.kind == "go2":
        from passive_info.anim.go2 import draw
        return draw(_W["stage"], d, smp, parts, parts_prev, smp_prev, _W["e_ref"], t_anim,
                    _W["nxt"], d.ks is None)
    from passive_info.anim.flow import draw
    return draw(_W["stage"], d, smp, parts, parts_prev, smp_prev, _W["e_ref"], t_anim,
                _W["nxt"], d.ks is None)


def render_frame(i: int) -> np.ndarray:
    job = _W["job"]
    n = len(job.shots)
    if job.cut == "info":
        img = _panel(i, 0, job.shots[i], job.shots[(i - 1) % n])
        from passive_info.anim.labels import info_overlay
        info_overlay(img, job, _W["sc"], job.shots[i], getattr(_W["stage"], "screen", {}))
        return style.finish(img, seed=i)
    if job.cut == "loop":
        img = _panel(i, 0, job.shots[i], job.shots[(i - 1) % n])
        if job.text:
            from passive_info.anim.labels import loop_overlay
            loop_overlay(img, job, _W["sc"], job.shots[i], getattr(_W["stage"], "screen", {}))
        return style.finish(img, seed=i)
    sh, prev = job.shots[i], job.shots[max(i - 1, 0)]
    halves = [_panel(i, s, sh[s], prev[s]) for s in (0, 1)]
    img = np.concatenate(halves, axis=1)
    from passive_info.anim.labels import talk_overlay
    talk_overlay(img, job, _W["sc"], sh, i, job.cards[i] if job.cards else 0.0)
    return style.finish(img, seed=i)


# ---------------------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------------------

def _ffmpeg(path: pathlib.Path, w: int, h: int, fps: float, codec: str):
    args = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", f"{fps:g}", "-i", "-"]
    if codec == "mp4":
        args += ["-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart"]
    else:
        args += ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "30", "-row-mt", "1",
                 "-pix_fmt", "yuv420p"]
    return subprocess.Popen(args + [str(path)], stdin=subprocess.PIPE)


def render(job: Job, name: str, workers: int | None = None, webm: bool = True,
           frames: list[int] | None = None, out_dir: pathlib.Path = OUT) -> list[pathlib.Path]:
    """Render `job` to `<name>.mp4` (and `.webm`), or only the listed `frames` to PNGs."""
    import multiprocessing as mp

    out_dir.mkdir(parents=True, exist_ok=True)
    workers = workers or max(1, min(4, os.cpu_count() or 1))
    idx = list(range(len(job.shots))) if frames is None else list(frames)
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init, initargs=(job,)) as pool:
        if frames is not None:
            from PIL import Image
            paths = []
            for i, img in zip(idx, pool.imap(render_frame, idx)):
                p = out_dir / f"{name}_{i:04d}.png"
                Image.fromarray(img).save(p)
                paths.append(p)
            return paths
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg is not on the PATH")
        paths = [out_dir / f"{name}.mp4"] + ([out_dir / f"{name}.webm"] if webm else [])
        procs = [_ffmpeg(p, job.width, job.height, job.fps, p.suffix[1:]) for p in paths]
        for k, img in enumerate(pool.imap(render_frame, idx, chunksize=2)):
            buf = np.ascontiguousarray(img).tobytes()
            for pr in procs:
                pr.stdin.write(buf)
            if k % 60 == 0:
                print(f"  {name}: frame {k}/{len(idx)}", flush=True)
        for pr in procs:
            pr.stdin.close()
            pr.wait()
        return paths


def contact_frames(job: Job, design: int, fracs=(0.5,)) -> list[int]:
    """Frame indices at fractions of one design's contact window (loop cuts)."""
    out = []
    for f in fracs:
        t = f * T_WINDOW
        best = min((i for i, s in enumerate(job.shots) if s.design == design and s.t >= 0),
                   key=lambda i: abs(job.shots[i].t - t))
        out.append(best)
    return out
