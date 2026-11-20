"""The design and model time that each video frame shows, and the particle states.

A timeline is a list of `Shot`s. Each design has a short pre-roll with no force, then the
contact window slowed down by `slow`, then the tail with no force. During the tail the display
returns to rest, so that one design follows another without a cut, and the end of a loop
matches its start.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from passive_info.anim.data import T_TAIL, Design, Scenario
from passive_info.anim.particles import FrameParticles, Transport
from passive_info.anim import style
from passive_info.information import FS, T_WINDOW


@dataclass(frozen=True)
class Shot:
    design: int          #: index into the scenario's designs
    t: float             #: model time, s (negative during the pre-roll)
    beta: float          #: display amplitude, 1 in contact, easing to 0 in the tail
    spec_level: float    #: spectrum visibility, 1 then fading in the tail
    seg_frac: float      #: 0..1 through this design's segment (for titles and fades)


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def segment(design: int, fps: float, slow: float, pre: float = 0.5, hold: float = 0.0,
            t_tail: float = T_TAIL) -> list[Shot]:
    """One design: pre-roll, contact, optional hold at full spectrum, then the tail."""
    shots = []
    n_pre = int(round(pre * fps))
    n_con = int(round(T_WINDOW * slow * fps))
    n_hold = int(round(hold * fps))
    n_tail = int(round(t_tail * slow * fps))
    total = n_pre + n_con + n_hold + n_tail
    i = 0
    for k in range(n_pre):
        shots.append(Shot(design, -pre + k / fps, 1.0, 0.0, i / total)); i += 1
    for k in range(n_con):
        shots.append(Shot(design, T_WINDOW * k / n_con, 1.0, 1.0, i / total)); i += 1
    for k in range(n_hold):
        shots.append(Shot(design, T_WINDOW, 1.0, 1.0, i / total)); i += 1
    for k in range(n_tail):
        x = (k + 1) / n_tail
        shots.append(Shot(design, T_WINDOW + t_tail * x, float(1.0 - _smooth(2 * x - 0.6)),
                          float(1.0 - _smooth(1.6 * x - 0.5)), i / total)); i += 1
    return shots


@dataclass
class Sample:
    """A design's state at one model time, already scaled for display by `beta`."""
    joint_dev: np.ndarray
    rotor_dev: np.ndarray
    energy: np.ndarray
    hue_w: np.ndarray
    spec: np.ndarray
    bits: float
    force: float
    t_ms: float
    part_bits: np.ndarray           #: bits in each element, scaled by the shot's spec_level
    post_mean: np.ndarray           #: centre of the band of consistent forces over the window
    post_sd: np.ndarray             #: its standard deviation
    info_level: float               #: visibility of the information marks, 0..1


def sample(d: Design, shot: Shot) -> Sample:
    s = np.clip(shot.t * FS, 0.0, len(d.t) - 1.0)
    i0 = int(np.floor(s))
    i1 = min(i0 + 1, len(d.t) - 1)
    f = s - i0
    lerp = lambda A: (1 - f) * A[i0] + f * A[i1]
    b, b2 = shot.beta, shot.beta ** 2
    on = 1.0 if shot.t >= 0 else 0.0
    return Sample(joint_dev=b * lerp(d.joint_q) * on, rotor_dev=b * lerp(d.rotor_q) * on,
                  energy=b2 * lerp(d.energy) * on, hue_w=lerp(d.hue_w),
                  spec=lerp(d.spec_t) * shot.spec_level * on,
                  bits=float(lerp(d.bits_t)) * on, force=float(lerp(d.force)) * on,
                  t_ms=max(0.0, min(shot.t, T_WINDOW)) * 1e3,
                  part_bits=lerp(d.part_bits) * shot.spec_level * on,
                  post_mean=lerp(d.post_mean) if on else d.post_mean[0],
                  post_sd=lerp(d.post_sd) if on else d.post_sd[0],
                  info_level=float(shot.spec_level))


def particle_track(sc: Scenario, shots: list[Shot], fps: float, n_quanta: int = 520,
                   entry: str = "ground", seed: int = 0) -> list[FrameParticles]:
    """Run the particle update over a whole timeline and return one state for each shot."""
    d0 = sc.designs[0]
    peak = max(float(d.energy.sum(1).max()) for d in sc.designs)
    K = d0.hue_w.shape[-1]
    tr = Transport(d0.elements, d0.edges, quantum=peak / n_quanta, entry=entry,
                   hues=style.mode_hues(K), seed=seed)
    out = []
    for sh in shots:
        smp = sample(sc.designs[sh.design], sh)
        out.append(tr.step(smp.energy, smp.hue_w, 1.0 / fps))
    return out


def energy_ref(sc: Scenario) -> float:
    """The largest energy of any element in any design, used to scale every element.

    We use one scale for all elements, not one for each, so that an element with little
    energy stays dim.
    """
    return float(max(d.energy.max() for d in sc.designs))


def info_track(sc: Scenario, shots: list[Shot], fps: float, n_dots: int = 320,
               color=None, seed: int = 0) -> list[FrameParticles]:
    """Dots for new information: one dot per fixed number of bits in an element.

    Bits never decrease during the contact, so every dot starts at the contact and moves to
    the element that gained it. The dots fade out only when the display fades at the end.
    """
    d0 = sc.designs[0]
    peak = max(float(d.part_bits.sum(1).max()) for d in sc.designs)
    color = style.INFO if color is None else np.asarray(color)
    tr = Transport(d0.elements, d0.edges, quantum=peak / n_dots, entry="ground",
                   hues=color[None, :], seed=seed, max_birth_per_frame=40, fade_time=0.9)
    ones = np.ones((len(d0.elements), 1))
    out = []
    for sh in shots:
        smp = sample(sc.designs[sh.design], sh)
        out.append(tr.step(smp.part_bits, ones, 1.0 / fps))
    return out
