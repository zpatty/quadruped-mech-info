"""Labels for the talk cut: design names, a bit count, the spectrum axis and the scale factors.

Every value comes from the `Design` that the frame uses. The count is `bits_t`, the bars are
`spec_t`, and the white ticks are the natural frequencies of the mechanism.
"""
from __future__ import annotations

import numpy as np

from passive_info.anim import style
from passive_info.anim.timeline import sample

TITLE = "Information about a contact force in the motion of a passive leg"
SUBTITLE = ("We apply the same force to a rigid leg and to a series-elastic leg,",
            "and compute how much information about the force is in the motion of each leg.")

DIM = (0.62, 0.66, 0.74)
FAINT = (0.36, 0.39, 0.45)
BRIGHT = (0.93, 0.94, 0.97)


def _pow10(ks: float) -> str:
    e = int(np.round(np.log10(ks)))
    if abs(10 ** e - ks) / ks < 1e-6:
        sup = str(e).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
        return f"10{sup}"
    return f"{ks:g}"


def design_name(d, kind: str) -> tuple[str, str]:
    unit = "N·m/rad" if kind == "go2" else "N·m/rad"
    if d.ks is None:
        return "Rigid transmission", "the rotor is geared directly to the joint"
    return "Series elastic", f"a spring of {_pow10(d.ks)} {unit} connects the rotor to the joint"


def spectrum_strip(img, x0, x1, y_base, height, d, smp, hues, px, show_axis=True):
    """Bars of bits per force mode on the same two-part scale as the ring, axis labelled."""
    K = len(d.mode_freqs)
    df = d.mode_freqs[1] - d.mode_freqs[0]
    fmax = d.mode_freqs[-1] + 0.5 * df
    bw = (x1 - x0) / K
    for k in range(K):
        xa, xb = x0 + k * bw + 0.18 * bw, x0 + (k + 1) * bw - 0.18 * bw
        hgt = height * style.bar_length(smp.spec[k])
        if hgt > 0.5:
            style.fill_poly(img, [(xa, y_base), (xa, y_base - hgt), (xb, y_base - hgt),
                                  (xb, y_base)], hues[k] * 0.9)
    if not show_axis:
        return
    ink = np.array(FAINT) * 0.5
    # bits axis: linear to the knee, then stretched -- marked with a break
    for b in (0, 3, 4, 5):
        y = y_base - height * style.bar_length(b)
        style.line(img, (x0 - 10 * px, y), (x1, y), ink * (0.5 if b else 1.0), 1.0 * px)
        style.text(img, (x0 - 22 * px, y), f"{b}", 13 * px, FAINT, "regular", "rm")
    yk = y_base - height * style.BAR_KNEE_FRAC
    for dy in (-3, 3):
        style.line(img, (x0 - 14 * px, yk + dy * px + 3 * px), (x0 - 2 * px, yk + dy * px - 3 * px),
                   np.array(BRIGHT) * 0.5, 1.0 * px)
    for f in (0, 50, 100, 150, 200):
        x = x0 + (x1 - x0) * f / fmax
        style.line(img, (x, y_base), (x, y_base + 6 * px), ink, px)
        style.text(img, (x, y_base + 10 * px), f"{f}", 13 * px, FAINT, "regular", "ma")
    style.text(img, (x1, y_base + 26 * px), "frequency of the force, Hz", 13 * px, FAINT, "regular", "ra")
    for f in d.natural_hz:
        if f <= fmax:
            x = x0 + (x1 - x0) * f / fmax
            style.line(img, (x, y_base - height - 4 * px), (x, y_base - height - 16 * px),
                       np.array(BRIGHT) * 0.7, 1.5 * px)
    style.text(img, (x0, y_base - height - 24 * px),
               "bits for each frequency of the force   |   white ticks: natural frequencies of the leg",
               14 * px, FAINT, "regular", "ls")


def talk_overlay(img, job, sc, pair, i, card_alpha):
    """Draw the talk-cut labels over a composed side-by-side frame, in place."""
    H, W = img.shape[:2]
    px = H / 1080.0
    pw = W // 2
    K = len(sc.designs[0].mode_freqs)
    hues = style.mode_hues(K)

    style.line(img, (pw, 40 * px), (pw, H - 40 * px), style.INK * 0.06, 1.0 * px)

    for side, shot in enumerate(pair):
        d = sc.designs[shot.design]
        smp = sample(d, shot)
        x0 = side * pw
        name, sub = design_name(d, job.kind)
        style.text(img, (x0 + 56 * px, 58 * px), name, 30 * px, BRIGHT, "medium")
        style.text(img, (x0 + 56 * px, 100 * px), sub, 17 * px, DIM, "regular")
        style.text(img, (x0 + pw - 56 * px, 58 * px), f"{smp.bits:5.1f}", 40 * px, BRIGHT,
                   "light", "ra")
        style.text(img, (x0 + pw - 56 * px, 108 * px), "bits of information about the force", 15 * px,
                   DIM, "regular", "ra")
        style.text(img, (x0 + pw - 56 * px, 128 * px), "in the motion of the leg so far",
                   15 * px, FAINT, "regular", "ra")
        spectrum_strip(img, x0 + 84 * px, x0 + pw - 60 * px, H - 92 * px, 110 * px, d, smp,
                       hues, px)

    smp0 = sample(sc.designs[pair[0].design], pair[0])
    note = (f"{smp0.t_ms:5.1f} ms of an 85 ms contact   ·   motion scaled up {job.gain:g} times   ·   "
            f"time slowed down {job.slow:g} times   ·   both legs receive the same force")
    style.text(img, (W / 2, H - 22 * px), note, 14 * px, FAINT, "regular", "ms")

    if card_alpha > 0:
        img *= (1.0 - 0.94 * card_alpha)
        style.text(img, (W / 2, H / 2 - 40 * px), TITLE, 46 * px, BRIGHT, "light", "ms",
                   card_alpha)
        for k, line in enumerate(SUBTITLE):
            style.text(img, (W / 2, H / 2 + (30 + 34 * k) * px), line, 22 * px, DIM, "regular",
                       "ms", card_alpha)


# ---------------------------------------------------------------------------------------
# The loop cut: a small amount of text, so that a viewer can tell what the marks show
# ---------------------------------------------------------------------------------------

LEGEND = ("Glow: energy stored in each part of the leg",
          "Dots: energy that moves between the parts",
          "White ticks: natural frequencies of the leg")

#: Where each callout's text sits relative to the point it labels, in pixels at 1080p:
#: (dx, dy, anchor). One table per style, because the two layouts differ.
CALLOUTS = {
    "go2": dict(contact=(-210, 70, "rm"), rotor=(-150, 40, "rm"), spring=(-120, -80, "rm"),
                ring=(190, -40, "lm")),
    "proto": dict(contact=(150, 70, "lm"), rotor=(-90, 90, "rm"), spring=(-170, -10, "rm"),
                  ring=(0, -30, "ms")),
}


def _callout(img, p, d, s, px, alpha, anchor):
    """Text at `p + d`, with a thin line from the text to the point `p`."""
    p = np.asarray(p, float)
    t = p + np.asarray(d, float) * px
    gap = 8 * px
    start = t + np.array([gap if anchor[0] == "l" else -gap if anchor[0] == "r" else 0.0,
                          -gap if anchor[1] == "s" else gap if anchor[1] == "a" else 0.0])
    if anchor[0] == "r":
        start[0] += 2 * gap
    elif anchor[0] == "l":
        start[0] -= 2 * gap
    style.line(img, start, p, np.array(DIM) * 0.12 * alpha, 1.0 * px)
    style.disc(img, p, 2.0 * px, np.array(DIM) * 0.25 * alpha)
    style.text(img, t, s, 16 * px, DIM, "regular", anchor, alpha)


def loop_overlay(img, job, sc, shot, screen: dict):
    """Draw the loop-cut text over a composed frame, in place."""
    H, W = img.shape[:2]
    px = H / 1080.0
    d = sc.designs[shot.design]
    smp = sample(d, shot)
    f = shot.seg_frac
    fade = float(np.clip(min(f / 0.05, (1.0 - f) / 0.05), 0.0, 1.0))

    name, sub = design_name(d, job.kind)
    if d.ks is not None:
        name = "Series-elastic transmission"
    style.text(img, (56 * px, 60 * px), name, 28 * px, BRIGHT, "medium", "la", fade)
    style.text(img, (56 * px, 100 * px), sub, 17 * px, DIM, "regular", "la", fade)

    on = fade * float(shot.t >= 0)
    style.text(img, (W - 56 * px, 56 * px), f"{smp.bits:.0f} bits", 34 * px, BRIGHT, "light",
               "ra", on)
    style.text(img, (W - 56 * px, 104 * px), "information about the force", 16 * px, DIM,
               "regular", "ra", on)
    style.text(img, (W - 56 * px, 126 * px), "in the motion of the leg so far", 16 * px, FAINT,
               "regular", "ra", on)

    for k, line in enumerate(LEGEND):
        style.text(img, (56 * px, H - (90 - 26 * k) * px), line, 16 * px, DIM, "regular", "la")
    style.text(img, (W - 56 * px, H - 64 * px), f"Time slowed down {job.slow:g} times", 15 * px,
               FAINT, "regular", "ra")
    style.text(img, (W - 56 * px, H - 38 * px), f"Motion scaled up {job.gain:g} times", 15 * px,
               FAINT, "regular", "ra")

    if not screen:
        return
    co = CALLOUTS["go2" if job.kind == "go2" else "proto"]
    _callout(img, screen["contact"], co["contact"][:2], "contact force", px, 1.0, co["contact"][2])
    _callout(img, screen["rotor"], co["rotor"][:2], "motor rotor", px, 1.0, co["rotor"][2])
    if d.ks is not None:
        _callout(img, screen["spring"], co["spring"][:2], "series spring", px, fade,
                 co["spring"][2])
    rx, ry, ra = co["ring"]
    caption = "information about the force at each frequency"
    if screen.get("ring_bar") is not None:
        _callout(img, screen["ring_bar"], (rx, ry), caption, px, 1.0, ra)
    else:
        m = np.asarray(screen["ring_mid"], float)
        style.text(img, (m[0] + rx * px, m[1] + ry * px), caption, 16 * px, DIM, "regular", ra)
    for key, word, dx in (("ring_low", "low", -14), ("ring_high", "high", 14)):
        q = np.asarray(screen[key], float)
        style.text(img, (q[0] + dx * px, q[1]), word, 15 * px, FAINT, "regular",
                   "rm" if dx < 0 else "lm")


# ---------------------------------------------------------------------------------------
# The single-design loop: what the glow means, how many bits, and the force plot
# ---------------------------------------------------------------------------------------

INFO_TITLE = "Information about a contact force in the motion of a robot leg"
INFO_LEGEND = ("Glow: information about the force in the motion of each part",
               "Dots: new information, moving from the contact into the leg")


def _force_plot(img, x0, y0, w, h, d, smp, px, alpha):
    """The applied force so far, and the band of forces consistent with the motion so far."""
    C = style.INFO
    n_w = d.n_contact
    f = d.force[:n_w]
    lo, hi = -100.0, 220.0
    X = lambda i: x0 + w * i / (n_w - 1)
    Y = lambda v: y0 + h * (hi - np.clip(v, lo, hi)) / (hi - lo)
    ink = np.array(FAINT) * 0.5 * alpha

    style.line(img, (x0, Y(0)), (x0 + w, Y(0)), ink, 1.0 * px)
    for v in (0, 100, 200):
        style.text(img, (x0 - 10 * px, Y(v)), f"{v}", 13 * px, FAINT, "regular", "rm", alpha)
    style.text(img, (x0 - 10 * px, y0 - 16 * px), "N", 13 * px, FAINT, "regular", "rm", alpha)
    for ms in (0, 20, 40, 60, 80):
        x = x0 + w * ms / 85.0
        style.line(img, (x, y0 + h), (x, y0 + h + 5 * px), ink, 1.0 * px)
        style.text(img, (x, y0 + h + 9 * px), f"{ms}", 13 * px, FAINT, "regular", "ma", alpha)
    style.text(img, (x0 + w, y0 + h + 26 * px), "time into the contact, ms", 13 * px, FAINT,
               "regular", "ra", alpha)

    # the band, over the whole window: wide ahead of the current time, narrow behind it
    i = np.arange(n_w)
    top = smp.post_mean + 2 * smp.post_sd
    bot = smp.post_mean - 2 * smp.post_sd
    poly = ([(X(k), Y(top[k])) for k in i] + [(X(k), Y(bot[k])) for k in i[::-1]])
    style.fill_poly(img, poly, C * 0.22 * alpha * smp.info_level)
    style.polyline(img, [(X(k), Y(top[k])) for k in i], C * 0.5 * alpha * smp.info_level, 1.0 * px)
    style.polyline(img, [(X(k), Y(bot[k])) for k in i], C * 0.5 * alpha * smp.info_level, 1.0 * px)

    # the applied force, up to now
    now = int(round(smp.t_ms / 85.0 * (n_w - 1)))
    if now >= 1 and smp.info_level > 0:
        style.polyline(img, [(X(k), Y(f[k])) for k in range(now + 1)],
                       np.array(BRIGHT) * 0.9 * alpha * smp.info_level, 1.8 * px)
    xc = X(now)
    style.line(img, (xc, y0 - 4 * px), (xc, y0 + h), np.array(BRIGHT) * 0.25 * alpha, 1.0 * px)

    # key
    ky = y0 - 30 * px
    style.line(img, (x0 + 10 * px, ky), (x0 + 34 * px, ky), np.array(BRIGHT) * 0.9 * alpha, 1.8 * px)
    style.text(img, (x0 + 42 * px, ky), "force applied at the foot", 14 * px, DIM, "regular", "lm",
               alpha)
    ky2 = ky - 24 * px
    style.fill_poly(img, [(x0 + 10 * px, ky2 - 5 * px), (x0 + 34 * px, ky2 - 5 * px),
                          (x0 + 34 * px, ky2 + 5 * px), (x0 + 10 * px, ky2 + 5 * px)],
                    C * 0.35 * alpha)
    style.text(img, (x0 + 42 * px, ky2), "forces consistent with the motion of the leg so far",
               14 * px, DIM, "regular", "lm", alpha)


def info_overlay(img, job, sc, shot, screen: dict):
    """Draw the text and the force plot of the single-design loop, in place."""
    H, W = img.shape[:2]
    px = H / 1080.0
    d = sc.designs[shot.design]
    smp = sample(d, shot)

    style.text(img, (56 * px, 60 * px), INFO_TITLE, 26 * px, BRIGHT, "medium", "la")
    style.text(img, (56 * px, 98 * px),
               "Go2 front-right leg with its motors off, rigid transmission", 16 * px, DIM,
               "regular", "la")

    on = smp.info_level * float(shot.t >= 0)
    style.text(img, (W - 56 * px, 56 * px), f"{smp.bits:.0f} bits", 34 * px, BRIGHT, "light",
               "ra", on)
    style.text(img, (W - 56 * px, 104 * px), "information about the force", 16 * px, DIM,
               "regular", "ra", on)
    style.text(img, (W - 56 * px, 126 * px), "in the motion of the whole leg so far", 16 * px,
               FAINT, "regular", "ra", on)

    for k, line in enumerate(INFO_LEGEND):
        style.text(img, (56 * px, H - (64 - 26 * k) * px), line, 16 * px, DIM, "regular", "la")
    style.text(img, (W / 2, H - 22 * px),
                f"Motion scaled up {job.gain:g} times   ·   time slowed down {job.slow:g} times",
                14 * px, FAINT, "regular", "ms")

    _force_plot(img, W - 640 * px, H - 250 * px, 580 * px, 170 * px, d, smp, px, 1.0)

    if not screen:
        return
    idx = {e: i for i, e in enumerate(d.elements)}
    pb = lambda e: smp.part_bits[idx[e]]
    _callout(img, screen["contact"], (-210, 70), f"foot contact   {pb('ground'):.0f} bits", px,
             1.0, "rm")
    _callout(img, screen["rotor"], (-150, 40), f"knee motor rotor   {pb('rotor_calf'):.0f} bits",
             px, 1.0, "rm")
    _callout(img, screen["calf"], (170, 10), f"calf   {pb('calf'):.0f} bits", px, 1.0, "lm")
