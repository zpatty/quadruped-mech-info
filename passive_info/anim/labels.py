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


def _callout(img, p, d, s, px, alpha, anchor, size=16, line_level=0.12, color=DIM):
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
    style.line(img, start, p, np.array(DIM) * line_level * alpha, max(1.0, size / 16) * px)
    style.disc(img, p, 2.0 * max(1.0, size / 16) * px, np.array(DIM) * 0.25 * alpha)
    style.text(img, t, s, size * px, color, "regular", anchor, alpha)


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
# The single-design loop, square, for a round frame on the website
#
# The website shows this video in a round shape about 300 px across, so the frame is scaled
# down to about 0.28 and its corners are cut off. Everything here therefore sits inside a
# circle of radius 0.35 of the frame, and every label is large.
# ---------------------------------------------------------------------------------------

#: Caption text for the website, which is not drawn in the frame.
INFO_TITLE = "Information about a contact force in the motion of a robot leg"
INFO_LEGEND = ("Glow: information about the force in the motion of each part",
               "Dots: new information, moving from the contact into the leg")

#: The lines under the counter. The last one is brighter: the information is in the mechanism.
INFO_CAPTION = ("of information", "about the force", "in the mechanism")

#: Layout in pixels for a 1080 x 1080 frame, relative to the frame centre. Scaled with size.
INFO_LAYOUT = dict(
    safe_r=378,             # the circle that everything must stay inside
    count=(40, -40),        # left end of the large counter's baseline
    count_size=120, unit_size=46, caption_size=36,
    part_size=34,
)


def info_overlay(img, job, sc, shot, screen: dict):
    """Draw the counter and the part labels of the square loop, in place."""
    H, W = img.shape[:2]
    S = min(W, H)
    px = S / 1080.0
    c = np.array([W / 2.0, H / 2.0])
    lay = INFO_LAYOUT
    d = sc.designs[shot.design]
    smp = sample(d, shot)

    # the information is shown at full strength during the contact and fades with the loop
    al = smp.info_level if shot.t >= 0 else 0.0
    raw_bits = smp.bits

    cx, cy = c + np.array(lay["count"]) * px
    a_txt = max(al, 0.35)
    style.text(img, (cx, cy), f"{raw_bits:.0f}", lay["count_size"] * px, BRIGHT, "light", "ls",
               a_txt)
    from passive_info.anim.style import font
    wnum = font("light", int(lay["count_size"] * px)).getlength(f"{raw_bits:.0f}")
    style.text(img, (cx + wnum + 12 * px, cy), "bits", lay["unit_size"] * px, BRIGHT, "light",
               "ls", a_txt)
    cap = lay["caption_size"] * px
    for k, line in enumerate(INFO_CAPTION):
        style.text(img, (cx + 4 * px, cy + (1.45 + 1.2 * k) * cap), line, cap,
                   BRIGHT if k == len(INFO_CAPTION) - 1 else DIM, "regular", "ls", a_txt)

    if not screen:
        return
    idx = {e: i for i, e in enumerate(d.elements)}
    # the numbers keep their values while the display fades, so that they never count down
    raw = smp.part_bits / smp.info_level if smp.info_level > 1e-6 else smp.part_bits * 0
    pb = lambda e: raw[idx[e]]
    ps = lay["part_size"]
    _callout(img, screen["contact"], (-140, 20), f"foot  {pb('ground'):.0f} bits", px, al,
             "rm", size=ps, line_level=0.2)
    _callout(img, screen["rotor"], (-20, 150), f"motor  {pb('rotor_calf'):.0f} bits", px, al,
             "lm", size=ps, line_level=0.2)
