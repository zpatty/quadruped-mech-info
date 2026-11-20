"""The 2D schematic: the 2-DoF prototype of Wensing et al. (2017) over a flow field.

The drawing follows the clamped bench in the paper. A rotor turns about a hip pivot that is
fixed to a hatched trunk. A lever of length r connects the pivot to the leg mass at the foot,
and the foot rests on the ground spring and damper. On a series-elastic design, the rotor is a
separate body, and a spiral torsion spring connects it to the hub of the leg. On the rigid
design, the rotor is fixed to the hub.

Energy and information come from `data.proto_design`, which uses the linear model of the
paper. The streamlines in the background are decoration and do not show data. Their
brightness follows the magnitude of the force.
"""
from __future__ import annotations

import numpy as np

from passive_info.anim import style


class FlowStage:
    """Layout for one panel; all geometry in pixels."""

    def __init__(self, width: int, height: int, gain: float = 250.0, seed: int = 3):
        self.W, self.H, self.gain = width, height, gain
        s = height / 1080.0
        self.s = s
        self.lever = min(480 * s, 0.34 * width)
        self.pivot = np.array([0.5 * width - 0.5 * self.lever, 0.50 * height])
        self.floor_y = self.pivot[1] + 260 * s
        self.rotor_r = 92 * s
        self.hub_r = 30 * s
        # the spectrum arc goes over the whole mechanism and does not overlap the rotor
        self.halo_c = np.array([self.pivot[0] + 0.5 * self.lever, self.pivot[1] + 150 * s])
        self.halo_r0 = float(np.hypot(0.5 * self.lever, 150 * s)) + self.rotor_r + 40 * s
        self.halo_len = 0.25 * self.halo_r0
        self._field(seed)

    # -----------------------------------------------------------------------------------
    def _field(self, seed):
        """Streamlines of a smooth field that converges on the contact, computed once."""
        rng = np.random.default_rng(seed)
        W, H, s = self.W, self.H, self.s
        target = np.array([self.pivot[0] + self.lever, self.floor_y])

        def vel(p):
            d = target - p
            r = np.linalg.norm(d) + 1e-6
            swirl = np.array([-d[1], d[0]]) / r
            n = np.array([np.sin(p[1] / (170 * s) + 1.3 * np.cos(p[0] / (260 * s))),
                          np.cos(p[0] / (190 * s) - 0.8 * np.sin(p[1] / (230 * s)))])
            v = 0.55 * d / r + 0.35 * swirl + 0.45 * n
            return v / (np.linalg.norm(v) + 1e-9)

        lines = []
        for _ in range(150):
            p = np.array([rng.uniform(-0.05, 1.05) * W, rng.uniform(-0.05, 1.05) * H])
            pts = [p.copy()]
            for _ in range(70):
                p = p + 9 * s * vel(p)
                pts.append(p.copy())
                if np.linalg.norm(p - target) < 40 * s:
                    break
            lines.append((np.array(pts), rng.random()))
        self.lines = lines

    # -----------------------------------------------------------------------------------
    def geometry(self, theta_j: float, theta_m: float) -> dict:
        """Positions at display angles (rad, already scaled)."""
        c = self.pivot
        foot = c + self.lever * np.array([np.cos(theta_j), np.sin(theta_j)])
        return dict(pivot=c, foot=foot, theta_j=theta_j, theta_m=theta_m,
                    contact=np.array([foot[0], self.floor_y]))


# ---------------------------------------------------------------------------------------
# Element anchors for the particles
# ---------------------------------------------------------------------------------------

def _zigzag(a, b, n=9, amp=16.0):
    t = np.linspace(0, 1, 2 * n + 1)
    pts = a[None] + (b - a)[None] * t[:, None]
    nrm = np.array([-(b - a)[1], (b - a)[0]])
    nrm /= np.linalg.norm(nrm) + 1e-9
    off = np.zeros_like(t)
    off[1:-1:2], off[2:-1:2] = amp, -amp
    off[1], off[-2] = amp * 0.5, -amp * 0.5
    return pts + off[:, None] * nrm[None]


def _spiral(c, r0, r1, phase, turns=3.2, n=160):
    s = np.linspace(0, 1, n)
    th = phase + 2 * np.pi * turns * s
    r = r0 + (r1 - r0) * s
    return c[None] + np.stack([r * np.cos(th), r * np.sin(th)], 1)


class _Geo:
    def __init__(self, st: FlowStage, g: dict, is_rigid: bool):
        self.st, self.g, self.rigid = st, g, is_rigid
        s = st.s
        top = g["foot"] + np.array([0, 18 * s])
        self.spring_pts = _zigzag(top, g["contact"] + np.array([-22 * s, 0]), 7, 14 * s)
        twist = g["theta_m"] - g["theta_j"]
        self.spiral_pts = _spiral(g["pivot"], st.hub_r + 4 * s, st.rotor_r * 0.72,
                                  g["theta_j"] + 6 * twist)

    def point(self, e: str, u: float, v: float, t: float) -> np.ndarray:
        st, g, s = self.st, self.g, self.st.s
        if e == "ground":
            P = self.spring_pts
            k = u * (len(P) - 1)
            i = int(min(k, len(P) - 2))
            return P[i] + (k - i) * (P[i + 1] - P[i]) + 3 * s * np.array([np.sin(9 * v + t), 0])
        if e == "leg":
            a, b = g["pivot"], g["foot"]
            d = (b - a) / np.linalg.norm(b - a)
            n = np.array([-d[1], d[0]])
            return a + (b - a) * (0.12 + 0.86 * u) + n * 9 * s * np.sin(2 * np.pi * v + 2 * t)
        if e == "spring":
            P = self.spiral_pts
            k = u * (len(P) - 1)
            i = int(min(k, len(P) - 2))
            return P[i] + (k - i) * (P[i + 1] - P[i])
        th = 2 * np.pi * u + g["theta_m"] * 3 + 0.5 * t          # rotor
        r = st.rotor_r * (0.78 + 0.18 * v)
        return g["pivot"] + r * np.array([np.cos(th), np.sin(th)])

    def center(self, e: str) -> np.ndarray:
        g, st = self.g, self.st
        return dict(ground=self.spring_pts[len(self.spring_pts) // 2],
                    leg=0.5 * (g["pivot"] + g["foot"]),
                    spring=g["pivot"] + np.array([0.0, st.rotor_r * 0.45]),
                    rotor=g["pivot"] + np.array([0.0, st.rotor_r * 0.85]))[e]


def _positions(geo: _Geo, elements, nxt, parts, t):
    from passive_info.anim.particles import INPUT, path
    P = np.zeros((len(parts.pid), 2))
    for i in range(len(parts.pid)):
        dst = int(parts.dst[i])
        end = geo.point(elements[dst], parts.u[i], parts.v[i], t)
        if parts.prog[i] >= 1.0:
            P[i] = end
            continue
        src = int(parts.src[i])
        if src == INPUT:
            start = geo.g["contact"]
            hops = path(nxt, elements.index("ground"), dst)
        else:
            start = geo.point(elements[src], parts.u[i], parts.v[i], t)
            hops = path(nxt, src, dst)[1:-1]
        pts = np.vstack([start] + [geo.center(elements[h]) for h in hops] + [end])
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        L = np.concatenate([[0], np.cumsum(seg)])
        p = parts.prog[i]
        x = p * p * (3 - 2 * p) * L[-1]
        k = int(np.clip(np.searchsorted(L, x) - 1, 0, len(seg) - 1))
        P[i] = pts[k] + (x - L[k]) / max(seg[k], 1e-9) * (pts[k + 1] - pts[k])
    return P


# ---------------------------------------------------------------------------------------
# The frame
# ---------------------------------------------------------------------------------------

def draw(stage: FlowStage, d, smp, parts, parts_prev, smp_prev, e_ref: float, t_anim: float,
         nxt, is_rigid: bool) -> np.ndarray:
    st, s = stage, stage.s
    W, H = st.W, st.H
    K = len(d.mode_freqs)
    hues = style.mode_hues(K)
    idx = {e: i for i, e in enumerate(d.elements)}
    lvl = lambda e: style.level(smp.energy[idx[e]], e_ref)
    col = lambda e: style.mix_hue(smp.hue_w[idx[e]], hues)
    elements = list(d.elements)

    g = st.geometry(st.gain * float(smp.joint_dev[0]), st.gain * float(smp.rotor_dev[0]))
    geo = _Geo(st, g, is_rigid)
    gp = st.geometry(st.gain * float(smp_prev.joint_dev[0]), st.gain * float(smp_prev.rotor_dev[0]))
    geo_prev = _Geo(st, gp, is_rigid)

    img = np.zeros((H, W, 3), np.float32) + style.BG.astype(np.float32)
    L = np.zeros((H, W, 3), np.float32)

    # background streamlines that move toward the contact, brighter when the force is larger
    work = min(1.0, abs(smp.force) / 70.0)
    c_in = style.mix_hue(smp.hue_w.sum(0), hues)
    for pts, ph in st.lines:
        n = len(pts)
        if n < 6:
            continue
        head = ((t_anim * 0.35 + ph) % 1.0) * (n - 1)
        a, b = int(max(head - 5, 0)), int(head) + 1
        if b - a >= 2:
            style.polyline(L, pts[a:b], style.INK * 0.018 + c_in * (0.22 * work), 1.0 * s)

    # the floor, and the trunk that the bench is clamped to
    style.line(img, (0, st.floor_y), (W, st.floor_y), style.INK * 0.05, 1.0 * s)
    for x in np.arange(-H, W, 22 * s):
        style.line(img, (x, st.floor_y + 1), (x - 16 * s, st.floor_y + 17 * s), style.INK * 0.02,
                   1.0 * s)
    pv = g["pivot"]
    ty = pv[1] - st.rotor_r - 44 * s
    style.line(img, (pv[0] - 150 * s, ty), (pv[0] + 150 * s, ty), style.INK * 0.10, 2.0 * s)
    for x in np.arange(pv[0] - 150 * s, pv[0] + 150 * s, 16 * s):
        style.line(img, (x, ty), (x + 12 * s, ty - 12 * s), style.INK * 0.05, 1.0 * s)
    style.line(img, (pv[0], ty), pv, style.INK * 0.06, 1.5 * s)

    # ground spring and damper under the foot
    a_g = lvl("ground")
    c_g = col("ground") if a_g > 1e-3 else style.INK
    style.polyline(L, geo.spring_pts, style.INK * 0.10 + c_g * 0.9 * a_g, 1.8 * s)
    fx = g["foot"][0] + 22 * s
    top, bot = g["foot"][1] + 18 * s, st.floor_y
    mid = 0.5 * (top + bot)
    style.line(L, (fx, top), (fx, mid + 6 * s), style.INK * 0.10, 1.6 * s)
    style.polyline(L, [(fx - 9 * s, mid - 8 * s), (fx - 9 * s, bot - 4 * s),
                       (fx + 9 * s, bot - 4 * s), (fx + 9 * s, mid - 8 * s)], style.INK * 0.08, 1.4 * s)
    style.line(L, (fx - 7 * s, mid + 6 * s), (fx + 7 * s, mid + 6 * s), style.INK * 0.12, 1.6 * s)
    style.line(L, (g["foot"][0] - 30 * s, top), (fx + 12 * s, top), style.INK * 0.12, 2.0 * s)

    # the leg: lever and mass
    a_l = lvl("leg")
    c_l = col("leg") if a_l > 1e-3 else style.INK
    style.line(L, pv, g["foot"], style.INK * 0.14 + c_l * 1.0 * a_l, 4.0 * s)
    style.disc(L, g["foot"], 15 * s, style.INK * 0.05 + c_l * 0.8 * a_l)
    style.disc(img, g["foot"], 13 * s, style.BG * 0.5)
    style.polyline(L, g["foot"] + 15 * s * np.stack([np.cos(np.linspace(0, 6.3, 40)),
                                                      np.sin(np.linspace(0, 6.3, 40))], 1),
                   style.INK * 0.18 + c_l * a_l, 1.4 * s, closed=True)

    # rotor, hub, and the connection between them
    a_r = lvl("rotor")
    c_r = col("rotor") if a_r > 1e-3 else style.INK
    ring = lambda r, ph=0.0: pv + r * np.stack([np.cos(np.linspace(0, 6.3, 96) + ph),
                                                np.sin(np.linspace(0, 6.3, 96) + ph)], 1)
    style.polyline(L, ring(st.rotor_r), style.INK * 0.14 + c_r * 1.0 * a_r, 2.0 * s, closed=True)
    style.polyline(L, ring(st.rotor_r * 0.9), style.INK * 0.05 + c_r * 0.4 * a_r, 1.0 * s, closed=True)
    for k in range(6):
        th = 3 * g["theta_m"] + 2 * np.pi * k / 6
        dv = np.array([np.cos(th), np.sin(th)])
        style.line(L, pv + dv * st.rotor_r * 0.8, pv + dv * st.rotor_r * 0.98,
                   style.INK * 0.2 + c_r * 1.3 * a_r, 2.0 * s)
    style.polyline(L, ring(st.hub_r), style.INK * 0.2 + c_l * 0.6 * a_l, 1.6 * s, closed=True)
    if is_rigid:
        for k in range(4):
            th = 3 * g["theta_j"] + 2 * np.pi * k / 4 + np.pi / 4
            dv = np.array([np.cos(th), np.sin(th)])
            style.line(L, pv + dv * st.hub_r, pv + dv * st.rotor_r * 0.8, style.INK * 0.10,
                       2.4 * s)
    else:
        a_s = lvl("spring")
        c_s = col("spring") if a_s > 1e-3 else style.INK
        style.polyline(L, geo.spiral_pts, style.INK * 0.12 + c_s * 1.1 * a_s, 1.5 * s)

    # bright spot at the contact, scaled by the force
    f_lvl = min(1.0, abs(smp.force) / 80.0)
    if f_lvl > 0.01:
        style.splat(L, g["contact"], 9 * s, style.WHITE * 0.9 * f_lvl)

    _halo(L, st, d, smp, hues, s)

    # screen positions for the text callouts of the loop cut
    c, r1 = st.halo_c, st.halo_r0 + st.halo_len
    a0, a1 = np.radians(200), np.radians(340)
    arc = lambda a: c + r1 * np.array([np.cos(a), np.sin(a)])
    st.screen = dict(contact=g["contact"], rotor=pv + np.array([0, st.rotor_r]),
                     spring=pv + np.array([0, 0.5 * st.rotor_r]),
                     ring_low=arc(a0), ring_high=arc(a1), ring_mid=arc(1.5 * np.pi),
                     ring_bar=None)

    # particles
    if len(parts.pid):
        P = _positions(geo, elements, nxt, parts, t_anim)
        prev = {}
        if len(parts_prev.pid):
            Pp = _positions(geo_prev, elements, nxt, parts_prev, t_anim - 1.0 / 30)
            prev = {int(k): Pp[j] for j, k in enumerate(parts_prev.pid)}
        for j in range(len(parts.pid)):
            c = parts.hue[j] * parts.alpha[j] * (0.55 if parts.dying[j] else 1.0)
            q = prev.get(int(parts.pid[j]))
            if q is not None and 0.5 < np.hypot(*(P[j] - q)) < 80 * s:
                style.line(L, q + 0.4 * (P[j] - q), P[j], c * 0.5, 1.4 * s)
            style.splat(L, P[j], 2.0 * s, c * 1.3)

    out = img + style.glow(L, s)
    return style.bloom(out, 12 * s, 0.7)


def _halo(L, st: FlowStage, d, smp, hues, s):
    """The spectrum as an arc of bars over the mechanism, low frequency on the left."""
    K = len(d.mode_freqs)
    c, r0, ln = st.halo_c, st.halo_r0, st.halo_len
    a0, a1 = np.radians(200), np.radians(340)
    angs = a0 + (a1 - a0) * (np.arange(K) + 0.5) / K
    dth = (a1 - a0) / K * 0.36
    for k in range(K):
        f = style.bar_length(smp.spec[k])
        if f <= 1e-3:
            continue
        r1 = r0 + ln * f
        q = [(c[0] + r * np.cos(a), c[1] + r * np.sin(a))
             for r, a in ((r0, angs[k] - dth), (r1, angs[k] - dth), (r1, angs[k] + dth),
                          (r0, angs[k] + dth))]
        style.fill_poly(L, q, hues[k] * (0.25 + 0.9 * f))
    arc = np.linspace(a0, a1, 160)
    for r, lv in ((r0 - 4 * s, 0.05), (r0 + style.BAR_KNEE_FRAC * ln, 0.025), (r0 + ln, 0.04)):
        style.polyline(L, np.stack([c[0] + r * np.cos(arc), c[1] + r * np.sin(arc)], 1),
                       style.INK * lv, 1.0 * s)
    fmax = d.mode_freqs[-1] + 0.5 * (d.mode_freqs[1] - d.mode_freqs[0])
    on = float(np.max(smp.spec) > 0.05)
    for f in d.natural_hz:
        if f <= fmax:
            a = a0 + (a1 - a0) * f / fmax
            r = r0 + ln + 6 * s
            style.line(L, (c[0] + r * np.cos(a), c[1] + r * np.sin(a)),
                       (c[0] + (r + 16 * s) * np.cos(a), c[1] + (r + 16 * s) * np.sin(a)),
                       style.WHITE * 0.5 * on, 1.6 * s)
