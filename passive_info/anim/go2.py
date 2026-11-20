"""The 3D view: the Go2 from MuJoCo Menagerie, restyled dark, posed from the FR-leg trajectory.

MuJoCo renders the robot in two passes, RGB and segmentation. The code then draws the overlays
in linear light: link glow, rotors, series springs, the contact, particles and the spectrum
ring. It places them by projecting the joint anchors and axes of the posed model through the
render camera.

The view shows the whole robot in the stance crouch of the paper. Only the front-right leg
moves, as on the clamped bench of the paper. Its motion comes from the linear model and is
multiplied by one display gain, because the real deflections are less than a milliradian.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


from passive_info.anim import style

LEG = "FR"
PARTS = ("hip", "thigh", "calf")


@dataclass
class View:
    lookat: tuple = (0.10, -0.10, 0.14)
    distance: float = 0.82
    azimuth: float = 132.0
    elevation: float = -12.0


def _restyle(spec, width: int, height: int):
    import mujoco
    spec.visual.global_.offwidth = max(width, 640)
    spec.visual.global_.offheight = max(height, 480)
    spec.visual.headlight.diffuse = [0.3, 0.31, 0.35]
    spec.visual.headlight.ambient = [0.06, 0.065, 0.08]
    spec.visual.headlight.specular = [0.05, 0.05, 0.05]
    spec.visual.quality.shadowsize = 4096
    spec.visual.map.fogstart = 1.2
    spec.visual.map.fogend = 4.0
    spec.visual.rgba.fog = [0.006, 0.007, 0.011, 1.0]
    spec.visual.rgba.haze = [0.0, 0.0, 0.0, 1.0]
    for t in spec.textures:
        if t.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            t.rgb1 = [0.0, 0.0, 0.0]
            t.rgb2 = [0.0, 0.0, 0.0]
        elif t.name == "groundplane":
            t.rgb1 = [0.07, 0.075, 0.09]
            t.rgb2 = [0.07, 0.075, 0.09]
            t.markrgb = [0.2, 0.24, 0.3]
    for mat in spec.materials:
        if mat.name == "groundplane":
            mat.reflectance = 0.12
            mat.texrepeat = [10, 10]
        else:
            mat.rgba = [0.22, 0.23, 0.26, 1.0]
            mat.specular = 0.55
            mat.shininess = 0.55
            mat.reflectance = 0.0
    for g in spec.geoms:
        if g.name != "floor":
            g.rgba = [0.22, 0.23, 0.26, 1.0]
    for li in list(spec.worldbody.lights):
        spec.delete(li)
    # key from front-right above, cool rim from behind-left
    for pos, dirn, dif, shadow in (((1.2, -1.0, 1.6), (-0.6, 0.5, -0.8), 0.55, True),
                                   ((-1.2, 0.9, 0.7), (0.8, -0.6, -0.3), 0.75, False),
                                   ((0.3, -1.5, 0.3), (-0.1, 1.0, -0.15), 0.18, False)):
        li = spec.worldbody.add_light()
        li.pos, li.dir = list(pos), list(dirn)
        li.diffuse = [dif * 0.85, dif * 0.92, dif]
        li.specular = [dif * 0.6] * 3
        li.castshadow = shadow
        try:
            li.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        except AttributeError:
            li.directional = True


class Go2Stage:
    """A posed, restyled Go2 and its camera."""

    def __init__(self, width: int, height: int, gain: float = 250.0, view: View | None = None):
        import mujoco
        from mechinfo import mjcf
        from passive_info.embodied import STANCE_QPOS

        self.mj = mujoco
        self.W, self.H, self.gain = width, height, gain
        spec = mujoco.MjSpec.from_file(mjcf.menagerie_scene("unitree_go2"))
        _restyle(spec, width, height)
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        m = self.m
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        self.jnt = {p: jid(f"{LEG}_{p}_joint") for p in PARTS}
        self.qadr = {p: m.jnt_qposadr[self.jnt[p]] for p in PARTS}
        self.body = {p: bid(f"{LEG}_{p}") for p in PARTS}
        self.foot = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, LEG)
        self.foot_r = float(m.geom_size[self.foot, 0])
        self.stance = np.array(STANCE_QPOS, float)

        # every leg in the paper's crouch, trunk lowered until the feet touch
        q0 = m.key_qpos[0].copy() if m.nkey else self.d.qpos.copy()
        for leg in ("FL", "FR", "RL", "RR"):
            for p, v in zip(PARTS, STANCE_QPOS):
                q0[m.jnt_qposadr[jid(f"{leg}_{p}_joint")]] = v
        self.d.qpos[:] = q0
        mujoco.mj_kinematics(m, self.d)
        q0[2] -= float(self.d.geom_xpos[self.foot][2] - self.foot_r)
        self.q0 = q0

        self.view = view or View()
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.set_view(self.view)
        self.r_rgb = mujoco.Renderer(m, height, width)
        self.r_seg = mujoco.Renderer(m, height, width)
        self.r_seg.enable_segmentation_rendering()
        for r in (self.r_rgb,):
            r.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
            r.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
        self.pose(np.zeros(3))

    def set_view(self, v: View):
        self.cam.lookat[:] = v.lookat
        self.cam.distance, self.cam.azimuth, self.cam.elevation = v.distance, v.azimuth, v.elevation

    # -----------------------------------------------------------------------------------
    def pose(self, joint_dev: np.ndarray):
        """Pose the FR leg at stance plus `gain * joint_dev` (hip, thigh, calf; rad)."""
        q = self.q0.copy()
        for i, p in enumerate(PARTS):
            q[self.qadr[p]] = self.stance[i] + self.gain * float(joint_dev[i])
        self.d.qpos[:] = q
        self.mj.mj_kinematics(self.m, self.d)
        self.mj.mj_comPos(self.m, self.d)

    def anchors(self) -> dict:
        """World positions and axes that the overlays use, at the current pose."""
        d = self.d
        out = {}
        for p in PARTS:
            out[f"{p}_pos"] = d.xanchor[self.jnt[p]].copy()
            out[f"{p}_axis"] = d.xaxis[self.jnt[p]].copy()
        fc = d.geom_xpos[self.foot].copy()
        out["foot"] = fc
        out["contact"] = fc - np.array([0.0, 0.0, self.foot_r])
        return out

    # -----------------------------------------------------------------------------------
    def _glcam(self, r):
        c0, c1 = r.scene.camera[0], r.scene.camera[1]
        pos = 0.5 * (np.array(c0.pos) + np.array(c1.pos))
        fwd = np.array(c0.forward)
        up = np.array(c0.up)
        return pos, fwd, up, c0

    def project(self, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World points `(..., 3)` to pixel coordinates `(..., 2)` and view depth `(...)`."""
        pos, fwd, up, c = self._cam
        right = np.cross(fwd, up)
        rel = np.asarray(P, float) - pos
        x, y, z = rel @ right, rel @ up, rel @ fwd
        z = np.maximum(z, 1e-4)
        hh = 0.5 * (c.frustum_top - c.frustum_bottom)
        cy = 0.5 * (c.frustum_top + c.frustum_bottom)
        hw = hh * self.W / self.H
        u = self.W / 2 + (x * c.frustum_near / z - c.frustum_center) / hw * self.W / 2
        v = self.H / 2 - (y * c.frustum_near / z - cy) / hh * self.H / 2
        return np.stack([u, v], -1), z

    def render(self) -> tuple[np.ndarray, np.ndarray]:
        """`(rgb_linear HxWx3, body_id HxW)` at the current pose; caches the camera."""
        self.r_rgb.update_scene(self.d, camera=self.cam)
        self._cam = self._glcam(self.r_rgb)
        rgb = style.to_linear(self.r_rgb.render())
        self.r_seg.update_scene(self.d, camera=self.cam)
        seg = self.r_seg.render()
        geom = seg[..., 0]
        is_geom = seg[..., 1] == self.mj.mjtObj.mjOBJ_GEOM
        body = np.where(is_geom & (geom >= 0), self.m.geom_bodyid[np.clip(geom, 0, None)], -1)
        return rgb, body

    def body_mask(self, body_ids: np.ndarray, part: str) -> np.ndarray:
        return (body_ids == self.body[part]).astype(np.float32)


# ---------------------------------------------------------------------------------------
# Overlay geometry, in world coordinates at the current pose
# ---------------------------------------------------------------------------------------

ROTOR_R = 0.032          #: m, drawn radius of a rotor
ROTOR_OFF = {"hip": 0.075, "thigh": 0.062, "calf": 0.062}   #: m, out along the joint axis
CONTACT_R = 0.035
RING_R0, RING_LEN = 0.085, 0.13     #: spectrum ring: inner radius, length of a full bar


def _frame(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors spanning the plane normal to `axis`."""
    a = axis / np.linalg.norm(axis)
    t = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(a, t)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(a, e1)


def _outward(axis: np.ndarray, part: str) -> np.ndarray:
    """Direction of the axis that points away from the trunk, toward the camera."""
    a = axis / np.linalg.norm(axis)
    if part == "hip":
        return a if a[0] > 0 else -a
    return a if a[1] < 0 else -a


class Geometry:
    """Where each element is drawn, from the stage's anchors."""

    def __init__(self, A: dict, rotor_dev: np.ndarray, joint_dev: np.ndarray, gain: float):
        self.A = A
        self.rot = {}
        for i, p in enumerate(PARTS):
            out = _outward(A[f"{p}_axis"], p)
            c = A[f"{p}_pos"] + ROTOR_OFF[p] * out
            e1, e2 = _frame(out)
            # the rotor marker turns with the rotor coordinate, and the joint turns with the link
            self.rot[p] = dict(c=c, out=out, e1=e1, e2=e2, joint=A[f"{p}_pos"],
                               spin=gain * float(rotor_dev[i]),
                               twist=gain * float(rotor_dev[i] - joint_dev[i]))

    def ring(self, p: str, n: int = 48, r: float = ROTOR_R, phase: float = 0.0):
        g = self.rot[p]
        th = np.linspace(0, 2 * np.pi, n) + phase
        return g["c"] + r * (np.cos(th)[:, None] * g["e1"] + np.sin(th)[:, None] * g["e2"])

    def helix(self, p: str, n: int = 90, turns: float = 5.0, r: float = 0.014):
        g = self.rot[p]
        s = np.linspace(0.0, 1.0, n)
        th = 2 * np.pi * turns * s + g["twist"] * (1 - s) * 6.0
        base = g["c"][None] * (1 - s[:, None]) + g["joint"][None] * s[:, None]
        return base + r * (np.cos(th)[:, None] * g["e1"] + np.sin(th)[:, None] * g["e2"])

    def element_point(self, e: str, u: np.ndarray, v: np.ndarray, t: float) -> np.ndarray:
        """Position `(p, 3)` of a particle that has arrived on element `e`, with parameters `(u, v)`."""
        A = self.A
        u, v = np.asarray(u)[:, None], np.asarray(v)[:, None]
        wob = 0.006 * np.sin(2 * np.pi * (v + 0.35 * t))
        if e == "ground":
            th = 2 * np.pi * u + 0.6 * t
            r = CONTACT_R * np.sqrt(0.15 + 0.85 * v)
            return A["contact"] + np.hstack([r * np.cos(th), r * np.sin(th), 0.002 + 0 * th])
        seg = dict(calf=("foot", "calf_pos"), thigh=("calf_pos", "thigh_pos"),
                   hip=("thigh_pos", "hip_pos"))
        if e in seg:
            a, b = A[seg[e][0]], A[seg[e][1]]
            n1, n2 = _frame(b - a)
            th = 2 * np.pi * v + 1.3 * t
            rad = 0.008 + 0.006 * np.sin(7 * u)
            return (a + (b - a) * (0.08 + 0.84 * u) + rad * (np.cos(th) * n1 + np.sin(th) * n2)
                    + wob * n1)
        kind, p = e.split("_")
        g = self.rot[p]
        if kind == "rotor":
            th = 2 * np.pi * u + g["spin"]
            r = ROTOR_R * (0.55 + 0.45 * v)
            return g["c"] + r * (np.cos(th) * g["e1"] + np.sin(th) * g["e2"])
        # spring: on the helix
        s = u
        th = 2 * np.pi * 5.0 * s + g["twist"] * (1 - s) * 6.0
        base = g["c"] * (1 - s) + g["joint"] * s
        return base + 0.014 * (np.cos(th) * g["e1"] + np.sin(th) * g["e2"])

    def element_center(self, e: str) -> np.ndarray:
        A = self.A
        if e == "ground":
            return A["contact"] + np.array([0, 0, 0.002])
        if e == "calf":
            return 0.5 * (A["foot"] + A["calf_pos"])
        if e == "thigh":
            return 0.5 * (A["calf_pos"] + A["thigh_pos"])
        if e == "hip":
            return 0.5 * (A["thigh_pos"] + A["hip_pos"])
        kind, p = e.split("_")
        g = self.rot[p]
        return g["c"] if kind == "rotor" else 0.5 * (g["c"] + g["joint"])


def particle_positions(geo: Geometry, elements, nxt, parts, t: float) -> np.ndarray:
    """World positions `(p, 3)` of every particle at this frame."""
    from passive_info.anim.particles import INPUT, path
    P = np.zeros((len(parts.pid), 3))
    for i in range(len(parts.pid)):
        dst = int(parts.dst[i])
        end = geo.element_point(elements[dst], parts.u[i:i + 1], parts.v[i:i + 1], t)[0]
        if parts.prog[i] >= 1.0:
            P[i] = end
            continue
        src = int(parts.src[i])
        if src == INPUT:
            start = geo.A["contact"] + np.array([0, 0, 0.002])
            hops = path(nxt, elements.index("ground"), dst)
        else:
            start = geo.element_point(elements[src], parts.u[i:i + 1], parts.v[i:i + 1], t)[0]
            hops = path(nxt, src, dst)[1:-1]
        pts = np.vstack([start] + [geo.element_center(elements[h]) for h in hops
                                   if elements[h] != "ground" or src != INPUT] + [end])
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        L = np.concatenate([[0], np.cumsum(seg)])
        x = style_ease(parts.prog[i]) * L[-1]
        k = int(np.clip(np.searchsorted(L, x) - 1, 0, len(seg) - 1))
        f = (x - L[k]) / max(seg[k], 1e-9)
        P[i] = pts[k] + f * (pts[k + 1] - pts[k])
    return P


def style_ease(x):
    return x * x * (3 - 2 * x)


# ---------------------------------------------------------------------------------------
# The frame
# ---------------------------------------------------------------------------------------

def draw(stage: Go2Stage, d, smp, parts, parts_prev, smp_prev, e_ref: float,
         t_anim: float, nxt, is_rigid: bool) -> np.ndarray:
    """Compose one linear-light frame (HxWx3 float32)."""
    import cv2

    W, H = stage.W, stage.H
    px = H / 1080.0                                   # line widths are specified at 1080p
    K = len(d.mode_freqs)
    hues = style.mode_hues(K)
    E = smp.energy
    idx = {e: i for i, e in enumerate(d.elements)}
    lvl = lambda e: style.level(E[idx[e]], e_ref)
    col = lambda e: style.mix_hue(smp.hue_w[idx[e]], hues)

    # pose the previous frame first, for the particle trails, then the current frame
    stage.pose(smp_prev.joint_dev)
    geo_prev = Geometry(stage.anchors(), smp_prev.rotor_dev, smp_prev.joint_dev, stage.gain)
    stage.pose(smp.joint_dev)
    rgb, body = stage.render()
    geo = Geometry(stage.anchors(), smp.rotor_dev, smp.joint_dev, stage.gain)
    proj = lambda P: stage.project(P)[0]

    # base: the body, dimmed, with ink outlines from the segmentation
    base = rgb * 3.2
    lum = rgb.mean(-1)
    shade = (0.35 + 0.65 * np.clip(lum / max(np.percentile(lum[body > 0], 95), 1e-6), 0, 1.4)
             if (body > 0).any() else np.ones_like(lum))
    b = np.where(body > 0, body, -1).astype(np.int32)     # the robot only, not the floor
    edge = np.zeros_like(b, bool)
    edge[:, 1:] |= b[:, 1:] != b[:, :-1]
    edge[1:, :] |= b[1:, :] != b[:-1, :]
    edge = cv2.dilate(edge.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(np.float32)
    leg = np.zeros_like(edge)
    for p in PARTS:
        leg += stage.body_mask(body, p)
    base += edge[..., None] * (0.035 + 0.05 * cv2.dilate(leg, np.ones((5, 5)))[..., None]) * style.INK

    L = np.zeros((H, W, 3), np.float32)               # light layer

    # each link glows in the colours of the force modes in its energy
    for p in PARTS:
        m = stage.body_mask(body, p)
        a = lvl(p)
        if a > 1e-3:
            c = col(p).astype(np.float32)
            L += (m * shade)[..., None] * c * (1.1 * a)
            L += style.blur(m, 9 * px)[..., None] * c * (0.45 * a)

    # rotors: one ring per joint, with ticks that turn with the rotor; then a spring or a shaft
    for i, p in enumerate(PARTS):
        g = geo.rot[p]
        a_r, a_s = lvl(f"rotor_{p}"), lvl(f"spring_{p}")
        c_r = col(f"rotor_{p}") if a_r > 1e-3 else style.INK
        ring = proj(geo.ring(p))
        style.polyline(L, ring, style.INK * 0.12 + c_r * 0.9 * a_r, 1.6 * px, closed=True)
        style.polyline(L, proj(geo.ring(p, r=ROTOR_R * 0.55)), style.INK * 0.05, 1.0 * px,
                       closed=True)
        for k in range(3):
            th = g["spin"] + 2 * np.pi * k / 3
            dvec = np.cos(th) * g["e1"] + np.sin(th) * g["e2"]
            a0, a1 = proj(g["c"] + 0.55 * ROTOR_R * dvec), proj(g["c"] + ROTOR_R * dvec)
            style.line(L, a0, a1, style.INK * 0.2 + c_r * 1.2 * a_r, 1.6 * px)
        if is_rigid:
            style.line(L, proj(g["c"]), proj(g["joint"]), style.INK * 0.10, 2.2 * px)
        else:
            c_s = col(f"spring_{p}") if a_s > 1e-3 else style.INK
            style.polyline(L, proj(geo.helix(p)), style.INK * 0.14 + c_s * 1.1 * a_s, 1.4 * px)

    # the contact: the point where the force is applied, and the ground spring
    cpt = geo.A["contact"]
    th = np.linspace(0, 2 * np.pi, 64)
    circ = lambda r: proj(cpt + np.stack([r * np.cos(th), r * np.sin(th), 0 * th], 1))
    a_g = lvl("ground")
    c_g = col("ground") if a_g > 1e-3 else style.INK
    style.fill_poly(L, circ(CONTACT_R), c_g * 0.35 * a_g)
    style.polyline(L, circ(CONTACT_R), style.INK * 0.10 + c_g * 0.8 * a_g, 1.2 * px, closed=True)
    for k in range(3):                               # ripples, brighter when the ground spring has more energy
        r = CONTACT_R * (1.0 + ((t_anim * 0.8 + k / 3) % 1.0) * 1.6)
        fade = 1.0 - ((t_anim * 0.8 + k / 3) % 1.0)
        style.polyline(L, circ(r), c_g * 0.5 * a_g * fade, 1.0 * px, closed=True)
    f_lvl = min(1.0, abs(smp.force) / 80.0)
    if f_lvl > 0.01:
        style.splat(L, proj(cpt), 7 * px, style.WHITE * 0.8 * f_lvl)

    # the spectrum ring on the floor: bits about each mode of the force
    _spectrum_ring(L, stage, geo, d, smp, hues, px)

    # particles, each with a trail back toward its position on the previous frame
    if len(parts.pid):
        P = proj(particle_positions(geo, list(d.elements), nxt, parts, t_anim))
        prev = {}
        if len(parts_prev.pid):
            Pp = proj(particle_positions(geo_prev, list(d.elements), nxt, parts_prev,
                                         t_anim - 1.0 / 30))
            prev = {int(k): Pp[j] for j, k in enumerate(parts_prev.pid)}
        for j in range(len(parts.pid)):
            c = parts.hue[j] * parts.alpha[j] * (0.55 if parts.dying[j] else 1.0)
            q = prev.get(int(parts.pid[j]))
            if q is not None and 0.5 < np.hypot(*(P[j] - q)) < 60 * px:
                style.line(L, q + 0.5 * (P[j] - q), P[j], c * 0.5, 1.3 * px)
            style.splat(L, P[j], 1.8 * px, c * 1.3)

    out = base + style.glow(L, px)
    return style.bloom(out, 12 * px, 0.7)


def _spectrum_ring(L, stage, geo, d, smp, hues, px):
    """34 radial bars around the contact, an arc facing the camera, low to high frequency."""
    K = len(d.mode_freqs)
    c = geo.A["contact"] + np.array([0.0, 0.0, 0.001])
    fwd = np.array([np.cos(np.radians(stage.view.azimuth)), np.sin(np.radians(stage.view.azimuth))])
    toward = np.degrees(np.arctan2(-fwd[1], -fwd[0]))
    span = 200.0
    angs = np.radians(toward - span / 2 + span * (np.arange(K) + 0.5) / K)
    dth = np.radians(span / K) * 0.36
    for k in range(K):
        fill = style.bar_length(smp.spec[k])
        for r1, colr in ((RING_R0 + RING_LEN * fill, hues[k] * (0.25 + 0.9 * fill)),):
            if r1 <= RING_R0 + 1e-4:
                continue
            a0, a1 = angs[k] - dth, angs[k] + dth
            quad = np.array([[RING_R0 * np.cos(a0), RING_R0 * np.sin(a0), 0],
                             [r1 * np.cos(a0), r1 * np.sin(a0), 0],
                             [r1 * np.cos(a1), r1 * np.sin(a1), 0],
                             [RING_R0 * np.cos(a1), RING_R0 * np.sin(a1), 0]]) + c
            style.fill_poly(L, stage.project(quad)[0], colr)
    # guide arcs: the inner edge, the knee where the bar's scale changes, a full bar
    arc = np.linspace(angs[0] - dth, angs[-1] + dth, 120)
    for r, lv in ((RING_R0 - 0.003, 0.05), (RING_R0 + style.BAR_KNEE_FRAC * RING_LEN, 0.025),
                  (RING_R0 + RING_LEN, 0.04)):
        pts = c + np.stack([r * np.cos(arc), r * np.sin(arc), 0 * arc], 1)
        style.polyline(L, stage.project(pts)[0], style.INK * lv, 1.0 * px)
    # natural frequencies of the mechanism that are inside the band
    fmax = d.mode_freqs[-1] + 0.5 * (d.mode_freqs[1] - d.mode_freqs[0])
    on = float(np.max(smp.spec) > 0.05)
    for f in d.natural_hz:
        if f > fmax:
            continue
        a = np.radians(toward - span / 2 + span * f / fmax)
        r = RING_R0 + RING_LEN + 0.006
        p = c + np.array([r * np.cos(a), r * np.sin(a), 0])
        q = c + np.array([(r + 0.016) * np.cos(a), (r + 0.016) * np.sin(a), 0])
        style.line(L, stage.project(p)[0], stage.project(q)[0], style.WHITE * 0.5 * on, 1.6 * px)
