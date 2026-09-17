"""The same passive question on a real leg: one Go2 leg instead of the 2-DoF prototype."""
from __future__ import annotations

import dataclasses

import numpy as np

from mechinfo import mjcf

from mechinfo.mechanics import (assemble, energy_metric, ground_damping,
                                locked_effective_mass)
from passive_info.systems import Passive, _readout, omega_ref, K_I, ZETA_CONTACT, ZETA_SERIES

#: Menagerie's reflected rotor inertia on every Go2 leg joint, kg m^2.
I_ROT_GO2 = 0.01


#: The Go2's names. The loader is `mechinfo.mjcf`, which knows no robot; these are all
#: that is left. No path here, because `menagerie_scene` clones when the model is absent.
GO2 = mjcf.RobotSpec(feet=("FL", "FR", "RL", "RR"))


def _go2(xml: str | None):
    """`GO2` with a scene attached -- the caller's MJCF, or Menagerie's Go2."""
    return dataclasses.replace(GO2, xml=xml or mjcf.menagerie_scene("unitree_go2"))


def _compile(xml, legs, vertical, contact="menagerie", spec_hook=None):
    """The one place this module turns a scene into a model."""
    return mjcf.compile_model(_go2(xml), armature=I_ROT_GO2, vertical=vertical,
                              legs=legs, trunk_scale=0.25, contact=contact,
                              contact_stiff=0.002,
                              contact_solimp=(0.9, 0.99, 1e-4, 0.5, 2.0),
                              spec_hook=spec_hook)


#: Menagerie's home crouch for one leg: hip, thigh, calf. The trunk slide is then lowered
#: until the foot sits exactly on the floor, so the pose is a stance rather than a guess.
STANCE_QPOS = (0.0, 0.9, -1.8)


def go2_mass_and_jacobian(stance=STANCE_QPOS, clamped: bool = True,
                          xml: str | None = None, leg_name: str = "FR"):
    """`(H, J)` at the nominal stance: mass matrix and vertical foot Jacobian."""
    import mujoco

    m, d = _compile(xml, (leg_name,), "fixed" if clamped else True)
    leg = mjcf.index_legs(m, GO2)[leg_name]
    d.qpos[leg.qpos] = stance
    mujoco.mj_forward(m, d)
    if not clamped:
        d.qpos[0] -= float(d.geom_xpos[leg.foot_geom][2] - leg.foot_radius)
    mujoco.mj_forward(m, d)

    H = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, H)
    Jf, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    p = d.geom_xpos[leg.foot_geom].copy()
    p[2] -= leg.foot_radius
    mujoco.mj_jac(m, d, Jf, jr, p, leg.foot_body)
    arm = np.array([m.dof_armature[i] for i in leg.dofs])
    return H, Jf[2].copy(), arm, np.array(leg.dofs)


def _assemble_embodied(H, K_struct, J, ks, label, m_e_ref, readout="energy",
                       k_i=K_I, zeta_contact=ZETA_CONTACT, D_struct=None) -> Passive:
    """Identical in structure to `systems._assemble`, on a mass matrix read from MuJoCo."""
    J = np.atleast_2d(J)
    A, B_F, K, _ = assemble(H, K_struct, J, k_i, m_e_ref, zeta_contact,
                            D_struct=D_struct)
    return Passive(A=A, B_F=B_F, C=_readout(readout, H, K, 1.0, omega_ref()),
                   H=H, K=K, J=J, ks=ks, label=label, readout=readout)


def go2_proprioceptive(readout="energy", clamped: bool = True, **kw) -> Passive:
    """Rotors geared rigidly to the links -- Menagerie's own model, armature included."""
    H, J, _, _ = go2_mass_and_jacobian(clamped=clamped)
    m_e = locked_effective_mass(H, J)
    return _assemble_embodied(H, np.zeros_like(H), J, None, "go2 proprioceptive",
                              m_e, readout, **kw)


def go2_series_elastic(ks: float, readout="energy", zeta_series=ZETA_SERIES,
                       clamped: bool = True, **kw) -> Passive:
    """Each leg joint's rotor moved onto its own coordinate behind a spring `ks`."""
    Hp, J, arm, dofs = go2_mass_and_jacobian(clamped=clamped)
    n0, nr = Hp.shape[0], len(dofs)
    n = n0 + nr

    H = np.zeros((n, n))
    H[:n0, :n0] = Hp
    for i, dof in enumerate(dofs):
        H[dof, dof] -= arm[i]           # armature leaves the link
        H[n0 + i, n0 + i] = arm[i]      # and becomes a rotor of its own

    # one difference spring per joint: ks * (theta_j - theta_m)
    S = np.zeros((nr, n))
    for i, dof in enumerate(dofs):
        S[i, dof], S[i, n0 + i] = 1.0, -1.0
    K_s = ks * (S.T @ S)
    b_s = 2.0 * zeta_series * np.sqrt(ks * float(np.mean(arm)))

    m_e = locked_effective_mass(Hp, J)   # Hp, not H: the locked reference
    return _assemble_embodied(H, K_s, np.concatenate([J, np.zeros(nr)]), float(ks),
                              "go2 SEA", m_e, readout, D_struct=b_s * (S.T @ S), **kw)


# ---------------------------------------------------------------------------------------
# The same question in the full simulation.
#
# Everything above is a linear model that takes its mass matrix from MuJoCo. The functions
# below linearise nothing: they integrate the nonlinear articulated dynamics against MuJoCo's
# contact solver and recover the response matrix by finite differences through the simulator.
#
# Every torque is zero, so the leg collapses slowly under its own weight -- 27 mm of trunk
# travel, 0.08 rad at the knee. That is the nominal trajectory, and differencing against the
# undisturbed run removes it exactly. Contact is held for the whole window, checked not assumed.
# ---------------------------------------------------------------------------------------

#: MuJoCo timestep. Menagerie's 0.25 ms is not enough -- the rotor is sub-integrated outside
#: MuJoCo and coupled back explicitly, unstable above roughly k_s = 1e5, and the failure is
#: silent in the output. `simulate` returns a stability flag every caller must check.
SIM_DT = 5.0e-5

#: The sign of series elasticity's effect is set by the ground: compliance destroys contact
#: information on a hard floor and helps on a soft one, crossing over near k_i = 3.5e4, which
#: is roughly Menagerie's own contact. A lab floor is the stiff one, so `k_i = 1e6` is the
#: physical setting -- and the simulation's default sits on the wrong side of the crossover.
#: Every simulated bit count must therefore state its floor, which is what `contact=` is for.
#:
#: The stiff-limit residual is a resolution floor, not a bug: comparing two differently
#: structured MuJoCo models costs about 0.3 bits at a 100 Hz band and about 4 at 200 Hz. Use
#: the simulation only for effects well above it.

#: Disturbance amplitude for the finite differences, in newtons. Small enough to stay linear
#: (bits move 0.09 across 2 -> 20 N, a factor of ten) and large enough to sit well above the
#: integrator's own noise. The rig weighs 37 N, so this is a light touch, not a kick.
SIM_EPS = 5.0


def _add_series_springs(spec, ks: float, zeta_series: float, leg: str = "FR"):
    """Move each leg joint's armature onto a new rotor coordinate behind a spring."""
    added = 0
    for b in spec.bodies:
        for j in list(b.joints):
            if j.name.startswith(f"{leg}_") and j.name.endswith("_joint"):
                # b_s from the joint's own armature, not a module constant: the armature
                # is the reflected rotor inertia, which is the premise of reading the
                # original joint as the rotor.
                i_rot = float(np.atleast_1d(j.armature)[0]) or I_ROT_GO2
                nj = b.add_joint()
                nj.name, nj.type = j.name + "_spring", j.type
                nj.axis, nj.pos = j.axis, j.pos
                nj.stiffness = [ks, 0.0, 0.0]
                nj.damping = [2.0 * zeta_series * np.sqrt(ks * i_rot), 0.0, 0.0]
                nj.armature = 0.0
                nj.limited = 0
                added += 1
                # Menagerie's default class carries frictionloss = 0.2 N m. Left alone the SEA
                # rig gets six dry-friction elements to the rigid one's three, and dry friction
                # does not vanish when the spring locks.
                nj.frictionloss = 0.0

    # Silence here used to mean "rigid": a model not following the `{leg}_*_joint`
    # convention matched nothing, got no springs, and reported itself as a series elastic
    # design. Not a crash -- a silently wrong experiment.
    if not added:
        names = sorted(j.name for b in spec.bodies for j in b.joints if j.name)
        raise ValueError(
            f"no joint matched '{leg}_*_joint', so no series springs were added and this "
            f"model would have run as a RIGID one. Joints present: {names[:8]}"
            f"{' ...' if len(names) > 8 else ''}")


def _foot_spring(m, d, leg, k_i, b_i, z0):
    """Explicit vertical foot spring, in place of MuJoCo's solved contact."""
    import mujoco
    Jf, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    pos = d.geom_xpos[leg.foot_geom].copy()
    pos[2] -= leg.foot_radius
    mujoco.mj_jac(m, d, Jf, jr, pos, leg.foot_body)
    return -k_i * (float(pos[2]) - z0) - b_i * float(Jf[2] @ d.qvel)


def _locked_dofs(m):
    """Velocity indices that survive when the series transmission is locked."""
    import mujoco
    keep = []
    for i in range(m.nv):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.dof_jntid[i]))
        if not (name or "").endswith("_spring"):
            keep.append(i)
    return np.asarray(keep, dtype=int)


def _ground_gains(m, d, leg, k_i):
    """(b_i, z0) for the explicit ground, using the analytical model's own rule."""
    import mujoco
    H = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, H)
    Jf, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    pos = d.geom_xpos[leg.foot_geom].copy()
    pos[2] -= leg.foot_radius
    mujoco.mj_jac(m, d, Jf, jr, pos, leg.foot_body)
    keep = _locked_dofs(m)
    Hk, Jk = H[np.ix_(keep, keep)], Jf[2][keep]
    m_e = 1.0 / float(Jk @ np.linalg.solve(Hk, Jk))
    return ground_damping(k_i, m_e, ZETA_CONTACT), float(pos[2])


def simulate(ks: float | None, amps, dt: float = SIM_DT, zeta_series=ZETA_SERIES,
             t_window: float | None = None, contact="menagerie", ground="spring",
             k_i: float = K_I, gravity: bool = True, vertical="fixed"):
    """Integrate the passive leg under a band-limited foot force; return the state history."""
    # Returns (X, ok). `ok` must be checked: an unstable run reports a plausible number
    # rather than failing -- 572 bits against a true 96.0.
    import mujoco
    from passive_info.information import T_WINDOW, mode_basis

    t_window = T_WINDOW if t_window is None else t_window
    m, d, leg = _sim_rig(ks, dt, zeta_series, contact, ground, gravity, vertical=vertical)
    n_t = int(round(t_window / m.opt.timestep))
    f = mode_basis(n_t, len(amps)) @ np.asarray(amps, float)
    b_i, z0 = _ground_gains(m, d, leg, k_i) if ground == "spring" else (0.0, 0.0)

    X, held = [], True
    for i in range(n_t):
        d.xfrc_applied[:] = 0.0
        d.xfrc_applied[leg.foot_body, 2] = f[i]
        if ground == "spring":
            d.xfrc_applied[leg.foot_body, 2] += _foot_spring(m, d, leg, k_i, b_i, z0)
        mujoco.mj_step(m, d)
        if ground != "spring":
            held &= bool(np.linalg.norm(mjcf.ground_reaction(m, d, leg)) > 1e-9)
        X.append([*d.qpos, *d.qvel])
    X = np.asarray(X)
    return X, bool(held and np.all(np.isfinite(X)))


def _sim_rig(ks, dt, zeta_series=ZETA_SERIES, contact="menagerie", ground="spring",
             gravity=True, vertical="fixed", xml: str | None = None,
             leg_name: str = "FR"):
    import mujoco

    def hook(sp):
        if ks is not None:
            _add_series_springs(sp, ks, zeta_series, leg=leg_name)
        if ground == "spring":
            for g in sp.geoms:                 # no solved contact anywhere
                g.contype = g.conaffinity = 0

    # `vertical="fixed"` clamps the trunk, for both experiments. The controlled one needs it:
    # with a free base a commanded force accelerates the trunk instead of reaching the ground.
    # `vertical=True` still builds the sliding rig, so free-body results stay reproducible.
    m, d = _compile(xml, (leg_name,), vertical, contact=contact, spec_hook=hook)
    leg = mjcf.index_legs(m, GO2)[leg_name]
    d.qpos[leg.qpos] = STANCE_QPOS
    mujoco.mj_forward(m, d)
    # Lower the trunk onto the floor -- only if there is a trunk to lower. Unconditionally,
    # `d.qpos[0]` is the slide's coordinate on the free rig and the hip angle on the clamped
    # one. Clamped, the pose fixes the foot's height and there is nothing to lower.
    slide = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "trunk_z")
    if slide >= 0:
        d.qpos[m.jnt_qposadr[slide]] -= float(d.geom_xpos[leg.foot_geom][2]
                                              - leg.foot_radius)
    elif ground != "spring":
        # A solved contact needs the foot on the floor, and welded there is no coordinate
        # that puts it there: the foot hangs 157 mm up, `ncon` is 0, and every bit count is
        # about a leg in mid-air. Refused here rather than discovered downstream.
        raise ValueError(
            f"ground={ground!r} needs the foot on the floor, and a clamped trunk "
            f"(vertical={vertical!r}) leaves it "
            f"{1e3 * float(d.geom_xpos[leg.foot_geom][2] - leg.foot_radius):.0f} mm above "
            f"it. Use ground='spring', or vertical=True to study the solved contact.")
    if dt is not None:
        m.opt.timestep = dt
    if not gravity:
        # Gravity only sets the operating point; the analytical model has no g term at all,
        # so switching it off makes the nominal exactly static and the finite differences a
        # linearisation about the stance pose rather than about a collapsing trajectory.
        m.opt.gravity[:] = 0.0
    mujoco.mj_forward(m, d)
    return m, d, leg


def _sim_metric(ks, dt=SIM_DT, zeta_series=ZETA_SERIES, contact="menagerie",
                ground="spring", gravity=True, vertical="fixed", k_i: float = K_I):
    """`E = diag(K, H)` for the simulated rig: the mechanism's own stored energy."""
    import mujoco

    m, d, leg = _sim_rig(ks, dt, zeta_series, contact, ground, gravity, vertical=vertical)
    n = m.nv
    H = np.zeros((n, n))
    mujoco.mj_fullM(m, d, H)

    K = np.zeros((n, n))
    for j in range(m.njnt):
        K[m.jnt_dofadr[j], m.jnt_dofadr[j]] = m.jnt_stiffness[j]
    Jf, jr = np.zeros((3, n)), np.zeros((3, n))
    pos = d.geom_xpos[leg.foot_geom].copy()
    pos[2] -= leg.foot_radius
    mujoco.mj_jac(m, d, Jf, jr, pos, leg.foot_body)
    Jz = Jf[2]
    K += k_i * np.outer(Jz, Jz)

    return energy_metric(K, H)


def sim_profile_bits(ks: float | None, eps: float = SIM_EPS, dt: float = SIM_DT,
                     sigma: float | None = None, zeta_series=ZETA_SERIES,
                     contact="menagerie", ground="spring", k_i: float = K_I,
                     gravity: bool = True, vertical="fixed") -> dict:
    """Bits about a band-limited foot force, measured through the full simulation."""
    from passive_info.information import SIGMA_A, SIGMA_ENERGY, n_modes

    K = n_modes()
    kw = dict(contact=contact, ground=ground, k_i=k_i, gravity=gravity, vertical=vertical)
    base, ok = simulate(ks, np.zeros(K), dt, zeta_series, **kw)
    cols = []
    for j in range(K):
        a = np.zeros(K)
        a[j] = eps
        xj, ok_j = simulate(ks, a, dt, zeta_series, **kw)
        ok &= ok_j
        cols.append((xj - base) / eps)
    if not ok:
        raise FloatingPointError(
            f"unusable run at k_s={ks}, dt={dt}: refuse to report a bit count")

    M = np.stack(cols, axis=-1)
    E = _sim_metric(ks, dt, zeta_series, contact, ground, gravity, vertical=vertical,
                    k_i=k_i)
    gram = sum(M[t].T @ E @ M[t] for t in range(M.shape[0]))
    gram *= (SIGMA_A / (SIGMA_ENERGY if sigma is None else sigma)) ** 2
    sign, logdet = np.linalg.slogdet(np.eye(K) + gram)
    eig = np.linalg.eigvalsh(gram)[::-1]
    per = 0.5 * np.log2(1.0 + np.clip(eig, 0.0, None))
    return dict(bits=float(0.5 * logdet / np.log(2.0)), eigbits=per,
                rank=int(np.sum(per > 1.0)), n_modes=K, dt=dt, stable=True)
