"""The leg commanded to produce a force, in the full simulation."""
from __future__ import annotations

import numpy as np
import warp as wp

from passive_info.control import F_CUT, SIGMA_DEFLECTION

#: Bit depth of the commanded torque, as issued through real hardware. The command needs a
#: declared resolution for the same reason the state does: for a proprioceptive leg it is a
#: deterministic function of the disturbance, so declaring it noise-free makes the quantity
#: ill-posed rather than exactly observed. 16 bits over Menagerie's actuator range.
#:
#: The reported information depends on it, and this design's action channel is small by
#: construction, so sweep it rather than quoting one value.
COMMAND_BITS = 16

#: Streaming settings, imported rather than restated, so the controlled and passive paths
#: cannot drift apart. They did, and at reuse 16 the controlled estimate stopped converging
#: at all rather than merely losing accuracy.
from passive_info.sim_info import STREAM_BUFFER, STREAM_REFRESH
from passive_info.embodied import SIM_DT, _ground_gains, _sim_rig
from passive_info.systems import K_I, ZETA_SERIES
from passive_info.warp_rig import _foot_point, _record


@wp.kernel
def _control_and_force(point: wp.array(dtype=wp.vec3), jacp: wp.array3d(dtype=float),
                       qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float),
                       motor_dof: wp.array(dtype=int), spring_qadr: wp.array(dtype=int),
                       tau_d: wp.array2d(dtype=float), fd_t: wp.array(dtype=float),
                       static_jac: int, noise: wp.array3d(dtype=float),
                       step: int, n_sub: int, ks: float, k_p: float, k_d: float,
                       alpha: float, dt_ctrl: float,
                       foot_body: int, k_i: float, b_i: float, z0: float,
                       fmode: wp.array2d(dtype=float),
                       deriv: wp.array2d(dtype=float), prev: wp.array2d(dtype=float),
                       held: wp.array2d(dtype=float),
                       u_out: wp.array3d(dtype=float),
                       f_out: wp.array2d(dtype=float),
                       qfrc: wp.array2d(dtype=float),
                       xfrc: wp.array2d(dtype=wp.spatial_vector)):
    """One control step: torque command per joint, plus the unknown force and the ground."""
    w = wp.tid()
    n_j = motor_dof.shape[0]

    # The controller runs slower than the integrator, by design. Updating every
    # integration step made the derivative term differentiate the sensor's white noise over
    # dt = 5e-5 and 63 of 64 worlds diverged. Between updates the command is held.
    update = (step % n_sub) == 0
    for i in range(n_j):
    # tau_d = J(q)^T f_d, evaluated at the CURRENT configuration -- what a real
    # implementation does. `static_jac` freezes it at the stance pose, which makes the
    # command identical in every world and I(a;u) zero by construction.
        if static_jac == 1:
            td = tau_d[step, i]
        else:
            td = jacp[w, 2, motor_dof[i]] * fd_t[step]
        u = td
        if ks > 0.0:
            if update:
                # sign: MuJoCo applies -k*q to a joint carrying stiffness, so the torque
                # delivered through the spring to the link is -k_s * theta_rel. Measuring
                # +k_s * theta_rel made `tau_d - meas` add the delivered torque to the command
                # -- positive feedback, and it diverged at every gain. Caught open loop, where
                # tau_d read [-1367, 3366, -982] against k_s*theta_rel [1596, -3513, 1202].
                meas = -ks * qpos[w, spring_qadr[i]] + noise[w, step, i]
                err = td - meas
                deriv[w, i] += alpha * ((err - prev[w, i]) / dt_ctrl - deriv[w, i])
                prev[w, i] = err
                held[w, i] = k_p * err + k_d * deriv[w, i]
            u = td + held[w, i]
        qfrc[w, motor_dof[i]] = u
        u_out[w, step, i] = u

    z = point[w][2]
    vz = float(0.0)
    for i in range(qvel.shape[1]):
        vz += jacp[w, 2, i] * qvel[w, i]
    # The force delivered into the ground is the ground's reaction, sign flipped -- the same
    # quantity `control.py` calls C_f. Recorded because "did the controller work" is answered
    # by comparing this against the command, not by any information measure.
    f_ground = -k_i * (z - z0) - b_i * vz
    f_out[w, step] = -f_ground
    f = fmode[w, step] + f_ground
    xfrc[w, foot_body] = wp.spatial_vector(0.0, 0.0, f, 0.0, 0.0, 0.0)


def _nominal_jacobian(m, d, leg):
    """Vertical foot Jacobian at the stance pose, for the static eq. (44) map."""
    import mujoco
    Jf, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    pos = d.geom_xpos[leg.foot_geom].copy()
    pos[2] -= leg.foot_radius
    mujoco.mj_jac(m, d, Jf, jr, pos, leg.foot_body)
    return Jf[2].copy()


def _spring_qadr(m, leg):
    """qpos addresses of the series-spring joints, in the same order as the motor dofs."""
    import mujoco
    out = []
    for dof in leg.dofs:
        j = int(m.dof_jntid[dof])
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_spring")
        if sid < 0:
            return None
        out.append(int(m.jnt_qposadr[sid]))
    return np.array(out, np.int32)


def rollout_controlled(ks: float | None, amps: np.ndarray, f_desired, dt: float = SIM_DT,
                       zeta_series: float = ZETA_SERIES, t_window: float | None = None,
                       k_i: float = K_I, gravity: bool = True, zeta_loop: float = 0.7,
                       static_jac: bool = False,
                       sigma_deflection: float = SIGMA_DEFLECTION, f_cut: float = F_CUT,
                       seed: int = 0, device: str | None = None):
    """Batched rollouts with a commanded force AND an unknown disturbance."""
    import mujoco
    import mujoco_warp as mjw

    from passive_info.information import T_WINDOW, mode_basis

    amps = np.atleast_2d(np.asarray(amps, float))
    nworld = len(amps)
    t_window = T_WINDOW if t_window is None else t_window

    # clamped. Not a convenience -- see `control.py`: with a free base, pressing the foot down
    # accelerates the trunk upward and the command has no steady state. Measured on the free
    # rig before this was fixed: 77.8 N delivered for a 40 N command (118% error) even OPEN
    # LOOP, and 264 kN on a soft spring.
    m, d, leg = _sim_rig(ks, dt, zeta_series, "menagerie", "spring", gravity,
                         vertical="fixed")
    n_t = int(round(t_window / m.opt.timestep))
    n_j = len(leg.dofs)
    b_i, z0 = _ground_gains(m, d, leg, k_i)

    Jz = _nominal_jacobian(m, d, leg)
    t = np.arange(n_t) * m.opt.timestep
    fd = np.asarray([float(f_desired(x)) for x in t])
    tau_d = np.outer(fd, Jz[np.asarray(leg.dofs)])          # (n_t, n_j), eq. (44)

    sq = _spring_qadr(m, leg)
    is_sea = ks is not None and sq is not None

    # Gated on the loop's own stability: past the boundary the design runs open loop rather
    # than diverging. A stiff enough spring is a coarse enough torque sensor that closing a
    # loop on it is worse than not. The analytic predictor and the simulation agree on where
    # that boundary is.
    loop_open = False
    if is_sea:
        from passive_info.control import closed_loop_spectral_radius, sea as _sea_rig
        rho = float(closed_loop_spectral_radius(_sea_rig(float(ks))))
        if rho >= 1.0:
            loop_open = True
            print(f"    k_s={ks:g}: torque loop spectral radius {rho:.3f} >= 1 -- running "
                  f"OPEN LOOP (no closed-loop system exists at this control rate)",
                  flush=True)
    sigma_tau = (float(ks) * sigma_deflection) if is_sea else 0.0
    # Gains scaled to the spring, as `control.gains_for`: a derivative term that damps a 13 Hz
    # resonance is far too large for a 190 Hz one, so k_d falls as the spring stiffens.
    closed = is_sea and not loop_open
    k_p = 1.0 if closed else 0.0
    k_d = (2.0 * zeta_loop * np.sqrt(0.01 / float(ks))) if closed else 0.0
    # Controller at FS_CTRL, integrator at 1/dt. n_sub is how many integration steps pass
    # between control updates; the command is held across them.
    from passive_info.control import FS_CTRL
    n_sub = max(1, int(round(1.0 / (FS_CTRL * m.opt.timestep))))
    dt_ctrl = n_sub * m.opt.timestep
    alpha = dt_ctrl / (dt_ctrl + 1.0 / (2.0 * np.pi * f_cut))

    rng = np.random.default_rng(seed)
    noise = (rng.normal(0.0, sigma_tau, size=(nworld, n_t, n_j)).astype(np.float32)
             if sigma_tau > 0 else np.zeros((nworld, n_t, n_j), np.float32))
    f = mode_basis(n_t, amps.shape[1]) @ amps.T             # (n_t, nworld)

    device = device or ("cuda" if wp.get_cuda_device_count() else "cpu")
    with wp.ScopedDevice(device):
        wm = mjw.put_model(m)
        wd = mjw.put_data(m, d, nworld=nworld)

        a_fmode = wp.array(np.ascontiguousarray(f.T, dtype=np.float32))
        a_tau = wp.array(np.ascontiguousarray(tau_d, dtype=np.float32))
        a_fd = wp.array(np.ascontiguousarray(fd, dtype=np.float32))
        a_noise = wp.array(np.ascontiguousarray(noise, dtype=np.float32))
        a_mdof = wp.array(np.asarray(leg.dofs, np.int32), dtype=int)
        a_sq = wp.array(sq if is_sea else np.zeros(n_j, np.int32), dtype=int)
        point = wp.zeros(nworld, dtype=wp.vec3)
        bodyid = wp.array(np.full(nworld, leg.foot_body, np.int32), dtype=int)
        jacp = wp.zeros((nworld, 3, m.nv), dtype=float)
        deriv = wp.zeros((nworld, n_j), dtype=float)
        prev = wp.zeros((nworld, n_j), dtype=float)
        held = wp.zeros((nworld, n_j), dtype=float)
        X = wp.zeros((nworld, n_t, m.nq + m.nv), dtype=float)
        U = wp.zeros((nworld, n_t, n_j), dtype=float)
        Fd = wp.zeros((nworld, n_t), dtype=float)

        for i in range(n_t):
            wp.launch(_foot_point, dim=nworld,
                      inputs=[wd.geom_xpos, leg.foot_geom, float(leg.foot_radius), point])
            mjw.jac(wm, wd, jacp, None, point, bodyid)
            wd.qfrc_applied.zero_()
            wp.launch(_control_and_force, dim=nworld,
                      inputs=[point, jacp, wd.qpos, wd.qvel, a_mdof, a_sq, a_tau, a_fd,
                              int(static_jac), a_noise,
                              i, int(n_sub), float(ks) if closed else 0.0,
                              float(k_p), float(k_d), float(alpha), float(dt_ctrl),
                              leg.foot_body, float(k_i), float(b_i), float(z0), a_fmode,
                              deriv, prev, held, U, Fd, wd.qfrc_applied,
                              wd.xfrc_applied])
            mjw.step(wm, wd)
            wp.launch(_record, dim=nworld, inputs=[wd.qpos, wd.qvel, i, X])
        wp.synchronize_device(device)
        Xh, Uh, Fh = X.numpy(), U.numpy(), Fd.numpy()

    ok = np.isfinite(Xh).all(axis=(1, 2)) & np.isfinite(Uh).all(axis=(1, 2))
    return Xh, Uh, Fh, ok


def _controlled_refill(ks, band_lo, band_hi, held, K, sigma_a, sigma, E_half, f_desired,
                       nworld, dt, zeta_series, device, rng, gen, state, want,
                       closed_loop, sigma_u):
    """One block of commanded rollouts, returning `(a_band, observation)` on the device."""
    import torch

    def refill(n):
        a = rng.normal(0.0, sigma_a, size=(n, K))
        if band_lo:
            a[:, :band_lo] = held
        X, U, F, ok = rollout_controlled(ks, a, f_desired, dt=dt, zeta_series=zeta_series,
                                         gravity=False, device=device)
        state["total"] += n
        state["lost"] += int((~ok).sum())
        if not ok.any():
            raise FloatingPointError(f"every controlled rollout non-finite at k_s={ks}")
        Xt = torch.tensor(X[ok], dtype=torch.float32, device=device)
        Ut = torch.tensor(U[ok], dtype=torch.float32, device=device)
        Ut = Ut.reshape(len(Ut), -1)
        # The command is observed at finite resolution, like every other signal here.
        Ut = Ut + sigma_u * torch.randn(Ut.shape, generator=gen, device=device,
                                        dtype=Ut.dtype)
        if want == "action":
            obs = Ut
        elif want == "state":
            Y = Xt @ E_half.T
            Y += sigma * torch.randn(Y.shape, generator=gen, device=device, dtype=Y.dtype)
            obs = Y.reshape(len(Y), -1)
        else:
            Y = Xt @ E_half.T
            Y += sigma * torch.randn(Y.shape, generator=gen, device=device, dtype=Y.dtype)
    # The command is always part of the joint observation: with the Jacobian at the current
    # pose it always varies with the disturbance. An earlier version appended it only when
    # the loop was closed, which was right only for the frozen-Jacobian model.
    #
    # The dust case is guarded in `critic._stats`, which neutralises near-constant columns
    # rather than rescaling rounding noise to unit variance.
            obs = torch.cat([Y.reshape(len(Y), -1), Ut], dim=1)
        at = torch.tensor(a[ok, band_lo:band_hi], dtype=torch.float32, device=device)
        return at, obs
    return refill


def sim_chain_bits_controlled(ks: float | None, band_hz: float = 100.0, group: int = 1,
                              steps: int = 800, batch: int = 1024,
                              buffer: int = STREAM_BUFFER,
                              refresh_every: int = STREAM_REFRESH, amp: float = 40.0, freq: float = 20.0,
                              sigma_a: float | None = None, sigma: float | None = None,
                              command_bits: int = COMMAND_BITS,
                              seed: int = 0, dt: float = SIM_DT,
                              zeta_series: float = ZETA_SERIES, device: str | None = None,
                              verbose: bool = True, **critic_kw) -> dict:
    """`I(a;x,u)`, `I(a;u)` and the conditional `I(a;x|u)` per band, through the simulation."""
    # The proprioceptive leg runs open loop, so I(a;z_u) = 0 and its three quantities must
    # coincide. If they do not, the estimator is inventing information.
    import numpy as np
    import torch

    from mechinfo.critic import infonce_bits_streaming
    from passive_info.information import SIGMA_A, SIGMA_ENERGY, n_modes
    from passive_info.sim_info import _energy_half

    sigma_a = SIGMA_A if sigma_a is None else sigma_a
    sigma = SIGMA_ENERGY if sigma is None else sigma
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # vertical="fixed": the controlled rig is clamped, so its metric is the clamped one.
    E_half = _energy_half(ks, dt, zeta_series, "menagerie", "spring", False,
                          device=device, vertical="fixed")
    K = n_modes(band_hz=band_hz)

    def f_desired(t):
        return amp * np.sin(2.0 * np.pi * freq * t)

    # Command resolution from the actuator's own torque range, at the declared bit depth.
    m_tmp, _, leg_tmp = _sim_rig(ks, dt, zeta_series, "menagerie", "spring", False,
                                 vertical="fixed")
    tau_span = float(np.max(np.abs(m_tmp.actuator_ctrlrange[:len(leg_tmp.dofs)]))) * 2.0
    sigma_u = tau_span / (2.0 ** command_bits)
    if verbose:
        print(f"  command resolution: {command_bits} bits over {tau_span:.2f} N m "
              f"-> sigma_u = {sigma_u:.3e} N m", flush=True)

    # Is the loop actually closed at this stiffness? Past the stability boundary the design
    # runs open loop, so its command carries nothing and the conditional is the unconditioned
    # information -- the same structural statement as the rigid design's.
    closed = ks is not None
    if closed:
        from passive_info.control import closed_loop_spectral_radius, sea as _sea_rig
        closed = float(closed_loop_spectral_radius(_sea_rig(float(ks)))) < 1.0

    edges = list(range(0, K, group)) + [K]
    joint, action, cmi, dropped = [], [], [], []

    for lo, hi in zip(edges[:-1], edges[1:]):
        rng = np.random.default_rng(seed + 1000 * lo)
        gen = torch.Generator(device=device).manual_seed(seed + 1000 * lo)
        held = rng.normal(0.0, sigma_a, size=lo)
        st = {"lost": 0, "total": 0}
        mk = lambda w: _controlled_refill(ks, lo, hi, held, K, sigma_a, sigma, E_half,
                                          f_desired, buffer, dt, zeta_series, device,
                                          rng, gen, st, w, closed, sigma_u)
        # split_at tells the critic where the state ends and the command begins, so each gets
        # its own encoder. Without it the joint estimate fell below the command-only estimate
        # on 10 of 17 bands -- impossible, since I(a;x,u) >= I(a;u).
        n_state = int(np.prod(E_half.shape[:1])) * int(round(0.085 / dt))
        rj = infonce_bits_streaming(mk("joint"), batch=batch, steps=steps, buffer=buffer,
                                    refresh_every=refresh_every, device=device, seed=seed,
                                    verbose=False, split_at=n_state, **critic_kw)
        # Always estimated, never assumed. An earlier version set I(a;u) = 0 whenever the
        # torque loop was open, calling it structural. It was not: the zero came from freezing
        # the Jacobian, and with J evaluated at the current pose the command responds to the
        # disturbance through the encoders whether or not a torque loop is closed. Whether that
        # channel is large enough to resolve is a measurement.
        ra = infonce_bits_streaming(mk("action"), batch=batch, steps=steps, buffer=buffer,
                                    refresh_every=refresh_every, device=device, seed=seed,
                                    verbose=False, **critic_kw)
        i_u = ra["bits"]
        joint.append(rj["bits"]); action.append(i_u); cmi.append(rj["bits"] - i_u)
        dropped.append(st["lost"] / max(st["total"], 1))
        if verbose:
            print(f"  band {lo:>2}-{hi:<2}  joint {rj['bits']:>7.4f}  action {i_u:>7.4f}  "
                  f"cmi {cmi[-1]:>7.4f}  dropped {dropped[-1]:.2%}", flush=True)

    return dict(joint=np.array(joint), action=np.array(action), cmi=np.array(cmi),
                closed_loop=closed, dropped=float(max(dropped)) if dropped else 0.0)
