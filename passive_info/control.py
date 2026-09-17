"""Commanding a force through each transmission, and measuring what arrives."""
from __future__ import annotations

from dataclasses import dataclass


import numpy as np
import scipy.linalg as sla

from passive_info.channel import chain, cond_mi_bits, gibbs_pair
from passive_info.information import (BAND_HZ, FS, SIGMA_A, SIGMA_ENERGY, T_WINDOW,
                                      mode_basis, n_modes)
from mechinfo.mechanics import (assemble, energy_metric, ground_damping,
                                rigid_plant, sea_plant, sym_sqrt, zoh)
from passive_info.systems import (I_ROT, K_I, M_L, R, ZETA_CONTACT, ZETA_SERIES,
                                  reference_m_e)


@dataclass(frozen=True)
class Rig:
    """A clamped-body leg with a motor-torque input and a ground under the foot."""
    A: np.ndarray
    B_u: np.ndarray          #: motor torque -> state
    C_f: np.ndarray          #: state -> force delivered into the ground
    C_tau: np.ndarray        #: state -> spring torque (SEA only; zeros for the rigid leg)
#: The mechanism itself, kept rather than recovered. `_disturbed_plant` used to rebuild H
#: from module constants and read K back out of A through an inverse, so a rig built with
#: non-default masses silently got the default mechanism's noise metric.
    H: np.ndarray
    K: np.ndarray
    J: np.ndarray
    #: Series stiffness per spring. `None` for a rigid transmission; a scalar for one
    #: spring; an array for a mechanism with several, which is what a multi-joint series
    #: elastic leg has and what `ks is None` could not express.
    ks: float | np.ndarray | None
    label: str
    r: float = R
    #: Commanded torque per unit desired foot force, one entry per command channel --
    #: Wensing eq. (44)'s `S_l J^T`. `None` uses the scalar `r`, which is only meaningful
    #: for a single command; see `tau_for`.
    tau_map: np.ndarray | None = None

    @property
    def n(self) -> int:
        return self.A.shape[0] // 2

    @property
    def n_springs(self) -> int:
        """How many series springs, hence how many torque sensors and command channels."""
        return 0 if self.ks is None else int(np.atleast_1d(self.ks).size)

    @property
    def has_sensor(self) -> bool:
        """Whether there is a spring deflection to read."""
        return self.ks is not None

    @property
    def has_loop(self) -> bool:
        """Whether a torque loop is closed around it. Needs something to measure."""
        return self.has_sensor

    def tau_for(self, f_d: float):
        """Wensing eq. (44): the open-loop torque for a desired foot force."""
        if self.tau_map is not None:
            return np.asarray(self.tau_map, float) * f_d
        n_u = self.B_u.shape[1]
        if n_u != 1:
            raise ValueError(
                f"{self.label}: {n_u} command channels but no tau_map, so the torque for a "
                "desired foot force is undefined. Supply tau_map = S_l J^T.")
        return self.r * f_d


def _assemble(H, K_struct, J, ks, label, D_struct=None, r=R,
              k_i: float = K_I, zeta_contact: float = ZETA_CONTACT,
              m_e_ref: float | None = None):
    """A `Rig` from a mechanism and a ground. Mirrors `systems._assemble`."""
    # m_e_ref must be the locked-transmission reference; see mechanics.locked_effective_mass.
    n = H.shape[0]
    J = np.atleast_2d(J)
    m_e_ref = reference_m_e(r=r) if m_e_ref is None else m_e_ref
    A, _, K, _ = assemble(H, K_struct, J, k_i, m_e_ref, zeta_contact,
                          D_struct=D_struct)
    # The GROUND's damping, not the assembled D: for a series elastic rig D also carries the
    # spring's own b_s, and `C_f` -- the force delivered into the ground -- must see only the
    # ground. Recovering it from D[0,0] instead moved the tracking error of every SEA and
    # left the rigid leg alone, which is how it was caught.
    b_i = ground_damping(k_i, m_e_ref, zeta_contact)
    Hinv = np.linalg.inv(H)
    e = np.zeros((n, 1))
    e[n - 1, 0] = 1.0                      # motor acts on the last coordinate
    B_u = np.vstack([np.zeros((n, 1)), Hinv @ e])
    # Force delivered into the ground is minus the ground's reaction on the foot.
    C_f = -np.hstack([k_i * J, b_i * J])
    C_tau = np.zeros((1, 2 * n))
    if ks is not None:                     # tau_s = k_s (theta_m - theta_j)
        C_tau[0, n - 2], C_tau[0, n - 1] = -ks, ks
    return Rig(A=A, B_u=B_u, C_f=C_f, C_tau=C_tau, H=H, K=K, J=J,
               ks=ks, label=label, r=r)


def rigid(m_l: float = M_L, r: float = R, i_rot: float = I_ROT) -> Rig:
    """Rotor geared rigidly to the leg, body clamped. One coordinate: theta."""
    H, K_s, J, D_s = rigid_plant(m_l, r, i_rot)
    return _assemble(H, K_s, J, None, "proprioceptive", D_struct=D_s, r=r)


def sea(ks: float, m_l: float = M_L, r: float = R, i_rot: float = I_ROT,
        zeta_series: float = ZETA_SERIES) -> Rig:
    """Rotor behind a series spring, body clamped. Coordinates [theta_j, theta_m]."""
    H, K_s, J, D_s = sea_plant(ks, m_l, r, i_rot, zeta_series)
    return _assemble(H, K_s, J, float(ks), "SEA", D_struct=D_s, r=r)


def static_gain(rig: Rig) -> float:
    """Foot force delivered per unit steady motor torque. Should be 1/r for both."""
    g = rig.C_f @ np.linalg.solve(-rig.A, rig.B_u)          # (1, m_u)
    return float(g.item()) if g.size == 1 else g.ravel()


#: Control and simulation rate, Hz. Matches the rest of the package.
FS_CTRL = 4000.0

#: Angular resolution of the spring-deflection reading, rad. The SEA's torque sensor is its
#: spring, so torque resolution is `k_s * SIGMA_DEFLECTION` and a stiffer spring is a coarser
#: sensor. 1e-5 rad is roughly a 19-bit absolute encoder.
#:
#: k_s >= 1e5 is unstable even with a perfect sensor, because the spring's mode has left the
#: loop's reach. See `F_CUT`.
SIGMA_DEFLECTION = 1.0e-5

#: Corner of a first-order low-pass on the derivative term, Hz. Differentiating a noisy
#: torque reading amplifies the noise -- a controller design choice, not a property of any
#: mechanism.
#:
#: It is also part of the high-stiffness limit, alongside the sampling rate, and the two bind
#: separately: a mode cannot be damped by a term that has filtered it out, and cannot be
#: resolved by a loop sampling near its period. Neither says a stiff transmission is worse,
#: only that it is harder to close a loop around.
F_CUT = 400.0


def sigma_tau_for(rig: Rig, sigma_deflection: float = SIGMA_DEFLECTION):
    """Torque-sensing resolution implied by reading the spring to a fixed angle."""
    if not rig.has_sensor:
        return 0.0
    ks = np.atleast_1d(rig.ks)
    out = ks * sigma_deflection
    return float(out.item()) if out.size == 1 else out


def _deriv_alpha(dt: float, f_cut: float) -> float:
    """First-order low-pass coefficient for the derivative term, written once."""
    return dt / (dt + 1.0 / (2.0 * np.pi * f_cut))


def simulate(rig: Rig, f_desired, t_end: float = T_WINDOW, fs: float = FS_CTRL,
             sigma_tau: float | None = None, k_p: float = 0.0, k_i: float = 0.0,
             k_d: float = 0.0, f_cut: float = F_CUT, seed: int = 0) -> dict:
    """Command a force profile and record what the foot actually delivers."""
    dt = 1.0 / fs
    if sigma_tau is None:
        sigma_tau = sigma_tau_for(rig)
    Ad, Bd = zoh(rig.A, rig.B_u, dt)
    rng = np.random.default_rng(seed)
    n_t = int(round(t_end * fs)) + 1

    C_tau = np.atleast_2d(rig.C_tau)
    m, m_u = C_tau.shape[0], Bd.shape[1]
    x = np.zeros((rig.A.shape[0], 1))
    integ = prev_err = deriv = np.zeros(m)
    sig = np.broadcast_to(np.atleast_1d(sigma_tau).astype(float), (m,))
    alpha = _deriv_alpha(dt, f_cut)
    t = np.arange(n_t) * dt
    fd, fa = np.empty(n_t), np.empty(n_t)
    tau_c, tau_m = np.empty((n_t, m_u)), np.empty((n_t, m))

    for k in range(n_t):
        f_d = float(f_desired(t[k]))
        tau_d = np.broadcast_to(np.atleast_1d(rig.tau_for(f_d)).astype(float), (m_u,))
        meas = (C_tau @ x).ravel()
        if rig.has_sensor:
            meas = meas + rng.normal(0.0, sig)
        # `k_p or k_i or k_d` was fine while the gains were scalars and raises
        # "truth value of an array is ambiguous" once they are per-channel.
        if rig.has_loop and (np.any(k_p) or np.any(k_i) or np.any(k_d)):
            err = tau_d[:m] - meas
            integ = integ + err * dt
            # The derivative of a noisy signal needs filtering; the cut-off is what limits
            # how much damping the sensor will support, which is the real design constraint.
            deriv = deriv + alpha * ((err - prev_err) / dt - deriv)
            prev_err = err
            u = tau_d + k_p * err + k_i * integ + k_d * deriv
        else:
            u = tau_d                                   # Wensing eq. (44), open loop
        fd[k], fa[k] = f_d, float((rig.C_f @ x).item())
        tau_c[k], tau_m[k] = u, meas
        x = Ad @ x + Bd @ np.asarray(u, float).reshape(-1, 1)

    # One channel stays 1-D: `figures.plate_force_control` plots tau_meas against t directly.
    sq = lambda v: v[:, 0] if v.shape[1] == 1 else v
    return dict(t=t, f_desired=fd, f_delivered=fa, tau_cmd=sq(tau_c), tau_meas=sq(tau_m),
                label=rig.label, ks=rig.ks)


def tracking_error(run: dict, settle_from: float = 0.5) -> dict:
    """RMS and steady-state error over the last part of the run, as a fraction of command."""
    m = run["t"] >= settle_from * run["t"][-1]
    ref = np.abs(run["f_desired"][m]).mean()
    err = run["f_delivered"][m] - run["f_desired"][m]
    return dict(rms=float(np.sqrt(np.mean(err ** 2))),
                rms_frac=float(np.sqrt(np.mean(err ** 2)) / max(ref, 1e-12)),
                bias=float(err.mean()), bias_frac=float(err.mean() / max(ref, 1e-12)))


def gains_for(rig: Rig, zeta: float = 0.7, i_rot: float = I_ROT) -> dict:
    """Torque-loop gains scaled to the spring, as `dict(k_p=..., k_d=...)`."""
    # The spring's mode is at sqrt(k_s / I_rot), so one fixed gain set cannot span the
    # sweep: k_d = 2 zeta sqrt(I_rot / k_s), gentler feedback on a stiffer spring.
    if not rig.has_loop:
        return dict(k_p=0.0, k_d=0.0)              # the rigid leg runs open loop
    ks = np.atleast_1d(rig.ks)
    k_d = 2.0 * zeta * np.sqrt(np.broadcast_to(np.atleast_1d(i_rot), ks.shape) / ks)
    if ks.size == 1:
        return dict(k_p=1.0, k_d=float(k_d.item()))
    return dict(k_p=np.ones(ks.size), k_d=k_d)


# ---------------------------------------------------------------------------------------
# What the state says about the world that the action does not already say.
#
#     I(a ; z_x | z_u) = I(a ; z_x, z_u) - I(a ; z_u)
#
# With a loop running, part of what the state reveals is redundant with what the command
# reveals -- the robot reacted, so its action already encodes some of what happened. The
# conditional removes exactly that overlap, separating knowing-through-mechanics from
# knowing-through-sensing.
#
# The action is observed exactly, because the controller sent it, which is what makes the
# conditioning non-trivial. It also means I(a;z_u) and I(a;z_x) are not on a common footing:
# one is limited by the sensor resolution and the other by the state resolution. The
# structure is robust; the relative magnitudes depend on both constants.
# ---------------------------------------------------------------------------------------


def _disturbed_plant(rig: Rig):
    """(B_f, C) for the clamped rig: unknown foot force in, energy-whitened state out."""
    n = rig.n
    B_f = np.vstack([np.zeros((n, 1)), np.linalg.inv(rig.H) @ rig.J.T])
    return B_f, sym_sqrt(energy_metric(rig.K, rig.H))


def _closed_loop_responses(rig, closed, fs, band, f_cut, fs_obs=None,
                           t_window: float = T_WINDOW, fs_enc: float = FS):
    """Linear maps from the force modes and from the sensor noise, to the state and command."""
    # fs is the control rate, fs_obs the observation rate. Separate, or a faster controller
    # would silently grant a finer readout.
    # The observer cannot read faster than the loop runs, so a slow controller pulls the
    # readout down with it; a fast one leaves it at the encoder rate.
    fs_obs = min(fs_enc, fs) if fs_obs is None else fs_obs
    stride = int(round(fs / fs_obs))
    if stride < 1 or abs(fs - stride * fs_obs) > 1e-9:
        raise ValueError(f"control rate {fs} Hz must be an integer multiple of the "
                         f"observation rate {fs_obs} Hz")

    dt = 1.0 / fs
    n_t = int(round(t_window * fs)) + 1
    Phi = mode_basis(int(round(t_window * fs_obs)) + 1, n_modes(t_window, band))
    B_f, C = _disturbed_plant(rig)
    Ad, Bd_f = zoh(rig.A, B_f, dt)
    _, Bd_u = zoh(rig.A, rig.B_u, dt)
    g = gains_for(rig) if (closed and rig.has_loop) else dict(k_p=0.0, k_d=0.0)
    alpha = _deriv_alpha(dt, f_cut)
    obs = np.arange(stride - 1, n_t - 1, stride)          # control steps that are observed
    m = C.shape[0]

    C_tau = np.atleast_2d(rig.C_tau)
    m_s, m_u = C_tau.shape[0], Bd_u.shape[1]             # torque sensors, command channels

    def _gain(v):
        return np.diag(np.broadcast_to(np.atleast_1d(v).astype(float), (m_s,)))

    Kp, Kd = _gain(g["k_p"]), _gain(g["k_d"])

    def run(f_seq, n_seq):
        """One pass of the loop. Returns the readout at every step, and the command."""
        x = np.zeros((rig.A.shape[0], 1))
        deriv = prev = np.zeros(m_s)
        xs = np.empty((n_t - 1, m))
        us = np.empty((n_t - 1, m_u))
        for k in range(n_t - 1):
            err = -((C_tau @ x).ravel() + n_seq[k])         # command part is known
            deriv = deriv + alpha * ((err - prev) / dt - deriv)
            prev = err
            u = Kp @ err + Kd @ deriv
            us[k] = u
            x = Ad @ x + Bd_f * f_seq[k] + Bd_u @ u.reshape(-1, 1)
            xs[k] = (C @ x).ravel()
        return xs, us

    zero = np.zeros((n_t, m_s))
    cols = [run(Phi[np.arange(n_t - 1) // stride, j], zero) for j in range(Phi.shape[1])]
    Ga_x = np.column_stack([c[0][obs].ravel() for c in cols])
    Ga_u = np.column_stack([c[1].ravel() for c in cols])

    # One impulse per noise channel, then shift it: the loop is lti and starts from rest, so
    # the response to an impulse at step k is the step-0 response delayed by k. That makes
    # the map block Toeplitz -- `h[i - k]` above the diagonal, zero below -- and writing it
    # out beats simulating once per noise sample. With several sensors it is one such block
    # per (output, noise) channel pair, which is why the impulse is now a loop.
    def _causal(h, rows):
        """(len(rows), q, n_steps) with entry [i, :, k] = h[rows[i] - k], zero if k > row."""
        lag = rows[:, None] - np.arange(n_t - 1)[None, :]
        return np.where((lag >= 0)[:, None, :],
                        h[np.clip(lag, 0, None)].transpose(0, 2, 1), 0.0)

    hx, hu = [], []
    for c in range(m_s):
        e0 = np.zeros((n_t, m_s))
        e0[0, c] = 1.0
        a, b = run(np.zeros(n_t), e0)
        hx.append(_causal(a, obs))                       # (n_obs, m, n_t-1)
        hu.append(_causal(b, np.arange(n_t - 1)))        # (n_t-1, m_u, n_t-1)
    # Columns are ordered (step, noise channel), matching the ravel of an (n_t-1, m_s) draw.
    Gn_x = np.stack(hx, axis=-1).reshape(len(obs) * m, (n_t - 1) * m_s)
    Gn_u = np.stack(hu, axis=-1).reshape((n_t - 1) * m_u, (n_t - 1) * m_s)
    return Ga_x, Ga_u, Gn_x, Gn_u


def _sensor_noise_blocks(Gn_x, Gn_u, sig_n):
    """`(Gn_x N Gn_x^T, Gn_u N Gn_u^T, Gn_x N Gn_u^T)` for sensor noise of scale `sig_n`."""
    s = np.atleast_1d(sig_n).astype(float)
    if s.size == 1:
        f = float(s.item()) ** 2
        return f * (Gn_x @ Gn_x.T), f * (Gn_u @ Gn_u.T), f * (Gn_x @ Gn_u.T)
    w = np.tile(s, Gn_x.shape[1] // s.size)
    Wx, Wu = Gn_x * w, Gn_u * w
    return Wx @ Wx.T, Wu @ Wu.T, Wx @ Wu.T


def conditional_information(rig: Rig, closed: bool = True, fs: float = FS_CTRL,
                            band: float = BAND_HZ, f_cut: float = F_CUT,
                            sigma: float | None = None,
                            sigma_a: float | None = None,
                            t_window: float = T_WINDOW) -> dict:
    """I(a;x), I(a;u), I(a;x,u) and the conditional I(a;x|u), in bits."""
    # With the loop open the conditional equals the unconditioned information exactly,
    # which is the structural check that this computes what it claims.
    sigma = SIGMA_ENERGY if sigma is None else sigma
    sigma_a = SIGMA_A if sigma_a is None else sigma_a

    Ga_x, Ga_u, Gn_x, Gn_u = _closed_loop_responses(rig, closed, fs, band, f_cut,
                                                    t_window=t_window)
    sig_n = sigma_tau_for(rig) if (closed and rig.has_sensor) else 0.0

    Sx, Su = sigma_a ** 2 * (Ga_x @ Ga_x.T), sigma_a ** 2 * (Ga_u @ Ga_u.T)
    Sxu = sigma_a ** 2 * (Ga_x @ Ga_u.T)
    Nxx, Nuu, Nxu = _sensor_noise_blocks(Gn_x, Gn_u, sig_n)
    Nx = Nxx + sigma ** 2 * np.eye(Ga_x.shape[0])
    Nu = Nuu + 1e-18 * np.eye(Ga_u.shape[0])

    def ld(M):
        s, v = np.linalg.slogdet(M)
        return v if s > 0 else -np.inf

    bits = lambda S, N: 0.5 * (ld(S + N) - ld(N)) / np.log(2.0)
    i_x, i_u = bits(Sx, Nx), bits(Su, Nu)
    top = np.block([[Sx + Nx, Sxu + Nxu], [(Sxu + Nxu).T, Su + Nu]])
    bot = np.block([[Nx, Nxu], [Nxu.T, Nu]])
    i_joint = 0.5 * (ld(top) - ld(bot)) / np.log(2.0)
    return dict(i_state=i_x, i_action=i_u, i_joint=i_joint, cmi=i_joint - i_u)


def closed_loop_spectral_radius(rig: Rig, fs: float = FS_CTRL,
                                f_cut: float = F_CUT) -> float:
    """Largest discrete pole magnitude of the torque loop. Above 1 the loop diverges."""
    # The loop carries two states beyond the plant, so stability cannot be read off the
    # mechanism. It is the sampled controller that fails first.
    F, _, _ = closed_loop_matrix(rig, fs, f_cut)
    return float(np.abs(np.linalg.eigvals(F)).max())


def closed_loop_matrix(rig: Rig, fs: float = FS_CTRL, f_cut: float = F_CUT):
    """`(F, L, n)` for the sampled torque loop, on the augmented state."""
    dt = 1.0 / fs
    Ad, Bd_u = zoh(rig.A, rig.B_u, dt)
    g = gains_for(rig)
    alpha = _deriv_alpha(dt, f_cut)
    n, C = Ad.shape[0], np.atleast_2d(rig.C_tau)
    m, m_u = C.shape[0], Bd_u.shape[1]           # sensors, command channels
    d0, e0 = n, n + m                            # where each controller block starts

    def _gain(v):
        """A per-channel gain as a diagonal `(m_u, m)`. A scalar broadcasts to every loop."""
        return np.diag(np.broadcast_to(np.atleast_1d(v).astype(float), (m,)))

    Kp, Kd = _gain(g["k_p"]), _gain(g["k_d"])

    F = np.zeros((n + 2 * m, n + 2 * m))         # [plant | filtered derivative | prev error]
    F[d0:e0, :n] = alpha * (-C) / dt             # d_{k+1} from the current error
    F[d0:e0, d0:e0] = (1.0 - alpha) * np.eye(m)
    F[d0:e0, e0:] = (-alpha / dt) * np.eye(m)
    F[e0:, :n] = -C                              # e_prev_{k+1} = e_k

    # u_k = Kp e_k + Kd d_{k+1}, and d_{k+1} is the derivative block of F applied to xi_k,
    # so the whole command is one linear map of the augmented state -- exactly as in the
    # scalar case, with the row becoming a block.
    u_x = Kp @ (-C) + Kd @ F[d0:e0, :n]
    u_d = Kd @ F[d0:e0, d0:e0]
    u_e = Kd @ F[d0:e0, e0:]
    F[:n, :n] = Ad + Bd_u @ u_x
    F[:n, d0:e0] = Bd_u @ u_d
    F[:n, e0:] = Bd_u @ u_e
    L = np.hstack([u_x, u_d, u_e])               # (m_u, n + 2m)
    return F, L, n


def conditional_spectrum(rig: Rig, closed: bool = True, fs: float = FS_CTRL,
                         band: float = BAND_HZ, f_cut: float = F_CUT,
                         sigma: float | None = None, sigma_a: float | None = None,
                         group: int = 1, t_window: float = T_WINDOW):
    """(freqs, cmi, joint, action) per frequency band, by the chain rule. Telescopes exactly."""

    sigma = SIGMA_ENERGY if sigma is None else sigma
    sigma_a = SIGMA_A if sigma_a is None else sigma_a
    Ga_x, Ga_u, Gn_x, Gn_u = _closed_loop_responses(rig, closed, fs, band, f_cut,
                                                    t_window=t_window)
    sig_n = sigma_tau_for(rig) if (closed and rig.has_sensor) else 0.0

    Nxx, Nuu, Nxu = _sensor_noise_blocks(Gn_x, Gn_u, sig_n)
    Nx = Nxx + sigma ** 2 * np.eye(Ga_x.shape[0])
    Nu = Nuu + 1e-18 * np.eye(Ga_u.shape[0])
    N_joint = np.block([[Nx, Nxu], [Nxu.T, Nu]])
    G_joint = np.vstack([Ga_x, Ga_u])

    def bits_for(G, N):
        """1/2 logdet(I + sigma_a^2 G^T N^-1 G) restricted to trailing mode blocks."""
        c, low = sla.cho_factor(N, lower=True)
        W = sla.cho_solve((c, low), G)                       # N^-1 G
        A = sigma_a ** 2 * (G.T @ W)                         # d x d, the small side
        K = A.shape[0]

        def ld(start):
            if start >= K:
                return 0.0
            S = A[start:, start:]
            return float(np.linalg.slogdet(np.eye(len(S)) + S)[1] / (2.0 * np.log(2.0)))
        return ld

    # Both terms telescope over the same bands, so their difference does too -- which is
    # why the per-band conditionals sum to what `conditional_information` returns, and is
    # the check this function exists to make available.
    K = n_modes(t_window, band)
    freqs = np.arange(K) / (2.0 * t_window)
    j_rows = chain(bits_for(G_joint, N_joint), K, group, freqs)
    a_rows = chain(bits_for(Ga_u, Nu), K, group, freqs)
    out_f = np.array([f for f, _ in j_rows])
    out_j = np.array([v for _, v in j_rows])
    out_a = np.array([v for _, v in a_rows])
    return out_f, out_j - out_a, out_j, out_a


def morphological_information(rig: Rig, fs: float = FS_CTRL, f_cut: float = F_CUT) -> dict:
    """`MC_W` of Zahedi and Ay on the closed torque loop, and what the command costs."""
    # X is the plant block alone: the two controller memories in the augmented state belong
    # to the controller, and counting them would credit the morphology with its memory.
    dt = 1.0 / fs
    Ad_open, _ = zoh(rig.A, rig.B_u, dt)
    _, Q = gibbs_pair(rig.K, rig.H, Ad_open, label=rig.label)
    F, L, n = closed_loop_matrix(rig, fs, f_cut)
    m = F.shape[0]

    rho = float(np.abs(np.linalg.eigvals(F)).max())
    if rho > 1.0:
        raise np.linalg.LinAlgError(
            f"{rig.label}: the torque loop is unstable at {fs:g} Hz (rho = {rho:.3f}), so it "
            "has no stationary law and MC_W is undefined. `solve_discrete_lyapunov` will "
            "still return the algebraic solution of the Lyapunov equation here -- it is not "
            "a covariance, and a power iteration from zero diverges instead of reaching it. "
            "Gate the sweep on `closed_loop_spectral_radius` as the controlled plates do.")

    Qa = np.zeros((m, m))
    Qa[:n, :n] = Q                        # the bath drives the plant; the controller is exact
    S_xi = sla.solve_discrete_lyapunov(F, Qa)
    S_xi = 0.5 * (S_xi + S_xi.T)

    Px = np.zeros((n, m))
    Px[:, :n] = np.eye(n)
    M = np.vstack([Px @ F, Px, L])        # [X_{k+1} ; X_k ; A_k], all from xi_k
    load = np.zeros((2 * n + 1, m))
    load[:n, :] = Px                      # only X_{k+1} picks up this step's noise
    S = M @ S_xi @ M.T + load @ Qa @ load.T
    S = 0.5 * (S + S.T)

    xp, x = np.arange(n), np.arange(n, 2 * n)
    free = cond_mi_bits(S, xp, x, np.empty(0, int))
    if not np.any(L):
        return dict(mc_w=free, unconditioned=free, command_cost=0.0, open_loop=True)
    cond = cond_mi_bits(S, xp, x, np.arange(2 * n, 2 * n + 1))
    return dict(mc_w=cond, unconditioned=free, command_cost=free - cond, open_loop=False)
