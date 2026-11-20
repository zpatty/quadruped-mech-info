"""Data for every animation frame, computed with the linear models of the paper.

We apply one band-limited contact force to each design. The models are the ones that the bit
counts use: `embodied.go2_*` for the Go2 leg and `systems.*` for the 2-DoF prototype, with
the energy readout. For each design this module returns, sample by sample:

  * `X`        the state under the force, followed by a tail with no force.
  * `energy`   the stored energy in each element. The quadratic forms of the elements sum
               to `E = diag(K, H)`, so the parts add up to the total `1/2 x^T E x`.
  * `hue_w`    the energy in each element due to each mode of the force, taken from the
               diagonal of the per-mode decomposition. The renderers use it for colour.
  * `bits_t`   the information about the force profile in the state up to time t,
               `1/2 logdet(I + gamma sum_{s<=t} M_s^T M_s)`. At the end of the contact
               window this equals `information.profile_bits`.
  * `spec_t`   `bits_t` split over the modes of the force. Each eigenchannel contributes in
               proportion to its squared loadings, so the spectrum sums to `bits_t`.

No sensor appears in this module. The information is in the state of the mechanism itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mechinfo.mechanics import zoh
from passive_info.information import (FS, SIGMA_A, T_WINDOW, mode_basis, n_modes,
                                      natural_sigma)

#: The band the paper's spectrum plates use. 2BT = 34 force modes over the 85 ms window.
BAND_HZ = 200.0

#: Force-free time simulated after the window, s. Enough for the contact mode to ring down.
T_TAIL = 0.030

#: Seed for the force profile. All designs use the same profile.
SEED = 7

#: Default designs: the rigid transmission, then two series-elastic ones from the results grid.
DEFAULT_KS = (None, 1.0e4, 1.0e3)

#: Go2 element names. On the rigid design the springs have zero energy, but the list is the
#: same for every design so that the renderers can use one layout.
GO2_ELEMENTS = ("ground", "calf", "thigh", "hip",
                "spring_calf", "spring_thigh", "spring_hip",
                "rotor_calf", "rotor_thigh", "rotor_hip")
#: Pairs of elements that exchange energy directly. The chain goes from the floor up through
#: the links, and each rotor connects to its link through the series spring.
GO2_EDGES = (("ground", "calf"), ("calf", "thigh"), ("thigh", "hip"),
             ("calf", "spring_calf"), ("spring_calf", "rotor_calf"),
             ("thigh", "spring_thigh"), ("spring_thigh", "rotor_thigh"),
             ("hip", "spring_hip"), ("spring_hip", "rotor_hip"))
GO2_JOINTS = ("hip", "thigh", "calf")           # coordinate order in the Go2 models

PROTO_ELEMENTS = ("ground", "leg", "spring", "rotor")
PROTO_EDGES = (("ground", "leg"), ("leg", "spring"), ("spring", "rotor"))


@dataclass
class Design:
    """One design's trajectory, energy partition and information, sample by sample."""
    ks: float | None
    label: str
    t: np.ndarray                   #: (n_t,) s; contact is [0, T_WINDOW]
    force: np.ndarray               #: (n_t,) N, zero in the tail
    X: np.ndarray                   #: (n_t, 2n) state [q; v]
    n: int                          #: configuration coordinates
    elements: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    energy: np.ndarray              #: (n_t, n_elem) J
    hue_w: np.ndarray               #: (n_t, n_elem, K) per-mode energy, unnormalised
    bits_t: np.ndarray              #: (n_t,) bits
    spec_t: np.ndarray              #: (n_t, K) bits per force mode
    mode_freqs: np.ndarray          #: (K,) Hz of each force mode
    natural_hz: np.ndarray          #: non-zero natural frequencies of the mechanism, Hz
    joint_q: np.ndarray             #: (n_t, n_joints) rad, link coordinates
    rotor_q: np.ndarray             #: (n_t, n_joints) rad, the rotors (= joints when rigid)
    total_energy: np.ndarray        #: (n_t,) 1/2 x^T E x, for checking the partition
    part_bits: np.ndarray           #: (n_t, n_elem) bits about the force in each element alone
    post_mean: np.ndarray           #: (n_t, n_w) centre of the band of consistent forces, N
    post_sd: np.ndarray             #: (n_t, n_w) its standard deviation, N
    extras: dict = field(default_factory=dict)

    @property
    def n_contact(self) -> int:
        """Samples inside the contact window."""
        return int(round(T_WINDOW * FS)) + 1


@dataclass
class Scenario:
    """The shared force and every design driven by it."""
    amps: np.ndarray                #: (K,) mode amplitudes, N
    band_hz: float
    designs: list[Design]

    def by_ks(self, ks) -> Design:
        for d in self.designs:
            if d.ks == ks:
                return d
        raise KeyError(ks)


# ---------------------------------------------------------------------------------------
# The force
# ---------------------------------------------------------------------------------------

def force_amplitudes(band_hz: float = BAND_HZ, seed: int = SEED) -> np.ndarray:
    """Mode amplitudes drawn from the prior of the paper, `N(0, SIGMA_A^2)` for each mode."""
    return np.random.default_rng(seed).normal(0.0, SIGMA_A, n_modes(T_WINDOW, band_hz))


#: Peak of the smooth part of the contact-shaped force, N, and the relative size of the
#: ripple that multiplies it.
CONTACT_PEAK, CONTACT_RIPPLE = 150.0, 0.15


def contact_amplitudes(band_hz: float = BAND_HZ, seed: int = SEED) -> np.ndarray:
    """Mode amplitudes of a force that looks like a contact: a half-sine times some ripple.

    Only the motion and the centre of the band of consistent forces depend on this choice.
    The bit counts and the width of the band depend only on the mechanism and the prior.
    """
    n_w = int(round(T_WINDOW * FS)) + 1
    K = n_modes(T_WINDOW, band_hz)
    Phi = mode_basis(n_w, K)
    bump = CONTACT_PEAK * np.sin(np.pi * (np.arange(n_w) + 0.5) / n_w)
    ripple = Phi @ force_amplitudes(band_hz, seed)
    ripple /= ripple.std()
    # the ripple scales the half-sine, so the force stays positive, as a contact force must
    return Phi.T @ (bump * (1.0 + CONTACT_RIPPLE * ripple))


# ---------------------------------------------------------------------------------------
# Simulation and the information spectrum
# ---------------------------------------------------------------------------------------

def _mode_responses(sys, band_hz: float, t_tail: float) -> tuple[np.ndarray, int]:
    """State response to each unit force mode, `(n_t, K, 2n)`, window then tail.

    Steps exactly as `information.response_matrix` does, so `C @ X` here is its output.
    """
    n_w = int(round(T_WINDOW * FS)) + 1
    n_tail = int(round(t_tail * FS))
    K = n_modes(T_WINDOW, band_hz)
    Phi = mode_basis(n_w, K)
    Ad, Bd = zoh(sys.A, sys.B_F, 1.0 / FS)
    ns = sys.A.shape[0]
    X = np.zeros((n_w + n_tail, K, ns))
    x = np.zeros((ns, K))
    for k in range(n_w + n_tail - 1):
        u = Phi[k] if k < n_w - 1 else np.zeros(K)
        x = Ad @ x + Bd * u[None, :]
        X[k + 1] = x.T
    return X, n_w


def _gamma(sys, sigma: float | None = None) -> float:
    sigma = natural_sigma(sys.readout) if sigma is None else sigma
    return (SIGMA_A / sigma) ** 2


def _bits_over_time(sys, Xj: np.ndarray, n_w: int, amps: np.ndarray,
                    sigma: float | None = None):
    """Cumulative bits, their per-mode spectrum, and the band of consistent forces.

    The band comes from the same linear Gaussian model as the bits. With prior
    `a ~ N(0, SIGMA_A^2 I)` and gram `G(t)`, the force profiles consistent with the state up
    to `t` have covariance `SIGMA_A^2 (I + G)^-1` in mode space. For a noise-free state the
    centre is `(I + G)^-1 G a`. Both are mapped to force samples with the mode basis.
    """
    gamma = _gamma(sys, sigma)
    Y = np.einsum("oi,tki->tko", sys.C, Xj)                 # readout per mode
    G = gamma * np.cumsum(np.einsum("tko,tjo->tkj", Y[:n_w], Y[:n_w]), axis=0)
    n_t, K = Xj.shape[0], Xj.shape[1]
    Phi = mode_basis(n_w, K)
    bits = np.zeros(n_t)
    spec = np.zeros((n_t, K))
    mean = np.zeros((n_t, n_w))
    sd = np.zeros((n_t, n_w))
    for i in range(n_t):
        lam, V = np.linalg.eigh(G[min(i, n_w - 1)])
        lam = np.clip(lam, 0.0, None)
        b = 0.5 * np.log2(1.0 + lam)
        bits[i] = b.sum()
        spec[i] = (V ** 2) @ b
        PV = Phi @ V
        mean[i] = PV @ ((lam / (1.0 + lam)) * (V.T @ amps))
        sd[i] = SIGMA_A * np.sqrt((PV ** 2) @ (1.0 / (1.0 + lam)))
    return bits, spec, mean, sd


def _part_bits(Xj: np.ndarray, n_w: int, Qs: np.ndarray, gamma: float) -> np.ndarray:
    """Bits about the force in each element on its own, `(n_t, n_elem)`.

    Element `e` is read out through its own energy form `Q_e`. The forms sum to `E`, so all
    elements read together give exactly the total bits.
    """
    n_t, K = Xj.shape[0], Xj.shape[1]
    out = np.zeros((n_t, len(Qs)))
    for e, Qe in enumerate(Qs):
        if not np.any(Qe):
            continue
        G = gamma * np.cumsum(np.einsum("tki,ij,tlj->tkl", Xj[:n_w], Qe, Xj[:n_w]), axis=0)
        for i in range(n_t):
            lam = np.clip(np.linalg.eigvalsh(G[min(i, n_w - 1)]), 0.0, None)
            out[i, e] = 0.5 * np.log2(1.0 + lam).sum()
    return out


def _build(sys, Q: dict[str, np.ndarray], elements, edges, amps, band_hz, t_tail,
           joint_idx, rotor_idx, label, ks, extras=None) -> Design:
    Xj, n_w = _mode_responses(sys, band_hz, t_tail)
    n_t, K, ns = Xj.shape
    X = np.einsum("k,tki->ti", amps, Xj)
    Phi = mode_basis(n_w, K)
    force = np.zeros(n_t)
    force[:n_w] = Phi @ amps

    Qs = np.stack([Q.get(e, np.zeros((ns, ns))) for e in elements])
    energy = 0.5 * np.einsum("ti,eij,tj->te", X, Qs, X)
    Yk = amps[None, :, None] * Xj
    hue_w = 0.5 * np.einsum("tki,eij,tkj->tek", Yk, Qs, Yk)

    from mechinfo.mechanics import energy_metric
    E = energy_metric(sys.K, sys.H)
    total = 0.5 * np.einsum("ti,ij,tj->t", X, E, X)

    bits, spec, post_mean, post_sd = _bits_over_time(sys, Xj, n_w, amps)
    part_bits = _part_bits(Xj, n_w, Qs, _gamma(sys))
    n = sys.n
    return Design(
        ks=ks, label=label, t=np.arange(n_t) / FS, force=force, X=X, n=n,
        elements=tuple(elements), edges=tuple(edges), energy=energy, hue_w=hue_w,
        bits_t=bits, spec_t=spec, mode_freqs=np.arange(K) / (2.0 * T_WINDOW),
        natural_hz=sys.nonzero_modes_hz(),
        joint_q=X[:, joint_idx], rotor_q=X[:, rotor_idx], total_energy=total,
        part_bits=part_bits, post_mean=post_mean, post_sd=post_sd, extras=extras or {})


def _block(ns: int, n: int, which: str, M: np.ndarray) -> np.ndarray:
    """Embed an n x n form on the configuration ('q') or velocity ('v') block."""
    Q = np.zeros((ns, ns))
    s = slice(0, n) if which == "q" else slice(n, 2 * n)
    Q[s, s] = M
    return Q


# ---------------------------------------------------------------------------------------
# Go2
# ---------------------------------------------------------------------------------------

def go2_body_inertias(leg_name: str = "FR") -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Per-body kinetic-energy forms `H_b` at the stance, and the rotor armatures.

    `sum_b H_b + diag(armature)` equals the mass matrix from `mj_fullM`, which is the `H` of
    the analytic `go2_*` models. `tests/test_anim_data.py` checks this.
    """
    import mujoco
    from mechinfo import mjcf
    from passive_info.embodied import GO2, STANCE_QPOS, _compile

    m, d = _compile(None, (leg_name,), "fixed")
    leg = mjcf.index_legs(m, GO2)[leg_name]
    d.qpos[leg.qpos] = STANCE_QPOS
    mujoco.mj_forward(m, d)
    out = {}
    for part in GO2_JOINTS:
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{leg_name}_{part}")
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jacBodyCom(m, d, jp, jr, b)
        R = d.ximat[b].reshape(3, 3)
        I_w = R @ np.diag(m.body_inertia[b]) @ R.T
        out[part] = m.body_mass[b] * (jp.T @ jp) + jr.T @ I_w @ jr
    arm = np.array([m.dof_armature[i] for i in leg.dofs])
    return out, arm


def go2_design(ks: float | None, amps: np.ndarray, band_hz: float = BAND_HZ,
               t_tail: float = T_TAIL) -> Design:
    """One Go2 leg design under the force `amps`."""
    from passive_info.embodied import go2_proprioceptive, go2_series_elastic

    sys = go2_proprioceptive() if ks is None else go2_series_elastic(ks)
    Hb, arm = go2_body_inertias()
    n, ns, nj = sys.n, 2 * sys.n, len(GO2_JOINTS)
    K_ground = sys.K - (0.0 if ks is None else _series_K(ks, n, nj))
    Q = {"ground": _block(ns, n, "q", K_ground)}
    for i, part in enumerate(GO2_JOINTS):
        Hl = np.zeros((n, n))
        Hl[:nj, :nj] = Hb[part]
        Q[part] = _block(ns, n, "v", Hl)
        rot = np.zeros((n, n))
        r = i if ks is None else nj + i           # rotor coordinate
        rot[r, r] = arm[i]
        Q[f"rotor_{part}"] = _block(ns, n, "v", rot)
        if ks is not None:
            S = np.zeros((1, n))
            S[0, i], S[0, nj + i] = 1.0, -1.0
            Q[f"spring_{part}"] = _block(ns, n, "q", ks * (S.T @ S))
    rotor_idx = list(range(nj)) if ks is None else list(range(nj, 2 * nj))
    label = "rigid" if ks is None else f"series elastic, k_s = {ks:g}"
    return _build(sys, Q, GO2_ELEMENTS, GO2_EDGES, amps, band_hz, t_tail,
                  list(range(nj)), rotor_idx, label, ks)


def _series_K(ks: float, n: int, nj: int) -> np.ndarray:
    S = np.zeros((nj, n))
    for i in range(nj):
        S[i, i], S[i, nj + i] = 1.0, -1.0
    return ks * (S.T @ S)


# ---------------------------------------------------------------------------------------
# The 2-DoF prototype of Wensing et al. (2017)
# ---------------------------------------------------------------------------------------

def proto_design(ks: float | None, amps: np.ndarray, band_hz: float = BAND_HZ,
                 t_tail: float = T_TAIL) -> Design:
    """The clamped prototype: one link of length r on a rotor, rigidly or through `ks`."""
    from passive_info.systems import I_ROT, K_I, M_L, R, proprioceptive, series_elastic

    sys = (proprioceptive(readout="energy") if ks is None
           else series_elastic(ks, readout="energy"))
    n, ns = sys.n, 2 * sys.n
    J = np.atleast_2d(sys.J)
    Q = {"ground": _block(ns, n, "q", K_I * (J.T @ J))}
    leg = np.zeros((n, n))
    rot = np.zeros((n, n))
    leg[0, 0] = M_L * R ** 2
    if ks is None:
        rot[0, 0] = I_ROT
    else:
        rot[1, 1] = I_ROT
        S = np.array([[1.0, -1.0]])
        Q["spring"] = _block(ns, n, "q", ks * (S.T @ S))
    Q["leg"] = _block(ns, n, "v", leg)
    Q["rotor"] = _block(ns, n, "v", rot)
    label = "rigid" if ks is None else f"series elastic, k_s = {ks:g}"
    return _build(sys, Q, PROTO_ELEMENTS, PROTO_EDGES, amps, band_hz, t_tail,
                  [0], [0] if ks is None else [1], label, ks,
                  extras=dict(r=R, m_l=M_L, i_rot=I_ROT, k_i=K_I))


def scenario(kind: str = "go2", ks_list=DEFAULT_KS, band_hz: float = BAND_HZ,
             seed: int = SEED, t_tail: float = T_TAIL, force: str = "prior") -> Scenario:
    """Every design under one shared force. `kind` is 'go2' or 'proto'.

    `force` is 'prior' for a draw from the prior of the paper, or 'contact' for a force that
    looks like a contact (`contact_amplitudes`).
    """
    amps = (contact_amplitudes if force == "contact" else force_amplitudes)(band_hz, seed)
    build = go2_design if kind == "go2" else proto_design
    return Scenario(amps=amps, band_hz=band_hz,
                    designs=[build(ks, amps, band_hz, t_tail) for ks in ks_list])
