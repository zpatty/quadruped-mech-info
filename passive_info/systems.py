"""The two passive transmissions of Wensing et al. (2017) Section III-A, as state space."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla

from mechinfo.mechanics import (assemble, energy_metric, locked_effective_mass,
                                rigid_plant, sea_plant, sym_sqrt)

#: The prototype of Wensing et al. (2017) Section III-a, in SI units.
M_B, M_L, R, I_ROT, K_I = 32.0, 3.0, 0.1, 0.010, 1.0e6

#: Ground damping as a ratio of critical on the contact mode, and joint damping.
ZETA_CONTACT = 0.15
JOINT_DAMPING = 0.0        # the prototype has none; kept explicit so it can be swept

#: Damping ratio across the series spring. Not optional: an undamped k_s is a lossless
#: oscillator whose resonance makes bits(k_s) non-monotone.
ZETA_SERIES = 0.10


@dataclass(frozen=True)
class Passive:
    """One transmission: `A`, `B_F`, `C` for the unactuated mechanism."""
    A: np.ndarray
    B_F: np.ndarray
    C: np.ndarray
    H: np.ndarray
    K: np.ndarray
    J: np.ndarray
    ks: float | None
    label: str
    readout: str = "motor"

    @property
    def n(self) -> int:
        return self.H.shape[0]

    def modes_hz(self) -> np.ndarray:
        """Undamped natural frequencies, sqrt(eig(K, H)) / 2pi, ascending."""
        w = sla.eigh(self.K, self.H, eigvals_only=True)
        return np.sqrt(np.clip(w, 0.0, None)) / (2.0 * np.pi)

    def n_zero_modes(self) -> int:
        """How many zero-frequency modes this mechanism has: the nullity of K."""
        return int(self.K.shape[0] - np.linalg.matrix_rank(self.K))

    def nonzero_modes_hz(self) -> np.ndarray:
        """Every mode that stores potential energy, zero modes dropped by count."""
        return self.modes_hz()[self.n_zero_modes():]

    def bandwidth(self) -> float:
        """The lowest non-zero mode -- eq. (20)'s omega_1, the one the paper reports."""
        f = self.modes_hz()
        z = self.n_zero_modes()
        return float(f[z]) if f.size > z else float("nan")


#: What is read out of the mechanism.
#:
#:   "motor"    the motor coordinate alone -- what an encoder on the rotor would give.
#:   "q"        every configuration coordinate.
#:   "state"    configuration and velocity.
#:   "energy"   the state whitened by the mechanism's own energy metric, `E = diag(K, H)`.
#:   "kinetic"  the same using only H, for a mechanism whose stiffness cannot be read.
#:
#: The gap between "motor" and "state" is the part of the contact the morphology responds
#: to but does not deliver to the actuator.
#:
#: Reported results use "energy". The dimensional readouts need a sigma per coordinate,
#: which asserts that a unit of motion in each is the same size; at large k_s that turns a
#: stiff direction into a second noisy look at one signal and manufactured most of the SEA's
#: apparent advantage. Energy is a scalar, so "noise of fixed energy" is basis-free: its
#: covariance is `sigma^2 E^-1`, and the readout is the whitening `x -> E^(1/2) x`.
#: apparent advantage.
READOUTS = ("motor", "q", "state", "energy", "kinetic")


def mode_coupling(sys: Passive, mode_index: int = -1) -> float:
    """|C (j w I - A)^-1 B_F| at the frequency of one undamped mode: how strongly an external
    disturbance excites and is observed through that mode, at resonance.
    """
    freqs = sys.modes_hz()
    w = 2.0 * np.pi * float(freqs[mode_index])
    n = sys.A.shape[0]
    G = sys.C @ np.linalg.solve(1j * w * np.eye(n) - sys.A, sys.B_F)
    return float(np.abs(G).max())


def omega_ref(m_b: float = M_B, m_l: float = M_L, r: float = R, i_rot: float = I_ROT,
              k_i: float = K_I, clamped: bool = True) -> float:
    """Reference rate: the rigid design's contact mode, sqrt(k_i / m_e)."""
    return float(np.sqrt(k_i / reference_m_e(m_b, m_l, r, i_rot, clamped=clamped)))


def reference_m_e(m_b: float = M_B, m_l: float = M_L, r: float = R,
                  i_rot: float = I_ROT, clamped: bool = True) -> float:
    """Effective mass at the foot with the transmission locked. One number for all designs."""
    if clamped:
        H = np.array([[m_l * r ** 2 + i_rot]])
        J = np.array([[-r]])
    else:
        H = np.array([[m_b + m_l, -r * m_l], [-r * m_l, m_l * r ** 2 + i_rot]])
        J = np.array([[1.0, -r]])
    return locked_effective_mass(H, J)


def _scaling(n: int, r: float, w: float, clamped: bool = True) -> np.ndarray:
    """Diagonal that maps the physical state to a dimensionless one."""
    pos = np.ones(n)
    if not clamped:
        pos[0] = 1.0 / r
    return np.concatenate([pos, pos / w])


def _readout(readout: str, H: np.ndarray, K: np.ndarray, r: float, w: float,
             clamped: bool = True) -> np.ndarray:
    n = H.shape[0]
    if readout == "motor":
        C = np.zeros((1, 2 * n))
        C[0, n - 1] = 1.0            # already an angle: dimensionless as it stands
        return C
    if readout in ("energy", "kinetic"):
        # "energy" is the mechanism's own stored energy and nothing else: elastic in the
        # configuration block, kinetic in the velocity block. "kinetic" keeps the reference
        # rate because it has no K to use -- see the note above `omega_ref`.
        return sym_sqrt(energy_metric(K, H, None if readout == "energy" else w))
    d = _scaling(n, r, w, clamped)
    if readout == "q":
        return np.hstack([np.diag(d[:n]), np.zeros((n, n))])
    if readout == "state":
        return np.diag(d)
    raise ValueError(f"readout must be one of {READOUTS}, got {readout!r}")


def _assemble(H: np.ndarray, K_struct: np.ndarray, J: np.ndarray, ks, label: str,
              k_i: float = K_I, zeta_contact: float = ZETA_CONTACT,
              joint_damping: float = JOINT_DAMPING, readout: str = "motor",
              r_scale: float = R, D_struct: np.ndarray | None = None,
              m_e_ref: float | None = None, clamped: bool = True) -> Passive:
    """Common assembly: add the ground spring, build A, B_F and the requested readout."""
    J = np.atleast_2d(J)
    if m_e_ref is None:
        m_e_ref = reference_m_e(r=r_scale, clamped=clamped)
    A, B_F, K, _ = assemble(H, K_struct, J, k_i, m_e_ref, zeta_contact,
                            joint_damping, D_struct)

    # The motor coordinate is the last one in both layouts: theta for the rigid design,
    # theta_m for the series elastic one.
    return Passive(A=A, B_F=B_F,
                   C=_readout(readout, H, K, r_scale, omega_ref(clamped=clamped),
                              clamped=clamped),
                   H=H, K=K, J=J, ks=ks, label=label, readout=readout)


def proprioceptive(m_b: float = M_B, m_l: float = M_L, r: float = R,
                   i_rot: float = I_ROT, clamped: bool = True, **kw) -> Passive:
    """Rotor geared rigidly to the leg, the rotor inertia inside H."""
    H, K_s, J, D_s = rigid_plant(m_l, r, i_rot, m_b=None if clamped else m_b)
    return _assemble(H, K_s, J, None, "proprioceptive", D_struct=D_s, clamped=clamped,
                     r_scale=r, **kw)


def series_elastic(ks: float, m_b: float = M_B, m_l: float = M_L, r: float = R,
                   i_rot: float = I_ROT, zeta_series: float = ZETA_SERIES,
                   clamped: bool = True, **kw) -> Passive:
    """Rotor on its own coordinate behind `ks`: q = [y_b, theta_j, theta_m]."""
    H, K_s, J, D_s = sea_plant(ks, m_l, r, i_rot, zeta_series,
                               m_b=None if clamped else m_b)
    return _assemble(H, K_s, J, float(ks), "SEA", D_struct=D_s, clamped=clamped,
                     r_scale=r, **kw)
