"""Shared primitives: discretisation, the energy metric, the ground, the state space."""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla


# ---------------------------------------------------------------------------------------
# Linear algebra
# ---------------------------------------------------------------------------------------

def zoh(A: np.ndarray, B: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact zero-order-hold pair `(Ad, Bd)` from one block matrix exponential."""
    n, m = A.shape[0], B.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n], M[:n, n:] = A, B
    E = sla.expm(M * dt)
    return E[:n, :n], E[:n, n:]


def sym_sqrt(M: np.ndarray) -> np.ndarray:
    """Symmetric square root, eigenvalues clipped at zero first."""
    # Energy metrics are singular where a mechanism stores no potential energy, so eigh
    # returns zeros that round negative. The clip is for that, not regularisation.
    lam, V = np.linalg.eigh(M)
    return (V * np.sqrt(np.clip(lam, 0.0, None))) @ V.T


def bits_from_gram(gram: np.ndarray) -> float:
    """`1/2 logdet(I + gram)` in bits: the Gaussian channel capacity."""
    sign, logdet = np.linalg.slogdet(np.eye(gram.shape[0]) + gram)
    if sign <= 0:
        raise FloatingPointError("non-positive-definite information matrix")
    return float(0.5 * logdet / np.log(2.0))


# ---------------------------------------------------------------------------------------
# The mechanism
# ---------------------------------------------------------------------------------------

def energy_metric(K: np.ndarray, H: np.ndarray, w: float | None = None) -> np.ndarray:
    """`E = diag(K, H)`: twice the mechanism's stored energy as a quadratic form."""
    n = H.shape[0]
    E = np.zeros((2 * n, 2 * n))
    E[:n, :n] = K if w is None else w ** 2 * H
    E[n:, n:] = H
    return E


def locked_effective_mass(H_locked: np.ndarray, J: np.ndarray) -> float:
    """`1 / (J H^-1 J^T)`: what the foot presents to the ground, transmission locked."""
    # H_locked is the mechanism with its transmission LOCKED, not the design's own H: an
    # SEA's own H excludes the rotor, which sits on a coordinate J does not touch.
    J2 = np.atleast_2d(J)
    return 1.0 / float((J2 @ np.linalg.solve(H_locked, J2.T)).item())


def ground_damping(k_i: float, m_e_ref: float, zeta_contact: float) -> float:
    """`b_i = 2 zeta sqrt(k_i m_e)`: the contact mode at a fixed fraction of critical."""
    # m_e_ref must be the locked-transmission mass. Per-design, an SEA meets a softer
    # ground than the rigid leg it becomes in the stiff limit.
    return 2.0 * zeta_contact * np.sqrt(k_i * m_e_ref)


def assemble(H: np.ndarray, K_struct: np.ndarray, J: np.ndarray, k_i: float,
             m_e_ref: float, zeta_contact: float, joint_damping: float = 0.0,
             D_struct: np.ndarray | None = None):
    """`(A, B_F, K, D)` for `H qddot + D qdot + K q = J^T f`, ground spring included."""
    n = H.shape[0]
    J = np.atleast_2d(J)
    K = K_struct + J.T @ (k_i * J)
    D = J.T @ (ground_damping(k_i, m_e_ref, zeta_contact) * J) + joint_damping * np.eye(n)
    if D_struct is not None:
        D = D + D_struct
    Hinv = np.linalg.inv(H)
    A = np.block([[np.zeros((n, n)), np.eye(n)],
                  [-Hinv @ K, -Hinv @ D]])
    B_F = np.vstack([np.zeros((n, 1)), Hinv @ J.T])
    return A, B_F, K, D


# ---------------------------------------------------------------------------------------
# The two transmissions of Wensing et al. (2017) Section III-A
#
# Returned as `(H, K_struct, J, D_struct)` rather than as an assembled system: the passive
# arm wraps them in a `Passive` with an energy readout and the controlled arm in a `Rig` with
# a motor input, and only the mechanism is shared.
#
# The motor coordinate is the last one in every layout -- `theta` for the rigid design,
# `theta_m` for the series elastic one. Both arms rely on that and neither said so.
# ---------------------------------------------------------------------------------------

def rigid_plant(m_l: float, r: float, i_rot: float, m_b: float | None = None):
    """Rotor geared rigidly to the leg. Clamped: `q = [theta]`."""
    if m_b is None:
        H = np.array([[m_l * r ** 2 + i_rot]])
        return H, np.zeros((1, 1)), np.array([-r]), None
    H = np.array([[m_b + m_l, -r * m_l],
                  [-r * m_l, m_l * r ** 2 + i_rot]])
    return H, np.zeros((2, 2)), np.array([1.0, -r]), None


def sea_plant(ks: float, m_l: float, r: float, i_rot: float, zeta_series: float,
              m_b: float | None = None):
    """Rotor on its own coordinate behind `ks`. Clamped `[theta_j, theta_m]`, free adds y_b."""
    # Not optional: an undamped ks is a lossless oscillator and makes bits(ks) non-monotone.
    b_s = 2.0 * zeta_series * np.sqrt(ks * i_rot)
    if m_b is None:
        H = np.diag([m_l * r ** 2, i_rot])
        couple = np.array([[1.0, -1.0], [-1.0, 1.0]])
        return H, ks * couple, np.array([-r, 0.0]), b_s * couple
    H = np.zeros((3, 3))
    H[:2, :2] = [[m_b + m_l, -r * m_l], [-r * m_l, m_l * r ** 2]]
    H[2, 2] = i_rot
    couple = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, -1.0], [0.0, -1.0, 1.0]])
    return H, ks * couple, np.array([1.0, -r, 0.0]), b_s * couple
