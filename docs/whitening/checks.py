"""Numerical checks for every result in `energy_whitening.tex`.

Each `check_*` function evaluates one theorem, proposition or demo on the repository's own
models and returns a dict of plain numbers. `make_figures.py` draws from these, and
`tests/test_energy_whitening.py` asserts on them. Nothing here changes library behaviour:
it only calls `mechinfo` and `passive_info`.

Run directly to print a report:

    uv run python docs/whitening/checks.py
"""
from __future__ import annotations

import dataclasses
from itertools import combinations

import numpy as np
import scipy.linalg as sla

from mechinfo.mechanics import (assemble, bits_from_gram, energy_metric, sea_plant,
                                sym_sqrt, zoh)
from passive_info.channel import gibbs_pair
from passive_info.information import (FS, SIGMA_A, SIGMA_ENERGY, T_WINDOW, mode_basis,
                                      n_modes, profile_bits, response_matrix)
from passive_info.spectrum import coordinate_bits, shapley_coordinate_bits
from passive_info.systems import (I_ROT, K_I, M_L, R, ZETA_CONTACT, ZETA_SERIES, Passive,
                                  omega_ref, proprioceptive, reference_m_e, series_elastic)


# ---------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------

def bits_of(sys: Passive, **kw) -> float:
    return profile_bits(sys, **kw)["bits"]


def reparametrized_sea(ks: float, T: np.ndarray, readout: str) -> Passive:
    """The clamped SEA written in new generalized coordinates `q = T q_new`.

    Mass, stiffness and damping transform as quadratic forms (`T^T X T`), the force Jacobian
    as a covector (`J T`). The ground is attached exactly as in `passive_info.systems`.
    `readout="energy"` whitens with the NEW coordinates' own metric; `readout="state"` applies
    the repository's per-coordinate rule (identity on positions, 1/omega_ref on velocities) to
    the new coordinates, which is what a per-coordinate noise model would do.
    """
    H, Ks, J, Ds = sea_plant(ks, M_L, R, I_ROT, ZETA_SERIES)
    J = np.atleast_2d(J)
    H2, K2, J2, D2 = T.T @ H @ T, T.T @ Ks @ T, J @ T, T.T @ Ds @ T
    A, B, K, _ = assemble(H2, K2, J2, K_I, reference_m_e(), ZETA_CONTACT, 0.0, D2)
    if readout == "energy":
        C = sym_sqrt(energy_metric(K, H2))
    elif readout == "state":
        w = omega_ref()
        C = np.diag([1.0, 1.0, 1.0 / w, 1.0 / w])
    else:
        raise ValueError(readout)
    return Passive(A=A, B_F=B, C=C, H=H2, K=K, J=J2, ks=ks, label="SEA(T)", readout=readout)


#: The reparametrizations used for Theorem (coordinate invariance).
REPARAMETRIZATIONS = {
    "as built $(\\theta_j,\\theta_m)$": np.eye(2),
    "degrees": np.diag([np.pi / 180.0, np.pi / 180.0]),
    "deflection $(\\theta_j,\\delta)$": np.array([[1.0, 0.0], [1.0, -1.0]]),
    "geared motor angle ($N=10$)": np.diag([1.0, 0.1]),
    "arbitrary $T$": np.array([[1.3, 0.4], [-0.2, 0.7]]),
}


# ---------------------------------------------------------------------------------------
# Section 2: what whitening is
# ---------------------------------------------------------------------------------------

def check_energy_identity(seed: int = 0) -> dict:
    """|E^{1/2} x_q|^2 equals q'Hq' + qKq for random states."""
    rng = np.random.default_rng(seed)
    s = series_elastic(1e4, readout="energy")
    n = s.n
    err = 0.0
    for _ in range(100):
        x = rng.standard_normal(2 * n)
        q, v = x[:n], x[n:]
        err = max(err, abs(np.sum((s.C @ x) ** 2) - (v @ s.H @ v + q @ s.K @ q))
                  / (v @ s.H @ v + q @ s.K @ q))
    return dict(max_rel_err=err)


def check_square_roots() -> dict:
    """Symmetric root, Cholesky factor and the bare Gram `M^T E M` give the same bits."""
    out = {}
    for name, s in (("rigid", proprioceptive(readout="energy")),
                    ("SEA 1e4", series_elastic(1e4, readout="energy"))):
        E = energy_metric(s.K, s.H)
        L = np.linalg.cholesky(E).T                     # L^T L = E, upper triangular
        raw = response_matrix(dataclasses.replace(s, C=np.eye(2 * s.n)))
        n_t = raw.shape[0] // (2 * s.n)
        X = raw.reshape(n_t, 2 * s.n, -1)
        G = sum(X[t].T @ E @ X[t] for t in range(n_t)) * (SIGMA_A / SIGMA_ENERGY) ** 2
        out[name] = dict(sym=bits_of(s), chol=bits_of(dataclasses.replace(s, C=L)),
                         gram=bits_from_gram(G))
    return out


def check_noise_equivalence(seed: int = 0) -> dict:
    """I(a; E^{1/2}x + n) = I(a; x + E^{-1/2} n): the noise model is the only content."""
    s = series_elastic(1e4, readout="energy")
    raw = response_matrix(dataclasses.replace(s, C=np.eye(2 * s.n)))   # (n_t*2n, K)
    n2 = 2 * s.n
    n_t = raw.shape[0] // n2
    E = energy_metric(s.K, s.H)
    # physical-coordinate noise covariance, block diagonal over time
    Ninv = np.kron(np.eye(n_t), E) / SIGMA_ENERGY ** 2
    G_phys = SIGMA_A ** 2 * raw.T @ Ninv @ raw
    return dict(whitened=bits_of(s), physical_noise=bits_from_gram(G_phys))


# ---------------------------------------------------------------------------------------
# Section 3: invariance and the noise law
# ---------------------------------------------------------------------------------------

def check_invariance(ks: float = 1e4) -> dict:
    rows = {}
    for name, T in REPARAMETRIZATIONS.items():
        rows[name] = dict(energy=bits_of(reparametrized_sea(ks, T, "energy")),
                          state=bits_of(reparametrized_sea(ks, T, "state")))
    e = [r["energy"] for r in rows.values()]
    s = [r["state"] for r in rows.values()]
    return dict(rows=rows, energy_spread=max(e) - min(e), state_spread=max(s) - min(s))


def check_maxent(seed: int = 0, trials: int = 2000) -> dict:
    """Among Gaussians with E[n^T E n] = d sigma^2, entropy peaks at sigma^2 E^{-1}.

    Random competitors are drawn and rescaled to the same mean energy; none may exceed the
    Gibbs entropy. (Non-Gaussian competitors are covered by the Gaussian max-entropy theorem
    and are not sampled here.)
    """
    rng = np.random.default_rng(seed)
    s = series_elastic(1e4, readout="energy")
    E = energy_metric(s.K, s.H)
    d = E.shape[0]
    sigma2 = SIGMA_ENERGY ** 2

    def h(S):
        return 0.5 * np.linalg.slogdet(2 * np.pi * np.e * S)[1]

    h_gibbs = h(sigma2 * np.linalg.inv(E))
    worst_gap = np.inf
    for _ in range(trials):
        A = rng.standard_normal((d, d))
        S = A @ A.T + 1e-3 * np.eye(d)
        S *= d * sigma2 / np.trace(E @ S)           # match mean energy: tr(E S) = d sigma^2
        worst_gap = min(worst_gap, h_gibbs - h(S))
    return dict(h_gibbs=h_gibbs, min_gap=worst_gap,
                effective_kT=sigma2, room_kT=1.380649e-23 * 300.0)


def check_fdt() -> dict:
    """Continuous FDT: k_B T E^{-1} solves A P + P A^T + B (2 k_B T D) B^T = 0.

    Also checks the discrete balance in `channel.gibbs_pair`: Q = P - A_d P A_d^T is PSD.
    """
    out = {}
    for name, s in (("rigid", proprioceptive(readout="energy")),
                    ("SEA 1e4", series_elastic(1e4, readout="energy"))):
        n = s.n
        D = -s.H @ s.A[n:, n:]                      # A[n:, n:] = -H^{-1} D
        P = np.linalg.inv(energy_metric(s.K, s.H))
        Bn = np.vstack([np.zeros((n, n)), np.linalg.inv(s.H)])
        resid = s.A @ P + P @ s.A.T + Bn @ (2.0 * D) @ Bn.T
        Ad, _ = zoh(s.A, s.B_F, 1.0 / FS)
        _, Q = gibbs_pair(s.K, s.H, Ad, label=name)
        out[name] = dict(lyap_resid=float(np.abs(resid).max() / np.abs(P).max()),
                         Q_min_eig=float(np.linalg.eigvalsh(Q)[0]))
    return out


# ---------------------------------------------------------------------------------------
# Section 4: structure of the whitened dynamics
# ---------------------------------------------------------------------------------------

def check_port_hamiltonian(ks_grid=(70.0, 1e3, 1e4, 1e5, 1e6)) -> dict:
    """E^{1/2} A E^{-1/2} = J - R with J skew, R = blkdiag(0, H^{-1/2} D H^{-1/2}) >= 0."""
    rows = []
    systems = [("rigid", proprioceptive(readout="energy"))] + [
        (f"SEA {k:g}", series_elastic(k, readout="energy")) for k in ks_grid]
    for name, s in systems:
        n = s.n
        Eh = sym_sqrt(energy_metric(s.K, s.H))
        At = Eh @ s.A @ np.linalg.inv(Eh)
        Sym = 0.5 * (At + At.T)
        D = -s.H @ s.A[n:, n:]
        Hmh = np.linalg.inv(sym_sqrt(s.H))
        R_pred = np.zeros_like(At)
        R_pred[n:, n:] = Hmh @ D @ Hmh
        sv = np.sort(np.linalg.svd(At[:n, n:], compute_uv=False))
        rows.append(dict(name=name,
                         sym_err=float(np.abs(Sym + R_pred).max() / np.abs(At).max()),
                         max_eig_sym=float(np.linalg.eigvalsh(Sym).max()),
                         sv=sv, omega=2 * np.pi * s.modes_hz()))
    return dict(rows=rows)


# ---------------------------------------------------------------------------------------
# Section 5: different numbers of degrees of freedom
# ---------------------------------------------------------------------------------------

def check_duplicate() -> dict:
    """Duplicating a readout with independent noise adds sum 1/2 log2((1+2l)/(1+l))."""
    prop = proprioceptive(readout="state")
    base = profile_bits(prop)
    dup = profile_bits(dataclasses.replace(prop, C=np.vstack([prop.C, prop.C])))
    M = response_matrix(prop)
    lam = np.linalg.eigvalsh((SIGMA_A / 1e-4) ** 2 * M.T @ M)
    predicted = float(np.sum(0.5 * np.log2((1 + 2 * lam) / (1 + lam))))
    return dict(gain=dup["bits"] - base["bits"], predicted=predicted,
                half_rank=0.5 * base["rank"], n_modes=base["n_modes"])


def check_locked_restriction(ks: float = 1e6) -> dict:
    """On the locked subspace q = (1,1) theta the SEA's energy form IS the rigid one."""
    s = series_elastic(ks, readout="energy")
    r = proprioceptive(readout="energy")
    V = np.array([[1.0], [1.0]])
    return dict(K_V=float((V.T @ s.K @ V).item()), K_rigid=float(r.K.item()),
                H_V=float((V.T @ s.H @ V).item()), H_rigid=float(r.H.item()),
                w_K_w=float(np.array([0, -1.0]) @ s.K @ np.array([0, -1.0])), ks=ks)


def check_stiff_limit(ks_grid=None, band_hz: float = 100.0) -> dict:
    """bits(SEA) - bits(rigid) for both readouts, and the off-subspace energy norms."""
    if ks_grid is None:
        ks_grid = np.logspace(1.5, 12, 43)
    p_e = bits_of(proprioceptive(readout="energy"), band_hz=band_hz)
    p_s = bits_of(proprioceptive(readout="state"), band_hz=band_hz)
    gap_e, gap_s, off_k, off_h = [], [], [], []
    for ks in ks_grid:
        s = series_elastic(float(ks), readout="energy")
        gap_e.append(bits_of(s, band_hz=band_hz) - p_e)
        gap_s.append(bits_of(series_elastic(float(ks), readout="state"),
                             band_hz=band_hz) - p_s)
        c1 = np.array([[np.sqrt(ks), -np.sqrt(ks), 0.0, 0.0]])
        c2 = np.array([[0.0, 0.0, np.sqrt(I_ROT), -np.sqrt(I_ROT)]])
        # Frobenius norm over every input mode and sample = sqrt(sum_t E_a |.|^2) / sigma_a
        off_k.append(np.linalg.norm(response_matrix(dataclasses.replace(s, C=c1),
                                                    band_hz=band_hz)))
        off_h.append(np.linalg.norm(response_matrix(dataclasses.replace(s, C=c2),
                                                    band_hz=band_hz)))
    ks_grid = np.asarray(ks_grid, float)
    big = ks_grid >= 1e7
    slope_k = np.polyfit(np.log10(ks_grid[big]), np.log10(np.array(off_k)[big]), 1)[0]
    slope_h = np.polyfit(np.log10(ks_grid[big]), np.log10(np.array(off_h)[big]), 1)[0]
    return dict(ks=ks_grid, gap_energy=np.array(gap_e), gap_state=np.array(gap_s),
                off_spring=np.array(off_k), off_rotor=np.array(off_h),
                slope_spring=slope_k, slope_rotor=slope_h,
                n_modes=n_modes(band_hz=band_hz))


def check_snr(mults=None) -> dict:
    """Bits against the noise scale, with the low- and high-SNR asymptotes (P2)."""
    if mults is None:
        mults = np.logspace(-3, 3, 49)
    out = {}
    for name, s in (("rigid", proprioceptive(readout="energy")),
                    ("SEA 1e4", series_elastic(1e4, readout="energy"))):
        M = response_matrix(s)
        G0 = SIGMA_A ** 2 * (M.T @ M)                 # gamma = 1 / sigma^2 pulled out
        lam = np.linalg.eigvalsh(G0)
        g = 1.0 / (SIGMA_ENERGY * np.asarray(mults)) ** 2
        exact = np.array([0.5 * np.sum(np.log2(1 + gi * lam)) for gi in g])
        low = g * np.trace(G0) / (2 * np.log(2))
        high = 0.5 * len(lam) * np.log2(g) + 0.5 * np.sum(np.log2(lam))
        out[name] = dict(exact=exact, low=low, high=high, lam=lam)
    gap = out["SEA 1e4"]["exact"] - out["rigid"]["exact"]
    det_gap = 0.5 * (np.sum(np.log2(out["SEA 1e4"]["lam"]))
                     - np.sum(np.log2(out["rigid"]["lam"])))
    tr_ratio = np.sum(out["SEA 1e4"]["lam"]) / np.sum(out["rigid"]["lam"])
    return dict(mults=np.asarray(mults), sys=out, gap=gap, det_gap=det_gap,
                trace_ratio=tr_ratio)


# ---------------------------------------------------------------------------------------
# Section 6: attribution
# ---------------------------------------------------------------------------------------

def check_subsets() -> dict:
    out = {}
    for name, s in (("rigid", proprioceptive(readout="energy")),
                    ("SEA 1e4", series_elastic(1e4, readout="energy"))):
        out[name] = dict(all_coords=coordinate_bits(s, list(range(2 * s.n))),
                         whitened=bits_of(s))
    shap = {ks: shapley_coordinate_bits(series_elastic(ks, readout="energy"))
            for ks in (70.0, 1e3, 1e4, 1e6, 1e8)}
    s8 = series_elastic(1e8, readout="energy")
    r = proprioceptive(readout="energy")
    fusion = dict(qj=coordinate_bits(s8, [0]), qm=coordinate_bits(s8, [1]),
                  qboth=coordinate_bits(s8, [0, 1]), q_rigid=coordinate_bits(r, [0]),
                  vj=coordinate_bits(s8, [2]), vm=coordinate_bits(s8, [3]),
                  vboth=coordinate_bits(s8, [2, 3]), v_rigid=coordinate_bits(r, [1]))
    # Noise correlation of (theta_j, theta_m) under sigma^2 E^{-1}, against k_s
    corr = {}
    for ks in (70.0, 1e3, 1e4, 1e6, 1e8):
        Kinv = np.linalg.inv(series_elastic(ks, readout="energy").K)
        corr[ks] = float(Kinv[0, 1] / np.sqrt(Kinv[0, 0] * Kinv[1, 1]))
    return dict(totals=out, shapley=shap, fusion=fusion, position_noise_corr=corr)


def check_morphological() -> dict:
    """MC_W needs a positive-definite E: defined on the clamped bench, refused on the free one."""
    from passive_info.channel import morphological_bits
    vals = {"rigid": morphological_bits(proprioceptive(readout="energy"))}
    for k in (70.0, 1e3, 1e4, 1e6):
        vals[f"SEA {k:g}"] = morphological_bits(series_elastic(k, readout="energy"))
    try:
        morphological_bits(proprioceptive(clamped=False, readout="energy"))
        free = "returned a number"
    except np.linalg.LinAlgError:
        free = "raises LinAlgError"
    return dict(values=vals, free_bench=free)


def check_mcw_closed_form() -> dict:
    """MC_W (no action) = -1/2 log2 det(I - A~ A~^T), with A~ = E^{1/2} A_d E^{-1/2}."""
    from passive_info.channel import morphological_bits
    rows = {}
    for name, s in (("rigid", proprioceptive(readout="energy")),
                    ("SEA 70", series_elastic(70.0, readout="energy")),
                    ("SEA 1e4", series_elastic(1e4, readout="energy"))):
        Ad, _ = zoh(s.A, s.B_F, 1.0 / FS)
        Eh = sym_sqrt(energy_metric(s.K, s.H))
        At = Eh @ Ad @ np.linalg.inv(Eh)
        sv = np.linalg.svd(At, compute_uv=False)
        closed = float(-0.5 * np.sum(np.log2(1.0 - sv ** 2)))
        rows[name] = dict(code=morphological_bits(s), closed=closed, sv_max=float(sv.max()),
                          sv_min=float(sv.min()))
    return rows


def check_sampling() -> dict:
    """Sampling-rate dependence under i.i.d. per-sample noise of fixed variance.

    "repo convention": the DCT modes are unit-norm over the SAMPLES (`mode_basis`), so a
    coefficient's force amplitude scales as fs^{-1/2}; the Gram, a sum over fs*T samples,
    then tends to an integral and the bits converge as fs grows.
    "fixed amplitude": the same force waveforms at every fs (amplitude scaled by
    sqrt(fs/FS), equivalent to sigma -> sigma sqrt(FS/fs)); now each doubling of fs is an
    independent second look and adds ~1/2 bit per well-excited mode, as in P1.
    """
    out = {}
    for name, s in (("rigid", proprioceptive(readout="energy")),
                    ("SEA 1e4", series_elastic(1e4, readout="energy"))):
        b = {}
        for fs in (2000.0, 4000.0, 8000.0):
            b[(fs, "repo convention")] = bits_of(s, fs=fs)
            b[(fs, "fixed amplitude")] = bits_of(s, fs=fs,
                                                 sigma=SIGMA_ENERGY * np.sqrt(FS / fs))
        out[name] = b
    return out


def check_degrees_prediction() -> dict:
    """Relabelling radians as degrees multiplies the per-coordinate SNR by (180/pi)^2."""
    inv = check_invariance()["rows"]
    names = list(inv)
    observed = inv[names[1]]["state"] - inv[names[0]]["state"]
    M = response_matrix(reparametrized_sea(1e4, np.eye(2), "state"))
    lam = np.linalg.eigvalsh((SIGMA_A / 1e-4) ** 2 * M.T @ M)
    c = (180.0 / np.pi) ** 2
    exact = float(0.5 * np.sum(np.log2((1 + c * lam) / (1 + lam))))
    return dict(observed=observed, exact=exact,
                high_snr=0.5 * len(lam) * np.log2(c), n_modes=len(lam))


# ---------------------------------------------------------------------------------------
# Section 7: related problems
# ---------------------------------------------------------------------------------------

ROD = dict(L=0.2, EA=1.0e3, rhoA=0.1, zeta1=0.05, band_hz=400.0,
           sigma_energy=1.0e-3, sigma_node=1.0e-6, sigma_a=1.0)


def rod_system(N: int, p: dict = ROD):
    """Fixed-free axial rod, N linear elements, consistent mass, tip force."""
    L, EA, rhoA = p["L"], p["EA"], p["rhoA"]
    h = L / N
    ke = EA / h * np.array([[1.0, -1.0], [-1.0, 1.0]])
    me = rhoA * h / 6.0 * np.array([[2.0, 1.0], [1.0, 2.0]])
    K = np.zeros((N + 1, N + 1))
    H = np.zeros((N + 1, N + 1))
    for e in range(N):
        K[e:e + 2, e:e + 2] += ke
        H[e:e + 2, e:e + 2] += me
    K, H = K[1:, 1:], H[1:, 1:]
    w1 = 0.5 * np.pi * np.sqrt(EA / rhoA) / L          # continuum first mode
    D = 2.0 * p["zeta1"] / w1 * K                       # stiffness-proportional damping
    Hi = np.linalg.inv(H)
    A = np.block([[np.zeros((N, N)), np.eye(N)], [-Hi @ K, -Hi @ D]])
    J = np.zeros((1, N))
    J[0, -1] = 1.0
    B = np.vstack([np.zeros((N, 1)), Hi @ J.T])
    return A, B, K, H, w1


def _response(A, B, C, band_hz, t_window=T_WINDOW, fs=FS):
    n_t = int(round(t_window * fs)) + 1
    Phi = mode_basis(n_t, n_modes(t_window, band_hz))
    Ad, Bd = zoh(A, B, 1.0 / fs)
    X = np.zeros((A.shape[0], Phi.shape[1]))
    out = [C @ X]
    for k in range(n_t - 1):
        X = Ad @ X + Bd * Phi[k][None, :]
        out.append(C @ X)
    return np.concatenate(out, axis=0)                  # (n_t * n_out, K)


def check_rod(Ns=(2, 4, 8, 16, 32, 64, 128), p: dict = ROD) -> dict:
    """Mesh refinement: energy-whitened bits converge, per-node isotropic bits do not."""
    e, nn, tip = [], [], []
    for N in Ns:
        A, B, K, H, w1 = rod_system(N, p)
        Ce = sym_sqrt(energy_metric(K, H))
        Cn = np.diag(np.r_[np.ones(N), np.ones(N) / w1])
        Ct = np.zeros((2, 2 * N))
        Ct[0, N - 1], Ct[1, 2 * N - 1] = 1.0, 1.0 / w1
        for C, sig, acc in ((Ce, p["sigma_energy"], e), (Cn, p["sigma_node"], nn),
                            (Ct, p["sigma_node"], tip)):
            M = _response(A, B, C, p["band_hz"])
            acc.append(bits_from_gram((p["sigma_a"] / sig) ** 2 * (M.T @ M)))
    Ns = np.asarray(Ns)
    k = n_modes(band_hz=p["band_hz"])
    late = Ns >= 16
    slope = np.polyfit(np.log2(Ns[late]), np.array(nn)[late], 1)[0]
    return dict(N=Ns, energy=np.array(e), per_node=np.array(nn), tip=np.array(tip),
                n_modes=k, per_node_bits_per_doubling=slope,
                predicted_per_doubling=0.5 * k)


def check_floating() -> dict:
    """Free bench: E is singular, yet nothing is lost when force and ground share J.

    A counterexample pushes on the trunk instead of the foot, which excites the zero-energy
    direction; there the singular metric does lose information.
    """
    f_e = proprioceptive(clamped=False, readout="energy")
    f_k = proprioceptive(clamped=False, readout="kinetic")
    lam, V = sla.eigh(f_e.K, f_e.H)
    phi0 = V[:, np.argmin(np.abs(lam))]
    J_phi0 = float((f_e.J @ phi0).item())
    # counterexample: unit force on the trunk coordinate y_b, not through the foot Jacobian
    n = f_e.n
    B_trunk = np.vstack([np.zeros((n, 1)), np.linalg.solve(f_e.H, np.eye(n)[:, [0]])])
    t_e = dataclasses.replace(f_e, B_F=B_trunk)
    t_k = dataclasses.replace(f_k, B_F=B_trunk)
    return dict(zero_eig=float(lam.min()), J_phi0=J_phi0,
                foot_energy=bits_of(f_e), foot_kinetic=bits_of(f_k),
                trunk_energy=bits_of(t_e), trunk_kinetic=bits_of(t_k),
                phi0_trunk_force=float(phi0[0]))


# =======================================================================================
# Part II: is energy the right choice?
# =======================================================================================

def _mpow(S: np.ndarray, p: float) -> np.ndarray:
    lam, V = np.linalg.eigh(S)
    return (V * np.clip(lam, 1e-300, None) ** p) @ V.T


def metric_Wp(K: np.ndarray, H: np.ndarray, p: float, w0: float) -> np.ndarray:
    """Frequency-weighted precision form sum_i (omega_i / w0)^p * (modal energy form i).

    p = 0 is the energy metric; p = -1 is action weighting (the quantum zero-point shape).
    Built in whitened coordinates, where the modal energies are the blocks of
    diag((Omega Omega^T)^{p/2}, (Omega^T Omega)^{p/2}), Omega = K^{1/2} H^{-1/2}.
    `w0` only makes the units match the energy metric; it cancels in every design gap at
    equal rank.
    """
    n = H.shape[0]
    Eh = sym_sqrt(energy_metric(K, H))
    Om = sym_sqrt(K) @ np.linalg.inv(sym_sqrt(H))
    mid = np.zeros((2 * n, 2 * n))
    mid[:n, :n] = _mpow(Om @ Om.T, p / 2.0) / w0 ** p
    mid[n:, n:] = _mpow(Om.T @ Om, p / 2.0) / w0 ** p
    return Eh @ mid @ Eh


def _bits_with_W(s: Passive, W: np.ndarray, **kw) -> float:
    return profile_bits(dataclasses.replace(s, C=sym_sqrt(W), readout="energy"), **kw)["bits"]


def _state_W(n: int) -> np.ndarray:
    w = omega_ref()
    return np.diag(np.r_[np.ones(n), np.ones(n) / w ** 2])


def _kinetic_W(s: Passive) -> np.ndarray:
    return energy_metric(s.K, s.H, w=omega_ref())


def check_uniqueness_flow(seed: int = 0) -> dict:
    """d/dt (a q'Kq + b qdot'H qdot) = 2(a - b) qdot'Kq along H qdd = -K q (Theorem)."""
    rng = np.random.default_rng(seed)
    s = series_elastic(1e4, readout="energy")
    worst = 0.0
    for a, b in ((1.0, 1.0), (2.0, 1.0), (0.5, 3.0)):
        for _ in range(50):
            q, v = rng.standard_normal(s.n), rng.standard_normal(s.n)
            acc = -np.linalg.solve(s.H, s.K @ q)
            dW = 2 * a * v @ s.K @ q + 2 * b * v @ s.H @ acc
            pred = 2 * (a - b) * v @ s.K @ q
            worst = max(worst, abs(dW - pred) / (abs(v @ s.K @ q) + 1e-30))
    return dict(max_rel_err=worst)


def check_flow_invariance() -> dict:
    """Which precision forms the undamped flow preserves: Phi^T W Phi = W ?"""
    s = series_elastic(1e4, zeta_series=0.0, zeta_contact=0.0, readout="energy")
    w0 = omega_ref()
    forms = {"energy": energy_metric(s.K, s.H), "action (p=-1)": metric_Wp(s.K, s.H, -1, w0),
             "p=+1": metric_Wp(s.K, s.H, 1, w0), "per-coordinate": _state_W(s.n),
             "kinetic": _kinetic_W(s)}
    out = {}
    for name, W in forms.items():
        dev = 0.0
        for t in np.linspace(1e-4, 2e-2, 40):
            Phi = sla.expm(s.A * t)
            dev = max(dev, np.abs(Phi.T @ W @ Phi - W).max() / np.abs(W).max())
        out[name] = dev
    out["det_Phi"] = float(np.linalg.det(sla.expm(s.A * 7e-3)))   # Liouville: = 1
    return out


def check_locality() -> dict:
    """Changing the GROUND spring changes the action metric's MOTOR-velocity entry.

    The energy metric's (theta_m-dot, theta_m-dot) entry is I_rot whatever k_i is: each
    element contributes its own term. A modal weighting is a matrix function of H^{-1}K,
    so every element influences every entry.
    """
    w0 = omega_ref()
    out = {}
    for ki in (1e5, 1e6, 1e7):
        s = series_elastic(1e4, k_i=ki, readout="energy")
        idx = 2 * s.n - 1
        out[ki] = dict(energy=float(energy_metric(s.K, s.H)[idx, idx]),
                       action=float(metric_Wp(s.K, s.H, -1, w0)[idx, idx] * w0))
    return out


def check_free_motion(t_max: float = 0.02, n: int = 400, bits0: float = 2.0) -> dict:
    """Information about a scalar signal stored in the state, under free motion.

    The signal Y enters as x(0) = v Y with v along the motor velocity. Each metric's noise
    scale is set so that I(0) = `bits0`. Undamped: energy and action stay flat (they are
    flow-invariant), while per-coordinate and kinetic oscillate, i.e. free motion "creates"
    information. Damped: energy can only decrease.
    """
    times = np.linspace(0.0, t_max, n)
    out = {"t": times}
    for damped in (False, True):
        kw = {} if damped else dict(zeta_series=0.0, zeta_contact=0.0)
        s = series_elastic(1e4, readout="energy", **kw)
        w0 = omega_ref()
        forms = {"energy": energy_metric(s.K, s.H), "action": metric_Wp(s.K, s.H, -1, w0),
                 "per-coordinate": _state_W(s.n), "kinetic": _kinetic_W(s)}
        res = {}
        E = forms["energy"]
        for name, W in forms.items():
            # start where W resolves the state worst relative to its energy: the
            # generalized eigenvector of (W, E) with the smallest eigenvalue. For the
            # energy metric every direction is equivalent; take the motor velocity.
            if name == "energy":
                v = np.zeros(4)
                v[3] = 1.0
            else:
                lam, V = sla.eigh(W, E)
                v = V[:, 0] / np.linalg.norm(V[:, 0])
            snr0 = 2.0 ** (2 * bits0) - 1.0
            scale = snr0 / float(v @ W @ v)
            I = []
            for t in times:
                u = sla.expm(s.A * t) @ v
                I.append(0.5 * np.log2(1.0 + scale * float(u @ W @ u)))
            res[name] = np.array(I)
        out["damped" if damped else "undamped"] = res
    return out


def check_alternatives_sea(ks_list=(1e3, 1e4, 1e6, 1e8, 1e10)) -> dict:
    """Stiff-limit gap and coordinate-invariance spread for each candidate rule."""
    w0 = omega_ref()
    r = proprioceptive(readout="energy")
    rules = {f"p={p}": (lambda s, p=p: metric_Wp(s.K, s.H, p, w0)) for p in (-2, -1, 0, 1, 2, 3)}
    rules["kinetic"] = _kinetic_W
    rules["per-coordinate"] = lambda s: _state_W(s.n)
    from passive_info.information import SIGMA_THETA
    gaps, spread = {}, {}
    for name, rule in rules.items():
        # the per-coordinate rule is quoted at the encoder resolution, as in the code's
        # "state" readout; the energy-unit rules at SIGMA_ENERGY
        kw = dict(sigma=SIGMA_THETA) if name == "per-coordinate" else {}
        base = _bits_with_W(r, rule(r), **kw)
        gaps[name] = np.array([_bits_with_W(sk, rule(sk), **kw) - base for sk in
                               (series_elastic(float(k), readout="energy") for k in ks_list)])
        vals = [_bits_with_W(sT, rule(sT), **kw) for sT in
                (reparametrized_sea(1e4, T, "energy") for T in REPARAMETRIZATIONS.values())]
        spread[name] = max(vals) - min(vals)
    return dict(ks=np.asarray(ks_list), gaps=gaps, spread=spread)


def check_alternatives_rod(Ns=(4, 8, 16, 32, 64), p: dict = ROD) -> dict:
    """Mesh refinement under each rule."""
    out = {}
    for name in ("p=-1", "p=0", "p=1", "kinetic", "per-node"):
        vals = []
        for N in Ns:
            A, B, K, H, w1 = rod_system(N, p)
            if name.startswith("p="):
                W = metric_Wp(K, H, float(name[2:]), w1)
                sig = p["sigma_energy"]
            elif name == "kinetic":
                W = energy_metric(K, H, w=w1)
                sig = p["sigma_energy"]
            else:
                W = np.diag(np.r_[np.ones(N), np.ones(N) / w1 ** 2])
                sig = p["sigma_node"]
            M = _response(A, B, sym_sqrt(W), p["band_hz"])
            vals.append(bits_from_gram((p["sigma_a"] / sig) ** 2 * (M.T @ M)))
        out[name] = np.array(vals)
    return dict(N=np.asarray(Ns), bits=out)


def check_data_driven_whitening(sigma_n: float = 0.1) -> dict:
    """Whitening the response by its own covariance erases the body.

    y = M w is whitened on the range of M by (sigma_w^2 M M^T)^{+1/2}, then isotropic noise
    sigma_n is added. Every full-rank body gives (d/2) log2(1 + sigma_n^-2).
    """
    out = {}
    for name, s in (("rigid", proprioceptive(readout="energy")),
                    ("SEA 70", series_elastic(70.0, readout="energy")),
                    ("SEA 1e4", series_elastic(1e4, readout="energy"))):
        M = response_matrix(s)
        U, sv, _ = np.linalg.svd(SIGMA_A * M, full_matrices=False)
        Wh = (U / sv) @ U.T                        # (sigma_w^2 M M^T)^{+1/2} on range
        G = (Wh @ (SIGMA_A * M)).T @ (Wh @ (SIGMA_A * M)) / sigma_n ** 2
        out[name] = bits_from_gram(G)
    d = n_modes()
    out["predicted"] = 0.5 * d * np.log2(1 + sigma_n ** -2)
    return out


# ---------------------------------------------------------------------------------------
# The Langevin body: noise injected through the dampers (fluctuation-dissipation)
# ---------------------------------------------------------------------------------------

def _damping(s: Passive) -> np.ndarray:
    return -s.H @ s.A[s.n:, s.n:]                    # A[n:, n:] = -H^{-1} D


def langevin_bits(s: Passive, C: np.ndarray, eps: float, sigma_T2: float | None = None,
                  fs: float = FS, band_hz: float = 100.0) -> float:
    """Bits about the force profile when the body is thermalized by its own dampers.

    H qdd + D qd + K q = J^T f + xi, E[xi xi^T] = 2 sigma_T^2 D delta, started from the
    Gibbs law N(0, sigma_T^2 E^{-1}), observed through C with extra readout noise eps.
    The noise trajectory is stationary Gauss-Markov: Cov(x_k, x_l) = A_d^{k-l} P (k >= l),
    which is exact because P is the FDT-stationary covariance (`gibbs_pair`).
    """
    sigma_T2 = SIGMA_ENERGY ** 2 if sigma_T2 is None else sigma_T2
    n2 = 2 * s.n
    Ad, _ = zoh(s.A, s.B_F, 1.0 / fs)
    P = sigma_T2 * np.linalg.inv(energy_metric(s.K, s.H))
    M = response_matrix(dataclasses.replace(s, C=C), fs=fs, band_hz=band_hz)
    m = C.shape[0]
    n_t = M.shape[0] // m
    powers = [np.eye(n2)]
    for _ in range(n_t):
        powers.append(Ad @ powers[-1])
    S = np.zeros((n_t * m, n_t * m))
    CP = C @ P
    for k in range(n_t):
        for l in range(k + 1):
            blk = C @ powers[k - l] @ CP.T
            S[k * m:(k + 1) * m, l * m:(l + 1) * m] = blk
            S[l * m:(l + 1) * m, k * m:(k + 1) * m] = blk.T
    S += eps ** 2 * np.eye(n_t * m)
    Z = np.linalg.solve(np.linalg.cholesky(S), M)
    return bits_from_gram(SIGMA_A ** 2 * (Z.T @ Z))


def langevin_innovation_g(s: Passive, fs: float = FS, sigma_T2: float | None = None) -> float:
    """g = B_d^T Q_d^{-1} B_d: the per-sample SNR of the force against the thermal kick."""
    sigma_T2 = SIGMA_ENERGY ** 2 if sigma_T2 is None else sigma_T2
    Ad, Bd = zoh(s.A, s.B_F, 1.0 / fs)
    _, Q = gibbs_pair(s.K, s.H, Ad, label=s.label)
    return float((Bd.T @ np.linalg.solve(sigma_T2 * Q, Bd)).item())


def langevin_innovation_bits(s: Passive, fs: float = FS, band_hz: float = 100.0) -> float:
    """Full-state observation: bits = 1/2 logdet(I + sigma_w^2 g Phi^T Phi)."""
    g = langevin_innovation_g(s, fs)
    n_t = int(round(T_WINDOW * fs)) + 1
    Phi = mode_basis(n_t, n_modes(T_WINDOW, band_hz))[:-1]   # last sample drives nothing
    return bits_from_gram(SIGMA_A ** 2 * g * (Phi.T @ Phi))


def check_langevin_innovations() -> dict:
    out = {}
    for name, s in (("rigid", proprioceptive(readout="energy")),
                    ("SEA 1e4", series_elastic(1e4, readout="energy"))):
        out[name] = dict(innovations=langevin_innovation_bits(s),
                         stacked=langevin_bits(s, np.eye(2 * s.n), 0.0))
    return out


def check_langevin_dt_limit(fs_list=(1e2, 1e3, 4e3, 16e3, 64e3)) -> dict:
    """g sigma_T^2 / (dt J D^+ J^T) -> 1/2 as dt -> 0, for every design."""
    out = {}
    designs = [("rigid", proprioceptive(readout="energy"))] + [
        (f"SEA {k:g}", series_elastic(k, readout="energy")) for k in (70.0, 1e4, 1e6)]
    for name, s in designs:
        JDJ = float((s.J @ np.linalg.pinv(_damping(s)) @ s.J.T).item())
        out[name] = dict(JDJ=JDJ, ratio=np.array([
            langevin_innovation_g(s, fs) * SIGMA_ENERGY ** 2 * fs / JDJ for fs in fs_list]))
    return dict(fs=np.asarray(fs_list), rows=out)


def check_langevin_designs(ks_list=(70.0, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5, 1e6)) -> dict:
    """SEA vs rigid under Langevin noise for several readouts, and the readout-only models."""
    from passive_info.information import SIGMA_THETA
    designs = [("rigid", None)] + [(k, float(k)) for k in ks_list]
    rows = {}
    for name, ks in designs:
        s = proprioceptive(readout="energy") if ks is None else series_elastic(ks, readout="energy")
        n2 = 2 * s.n
        Ce = sym_sqrt(energy_metric(s.K, s.H))
        Cm = np.zeros((1, n2))
        Cm[0, s.n - 1] = 1.0
        rows[name] = dict(
            full=langevin_bits(s, np.eye(n2), 0.0),
            full_readout=langevin_bits(s, Ce, SIGMA_ENERGY),
            motor=langevin_bits(s, Cm, 1e-9),
            motor_encoder=langevin_bits(s, Cm, SIGMA_THETA),
            readout_energy=profile_bits(s)["bits"],
            readout_motor=profile_bits(dataclasses.replace(s, C=Cm, readout="motor"))["bits"])
    return dict(ks=np.asarray(ks_list, float), rows=rows)


# ---------------------------------------------------------------------------------------

def report() -> None:
    np.set_printoptions(precision=4, suppress=False)
    print("energy identity", check_energy_identity())
    print("square roots", check_square_roots())
    print("noise equivalence", check_noise_equivalence())
    inv = check_invariance()
    for k, v in inv["rows"].items():
        print(f"  {k:40s} energy {v['energy']:.9f}   state {v['state']:.4f}")
    print("  spreads", inv["energy_spread"], inv["state_spread"])
    print("maxent", check_maxent())
    print("fdt", check_fdt())
    for r in check_port_hamiltonian()["rows"]:
        print("  pH", r["name"], r["sym_err"], r["max_eig_sym"], r["sv"], r["omega"])
    print("duplicate", check_duplicate())
    print("locked restriction", check_locked_restriction())
    st = check_stiff_limit(np.array([1e2, 1e4, 1e6, 1e8, 1e10, 1e12]))
    print("stiff", st["gap_energy"], st["gap_state"], st["slope_spring"], st["slope_rotor"])
    sn = check_snr(np.array([1e-3, 1e-2, 1, 10, 100]))
    print("snr gap", sn["gap"], "det gap", sn["det_gap"], "trace ratio", sn["trace_ratio"])
    sub = check_subsets()
    print("subsets", sub["totals"], sub["fusion"], sub["position_noise_corr"])
    for k, v in sub["shapley"].items():
        print("  shapley", k, v, v.sum())
    print("MC_W", check_morphological())
    print("rod", check_rod())
    print("floating", check_floating())
    print("--- Part II")
    print("uniqueness identity", check_uniqueness_flow())
    print("flow invariance", check_flow_invariance())
    print("locality", check_locality())
    alt = check_alternatives_sea()
    for k in alt["gaps"]:
        print(f"  {k:15s} stiff gaps {alt['gaps'][k]}  invariance spread {alt['spread'][k]:.2e}")
    print("rod by rule", check_alternatives_rod()["bits"])
    print("data-driven", check_data_driven_whitening())
    print("langevin innovations", check_langevin_innovations())
    print("langevin dt limit", check_langevin_dt_limit())
    for k, v in check_langevin_designs()["rows"].items():
        print("  langevin", k, {kk: round(vv, 3) for kk, vv in v.items()})


if __name__ == "__main__":
    report()
