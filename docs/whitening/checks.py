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


if __name__ == "__main__":
    report()
