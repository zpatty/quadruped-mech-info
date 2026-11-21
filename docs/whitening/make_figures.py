"""Draw every figure in `energy_whitening.tex` from `checks.py`.

    uv run python docs/whitening/make_figures.py

Writes PDFs to docs/whitening/figures/ and prints the numbers quoted in the note.
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import checks  # noqa: E402

OUT = HERE / "figures"

# Categorical slots, fixed order (blue, orange, aqua, yellow); text never wears series color.
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.5, "lines.markersize": 4, "legend.frameon": False,
    "pdf.fonttype": 42, "font.family": "serif", "mathtext.fontset": "cm",
})


def _save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def fig_invariance():
    r = checks.check_invariance()
    names = list(r["rows"])
    e = [r["rows"][n]["energy"] for n in names]
    s = [r["rows"][n]["state"] for n in names]
    y = np.arange(len(names))[::-1]
    fig, ax = plt.subplots(figsize=(5.6, 2.1))
    ax.scatter(s, y, color=C2, s=28, zorder=3, label="per-coordinate noise (\"state\")",
               edgecolor="white", linewidth=0.8)
    ax.scatter(e, y, color=C1, s=28, zorder=4, marker="D",
               label="energy-whitened noise", edgecolor="white", linewidth=0.8)
    ax.set_yticks(y, [n.replace("$", "$") for n in names])
    ax.set_xlabel("bits about the contact-force profile")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    ax.set_xlim(60, 190)
    _save(fig, "invariance")
    return r


def fig_stiff_limit():
    r = checks.check_stiff_limit()
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.5))
    a.semilogx(r["ks"], r["gap_state"], color=C2)
    a.semilogx(r["ks"], r["gap_energy"], color=C1)
    a.axhline(0.5 * r["n_modes"], color=INK2, lw=0.8, ls=":")
    a.axhline(0.0, color=INK2, lw=0.6)
    a.text(1e10, 0.5 * r["n_modes"] - 1.3, "8.5 bits = 17 modes / 2", color=INK2,
           fontsize=7, ha="center")
    a.text(2e5, 9.6, "per-coordinate", color=INK, fontsize=7.5)
    a.text(1e8, 0.7, "energy-whitened", color=INK, fontsize=7.5)
    a.set_xlabel(r"series stiffness $k_s$ (N m/rad)")
    a.set_ylabel("bits(SEA) $-$ bits(rigid)")
    a.set_title("(a) the redundant coordinate", loc="left")

    b.loglog(r["ks"], r["off_spring"], color=C1)
    b.loglog(r["ks"], r["off_rotor"], color=C3)
    ks = r["ks"]
    ref = r["off_spring"][-1] * (ks / ks[-1]) ** -0.5
    b.loglog(ks[ks > 1e6], ref[ks > 1e6], color=INK2, lw=0.8, ls="--")
    ref2 = r["off_rotor"][-1] * (ks / ks[-1]) ** -1.0
    b.loglog(ks[ks > 1e6], ref2[ks > 1e6], color=INK2, lw=0.8, ls="--")
    b.text(3e9, 3e-6, r"$\sqrt{k_s}\,\delta\ \ (\propto k_s^{-1/2})$", fontsize=7.5)
    b.text(3e5, 2e-9, r"$\sqrt{I_{\rm rot}}\,\dot\delta\ \ (\propto k_s^{-1})$", fontsize=7.5)
    b.set_xlabel(r"series stiffness $k_s$ (N m/rad)")
    b.set_ylabel(r"$\|\cdot\|$ over window and modes")
    b.set_title("(b) its whitened size vanishes", loc="left")
    fig.tight_layout()
    _save(fig, "stiff_limit")
    return r


def fig_snr():
    r = checks.check_snr()
    m = r["mults"]
    rig = r["sys"]["rigid"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.5))
    a.loglog(m, rig["exact"], color=C1)
    ok = rig["high"] > 0
    a.loglog(m[ok], rig["high"][ok], color=INK2, lw=0.8, ls="--")
    lo = rig["low"] < 1e3
    a.loglog(m[lo], rig["low"][lo], color=INK2, lw=0.8, ls=":")
    a.text(1.2e-3, 60, r"$\frac{r}{2}\log_2\sigma^{-2} + \frac{1}{2}\log_2\det G_0$", fontsize=7.5)
    a.text(20, 0.03, r"$\mathrm{tr}\,G_0 / (2\sigma^2\ln 2)$", fontsize=7.5)
    a.set_xlabel(r"noise scale $\sigma/\sigma_{\rm ref}$")
    a.set_ylabel("bits (rigid leg)")
    a.set_title("(a) two regimes", loc="left")

    b.semilogx(m, r["gap"], color=C1)
    b.axhline(r["det_gap"], color=INK2, lw=0.8, ls="--")
    b.axhline(0.0, color=INK2, lw=0.6)
    b.text(1.5e-3, r["det_gap"] + 0.003, r"$\frac{1}{2}\log_2\frac{\det G_{\rm SEA}}{\det G_{\rm rigid}}$",
           fontsize=7.5)
    b.set_xlabel(r"noise scale $\sigma/\sigma_{\rm ref}$")
    b.set_ylabel(r"bits(SEA $10^4$) $-$ bits(rigid)")
    b.set_title("(b) the design gap", loc="left")
    fig.tight_layout()
    _save(fig, "snr")
    return r


def fig_shapley():
    ks = np.logspace(1.8, 9, 25)
    phi = np.array([checks.shapley_coordinate_bits(checks.series_elastic(float(k),
                                                                         readout="energy"))
                    for k in ks])
    fig, ax = plt.subplots(figsize=(4.2, 2.5))
    labels = [r"$\theta_j$", r"$\theta_m$", r"$\dot\theta_j$", r"$\dot\theta_m$"]
    for i, (c, ls) in enumerate(zip((C1, C2, C3, C4), ("-", "--", "-", "--"))):
        ax.semilogx(ks, phi[:, i], color=c, ls=ls)
        ax.text(ks[-1] * 1.3, phi[-1, i] + (1.6 if i == 0 else -1.6 if i == 1 else 0),
                labels[i], color=INK, fontsize=8, va="center")
    ax.set_xlim(ks[0], ks[-1] * 6)
    ax.set_xlabel(r"series stiffness $k_s$ (N m/rad)")
    ax.set_ylabel("Shapley bits")
    _save(fig, "shapley")
    return phi


def fig_rod():
    r = checks.check_rod()
    N = r["N"]
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.semilogx(N, r["per_node"], color=C2, marker="o", base=2)
    ax.semilogx(N, r["tip"], color=C3, marker="s", base=2)
    ax.semilogx(N, r["energy"], color=C1, marker="D", base=2)
    ax.text(N[-1] * 1.15, r["per_node"][-1], "per-node isotropic", va="center", fontsize=7.5)
    ax.text(N[-1] * 1.15, r["tip"][-1], "tip sensor only", va="center", fontsize=7.5)
    ax.text(N[-1] * 1.15, r["energy"][-1], "energy-whitened", va="center", fontsize=7.5)
    ax.set_xlim(N[0] / 1.3, N[-1] * 12)
    ax.set_xlabel("finite elements $N$")
    ax.set_ylabel("bits about the tip force")
    _save(fig, "rod")
    return r


def fig_ellipse():
    """Mass on a spring: energy-shaped noise follows the constant-energy ellipses."""
    k, m, sig = 4.0, 1.0, 0.25
    rng = np.random.default_rng(1)
    th = np.linspace(0, 2 * np.pi, 200)
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.7), sharex=True, sharey=True)
    for ax, title, cov in ((axes[0], "(a) energy-shaped noise", np.diag([sig ** 2 / k, sig ** 2 / m])),
                           (axes[1], "(b) per-coordinate noise", np.diag([0.09, 0.09]) * sig ** 2 / 0.25)):
        for e in (0.5, 1.0, 1.5):
            ax.plot(e * sig * np.cos(th) / np.sqrt(k), e * sig * np.sin(th) / np.sqrt(m),
                    color=INK2, lw=0.7)
        pts = rng.multivariate_normal([0, 0], cov, 400)
        ax.scatter(pts[:, 0], pts[:, 1], s=3, color=C1 if ax is axes[0] else C2, alpha=0.6,
                   linewidths=0)
        ax.set_title(title, loc="left")
        ax.set_xlabel("position $q$")
        ax.set_aspect("equal")
    axes[0].set_ylabel(r"velocity $\dot q$")
    fig.tight_layout()
    _save(fig, "ellipse")


def fig_free_motion():
    r = checks.check_free_motion()
    t = r["t"] * 1e3
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.5), sharey=True)
    for ax, key, title in ((a, "undamped", "(a) undamped"), (b, "damped", "(b) with the leg's damping")):
        res = r[key]
        ax.plot(t, res["per-coordinate"], color=C2, label="per-coordinate")
        ax.plot(t, res["kinetic"], color=C4, ls="--", label=r"kinetic ($\omega_{\rm ref}$)")
        ax.plot(t, res["action"], color=C3, ls=":", label="action ($p=-1$)")
        ax.plot(t, res["energy"], color=C1, label="energy")
        ax.axhline(2.0, color=INK2, lw=0.6)
        ax.set_xlabel("time since the signal was stored (ms)")
        ax.set_title(title, loc="left")
    a.set_ylabel("bits about the stored signal")
    a.legend(loc="lower right", ncol=2, fontsize=7)
    a.set_ylim(0, 4.1)
    fig.tight_layout()
    _save(fig, "free_motion")
    return r


def fig_alternatives():
    sea = checks.check_alternatives_sea()
    rod = checks.check_alternatives_rod()
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.6))
    ks = sea["ks"]
    styles = {"p=-1": (C3, ":"), "p=0": (C1, "-"), "p=1": (C4, "--"), "p=2": (INK2, "-."),
              "p=3": (C2, "-")}
    for name, (c, ls) in styles.items():
        a.semilogx(ks, np.abs(sea["gaps"][name]) + 1e-6, color=c, ls=ls)
    a.set_yscale("log")
    a.set_ylim(1e-6, 2e2)
    a.text(1.4e9, 120, "$p=3$", fontsize=7.5)
    a.text(1.4e9, 2.2, "$p=2$", fontsize=7.5)
    a.text(1.4e9, 1.4e-3, "$p=1$", fontsize=7.5)
    a.text(2e6, 3e-6, "$p\\leq0$", fontsize=7.5)
    a.set_xlabel(r"series stiffness $k_s$ (N m/rad)")
    a.set_ylabel(r"$|$bits(SEA) $-$ bits(rigid)$|$")
    a.set_title(r"(a) stiff limit, $W_p=\sum_i\omega_i^p E_i$", loc="left")

    N = rod["N"]
    for name, (c, ls, lab) in {"p=-1": (C3, ":", "action $p=-1$"), "p=0": (C1, "-", "energy $p=0$"),
                               "p=1": (C4, "--", "$p=+1$"), "kinetic": (INK2, "-.", "kinetic")}.items():
        v = rod["bits"][name]
        b.semilogx(N, v - v[-1], color=c, ls=ls, marker="o", markersize=3, base=2)
        b.text(N[-1] * 1.15, v[0] - v[-1] if name == "p=1" else (v[-1] - v[-1]), "", fontsize=7)
    b.text(5, -11, "$p=+1$ keeps growing", fontsize=7.5)
    b.text(17, -2.2, "$p\\leq 0$, kinetic: converge", fontsize=7.5)
    b.set_xlabel("finite elements $N$")
    b.set_ylabel(r"bits $-$ bits($N=64$)")
    b.set_title("(b) finite-element rod", loc="left")
    fig.tight_layout()
    _save(fig, "alternatives")
    return sea, rod


def fig_langevin():
    r = checks.check_langevin_designs()
    ks = r["ks"]
    rows = r["rows"]
    get = lambda key: np.array([rows[k][key] for k in ks]) - rows["rigid"][key]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.6))
    a.semilogx(ks, get("full"), color=INK2, ls="-.")
    a.semilogx(ks, get("motor"), color=C1, marker="o", markersize=3)
    a.semilogx(ks, get("readout_motor"), color=C2, marker="s", markersize=3)
    a.axhline(0, color=INK2, lw=0.6)
    a.text(2e4, -4.5, "Langevin, full state", fontsize=7.5)
    a.text(1.2e3, -22, "Langevin, motor encoder", fontsize=7.5)
    a.text(1.3e4, 4.5, "readout noise, motor", fontsize=7.5)
    a.set_xlabel(r"series stiffness $k_s$ (N m/rad)")
    a.set_ylabel("bits(SEA) $-$ bits(rigid)")
    a.set_title("(a) design gap by noise model", loc="left")

    eff = np.array([rows[k]["motor"] / rows[k]["full"] for k in ks])
    b.semilogx(ks, eff, color=C1, marker="o", markersize=3)
    b.axhline(rows["rigid"]["motor"] / rows["rigid"]["full"], color=INK2, lw=0.8, ls="--")
    b.text(80, rows["rigid"]["motor"] / rows["rigid"]["full"] - 0.08, "rigid transmission", fontsize=7.5)
    b.set_ylim(0, 1.05)
    b.set_xlabel(r"series stiffness $k_s$ (N m/rad)")
    b.set_ylabel("motor bits / full-state bits")
    b.set_title("(b) routing efficiency", loc="left")
    fig.tight_layout()
    _save(fig, "langevin")
    return r


if __name__ == "__main__":
    fig_invariance()
    fig_stiff_limit()
    fig_snr()
    fig_shapley()
    fig_rod()
    fig_ellipse()
    fig_free_motion()
    fig_alternatives()
    fig_langevin()
