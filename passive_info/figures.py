"""Plots for the passive rebuild."""
from __future__ import annotations

import pathlib

import numpy as np

from passive_info.information import (BAND_ENSEMBLE_HZ, ensemble_summary, profile_bits)
from passive_info.systems import mode_coupling, proprioceptive, series_elastic

RESULTS = pathlib.Path(__file__).parent / "results"

INK, MUTED, INK_BG = "#1a1a1a", "#6b6b6b", "#ffffff"
PEAK = "#c8102e"                       # a single reserved accent for "the peak", used nowhere else
#: Sequential ramp for the ordered k_s sweep, light (soft) to dark (stiff); proprioceptive
#: is drawn separately in INK so it reads as the reference case rather than another rung.
KS_SEQUENCE = (70.0, 300.0, 1000.0, 2728.0, 1e4, 1e5, 1e6)
_SEQ_COLORS = ["#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#0b559f", "#08306b"]
COLORS = dict(zip(KS_SEQUENCE, _SEQ_COLORS))


def _style():
    import matplotlib.pyplot as plt  # noqa: F401
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        # Spines match the ticks. A grey frame around black ticks and labels
        # reads as an oversight rather than a choice, and washes out in print.
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        # Ticks and their labels are INK, not MUTED. The muted grey was chosen when these
        # plates were read on screen; at print size it reads washed out, and a reader
        # checking a value against an axis should not have to work for it. The axis spines
        # stay muted -- they are a frame, not data.
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "axes.linewidth": 0.7,
    })
    return plt


def _finish(fig, axes, name: str, fs: float = 9.0, exact_size: bool = False,
            dpi: int = 200) -> pathlib.Path:
    """Style the axes and write the plate."""
    for a in np.atleast_1d(axes).ravel():
        a.grid(True, alpha=0.18, lw=0.6)
        a.tick_params(labelsize=fs - 1.5)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    if not exact_size:
        fig.tight_layout(w_pad=1.8)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{name}.png"
    fig.savefig(out, dpi=dpi, bbox_inches=None if exact_size else "tight")
    print(f"  wrote {out.name}")
    return out


def plate_coupling():
    """Frequency keeps climbing with k_s; coupling to that same mode does not."""
    plt = _style()
    fs = 9
    ks_grid = np.logspace(2, 6.3, 80)
    freq = np.array([series_elastic(k).modes_hz()[-1] for k in ks_grid])
    coup = np.array([mode_coupling(series_elastic(k)) for k in ks_grid])
    j = int(np.argmax(coup))

    fig, ax = plt.subplots(2, 1, figsize=(5.2, 5.4), sharex=True)

    a = ax[0]
    a.loglog(ks_grid, freq, color=INK, lw=1.8)
    a.set_ylabel("2nd mode frequency (Hz)", fontsize=fs)
    a.set_title("(a)  the mode keeps climbing in frequency", fontsize=fs, loc="left")
    a.text(0.03, 0.92, "monotone: stiffer always means faster",
           transform=a.transAxes, fontsize=fs - 2, color=MUTED, va="top")

    b = ax[1]
    b.loglog(ks_grid, coup, color=INK, lw=1.8)
    b.axvline(ks_grid[j], color=PEAK, lw=1.0, ls="--")
    b.plot([ks_grid[j]], [coup[j]], marker="o", ms=6, mfc=PEAK, mec=PEAK, ls="none")
    b.annotate(f"peak coupling\n$k_s$ = {ks_grid[j]:.0f} N·m/rad",
               xy=(ks_grid[j], coup[j]), xytext=(ks_grid[j] * 6.0, coup[j] * 0.55),
               fontsize=fs - 2.5, color=PEAK, ha="left", va="top",
               arrowprops=dict(arrowstyle="-", color=PEAK, lw=0.8))
    b.text(0.03, 0.08, "but how strongly that mode is\nexcited RISES, THEN FALLS",
           transform=b.transAxes, fontsize=fs - 2, color=MUTED, va="bottom")
    b.set_xlabel(r"Series stiffness $k_s$ (N$\cdot$m/rad)", fontsize=fs)
    b.set_ylabel(r"Mode coupling  $|C(j\omega I{-}A)^{-1}B_F|$", fontsize=fs)
    b.set_title("(b)  ...but excitability of that mode peaks", fontsize=fs, loc="left")

    return _finish(fig, ax, "plate_coupling", fs)


def plate_readout_comparison():
    """The interior peak in coupling is specific to the motor channel, and this shows it."""
    plt = _style()
    fs = 9
    ks_grid = np.logspace(2, 6.3, 60)
    motor = np.array([mode_coupling(series_elastic(k, readout="motor")) for k in ks_grid])
    state = np.array([mode_coupling(series_elastic(k, readout="state")) for k in ks_grid])

    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.6))

    a = ax[0]
    j = int(np.argmax(motor))
    a.loglog(ks_grid, motor, color=INK, lw=1.8)
    a.axvline(ks_grid[j], color=PEAK, lw=1.0, ls="--")
    a.plot([ks_grid[j]], [motor[j]], marker="o", ms=6, mfc=PEAK, mec=PEAK, ls="none")
    a.set_xlabel(r"Series stiffness $k_s$ (N$\cdot$m/rad)", fontsize=fs)
    a.set_ylabel("Coupling (motor channel only)", fontsize=fs)
    a.set_title("(a)  motor channel: an interior peak", fontsize=fs, loc="left")
    a.text(0.97, 0.06, "must cross the spring\nto see anything",
          transform=a.transAxes, fontsize=fs - 2.5, color=MUTED, ha="right")

    b = ax[1]
    b.loglog(ks_grid, state, color=INK, lw=1.8)
    b.set_xlabel(r"Series stiffness $k_s$ (N$\cdot$m/rad)", fontsize=fs)
    b.set_ylabel("Coupling (best of all 6 state channels)", fontsize=fs)
    b.set_title("(b)  full state: no interior peak", fontsize=fs, loc="left")
    b.text(0.97, 0.92, r"$y_b,\ \theta_j$ are driven directly,"
          "\nso the mode is always reachable", transform=b.transAxes,
          fontsize=fs - 2.5, color=MUTED, ha="right", va="top")

    return _finish(fig, ax, "plate_readout_comparison", fs)


def plate_band_crossing():
    """Bits at a single assumed bandwidth, swept over that bandwidth: the curves cross."""
    plt = _style()
    fs = 9
    bands = np.logspace(np.log10(15), np.log10(400), 30)
    designs = [(None, "proprioceptive", INK)] + [(k, f"$k_s$={k:.3g}", COLORS[k])
              for k in (70.0, 1000.0, 2728.0, 1e4, 1e6)]

    fig, ax = plt.subplots(1, 1, figsize=(5.6, 4.0))
    for ks, label, color in designs:
        s = proprioceptive(readout="motor") if ks is None else series_elastic(
            ks, readout="motor")
        b = [profile_bits(s, band_hz=float(bh))["bits"] for bh in bands]
        ls = "--" if ks is None else "-"
        ax.loglog(bands, b, color=color, lw=1.8, ls=ls, label=label)
    for bh in BAND_ENSEMBLE_HZ:
        ax.axvline(bh, color=MUTED, lw=0.6, ls=":", alpha=0.6)
    ax.set_xlabel("Assumed disturbance bandwidth (Hz)", fontsize=fs)
    ax.set_ylabel("Information about the GRF (bits)", fontsize=fs)
    ax.set_title("no design's curve stays on top", fontsize=fs, loc="left")
    ax.legend(fontsize=fs - 2.5, frameon=False, loc="upper left", handlelength=1.6)
    # log-log with matplotlib's default minor ticks turns this axis into a smear at this
    # width; fixed ticks at the ensemble points read cleanly instead
    ax.minorticks_off()
    ax.set_xticks(list(BAND_ENSEMBLE_HZ))
    ax.set_xticklabels([f"{int(b)}" for b in BAND_ENSEMBLE_HZ])
    ax.text(0.97, 0.04, "dotted verticals: the 5-point ensemble",
           transform=ax.transAxes, fontsize=fs - 3, color=MUTED, ha="right")
    return _finish(fig, ax, "plate_band_crossing", fs)


def plate_ensemble_mean_vs_worst():
    """The reversal: mean bits over the ensemble peaks mid-sweep, worst-case falls monotonically."""
    plt = _style()
    fs = 9
    ks_axis = list(KS_SEQUENCE)
    designs = [series_elastic(k, readout="motor") for k in ks_axis]
    means = [ensemble_summary(s)["mean"] for s in designs]
    worsts = [ensemble_summary(s)["worst"] for s in designs]
    prop_mean = ensemble_summary(proprioceptive(readout="motor"))["mean"]
    prop_worst = ensemble_summary(proprioceptive(readout="motor"))["worst"]

    fig, ax = plt.subplots(2, 1, figsize=(5.2, 5.6), sharex=True)

    a = ax[0]
    a.semilogx(ks_axis, means, marker="o", ms=5, lw=1.8, color=INK)
    a.axhline(prop_mean, color=MUTED, lw=1.0, ls=":")
    a.text(0.03, 0.93, "proprioceptive", transform=a.transAxes, fontsize=fs - 2.5,
          color=MUTED, va="top")
    a.set_ylabel("Mean information over ensemble (bits)", fontsize=fs)
    a.set_title("(a)  the mean peaks at an intermediate $k_s$", fontsize=fs, loc="left")

    b = ax[1]
    b.semilogx(ks_axis, worsts, marker="s", ms=5, lw=1.8, color=INK)
    b.axhline(prop_worst, color=MUTED, lw=1.0, ls=":")
    b.text(0.97, 0.90, "proprioceptive", transform=b.transAxes, fontsize=fs - 2.5,
          color=MUTED, va="top", ha="right")
    b.set_xlabel(r"Series stiffness $k_s$ (N$\cdot$m/rad)", fontsize=fs)
    b.set_ylabel("Worst-case information over ensemble (bits)", fontsize=fs)
    b.set_title("(b)  the worst case falls monotonically instead", fontsize=fs, loc="left")
    b.set_ylim(9.7, 11.3)

    return _finish(fig, ax, "plate_ensemble_reversal", fs)


def plate_bandwidth_landscape():
    """The landscape: what a series spring is worth, against stiffness AND contact bandwidth."""
    plt = _style()
    from passive_info.spectrum import marginal_modal_bits

    ks_grid = np.logspace(np.log10(70.0), 6.0, 22)
    bands = np.logspace(np.log10(15.0), np.log10(300.0), 18)
    rigid = {b: profile_bits(proprioceptive(readout="energy"), band_hz=b)["bits"]
             for b in bands}
    Z = np.array([[profile_bits(series_elastic(k, readout="energy"), band_hz=b)["bits"]
                   - rigid[b] for k in ks_grid] for b in bands])

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.2, 3.9))
    lim = float(np.abs(Z).max())
    im = ax.pcolormesh(ks_grid, bands, Z, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="gouraud")
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Information, series elastic $-$ rigid (bits)", fontsize=8.5)
    cb.ax.tick_params(labelsize=7)

    # Same overlay, drawn on a BANDWIDTH axis rather than a frequency one -- a point lies
    # above a curve exactly when the disturbance band has reached that mode.
    for col in _mode_curves(ks_grid).T:
        ax.plot(ks_grid, col, color="white", lw=1.9)
        ax.plot(ks_grid, col, color=INK, lw=0.9, ls="--")
    ax.text(2.2e2, 20.0, "mode 1", color=INK, fontsize=7.5, rotation=8)
    ax.text(1.1e3, 150.0, "mode 2", color=INK, fontsize=7.5, rotation=32)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(bands[0], bands[-1])
    ax.set_yticks([15, 30, 50, 100, 200, 300])
    ax.set_yticklabels(["15", "30", "50", "100", "200", "300"])
    ax.minorticks_off()
    ax.set_xlabel("Series stiffness $k_s$ (N m/rad)", fontsize=9)
    ax.set_ylabel("Contact bandwidth (Hz)", fontsize=9)
    ax.set_title("(a) a mode is worth nothing until the contact reaches it",
                 fontsize=9.5, loc="left")

    probe = np.logspace(np.log10(15.0), np.log10(300.0), 14)
    for ks, colour, dash in ((1e4, COLORS[1e4], "-"), (70.0, COLORS[70.0], "--")):
        sys = series_elastic(ks, readout="energy")
        curves = np.array([[m for _, m in marginal_modal_bits(sys, band_hz=b)]
                           for b in probe])
        freqs = [f for f, _ in marginal_modal_bits(sys, band_hz=100.0)]
        # Mode indices are counted from the first non-zero mode, not from a fixed 1: the
        # clamped bench has no zero mode, so a hardcoded (1, 2) ran off the end of a
        # two-mode system. The highest mode is drawn solid, the rest faint.
        idx = list(range(sys.n_zero_modes(), len(freqs)))
        for j in idx:
            bx.plot(probe, curves[:, j], dash, lw=1.5, color=colour,
                    alpha=1.0 if j == idx[-1] else 0.45)
            bx.axvline(freqs[j], color=colour, lw=0.8, ls=":", alpha=0.7)
        bx.plot([], [], dash, lw=1.5, color=colour,
                label=f"$k_s = 10^4$" if ks == 1e4 else "$k_s = 70$")
    bx.annotate("mode 2 switches on\nas the band reaches it", xy=(200, 11.5),
                xytext=(34, 42), fontsize=7.5, color=INK,
                arrowprops=dict(arrowstyle="->", lw=0.8, color=INK,
                                connectionstyle="arc3,rad=-0.18"))
    bx.text(80, 1.5, "mode 1 (faint)\nflat throughout", fontsize=7, color=MUTED)
    bx.set_xscale("log")
    bx.set_xticks([15, 30, 50, 100, 200, 300])
    bx.set_xticklabels(["15", "30", "50", "100", "200", "300"])
    bx.minorticks_off()
    bx.set_xlabel("Contact bandwidth (Hz)      $\\cdots$ = that design's mode frequencies",
                  fontsize=8.5)
    bx.set_ylabel("Marginal information from each mode (bits)", fontsize=9)
    bx.set_title("(b) each mode pays only once the band arrives", fontsize=9.5, loc="left")
    bx.legend(fontsize=7.5, frameon=False, loc="upper left")
    return _finish(fig, (ax, bx), "plate_bandwidth_landscape")


def plate_information_spectrum(band_hz: float = 200.0, sigma_mult: float = 1.0,
                               fixed_window: bool = False):
    """Where the bits are, across stiffness, at the finest frequency resolution available."""
    plt = _style()
    from passive_info.information import SIGMA_ENERGY
    from passive_info.spectrum import chain_bits

    ks_grid = np.logspace(np.log10(70.0), 6.0, 30)
    sigma = SIGMA_ENERGY * sigma_mult
    freqs, Z = _chain_grid(ks_grid, band_hz, sigma)          # (freq, k_s)
    totals = Z.sum(axis=0)

    # Bits below each design's OWN first mode: what the leg knows about contact slower than
    # its own resonance. See the docstring -- this deliberately lets the window move with the
    # design, because a wider sub-resonant window is itself part of what a stiff transmission
    # buys. `plate_information_spectrum(fixed_window=True)` holds it at the rigid limit's f_1
    # instead, which isolates the mechanism from the width.
    f1 = np.array([series_elastic(k, readout="energy").bandwidth() for k in ks_grid])
    cut = proprioceptive(readout="energy").bandwidth()
    sub = np.array([Z[freqs < (cut if fixed_window else f1[i]), i].sum()
                    for i in range(len(ks_grid))])

    fig, (tx, sx, ax) = plt.subplots(
        3, 1, figsize=(7.4, 6.2), sharex=True,
        gridspec_kw=dict(height_ratios=[1, 1, 3.4], hspace=0.10))

    tx.plot(ks_grid, totals, "-", lw=1.6, color=INK)
    tx.set_ylabel("Total (bits)", fontsize=8.5)
    tx.set_title(f"information spectrum, kernel fixed band-limited at {band_hz:.0f} Hz",
                 fontsize=9.5, loc="left")
    tx.tick_params(labelbottom=False)

    sx.axhline(sub[-1], color=MUTED, lw=0.8, ls="--")
    sx.plot(ks_grid, sub, "-", lw=1.6, color=PEAK)
    if fixed_window:
        j = int(np.argmax(sub))
        sx.plot([ks_grid[j]], [sub[j]], "o", ms=4.5, color=PEAK)
        sx.annotate(f"+{sub[j] - sub[-1]:.2f} bits over rigid", xy=(ks_grid[j], sub[j]),
                    xytext=(11, 4), textcoords="offset points", fontsize=7.5, color=PEAK)
    else:
        sx.annotate("$I(a_{f<f_1};z)$", xy=(3.0e2, sub[6]), xytext=(24, -16),
                    textcoords="offset points", fontsize=8, color=PEAK,
                    arrowprops=dict(arrowstyle="->", lw=0.8, color=PEAK))
    sx.set_ylabel("Information below\n{}".format(f"{cut:.0f} Hz" if fixed_window
                                                  else "first mode"), fontsize=8.5)
    sx.tick_params(labelbottom=False)

    im = _mesh(ax, ks_grid, freqs, Z)
    cb = fig.colorbar(im, ax=(tx, sx, ax), pad=0.02, fraction=0.055)
    _label_bar(cb, "Information about the GRF, per band (bits)", 8.5, tick=7)
    _finish_heatmap(ax, ks_grid, freqs, fs=9, lw=2.4, thin=1.1)
    ax.plot([2.0e4, 6.0e4], [18, 18], color="white", lw=2.4)
    ax.plot([2.0e4, 6.0e4], [18, 18], color="#101010", lw=1.1, ls=(0, (4, 3)))
    ax.text(7.5e4, 14, "mechanical modes", color="white", fontsize=8)
    return _finish(fig, (tx, sx, ax), "plate_information_spectrum")


def plate_force_control(ks_show: float = 70.0):
    """Commanding a force through each transmission, and what arrives."""
    plt = _style()
    from passive_info.control import (gains_for, rigid, sea, sigma_tau_for, simulate,
                                      tracking_error)
    from passive_info.systems import I_ROT

    def cmd(t, rise=0.005, t0=0.005, amp=100.0):
        if t < t0:
            return 0.0
        u = min((t - t0) / rise, 1.0)
        return amp * (0.5 - 0.5 * np.cos(np.pi * u))

    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(11.6, 3.4))

    rg = sea(ks_show)
    op = simulate(rg, cmd)
    cl = simulate(rg, cmd, **gains_for(rg))
    rp = simulate(rigid(), cmd)
    ms = 1e3 * op["t"]
    ax.plot(ms, op["f_desired"], color=INK, lw=1.6, ls=(0, (5, 2)), label="commanded")
    ax.plot(ms, op["f_delivered"], color=COLORS[70.0], lw=1.4, label="SEA, open loop")
    ax.plot(ms, cl["f_delivered"], color=PEAK, lw=1.4, label="SEA, closed loop")
    ax.plot(ms, rp["f_delivered"], color=MUTED, lw=1.1, label="proprioceptive")
    ax.set_xlabel("Time (ms)", fontsize=9)
    ax.set_ylabel("Foot force (N)", fontsize=9)
    ax.set_title(f"(a) tracking, $k_s = {ks_show:.0f}$", fontsize=9.5, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="lower right")

    for ks_b, colour in ((1e3, COLORS[1000.0]), (3e4, COLORS[1e4])):
        rb = sea(ks_b)
        run = simulate(rb, cmd, **gains_for(rb))
        bx.plot(1e3 * run["t"], run["tau_meas"], color=colour, lw=0.7,
                label=f"$k_s={ks_b:.0e}$".replace("e+0", r"\\times 10^{") + "}$, "
                      + f"$\\sigma_\\tau={sigma_tau_for(rb):.2g}$ N m")
    clean = simulate(sea(1e3), cmd, sigma_tau=0.0, **gains_for(sea(1e3)))
    bx.set_ylim(-2, 16)
    bx.plot(1e3 * clean["t"], clean["tau_meas"], color=INK, lw=1.4, label="true torque")
    bx.set_xlabel("Time (ms)", fontsize=9)
    bx.set_ylabel("Spring torque (N m)", fontsize=9)
    bx.set_title("(b) the sensor is the spring", fontsize=9.5, loc="left")
    bx.legend(fontsize=7, frameon=False, loc="lower right")

    ks_grid = np.logspace(np.log10(50.0), 6.0, 26)
    op_e, cl_e = [], []
    for k in ks_grid:
        r = sea(k)
        op_e.append(100 * tracking_error(simulate(r, cmd))["rms_frac"])
        e = 100 * tracking_error(simulate(r, cmd, **gains_for(r)))["rms_frac"]
        cl_e.append(e if np.isfinite(e) and e < 1e3 else np.nan)
    rigid_e = 100 * tracking_error(rp)["rms_frac"]
    cx.loglog(ks_grid, op_e, color=COLORS[70.0], lw=1.6, label="SEA, open loop")
    cx.loglog(ks_grid, cl_e, color=PEAK, lw=1.6, label="SEA, closed loop")
    cx.axhline(rigid_e, color=MUTED, lw=1.0, ls="--")
    cx.text(6e4, rigid_e * 1.15, "proprioceptive", fontsize=7.5, color=MUTED)
    first_bad = ks_grid[np.argmax(np.isnan(cl_e))] if np.any(np.isnan(cl_e)) else None
    if first_bad is not None:
        cx.axvspan(first_bad, ks_grid[-1], color=MUTED, alpha=0.12, lw=0)
        cx.text(first_bad * 1.6, 0.35, "loop not closable at\n4 kHz, 400 Hz filter",
                fontsize=7, color=MUTED)
    cx.set_xlabel("Series stiffness $k_s$ (N m/rad)", fontsize=9)
    cx.set_ylabel("RMS force error (% of command)", fontsize=9)
    cx.set_title("(c) where each approach works", fontsize=9.5, loc="left")
    cx.legend(fontsize=7.5, frameon=False, loc="lower left")
    return _finish(fig, (ax, bx, cx), "plate_force_control")


def plate_controlled_stiffness(fs: float = 20000.0, band: float = 100.0,
                               f_cut: float = 2000.0):
    """With the loop closed: everything the robot has, against what the state alone adds."""
    plt = _style()
    from passive_info.control import (closed_loop_spectral_radius, conditional_information,
                                      rigid, sea)

    ks_grid = np.logspace(np.log10(70.0), 6.0, 40)
    ks_grid = np.array([k for k in ks_grid
                        if closed_loop_spectral_radius(sea(k), fs, f_cut) <= 1.0])
    joint, cmi = [], []
    for k in ks_grid:
        r = conditional_information(sea(k), fs=fs, band=band, f_cut=f_cut)
        joint.append(r["i_joint"])
        cmi.append(r["cmi"])
    ref = conditional_information(rigid(), fs=fs, band=band, f_cut=f_cut)

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(ks_grid, joint, "-", lw=1.8, color="#2171b5",
            label="$I(a\\,;z_x,z_u)$   everything available")
    ax.plot(ks_grid, cmi, "-", lw=1.8, color=PEAK,
            label="$I(a\\,;z_x\\mid z_u)$   unique to the state")
    ax.axhline(ref["i_joint"], ls=":", lw=1.5, color=INK)
    ax.text(ks_grid[0] * 1.15, ref["i_joint"] + 2.5,
            "proprioceptive (no sensor: both series coincide)", fontsize=7.5, color=INK)
    ax.fill_between(ks_grid, cmi, joint, color=PEAK, alpha=0.07, lw=0)
    j = len(ks_grid) // 2
    ax.annotate("the gap is exactly $I(a\\,;z_u)$:\nwhat the command alone carries",
                xy=(ks_grid[j], 0.5 * (joint[j] + cmi[j])),
                xytext=(1.05e2, 103.0), fontsize=7.5, color=MUTED, ha="left",
                arrowprops=dict(arrowstyle="->", lw=0.8, color=MUTED,
                                connectionstyle="arc3,rad=-0.18"))
    ax.set_xscale("log")
    ax.set_xlabel("Series stiffness $k_s$ (N m/rad)", fontsize=9)
    ax.set_ylabel("Information about the GRF (bits)", fontsize=9)
    ax.set_title(f"closed-loop force control, {band:.0f} Hz contact band, "
                 f"{fs / 1e3:.0f} kHz loop", fontsize=9.5, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="center left")
    return _finish(fig, ax, "plate_controlled_stiffness")


#: Cache for the sampled panel of `plate_method_comparison`; the KSG column is ~40 minutes.
# ---------------------------------------------------------------------------------------
# The grids, and the one heatmap this package draws five times.
#
# Separating "compute the grid" from "draw the grid" is what makes the arrays testable:
# `_chain_grid` is callable from a test and returns exactly what the figure shows.
# ---------------------------------------------------------------------------------------

def _chain_grid(ks_grid, band_hz, sigma=None):
    """(freqs, Z) with Z[i, j] the chain-rule bits of band i for stiffness j."""
    from passive_info.spectrum import chain_bits
    cols = [chain_bits(series_elastic(k, readout="energy"), band_hz=band_hz, sigma=sigma)
            for k in ks_grid]
    freqs = np.array([f for f, _ in cols[0]])
    return freqs, np.array([[b for _, b in c] for c in cols]).T


def _mode_curves(ks_grid):
    """(n_ks, n_modes) of each design's mechanical mode frequencies."""
    return np.array([series_elastic(k, readout="energy").nonzero_modes_hz() for k in ks_grid])


def _draw_modes(ax, ks_grid, lw=2.0, thin=0.9):
    """The design's own modes over a heatmap: a white halo under a dark dashed line."""
    for col in _mode_curves(ks_grid).T:
        ax.plot(ks_grid, col, color="white", lw=lw, alpha=0.95)
        ax.plot(ks_grid, col, color="#101010", lw=thin, ls=(0, (4, 3)))


def _mesh(ax, ks_grid, freqs, Z, **mesh_kw):
    """The stiffness x frequency image. Returns the mappable for a colour bar."""
    return ax.pcolormesh(ks_grid, freqs, Z, cmap="magma", shading="gouraud", **mesh_kw)


def _finish_heatmap(ax, ks_grid, freqs, fs, lw=2.0, thin=0.9):
    """Mode overlay and axes, after the colour bar has been placed."""
    if lw:
        _draw_modes(ax, ks_grid, lw, thin)
    ax.set_ylim(0, freqs[-1])
    ax.set_xscale("log")
    ax.set_xlabel("Series stiffness $k_s$ (N m/rad)", fontsize=fs)
    ax.set_ylabel("Contact frequency (Hz)", fontsize=fs)


def _label_bar(cb, text, fs, tick=None):
    """Colour bar label and tick size."""
    cb.set_label(text, fontsize=fs)
    cb.ax.tick_params(labelsize=fs - 1 if tick is None else tick)


def _cache_hit(path, key):
    """The cached arrays if the key matches exactly, else None. Written out three times."""
    if not path.exists():
        return None
    z = np.load(path)
    return z if (z["key"].shape == key.shape and np.allclose(z["key"], key)) else None


def _model_fingerprint() -> float:
    """A checksum of the model, to be mixed into every cache key."""
    from passive_info.spectrum import chain_bits
    probe = 0.0
    for ks in (7.0e1, 1.0e4):
        sysk = series_elastic(ks, readout="energy")
        for band in (100.0, 200.0):
            probe += sum(b for _, b in chain_bits(sysk, band_hz=band))
    return float(probe)


_METHOD_CACHE = RESULTS / "method_comparison.npz"


def _method_grids(ks_grid, band_hz, n_samples, n_blocks):
    """(chain, Fisher, ksg) spectra on a k_s x band grid. Only the last one is sampled."""
    from mechinfo.estimators import ksg_chain_bits
    from passive_info.information import FS, SIGMA_A, SIGMA_ENERGY, T_WINDOW
    from passive_info.spectrum import chain_bits, fisher_bits, response_matrix

    key = np.concatenate([ks_grid, [band_hz, n_samples, n_blocks,
                                    _model_fingerprint()]])
    z = _cache_hit(_METHOD_CACHE, key)
    if z is not None and np.all(np.isfinite(z["ksg"])):
        return z["freqs"], z["chain"], z["fisher"], z["ksg"]

    # Saved after every stiffness, not once at the end. The sampled column is roughly
    # 45 minutes of k-NN, and a run interrupted at 11 of 12 should not start over: a partial
    # cache is reloaded and only the missing columns are recomputed.
    n_b = len(chain_bits(series_elastic(ks_grid[0], readout="energy"), band_hz=band_hz))
    chain = np.full((len(ks_grid), n_b), np.nan)
    fisher, ksg, freqs = np.full_like(chain, np.nan), np.full_like(chain, np.nan), None
    z = _cache_hit(_METHOD_CACHE, key)
    if z is not None:
        chain, fisher = z["chain"].T.copy(), z["fisher"].T.copy()
        ksg, freqs = z["ksg"].T.copy(), z["freqs"]

    RESULTS.mkdir(exist_ok=True)
    for i, k in enumerate(ks_grid):
        if np.all(np.isfinite(ksg[i])):
            continue
        sysk = series_elastic(k, readout="energy")
        c = chain_bits(sysk, band_hz=band_hz)
        freqs = np.array([f for f, _ in c])
        chain[i] = [b for _, b in c]
        fisher[i] = [b for _, b in fisher_bits(sysk, band_hz=band_hz)]
        M = response_matrix(sysk, T_WINDOW, FS, band_hz)
        ksg[i] = ksg_chain_bits(M, SIGMA_ENERGY, SIGMA_A, n_samples=n_samples,
                                n_blocks=n_blocks, seed=i)
        np.savez(_METHOD_CACHE, key=key, freqs=freqs,
                 chain=chain.T, fisher=fisher.T, ksg=ksg.T)
        print(f"    k_s = {k:9.3g} done ({i + 1}/{len(ks_grid)})", flush=True)
    return freqs, chain.T, fisher.T, ksg.T


def plate_method_comparison(band_hz: float = 200.0, n_samples: int = 10_000,
                            n_blocks: int = 1):
    """The headline spectrum computed three ways. A check on the method, not a new result."""
    plt = _style()
    ks_grid = np.logspace(np.log10(70.0), 6.0, 12)
    freqs, chain, fisher, ksg = _method_grids(ks_grid, band_hz, n_samples, n_blocks)

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.0))
    (ax, bx), (cx, dx) = axes
    lim = float(max(chain.max(), fisher.max()))
    for a, Z, title, vmax in ((ax, chain, "(a) exact  $\\frac{1}{2}\\log\\det(I+G)$", lim),
                              (bx, fisher, "(b) Fisher  $\\frac{1}{2}\\log\\det G$", lim),
                              (cx, ksg, f"(c) KSG, N = {n_samples // 1000}k  "
                                        "(own scale)", None)):
        im = _mesh(a, ks_grid, freqs, Z, vmin=0.0 if vmax else None, vmax=vmax)
        _label_bar(fig.colorbar(im, ax=a, pad=0.02), "Information in band (bits)", 8)
        _finish_heatmap(a, ks_grid, freqs, fs=8.5, lw=2.2, thin=1.0)
        a.set_title(title, fontsize=9.5, loc="left")

    dx.plot([0, lim], [0, lim], "-", lw=0.9, color=MUTED)
    dx.plot(chain.ravel(), fisher.ravel(), ".", ms=2.5, color="#2171b5",
            label="Fisher")
    dx.plot(chain.ravel(), ksg.ravel(), ".", ms=2.5, color=PEAK, label="KSG")
    dx.set_xlabel("Exact information in band (bits)", fontsize=8.5)
    dx.set_ylabel("Information reported by the method (bits)", fontsize=8.5)
    dx.set_title("(d) agreement, one point per grid cell", fontsize=9.5, loc="left")
    dx.legend(fontsize=8, frameon=False, loc="upper left")
    return _finish(fig, axes, "plate_method_comparison")


def plate_spectrum_column(band_hz: float = 200.0, width_in: float = 3.5,
                          height_in: float = 4.3, fs: float = 10.0):
    """Total information and where it sits in frequency, on one stiffness axis."""
    plt = _style()
    from passive_info.spectrum import chain_bits

    ks_grid = np.logspace(np.log10(70.0), 6.0, 30)
    freqs, Z = _chain_grid(ks_grid, band_hz)

    fig, (tx, ax) = plt.subplots(
        2, 1, figsize=(width_in, height_in), sharex=True, layout="constrained",
        gridspec_kw=dict(height_ratios=[1, 2.4]))
    fig.get_layout_engine().set(hspace=0.02, wspace=0.02, h_pad=0.02, w_pad=0.02)

    tx.plot(ks_grid, Z.sum(axis=0), "-", lw=1.5, color=INK)
    tx.set_ylabel("Total (bits)", fontsize=fs)
    tx.tick_params(labelbottom=False)

    im = _mesh(ax, ks_grid, freqs, Z)
    cb = fig.colorbar(im, ax=(tx, ax), pad=0.03, fraction=0.06)
    _label_bar(cb, "Information about the GRF, per band (bits)", fs - 1)
    _finish_heatmap(ax, ks_grid, freqs, fs=fs)
    return _finish(fig, (tx, ax), "plate_spectrum_column", fs=fs + 1.0,
                   exact_size=True, dpi=400)


def plate_coordinate_information(band_hz: float = 200.0, width_in: float = 3.5,
                                 height_in: float = 4.3, fs: float = 10.0):
    """Where the contact is written into the mechanism, accumulated coordinate by coordinate."""
    plt = _style()
    from passive_info.spectrum import coordinate_bits

    ks_grid = np.logspace(np.log10(70.0), 6.0, 13)
    names = {0: r"$\theta_j$", 1: r"$\theta_m$",
             2: r"$\dot\theta_j$", 3: r"$\dot\theta_m$"}
    colour = {0: "#2a78d6", 2: "#8fbce8", 1: "#eb6834", 3: "#f2b199"}
    orders = ([0, 2, 1, 3], [1, 3, 0, 2])          # link first, rotor first

    def stack(order):
        out = np.zeros((len(order), len(ks_grid)))
        for i, k in enumerate(ks_grid):
            sysk = series_elastic(float(k), readout="energy")
            cum = [coordinate_bits(sysk, order[:j + 1], band_hz=band_hz)
                   for j in range(len(order))]
            out[:, i] = np.diff([0.0] + cum)
        return out

    ref = coordinate_bits(proprioceptive(readout="energy"), [0, 1], band_hz=band_hz)

    fig, axes = plt.subplots(2, 1, figsize=(width_in, height_in), sharex=True,
                             layout="constrained")
    fig.get_layout_engine().set(hspace=0.03, h_pad=0.02, w_pad=0.02)

    for ax, order, lead in zip(axes, orders, ("link first", "rotor first")):
        inc = stack(order)
        ax.stackplot(ks_grid, inc, colors=[colour[o] for o in order],
                     labels=[names[o] for o in order], edgecolor="white", lw=0.4)
        ax.axhline(ref, color=INK, lw=1.0, ls=(0, (4, 2)))
        ax.set_xscale("log")
        ax.set_ylabel("Cumulative information (bits)", fontsize=fs)
        ax.text(0.97, 0.94, lead, transform=ax.transAxes, fontsize=fs - 2, color=INK,
                va="top", ha="right")
        ax.set_ylim(0, None)

    # A framed legend, because the swatches sit on top of the fills they name and an
    # unframed blue patch on a blue field reads as an empty box.
    axes[0].legend(fontsize=fs - 2.5, ncol=4, loc="lower center", frameon=True,
                   facecolor="white", edgecolor="none", framealpha=0.9,
                   handlelength=1.0, columnspacing=0.7, handletextpad=0.35,
                   borderpad=0.3)
    axes[1].set_xlabel("Series stiffness $k_s$ (N m/rad)", fontsize=fs)
    return _finish(fig, axes, "coordinate_information", fs=fs + 1.0,
                   exact_size=True, dpi=400)


def plate_shapley_coordinates(band_hz: float = 200.0, ks_show: float = 1.0e4,
                              width_in: float = 3.5, height_in: float = 4.3,
                              fs: float = 10.0):
    """Where the contact information lives, attributed without choosing an order."""
    plt = _style()
    from passive_info.spectrum import shapley_coordinate_bits, temporal_coordinate_shapley

    order = [0, 2, 1, 3]                       # link angle, link rate, rotor angle, rotor rate
    names = {0: r"$\theta_j$", 1: r"$\theta_m$",
             2: r"$\dot\theta_j$", 3: r"$\dot\theta_m$"}
    colour = {0: "#2a78d6", 2: "#8fbce8", 1: "#eb6834", 3: "#f2b199"}

    ks_grid = np.logspace(np.log10(70.0), 6.0, 13)
    phi = np.array([shapley_coordinate_bits(series_elastic(float(k), readout="energy"),
                                            band_hz=band_hz) for k in ks_grid]).T

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(width_in, height_in), layout="constrained")
    fig.get_layout_engine().set(hspace=0.06, h_pad=0.02, w_pad=0.02)

    ax.stackplot(ks_grid, phi[order], colors=[colour[o] for o in order],
                 labels=[names[o] for o in order], edgecolor="white", lw=0.4)
    ax.set_xscale("log")
    ax.set_xlabel("Series stiffness $k_s$ (N m/rad)", fontsize=fs)
    ax.set_ylabel("Shapley bits", fontsize=fs)
    ax.set_ylim(0, None)
    ax.legend(fontsize=fs - 2.5, ncol=4, loc="lower center", frameon=True,
              facecolor="white", edgecolor="none", framealpha=0.9, handlelength=1.0,
              columnspacing=0.7, handletextpad=0.35, borderpad=0.3)

    t, tphi = temporal_coordinate_shapley(series_elastic(ks_show, readout="energy"),
                                          band_hz=band_hz)
    share = 100.0 * tphi / tphi.sum(axis=1, keepdims=True)
    keep = t > 0                                # t = 0 carries no increment
    bx.stackplot(t[keep] * 1e3, share[keep].T[order],
                 colors=[colour[o] for o in order], edgecolor="none")
    bx.set_xscale("log")
    bx.set_xlim(0.25, t[-1] * 1e3)
    bx.set_ylim(0, 100)
    bx.set_xlabel("Time within contact (ms)", fontsize=fs)
    bx.set_ylabel("Share of new information (%)", fontsize=fs)
    bx.text(0.97, 0.06, f"$k_s = 10^{{{int(np.log10(ks_show))}}}$",
            transform=bx.transAxes, fontsize=fs - 2, color="white",
            va="bottom", ha="right")
    return _finish(fig, (ax, bx), "plate_shapley_coordinates", fs=fs + 1.0,
                   exact_size=True, dpi=400)


#: Cache for the controlled column plate; one closed-loop spectrum per stiffness.
_CTRL_COL_CACHE = RESULTS / "controlled_column.npz"


def _controlled_column(ks_grid, band_hz, fs, f_cut):
    """(freqs, cmi, joint, action, ref) on the stiffness grid. One spectrum per design."""
    from passive_info.control import (closed_loop_spectral_radius, conditional_spectrum,
                                      rigid, sea)
    key = np.concatenate([ks_grid, [band_hz, fs, f_cut, _model_fingerprint()]])
    z = _cache_hit(_CTRL_COL_CACHE, key)
    if z is not None:
        return z["freqs"], z["cmi"], z["joint"], z["action"], float(z["ref"])

    freqs, cmi, joint, action = None, [], [], []
    for k in ks_grid:
        rig = sea(float(k))
        if closed_loop_spectral_radius(rig, fs, f_cut) > 1.0:
            cmi.append(np.full_like(cmi[-1], np.nan))
            joint.append(np.full_like(joint[-1], np.nan))
            action.append(np.full_like(action[-1], np.nan))
            continue
        f, c, j, a = conditional_spectrum(rig, fs=fs, band=band_hz, f_cut=f_cut)
        freqs, _ = f, cmi.append(c)
        joint.append(j)
        action.append(a)
        print(f"    k_s = {k:9.3g} done", flush=True)
    ref = conditional_spectrum(rigid(), fs=fs, band=band_hz, f_cut=f_cut)[1].sum()
    cmi, joint, action = (np.asarray(v).T for v in (cmi, joint, action))
    RESULTS.mkdir(exist_ok=True)
    np.savez(_CTRL_COL_CACHE, key=key, freqs=freqs, cmi=cmi, joint=joint,
             action=action, ref=ref)
    return freqs, cmi, joint, action, float(ref)


def plate_controlled_column(band_hz: float = 200.0, fs: float = 20000.0,
                            f_cut: float = 2000.0, width_in: float = 3.5,
                            height_in: float = 4.3, gfs: float = 10.0):
    """The closed-loop result in the same shape as the passive one."""
    plt = _style()
    ks_grid = np.logspace(np.log10(70.0), 6.0, 24)
    freqs, cmi, joint, action, ref = _controlled_column(ks_grid, band_hz, fs, f_cut)

    fig, (tx, ax) = plt.subplots(
        2, 1, figsize=(width_in, height_in), sharex=True, layout="constrained",
        gridspec_kw=dict(height_ratios=[1, 2.4]))
    fig.get_layout_engine().set(hspace=0.02, h_pad=0.02, w_pad=0.02)

    # W the unknown world force, X the state, A the action. See the module note on why the
    # action is observed exactly.
    tx.plot(ks_grid, joint.sum(axis=0), "-", lw=1.5, color="#2171b5",
            label="$I(W;X,A)$")
    tx.plot(ks_grid, cmi.sum(axis=0), "-", lw=1.5, color=PEAK, label="$I(W;X\\mid A)$")
    tx.axhline(ref, ls=":", lw=1.3, color=INK)
    tx.set_ylabel("Total (bits)", fontsize=gfs)
    tx.tick_params(labelbottom=False)
    tx.legend(fontsize=gfs - 2.5, frameon=False, loc="center left", handlelength=1.2)

    # No mode overlay here: this plate's y-axis is the controlled conditional and the
    # mechanical modes are not what shapes it. lw=0 draws none.
    im = _mesh(ax, ks_grid, freqs, cmi)
    cb = fig.colorbar(im, ax=(tx, ax), pad=0.03, fraction=0.06)
    _label_bar(cb, "Information unique to the state (bits)", gfs - 1)
    _finish_heatmap(ax, ks_grid, freqs, fs=gfs, lw=0.0)
    return _finish(fig, (tx, ax), "plate_controlled_column", fs=gfs + 1.0,
                   exact_size=True, dpi=400)


def plate_morphological(fs_hz: float = 20000.0, n_ks: int = 60, f_cut: float | None = None,
                        width_in: float = 3.5, height_in: float = 2.6, fs: float = 10.0):
    """MC_W under closed-loop force control, and what conditioning on the command costs."""
    from passive_info.control import (F_CUT, closed_loop_spectral_radius,
                                      morphological_information, rigid, sea)
    f_cut = F_CUT if f_cut is None else f_cut
    plt = _style()
    grid = np.logspace(np.log10(70.0), 6.0, n_ks)
    ks_grid = np.array([k for k in grid
                        if closed_loop_spectral_radius(sea(k), fs_hz, f_cut) <= 1.0])
    rows = [morphological_information(sea(k), fs=fs_hz, f_cut=f_cut) for k in ks_grid]
    mc = np.array([r["mc_w"] for r in rows])
    free = np.array([r["unconditioned"] for r in rows])
    ref = morphological_information(rigid(), fs=fs_hz, f_cut=f_cut)["mc_w"]

    fig, ax = plt.subplots(figsize=(width_in, height_in), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.03, w_pad=0.03)
    ax.fill_between(ks_grid, mc, free, color=MUTED, alpha=0.16, lw=0.0)
    ax.plot(ks_grid, free, color=MUTED, lw=1.4, ls=(0, (5, 2)), label="no conditioning")
    ax.plot(ks_grid, mc, color=INK, lw=1.9, label="$I(X_{k+1};X_k \\mid A)$")
    ax.axhline(ref, color=INK, lw=1.0, ls=":")
    if len(ks_grid) < len(grid):          # only when the stability gate actually bit
        ax.axvline(ks_grid[-1], color=PEAK, lw=1.0, ls="-.")
        ax.text(ks_grid[-1] * 0.82, ref + 0.15 * (float(max(free.max(), mc.max())) - ref),
                "loop unstable $\\rightarrow$", fontsize=fs - 2.0, color=PEAK,
                ha="right", va="center")
    ax.text(0.035, 0.03, "dotted: rigid leg (open loop)", transform=ax.transAxes,
            fontsize=fs - 1.5, color=INK, va="bottom")
    ax.legend(fontsize=fs - 1.5, frameon=False, loc="upper center", ncol=2,
              handlelength=1.5, columnspacing=1.2, borderaxespad=0.15)
    ax.set_xscale("log")
    ax.set_xlim(ks_grid[0], grid[-1])
    # Explicit, because the last stable point sits at rho = 0.999 and its spike would
    # otherwise set the range and clip the curve that produced it.
    top = float(max(free.max(), mc.max()))
    ax.set_ylim(ref - 0.45, top + 0.20 * (top - ref))
    ax.set_xlabel("Series stiffness $k_s$ (N m/rad)", fontsize=fs)
    ax.set_ylabel(f"Bits per {fs_hz / 1e3:g} kHz step", fontsize=fs)
    return _finish(fig, (ax,), f"plate_morphological_{fs_hz / 1e3:g}kHz",
                   fs=fs + 1.0, exact_size=True, dpi=400)


# ---------------------------------------------------------------------------------------
# Entry point. Must stay at the END of the file: it used to sit at line 232, above eleven of
# the fourteen plates, so nothing regenerated their committed PNGs.
# ---------------------------------------------------------------------------------------

#: Every plate that is closed form and cheap -- seconds to a couple of minutes each.
CHEAP_PLATES = (
    plate_coupling, plate_readout_comparison, plate_band_crossing,
    plate_ensemble_mean_vs_worst, plate_bandwidth_landscape, plate_information_spectrum,
    plate_force_control, plate_controlled_stiffness, plate_spectrum_column,
    plate_coordinate_information, plate_shapley_coordinates, plate_controlled_column,
    plate_morphological,
)

#: `plate_method_comparison` is NOT in the cheap set and is opt-in for a reason. Its k-NN
#: column is about 45 minutes, and `_method_grids` keys its cache on `_model_fingerprint()`
#: -- so any change that moves a reported number invalidates it and a plain `python -m
#: passive_info.figures` would silently start a 45-minute recompute. It has a committed cache
#: (`results/method_comparison.npz`); draw it with `--slow` when the model has changed.
SLOW_PLATES = (plate_method_comparison,)


def main(slow: bool = False) -> None:
    """Draw the analytic plates. `slow=True` adds the k-NN comparison -- see SLOW_PLATES."""
    for plate in CHEAP_PLATES + (SLOW_PLATES if slow else ()):
        plate()


if __name__ == "__main__":
    import sys

    main(slow="--slow" in sys.argv[1:])
