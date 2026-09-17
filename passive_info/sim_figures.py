"""Plates drawn from the cached simulation runs in `results/`."""
from __future__ import annotations

import glob
import json
import pathlib

import numpy as np

from passive_info.figures import INK, MUTED, _finish, _style

RESULTS = pathlib.Path(__file__).parent / "results"

# Three axes of comparison, three colour families, and they must not share. These plates sit
# next to each other and each varies something different -- the route, the quantity, the
# design -- so one palette across all three made the same blue mean two things.

#: Quantities -- the controlled plate. Blue/red/black.
Q_JOINT, Q_ACTION, Q_COND = "#08306b", "#c8102e", INK
#: Routes -- the passive plate. Green and orange: deliberately neither blue (which belongs to
#: the stiffness ramp and to I(W;X,A)) nor red (I(W;A)) nor black (the conditional, and the
#: rigid leg).
R_EXACT, R_SAMPLED = "#1b7837", "#e08214"
SEA_KS = (70.0, 1000.0, 1e4, 1e6)


_GROUND_REF = None


def _reference_ground() -> float:
    """The contact damping the CURRENT model specifies, `2 zeta sqrt(k_i m_e)`."""
    global _GROUND_REF
    if _GROUND_REF is None:
        from passive_info.embodied import go2_mass_and_jacobian
        from mechinfo.mechanics import ground_damping
        from passive_info.systems import K_I, ZETA_CONTACT
        Hp, J, _, _ = go2_mass_and_jacobian(clamped=True)
        _GROUND_REF = ground_damping(K_I, 1.0 / float(J @ np.linalg.solve(Hp, J)),
                                     ZETA_CONTACT)
    return _GROUND_REF


def _current_runs(pattern: str, band_hz: float) -> dict:
    """Runs at `band_hz` whose ground matches the current model, keyed by stiffness."""
    ref, rows, stale = _reference_ground(), [], []
    for f in sorted(glob.glob(str(RESULTS / pattern))):
        r = json.load(open(f))
        if float(r.get("band_hz", 100.0)) != band_hz:
            continue
        b = r.get("ground_b_i")
        if b is None or abs(float(b) - ref) > 1e-3 * ref:
            stale.append((pathlib.Path(f).name, b))
            continue
        tail = pathlib.Path(f).stem.rsplit(".", 1)[-1]
        rows.append((int(tail) if tail.isdigit() else 0, r))
    if stale:
        print(f"  ignoring {len(stale)} run(s) not on the current ground "
              f"(b_i = {ref:.2f}): " + ", ".join(f"{n} [{b}]" for n, b in stale))
    rows.sort(key=lambda t: (t[1]["ks"] is not None, t[1]["ks"] or 0.0, t[0]))
    by_ks = {r["ks"]: r for _, r in rows}
    if None not in by_ks:
        raise SystemExit(f"no proprioceptive run at {band_hz:g} Hz on the current ground")
    return by_ks


def _passive_rows(band_hz: float) -> dict:
    """Every passive grid at `band_hz` on the current ground, keyed by stiffness."""
    return _current_runs("grid_*.json", band_hz)


def _analytic_total(ks: float | None, band_hz: float) -> float:
    from passive_info.embodied import go2_proprioceptive, go2_series_elastic
    from passive_info.spectrum import chain_bits

    sysk = go2_proprioceptive() if ks is None else go2_series_elastic(float(ks))
    return float(sum(b for _, b in chain_bits(sysk, band_hz=band_hz)))


def _draw_passive(ax, band_hz: float, fs: float, note: bool = True,
                  linearised: bool = False) -> None:
    """The passive measurement: what the transmission costs, two ways."""
    by_ks = _passive_rows(band_hz)

    ks = np.array([k for k in SEA_KS if k in by_ks])
    routes = {
        "analytic": (np.array([_analytic_total(k, band_hz) for k in ks]),
                     _analytic_total(None, band_hz), R_EXACT, "o", "-"),
        "sampled": (np.array([by_ks[k]["total"] for k in ks]),
                    by_ks[None]["total"], R_SAMPLED, "D", "-"),
    }
    if linearised:
        routes["linearised"] = (np.array([by_ks[k]["linearised"] for k in ks]),
                                by_ks[None]["linearised"], MUTED, "s", "--")
    for name, (tot, ref, colour, mark, dash) in routes.items():
        ax.plot(ks, tot, dash, marker=mark, ms=3.4, lw=1.3, color=colour, label=name)
        ax.axhline(ref, color=colour, lw=0.8, ls=":", alpha=0.8)

    ax.set_xscale("log")
    ax.set_ylabel("$I(W;X)$ (bits)", fontsize=fs)
    # The key sits in the empty band between the sampled route and the exact ones -- the same
    # ~40 bit gap the panel exists to show. Anchored in axes fractions rather than by `loc`,
    # because every corner of this panel has data in it.
    leg = ax.legend(fontsize=fs - 2.5, frameon=False, loc="upper left",
                    bbox_to_anchor=(0.01, 0.52), handlelength=1.4, borderpad=0.2,
                    labelspacing=0.25, ncol=3, columnspacing=1.1, handletextpad=0.4,
                    title="dotted: rigid leg" if note else None)
    if note:
        leg.get_title().set_fontsize(fs - 3.0)
        leg.get_title().set_color(INK)


def _controlled_rows(band_hz: float) -> dict:
    """Every controlled run at `band_hz` on the current ground, keyed by stiffness."""
    return _current_runs("ctrl_*.json", band_hz)


def _draw_controlled(ax, band_hz: float, fs: float, note: bool = True) -> None:
    """The controlled measurement: what the state adds beyond the command."""
    # The band filter is load-bearing, as in `_passive_rows`: `ctrl_ks70.json` sorts before
    # `ctrl_ks70_b200.json`, so without it the plate draws one band while its caption names
    # the other.
    by_ks = _controlled_rows(band_hz)

    ks = np.array([k for k in SEA_KS if k in by_ks])
    joint = np.array([by_ks[k]["joint_total"] for k in ks])
    action = np.array([by_ks[k]["action_total"] for k in ks])
    cmi = joint - action
    ref = by_ks[None]

    ax.plot(ks, joint, "-", marker="o", ms=3.4, lw=1.3, color=Q_JOINT,
            label="$I(W;X,A)$")
    ax.plot(ks, action, "--", marker="s", ms=3.4, lw=1.3, color=Q_ACTION,
            label="$I(W;A)$")
    ax.plot(ks, cmi, "-", marker="^", ms=3.4, lw=1.3, color=INK, zorder=2,
            label="$I(W;X\\,|\\,A)$")
    for value, colour in ((ref["joint_total"], Q_JOINT), (ref["action_total"], Q_ACTION),
                          (ref["joint_total"] - ref["action_total"], INK)):
        ax.axhline(value, color=colour, lw=0.8, ls=":", alpha=0.8)
    ax.axhline(0.0, color=INK, lw=0.6, alpha=0.3)

    ax.set_xscale("log")
    ax.set_ylabel("Information (bits)", fontsize=fs)
    # Headroom rather than a smaller key. Three curves that cross leave no empty corner --
    # every placement inside the data sat on the command curve or on a reference line -- so
    # the axis is extended above the data and the key laid out in one row in the band that
    # opens up. The cost is vertical resolution, which this panel can afford: the effect it
    # has to show spans 0 to 29 bits on a 64 bit axis.
    ax.set_ylim(float(cmi.min()) - 4.0, float(max(joint.max(), action.max())) * 1.45)
    leg = ax.legend(fontsize=fs - 2.5, frameon=False, loc="upper left",
                    bbox_to_anchor=(0.0, 1.02), handlelength=1.4, borderpad=0.2,
                    labelspacing=0.25, ncol=3, columnspacing=1.0, handletextpad=0.4,
                    title="dotted: rigid leg" if note else None)
    if note:
        leg.get_title().set_fontsize(fs - 3.0)
        leg.get_title().set_color(INK)


def plate_sim_controlled_spectrum(band_hz: float = 200.0, width_in: float = 3.5,
                                  height_in: float = 2.6, fs: float = 10.0):
    """Every design's conditional spectrum on ONE axis, which is where the result lives."""
    plt = _style()
    by_ks = _controlled_rows(band_hz)
    from passive_info.figures import COLORS

    fig, ax = plt.subplots(figsize=(width_in, height_in), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.03, w_pad=0.03)

    n_b = len(by_ks[None]["cmi"])
    freqs = np.arange(n_b) / (2.0 * 0.085)
    ax.axhline(0.0, color=INK, lw=0.7, alpha=0.5)

    # Rigid leg in INK, the stiffness sweep on the package's own light-to-dark ramp, so the
    # ordering of the curves is legible before the legend is read.
    for k in list(SEA_KS) + [None]:
        if k not in by_ks:
            continue
        row = by_ks[k]
        colour = INK if k is None else COLORS[k]
        # 70 is not a power of ten and must not be rounded into one: `log10(70)` rounds to
        # 2, which would label the softest design 10^2 and misstate it by 30%.
        label = ("rigid" if k is None else
                 f"$k_s\\!=\\!10^{{{int(round(np.log10(k)))}}}$" if k >= 1e3 else
                 f"$k_s\\!=\\!{k:.0f}$")
        ax.plot(freqs, row["cmi"], "-", lw=1.4, color=colour, label=label,
                zorder=3 if k is None else 2)

    ax.set_xlabel("Contact frequency (Hz)", fontsize=fs)
    ax.set_ylabel("$I(W;X\\,|\\,A)$ per band (bits)", fontsize=fs)
    ax.set_xlim(0, freqs[-1])
    # Headroom for the key, as on `plate_sim_controlled`: five curves that all start near
    # 2.3 bits and fan out leave no empty corner, so the axis is extended above the data
    # rather than the key being squeezed into the middle of it.
    lo = min(min(by_ks[k]["cmi"]) for k in by_ks)
    hi = max(max(by_ks[k]["cmi"]) for k in by_ks)
    ax.set_ylim(lo - 0.3, hi * 1.7)
    leg = ax.legend(fontsize=fs - 2.5, frameon=False, loc="upper left",
                    bbox_to_anchor=(0.0, 1.02), ncol=3,
                    handlelength=1.4, columnspacing=1.0, handletextpad=0.4,
                    labelspacing=0.25, borderpad=0.2)
    leg.set_zorder(5)
    return _finish(fig, (ax,), f"plate_sim_controlled_spectrum_b{band_hz:g}", fs=fs + 1.0,
                   exact_size=True, dpi=400)


def plate_sim_column(band_hz: float = 200.0, width_in: float = 3.5,
                     height_in: float = 4.6, fs: float = 10.0):
    """Both simulated experiments, one column, sharing the stiffness axis."""
    plt = _style()
    fig, (tx, bx) = plt.subplots(
        2, 1, figsize=(width_in, height_in), sharex=True, layout="constrained",
        gridspec_kw=dict(height_ratios=[1.0, 1.0]))
    fig.get_layout_engine().set(hspace=0.03, h_pad=0.03, w_pad=0.03)

    _draw_passive(tx, band_hz, fs)
    _draw_controlled(bx, band_hz, fs, note=False)
    tx.tick_params(labelbottom=False)
    bx.set_xlabel("Series stiffness $k_s$ (N m/rad)", fontsize=fs)
    return _finish(fig, (tx, bx), "plate_sim_column", fs=fs + 1.0,
                   exact_size=True, dpi=400)


def plate_sim_passive(band_hz: float = 200.0, width_in: float = 3.5,
                      height_in: float = 2.6, fs: float = 10.0,
                      linearised: bool = False):
    """The passive plate on its own. See `_draw_passive` for what it shows."""
    plt = _style()
    fig, ax = plt.subplots(figsize=(width_in, height_in), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.03, w_pad=0.03)
    _draw_passive(ax, band_hz, fs, linearised=linearised)
    ax.set_xlabel("Series stiffness $k_s$ (N m/rad)", fontsize=fs)
    return _finish(fig, (ax,), f"plate_sim_passive_b{band_hz:g}", fs=fs + 1.0,
                   exact_size=True, dpi=400)


def plate_sim_controlled(band_hz: float = 200.0, width_in: float = 3.5,
                         height_in: float = 2.6, fs: float = 10.0):
    """The controlled plate on its own. See `_draw_controlled` for what it shows."""
    plt = _style()
    fig, ax = plt.subplots(figsize=(width_in, height_in), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.03, w_pad=0.03)
    _draw_controlled(ax, band_hz, fs)
    ax.set_xlabel("Series stiffness $k_s$ (N m/rad)", fontsize=fs)
    return _finish(fig, (ax,), "plate_sim_controlled", fs=fs + 1.0,
                   exact_size=True, dpi=400)


def main() -> None:
    """The simulated plates: the stacked column, and each half on its own."""
    import matplotlib
    matplotlib.use("Agg")
    plate_sim_column()
    plate_sim_controlled_spectrum()
    plate_sim_passive(200.0)
    plate_sim_controlled()


if __name__ == "__main__":
    main()
