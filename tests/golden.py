"""Every analytic number this package reports, as one flat dict of scalars."""
from __future__ import annotations

import json
import pathlib

import numpy as np

OUT = pathlib.Path(__file__).parent / "golden_analytic.json"


def _designs():
    from passive_info.systems import proprioceptive, series_elastic
    for clamped in (True, False):
        tag = "clamped" if clamped else "free"
        yield f"prop/{tag}", (lambda c=clamped, **kw: proprioceptive(clamped=c, **kw))
        for ks in (70.0, 1e3, 1e4, 1e6):
            yield (f"sea{ks:.0e}/{tag}",
                   (lambda c=clamped, k=ks, **kw: series_elastic(k, clamped=c, **kw)))


def collect() -> dict:
    from passive_info.control import (closed_loop_spectral_radius, conditional_information,
                                      conditional_spectrum, gains_for, rigid, sea,
                                      simulate, static_gain, tracking_error)
    from passive_info.information import (bits, ensemble_summary, profile_bits,
                                          response_matrix, sensitivity, time_to_resolve)
    from passive_info.channel import morphological_bits
    from passive_info.spectrum import (chain_bits, coordinate_bits, fisher_bits,
                                       marginal_modal_bits, modal_bits,
                                       shapley_coordinate_bits, temporal_coordinate_shapley)
    from passive_info.systems import mode_coupling, omega_ref, reference_m_e

    g: dict[str, float] = {}

    # ---- the mechanisms themselves -------------------------------------------------
    for tag, build in _designs():
        sys = build(readout="energy")
        g[f"{tag}/bandwidth"] = float(sys.bandwidth())
        g[f"{tag}/n_zero_modes"] = float(sys.n_zero_modes())
        for i, f in enumerate(sys.modes_hz()):
            g[f"{tag}/mode{i}"] = float(f)
        # A checksum on the state matrices: catches an assembly change that happens to
        # leave the bit counts alone, which a pure information test would not.
        for name, M in (("A", sys.A), ("B_F", sys.B_F), ("H", sys.H), ("K", sys.K)):
            g[f"{tag}/{name}.sum"] = float(np.sum(M))
            g[f"{tag}/{name}.absmax"] = float(np.max(np.abs(M)))
    for clamped in (True, False):
        tag = "clamped" if clamped else "free"
        g[f"ref/{tag}/m_e"] = float(reference_m_e(clamped=clamped))
        g[f"ref/{tag}/omega"] = float(omega_ref(clamped=clamped))

    # ---- readouts, the step case, and the profile case -----------------------------
    for tag, build in _designs():
        for readout in ("motor", "q", "state", "energy", "kinetic"):
            sys = build(readout=readout)
            g[f"{tag}/{readout}/step_bits"] = float(bits(sys))
            g[f"{tag}/{readout}/t_resolve_ms"] = float(time_to_resolve(sys))
            for band in (100.0, 200.0):
                r = profile_bits(sys, band_hz=band)
                g[f"{tag}/{readout}/b{band:.0f}/bits"] = float(r["bits"])
                g[f"{tag}/{readout}/b{band:.0f}/rank"] = float(r["rank"])
        sys = build(readout="energy")
        g[f"{tag}/sensitivity.absmax"] = float(np.max(np.abs(sensitivity(sys))))
        g[f"{tag}/M.sum"] = float(np.sum(response_matrix(sys)))
        e = ensemble_summary(sys)
        g[f"{tag}/ens/mean"] = float(e["mean"])
        g[f"{tag}/ens/worst"] = float(e["worst"])
        if sys.ks is not None:
            g[f"{tag}/coupling"] = float(mode_coupling(sys))

    # ---- the decompositions --------------------------------------------------------
    for tag, build in _designs():
        sys = build(readout="energy")
        for band in (100.0, 200.0):
            for name, rows in (("chain", chain_bits(sys, band_hz=band)),
                               ("fisher", fisher_bits(sys, band_hz=band)),
                               ("modal", modal_bits(sys, band_hz=band)),
                               ("marginal", marginal_modal_bits(sys, band_hz=band))):
                v = np.array([b for _, b in rows], float)
                g[f"{tag}/b{band:.0f}/{name}.sum"] = float(v.sum())
                g[f"{tag}/b{band:.0f}/{name}.absmax"] = float(np.max(np.abs(v)))
                g[f"{tag}/b{band:.0f}/{name}.first"] = float(v[0])
                g[f"{tag}/b{band:.0f}/{name}.last"] = float(v[-1])
            # The coordinate decompositions need a nonsingular K. `_coordinate_pieces`
            # inverts `E = diag(K, H)` to get the noise covariance, and a null direction of
            # E is a direction of infinite noise -- a coordinate the readout cannot see. A
            # pseudo-inverse would hand it zero noise instead, which is the opposite, so the
            # hard inverse raising is the correct behaviour and not a defect to paper over.
            # It rules the free bench out (one zero mode), and the Go2 plants with it.
            if sys.n_zero_modes():
                continue
            phi = np.asarray(shapley_coordinate_bits(sys, band_hz=band), float)
            g[f"{tag}/b{band:.0f}/shapley.sum"] = float(phi.sum())
            for i, p in enumerate(phi):
                g[f"{tag}/b{band:.0f}/shapley{i}"] = float(p)
            n2 = 2 * sys.n
            g[f"{tag}/b{band:.0f}/coord.all"] = float(coordinate_bits(sys, range(n2),
                                                                     band_hz=band))
            for i in range(n2):
                g[f"{tag}/b{band:.0f}/coord{i}"] = float(coordinate_bits(sys, [i],
                                                                        band_hz=band))
        if not sys.n_zero_modes():
            t, tphi = temporal_coordinate_shapley(sys, band_hz=100.0)
            g[f"{tag}/temporal.sum"] = float(np.sum(tphi))
            g[f"{tag}/temporal.absmax"] = float(np.max(np.abs(tphi)))

    # ---- morphological computation ---------------------------------------------------
    # MC_W is quoted at a step and diverges as that step shrinks, so two are pinned: a
    # single value could not catch a change that is really a change in the discretisation.
    for tag, build in _designs():
        sys = build(readout="energy")
        if sys.n_zero_modes():
            continue          # singular E, no equilibrium covariance -- see morphological_bits
        for fs in (4000.0, 20000.0):
            g[f"{tag}/mc_w/fs{fs:.0f}"] = float(morphological_bits(sys, dt=1.0 / fs))

    # ---- the closed loop -----------------------------------------------------------
    cmd = lambda t: 40.0 * np.sin(2.0 * np.pi * 20.0 * t)
    for name, rig in [("rigid", rigid())] + [(f"sea{k:.0e}", sea(k))
                                             for k in (70.0, 1e3, 1e4, 1e6)]:
        g[f"ctrl/{name}/static_gain"] = float(static_gain(rig))
        g[f"ctrl/{name}/rho"] = float(closed_loop_spectral_radius(rig))
        gains = gains_for(rig)
        g[f"ctrl/{name}/k_d"] = float(gains["k_d"])
        for tag, kw in (("open", {}), ("closed", gains)):
            g[f"ctrl/{name}/{tag}/rms_frac"] = float(
                tracking_error(simulate(rig, cmd, **kw))["rms_frac"])
        for closed in (False, True):
            d = conditional_information(rig, closed=closed, fs=20000.0, band=100.0)
            for k, v in d.items():
                g[f"ctrl/{name}/{int(closed)}/{k}"] = float(v)
        f, c, j, a = conditional_spectrum(rig, closed=True, fs=20000.0, band=100.0)
        g[f"ctrl/{name}/spec/cmi.sum"] = float(np.sum(c))
        g[f"ctrl/{name}/spec/joint.sum"] = float(np.sum(j))
        g[f"ctrl/{name}/spec/action.sum"] = float(np.sum(a))
        g[f"ctrl/{name}/spec/cmi.first"] = float(c[0])

    # ---- morphological computation under control -------------------------------------
    # Only where the loop is stable. An unstable loop has no stationary law, and
    # `morphological_information` raises there rather than reporting what
    # `solve_discrete_lyapunov` would still return, so the gate keeps the key set fixed.
    from passive_info.control import morphological_information
    # Both control rates: at 4 kHz the loop is unstable past ~4.6e4 and the stiff designs
    # drop out, at 20 kHz the whole sweep survives. That difference is itself a result, so
    # the key set differing between the two rates is intended rather than an accident.
    for name, rig in [("rigid", rigid())] + [(f"sea{k:.0e}", sea(k))
                                             for k in (70.0, 1e3, 1e4, 1e6)]:
        for fs_c in (4000.0, 20000.0):
            if closed_loop_spectral_radius(rig, fs_c) > 1.0:
                continue
            d = morphological_information(rig, fs=fs_c)
            for key in ("mc_w", "unconditioned", "command_cost"):
                g[f"ctrl/{name}/mc/fs{fs_c:.0f}/{key}"] = float(d[key])

    # ---- the Go2 linearisation -----------------------------------------------------
    from passive_info.embodied import go2_proprioceptive, go2_series_elastic
    for name, build in [("go2prop", lambda **kw: go2_proprioceptive(**kw))] + [
            (f"go2sea{k:.0e}", (lambda k=k, **kw: go2_series_elastic(k, **kw)))
            for k in (70.0, 1e3, 1e4, 1e6)]:
        sysk = build()
        g[f"{name}/bandwidth"] = float(sysk.bandwidth())
        g[f"{name}/A.sum"] = float(np.sum(sysk.A))
        for band in (100.0, 200.0):
            g[f"{name}/b{band:.0f}/bits"] = float(profile_bits(sysk, band_hz=band)["bits"])
            g[f"{name}/b{band:.0f}/chain.sum"] = float(
                sum(b for _, b in chain_bits(sysk, band_hz=band)))
    return g


def main() -> None:
    g = collect()
    OUT.write_text(json.dumps(g, indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT.name}: {len(g)} scalars")


if __name__ == "__main__":
    main()
