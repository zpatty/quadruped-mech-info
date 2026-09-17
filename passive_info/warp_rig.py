"""Batched rollouts of the passive leg through MuJoCo Warp, many worlds at once."""
from __future__ import annotations

import numpy as np
import warp as wp

from passive_info.embodied import SIM_DT, _ground_gains, _sim_rig
from passive_info.systems import K_I, ZETA_SERIES


@wp.kernel
def _foot_point(geom_xpos: wp.array2d(dtype=wp.vec3), foot_geom: int, radius: float,
                point: wp.array(dtype=wp.vec3)):
    """The contact point: the foot geom's centre, dropped by its radius. Per world."""
    w = wp.tid()
    p = geom_xpos[w, foot_geom]
    point[w] = wp.vec3(p[0], p[1], p[2] - radius)


@wp.kernel
def _apply_force(point: wp.array(dtype=wp.vec3), jacp: wp.array3d(dtype=float),
                 qvel: wp.array2d(dtype=float), fmode: wp.array2d(dtype=float), step: int,
                 foot_body: int, k_i: float, b_i: float, z0: float,
                 xfrc: wp.array2d(dtype=wp.spatial_vector)):
    """`f = f_mode(t) - k_i (z - z0) - b_i (J_z . qvel)`, written into the foot body."""
    w = wp.tid()
    z = point[w][2]
    vz = float(0.0)
    for i in range(qvel.shape[1]):
        vz += jacp[w, 2, i] * qvel[w, i]
    f = fmode[w, step] - k_i * (z - z0) - b_i * vz
    xfrc[w, foot_body] = wp.spatial_vector(0.0, 0.0, f, 0.0, 0.0, 0.0)


@wp.kernel
def _record(qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float), step: int,
            out: wp.array3d(dtype=float)):
    """Append `[qpos, qvel]` for every world. On device, so the loop never syncs to host."""
    w = wp.tid()
    nq = qpos.shape[1]
    for i in range(nq):
        out[w, step, i] = qpos[w, i]
    for i in range(qvel.shape[1]):
        out[w, step, nq + i] = qvel[w, i]


#: No model cache. Measured, it saved nothing at the widths that matter, and a graded version
#: coincided with a 30x regression on one design and not the other. The real lesson was about
#: batch width: a refill costs about the same at 1024 worlds as at 16384, so ask for as many
#: as fit.


#: Captured CUDA graphs, keyed by everything that fixes the instruction stream.
#:
#: A refill issues ~8500 kernel launches from Python, which is why width is nearly free and
#: why a busy node still hurts. Capturing the loop and replaying it is one launch.
#:
#: The risk is state, not speed. Replaying needs the same `Data` every time, and hand-resetting
#: qpos/qvel/xfrc_applied is NOT enough -- MJWarp's Data carries integrator and solver state
#: beyond those, and a partial reset fails silently as a subtly wrong trajectory. The reset
#: here is MuJoCo's own `reset_data` plus re-assigning the stance pose.
#:
#: Second risk, unverifiable by reading: the constraint solver may branch on convergence, and
#: a captured graph freezes whichever path capture took. `use_graph` is opt-in and gated on
#: the trajectory-parity tests; if those fail it must be abandoned rather than tuned.
_GRAPH: dict = {}


def clear_graph_cache():
    """Drop captured graphs. Call if anything model-affecting changes between runs."""
    _GRAPH.clear()


def rollout(ks: float | None, amps: np.ndarray, dt: float = SIM_DT,
            zeta_series: float = ZETA_SERIES, t_window: float | None = None,
            contact: str = "menagerie", ground: str = "spring", k_i: float = K_I,
            gravity: bool = True, device: str | None = None,
            as_torch: bool = False, use_graph: bool = False, vertical="fixed"):
    """Batched passive rollouts. `amps` is (nworld, K) -- one mode-coefficient vector per world."""
    import mujoco_warp as mjw

    from passive_info.information import T_WINDOW, mode_basis

    amps = np.atleast_2d(np.asarray(amps, float))
    nworld = len(amps)
    t_window = T_WINDOW if t_window is None else t_window

    m, d, leg = _sim_rig(ks, dt, zeta_series, contact, ground, gravity, vertical=vertical)
    n_t = int(round(t_window / m.opt.timestep))
    f = mode_basis(n_t, amps.shape[1]) @ amps.T                 # (n_t, nworld)
    b_i, z0 = _ground_gains(m, d, leg, k_i) if ground == "spring" else (0.0, 0.0)

    device = device or ("cuda" if wp.get_cuda_device_count() else "cpu")
    key = (ks, dt, zeta_series, contact, ground, gravity, device, nworld, n_t,
           float(k_i), amps.shape[1], vertical)
    graph_ok = use_graph and str(device).startswith("cuda")

    with wp.ScopedDevice(device):
        if graph_ok and key in _GRAPH:
            wm, wd, fmode, out, graph, qpos0 = _GRAPH[key]
            # MuJoCo's OWN reset, not a hand-picked list of fields -- see the note on _GRAPH.
            mjw.reset_data(wm, wd)
            wd.qpos.assign(qpos0)                 # reset_data returns qpos to the model's qpos0
            fmode.assign(np.ascontiguousarray(f.T, dtype=np.float32))
            wp.capture_launch(graph)
            wp.synchronize_device(device)
        else:
            wm = mjw.put_model(m)
            wd = mjw.put_data(m, d, nworld=nworld)
            fmode = wp.array(np.ascontiguousarray(f.T, dtype=np.float32))   # (nworld, n_t)
            point = wp.zeros(nworld, dtype=wp.vec3)
            bodyid = wp.array(np.full(nworld, leg.foot_body, np.int32), dtype=int)
            jacp = wp.zeros((nworld, 3, m.nv), dtype=float)
            out = wp.zeros((nworld, n_t, m.nq + m.nv), dtype=float)
            qpos0 = np.ascontiguousarray(np.tile(d.qpos, (nworld, 1)), dtype=np.float32)

            def _loop():
                for i in range(n_t):
                    wp.launch(_foot_point, dim=nworld,
                              inputs=[wd.geom_xpos, leg.foot_geom, float(leg.foot_radius),
                                      point])
                    if ground == "spring":
                        mjw.jac(wm, wd, jacp, None, point, bodyid)
                        wp.launch(_apply_force, dim=nworld,
                                  inputs=[point, jacp, wd.qvel, fmode, i, leg.foot_body,
                                          float(k_i), float(b_i), float(z0),
                                          wd.xfrc_applied])
                    mjw.step(wm, wd)
                    wp.launch(_record, dim=nworld, inputs=[wd.qpos, wd.qvel, i, out])

            if graph_ok:
                with wp.ScopedCapture(device) as cap:
                    _loop()
                _GRAPH[key] = (wm, wd, fmode, out, cap.graph, qpos0)
                # Capture records without executing, so the first call must also replay.
                mjw.reset_data(wm, wd)
                wd.qpos.assign(qpos0)
                wp.capture_launch(cap.graph)
            else:
                _loop()
            wp.synchronize_device(device)

        if as_torch:
    # Zero-copy handoff: `out` already lives in device memory and `wp.to_torch` wraps it
    # rather than copying. The numpy round-trip made the pipeline CPU-bound, and on a shared
    # node that means hostage to whoever else is on it.
            X = wp.to_torch(out)   # `wp.to_torch` is top-level; there is no `warp.torch`
            return X, None
        X = out.numpy()

    return X, np.isfinite(X).all(axis=(1, 2))


def sim_profile_bits_gpu(ks: float | None, eps: float | None = None, dt: float = SIM_DT,
                         sigma: float | None = None, zeta_series: float = ZETA_SERIES,
                         contact: str = "menagerie", ground: str = "spring",
                         k_i: float = K_I, gravity: bool = True,
                         device: str | None = None, vertical="fixed") -> dict:
    """`embodied.sim_profile_bits`, with all K+1 finite-difference columns in one batch."""
    from passive_info.embodied import SIM_EPS, _sim_metric
    from passive_info.information import SIGMA_A, SIGMA_ENERGY, n_modes

    eps = SIM_EPS if eps is None else eps
    K = n_modes()
    amps = np.zeros((K + 1, K))
    amps[1:] = eps * np.eye(K)

    X, ok = rollout(ks, amps, dt, zeta_series, contact=contact, ground=ground, k_i=k_i,
                    gravity=gravity, device=device, vertical=vertical)
    if not ok.all():
        raise FloatingPointError(
            f"unusable world(s) {np.flatnonzero(~ok).tolist()} at k_s={ks}, dt={dt}: "
            f"refuse to report a bit count")

    M = np.stack([(X[j + 1] - X[0]) / eps for j in range(K)], axis=-1)
    E = _sim_metric(ks, dt, zeta_series, contact, ground, gravity, vertical=vertical,
                    k_i=k_i)
    gram = sum(M[t].T @ E @ M[t] for t in range(M.shape[0]))
    gram *= (SIGMA_A / (SIGMA_ENERGY if sigma is None else sigma)) ** 2
    eig = np.linalg.eigvalsh(gram)[::-1]
    per = 0.5 * np.log2(1.0 + np.clip(eig, 0.0, None))
    return dict(bits=float(0.5 * np.linalg.slogdet(np.eye(K) + gram)[1] / np.log(2.0)),
                eigbits=per, rank=int(np.sum(per > 1.0)), n_modes=K, dt=dt, stable=True)


def sample_rollouts(ks: float | None, n: int, sigma_a: float | None = None,
                    band_hz: float | None = None, seed: int = 0, dt: float = SIM_DT,
                    stride: int = 1, device: str | None = None, **kw):
    """`n` rollouts with force profiles drawn from the prior. The production path."""
    from passive_info.information import SIGMA_A, BAND_HZ, n_modes

    sigma_a = SIGMA_A if sigma_a is None else sigma_a
    band_hz = BAND_HZ if band_hz is None else band_hz
    K = n_modes(band_hz=band_hz)
    a = np.random.default_rng(seed).normal(0.0, sigma_a, size=(n, K))
    X, ok = rollout(ks, a, dt=dt, device=device, **kw)
    return a, X[:, ::stride], ok


def benchmark(ks: float | None = None, worlds=(64, 256, 1024, 4096), dt: float = SIM_DT,
              device: str | None = None):
    """Rollouts per second against batch size. The number that decides feasibility."""
    import time

    from passive_info.information import n_modes
    K = n_modes()
    print(f"{'worlds':>8} {'secs':>9} {'roll/s':>10} {'steps/s':>12} {'vs CPU':>9}")
    rows = []
    for w in worlds:
        a = np.random.default_rng(0).normal(0.0, 100.0, size=(w, K))
        t0 = time.time()
        X, ok = rollout(ks, a, dt=dt, device=device)
        el = time.time() - t0
        n_t = X.shape[1]
        rows.append(dict(worlds=w, secs=el, per_s=w / el))
        print(f"{w:>8} {el:>9.2f} {w / el:>10.1f} {w * n_t / el:>12.3g} "
              f"{(w / el) * 0.156:>8.0f}x", flush=True)
        if not ok.all():
            print(f"         WARNING: {(~ok).sum()} non-finite worlds")
    best = max(r["per_s"] for r in rows)
    for target in (1e7, 1e8):
        print(f"  {target:.0e} rollouts -> {target / best / 3600:.1f} GPU-hours")
    return rows


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--ks", type=float, default=None)
    p.add_argument("--worlds", type=int, nargs="+", default=[64, 256, 1024, 4096])
    args = p.parse_args()
    wp.init()
    dev = "cuda" if wp.get_cuda_device_count() else "cpu"
    print(f"device: {dev}\n")
    if args.benchmark:
        benchmark(args.ks, tuple(args.worlds), device=dev)
