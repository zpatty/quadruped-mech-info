"""An InfoNCE critic for the chain terms, where the k-NN estimator cannot reach."""
from __future__ import annotations

import numpy as np


def _torch():
    import torch
    return torch


class SeparableCritic:
    """Two MLPs scoring `(a, s)` pairs by dot product, so a batch costs 2B passes."""

    def __init__(self, d_a: int, d_s: int, width: int = 256, depth: int = 2,
                 embed: int = 32, device="cuda", seed: int = 0):
        torch = _torch()
        torch.manual_seed(seed)

        def mlp(d_in):
            layers, d = [], d_in
            for _ in range(depth):
                layers += [torch.nn.Linear(d, width), torch.nn.ReLU()]
                d = width
            layers += [torch.nn.Linear(d, embed)]
            return torch.nn.Sequential(*layers)

        self.g = mlp(d_a).to(device)
        self.h = mlp(d_s).to(device)
        self.device = device

    def parameters(self):
        return list(self.g.parameters()) + list(self.h.parameters())

    def scores(self, a, s):
        """`K x K` matrix of critic scores; entry (i, j) is `f(a_i, s_j)`."""
        return self.g(a) @ self.h(s).T


class SplitCritic(SeparableCritic):
    """A separable critic that encodes state and command separately, then sums."""

    def __init__(self, d_a: int, d_x: int, d_u: int, width: int = 256, depth: int = 2,
                 embed: int = 32, device="cuda", seed: int = 0):
        torch = _torch()
        super().__init__(d_a, d_x, width, depth, embed, device, seed)
        self.d_x = d_x
        torch.manual_seed(seed + 1)
        layers, d = [], d_u
        for _ in range(depth):
            layers += [torch.nn.Linear(d, width), torch.nn.ReLU()]
            d = width
        layers += [torch.nn.Linear(d, embed)]
        self.h_u = torch.nn.Sequential(*layers).to(device)

    def parameters(self):
        return super().parameters() + list(self.h_u.parameters())

    def scores(self, a, s):
        x, u = s[:, :self.d_x], s[:, self.d_x:]
        return self.g(a) @ (self.h(x) + self.h_u(u)).T


def infonce_bits(a: np.ndarray, s: np.ndarray, batch: int = 1024, steps: int = 2000,
                 lr: float = 1e-3, width: int = 256, depth: int = 2, embed: int = 32,
                 device: str | None = None, seed: int = 0, holdout: float = 0.2,
                 weight_decay: float = 0.0, val_frac: float = 0.2,
                 patience: int | None = None) -> dict:
    """Lower bound on I(a; s) in bits, from a separable critic on the InfoNCE objective."""
    torch = _torch()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    a = np.asarray(a, np.float64).reshape(len(a), -1)
    s = np.asarray(s, np.float64).reshape(len(s), -1)
    a = (a - a.mean(0)) / np.where(a.std(0) > 0, a.std(0), 1.0)
    s = (s - s.mean(0)) / np.where(s.std(0) > 0, s.std(0), 1.0)

    # Three-way split: `val` selects when to stop, `test` is what gets reported. One split for
    # both would report the maximum of a noisy curve over hundreds of evaluations, which is
    # selection bias dressed as an estimate.
    n_hold = int(len(a) * holdout)
    n_val = int(len(a) * val_frac) if patience else 0
    A = torch.tensor(a, dtype=torch.float32, device=device)
    S = torch.tensor(s, dtype=torch.float32, device=device)
    A_te, S_te = A[:n_hold], S[:n_hold]
    A_va, S_va = A[n_hold:n_hold + n_val], S[n_hold:n_hold + n_val]
    # holdout splits before any training step: the training objective would let the critic
    # memorise the batch and return log K regardless of the truth.
    A_tr, S_tr = A[n_hold + n_val:], S[n_hold + n_val:]

    critic = SeparableCritic(A.shape[1], S.shape[1], width, depth, embed, device, seed)
    # Weight decay is not cosmetic: unregularised, the held-out bound goes negative.
    opt = torch.optim.Adam(critic.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator(device=device).manual_seed(seed)
    ln2 = float(np.log(2.0))
    curve = []
    best_val, bad, best_state = -float("inf"), 0, None

    for step in range(steps):
        idx = torch.randint(len(A_tr), (batch,), device=device, generator=gen)
        scores = critic.scores(A_tr[idx], S_tr[idx])
        # InfoNCE: the diagonal is the positive pair, the row is the K-sample partition.
        loss = torch.nn.functional.cross_entropy(
            scores, torch.arange(batch, device=device))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, steps // 20) == 0 or step == steps - 1:
            curve.append(float((np.log(batch) - loss.item()) / ln2))
            if patience and n_val:
                with torch.no_grad():
                    vi = torch.randint(len(A_va), (batch,), device=device, generator=gen)
                    vs = critic.scores(A_va[vi], S_va[vi])
                    vl = torch.nn.functional.cross_entropy(
                        vs, torch.arange(batch, device=device))
                    v = float((np.log(batch) - vl.item()) / ln2)
                if v > best_val:
                    best_val, bad = v, 0
                    best_state = {k: p.detach().clone()
                                  for k, p in enumerate(critic.parameters())}
                else:
                    bad += 1
                    if bad >= patience:
                        break

    if best_state is not None:
        with torch.no_grad():
            for k, prm in enumerate(critic.parameters()):
                prm.copy_(best_state[k])

    with torch.no_grad():
        vals = []
        for _ in range(32):
            idx = torch.randint(len(A_te), (batch,), device=device, generator=gen)
            scores = critic.scores(A_te[idx], S_te[idx])
            loss = torch.nn.functional.cross_entropy(
                scores, torch.arange(batch, device=device))
            vals.append(float((np.log(batch) - loss.item()) / ln2))

    # Saturation near 1 means the estimate IS the ceiling and the truth is above it.
    bits = float(np.mean(vals))
    ceiling = float(np.log2(batch))
    return dict(bits=bits, sd=float(np.std(vals)), ceiling=ceiling,
                saturation=bits / ceiling, curve=curve, batch=batch, steps=steps,
                best_val=best_val if best_state is not None else None)


def infonce_chain_bits(M: np.ndarray, sigma: float, sigma_a: float, group: int = 1,
                       n_samples: int = 200_000, n_blocks: int = 1, seed: int = 0,
                       device: str | None = None, **kw) -> np.ndarray:
    """`estimators.ksg_chain_bits` with the critic in place of k-NN."""
    G = M.T @ M
    d = len(G)
    L = np.linalg.cholesky(G + 1e-12 * np.eye(d))
    rng = np.random.default_rng(seed)
    edges = list(range(0, d, group)) + [d]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        acc = []
        for _ in range(n_blocks):
            A = rng.normal(0.0, sigma_a, size=(n_samples, d))
            A[:, :lo] = rng.normal(0.0, sigma_a, size=lo)
            S = A @ G.T + sigma * rng.normal(size=(n_samples, d)) @ L.T
            acc.append(infonce_bits(A[:, lo:hi], S, device=device, seed=seed, **kw)["bits"])
        out.append(float(np.mean(acc)))
    return np.array(out)


def validate(device=None, batch: int = 1024, steps: int = 3000, n: int = 200_000,
             seed: int = 0):
    """The critic against exact linear-Gaussian answers, spanning the `log2(batch)` ceiling."""
    from mechinfo.estimators import gaussian_mutual_information

    rng = np.random.default_rng(seed)
    d_a, d_s = 1, 34
    print(f"batch {batch} -> ceiling {np.log2(batch):.2f} bits | "
          f"observation dimension {d_s} | N = {n}\n")
    print(f"{'target':>8} {'exact':>9} {'infonce':>9} {'sd':>7} {'err':>8} {'sat':>6}")
    rows = []
    for snr in (0.5, 2.0, 8.0, 64.0):
        Mt = np.zeros((d_s, d_a))
        Mt[0, 0] = snr
        exact = gaussian_mutual_information(Mt, 1.0, 1.0)
        a = rng.normal(size=(n, d_a))
        s = a @ Mt.T + rng.normal(size=(n, d_s))
        r = infonce_bits(a, s, batch=batch, steps=steps, device=device, seed=seed)
        print(f"{snr:>8.1f} {exact:>9.4f} {r['bits']:>9.4f} {r['sd']:>7.4f} "
              f"{r['bits'] - exact:>+8.4f} {r['saturation']:>6.3f}")
        rows.append(dict(snr=snr, exact=exact, **{k: r[k] for k in
                                                  ("bits", "sd", "saturation")}))
    return rows


# Reuse is refresh_every * batch / buffer; the useful band is 4-32. A fixed set is
# memorised outright. Normalisation is frozen from one calibration draw.
def infonce_bits_streaming(refill, batch: int = 1024, steps: int = 4000,
                           refresh_every: int = 256, buffer: int = 16_384,
                           lr: float = 1e-3, width: int = 256, depth: int = 2,
                           embed: int = 32, device: str | None = None, seed: int = 0,
                           n_eval: int = 16, verbose: bool = False,
                           trace_every: int | None = None,
                           split_at: int | None = None) -> dict:
    """InfoNCE where `refill(n) -> (a, s)` generates samples instead of holding them."""
    torch = _torch()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ln2 = float(np.log(2.0))

    def _dev(v):
        """Accept numpy or an existing device tensor; never round-trip through the host."""
        if torch.is_tensor(v):
            return v.reshape(len(v), -1).float()
        return torch.as_tensor(np.ascontiguousarray(np.asarray(v, np.float32)),
                               device=device).reshape(len(v), -1)

    a_c, s_c = refill(min(buffer, 8192))
    a_c, s_c = _dev(a_c), _dev(s_c)

    def _stats(v):
        """Mean and scale, with near-constant columns neutralised rather than amplified."""
        mu, sd = v.mean(0), v.std(0)
        dead = sd < 1e-6 * sd.max().clamp_min(1e-30)
        return mu, torch.where(dead, torch.ones_like(sd), sd), int(dead.sum())

    a_mu, a_sd, _ = _stats(a_c)
    s_mu, s_sd, n_dead = _stats(s_c)
    if n_dead and verbose:
        print(f"    {n_dead}/{s_c.shape[1]} observation columns are near-constant; "
              f"left unscaled rather than amplified", flush=True)

    # Normalise on the GPU in float32: the float64 round-trip was the dominant cost.
    def to_t(a, s):
        return (_dev(a) - a_mu) / a_sd, (_dev(s) - s_mu) / s_sd

    if split_at is None:
        critic = SeparableCritic(int(a_c.shape[1]), int(s_c.shape[1]), width, depth, embed,
                                 device, seed)
    else:
        critic = SplitCritic(int(a_c.shape[1]), int(split_at),
                             int(s_c.shape[1]) - int(split_at), width, depth, embed,
                             device, seed)
    opt = torch.optim.Adam(critic.parameters(), lr=lr)      # no weight decay: see docstring
    gen = torch.Generator(device=device).manual_seed(seed)
    tgt = torch.arange(batch, device=device)

    import time as _time
    A, S = to_t(*refill(buffer))
    generated = len(a_c) + buffer
    curve = []
    t_start, t_refill, n_refill = _time.time(), 0.0, 0
    trace = []
    for step in range(steps):
        if step and step % refresh_every == 0:
            _t = _time.time()
            A, S = to_t(*refill(buffer))
            t_refill += _time.time() - _t
            n_refill += 1
            generated += buffer
            if verbose:
                # Per refill, not per band: a run printing only on band completion gave
                # no output at all when it stalled inside band 0.
                print(f"    step {step:>5}/{steps}  refill {n_refill}  "
                      f"{t_refill / n_refill:.1f}s/refill  "
                      f"elapsed {_time.time() - t_start:.0f}s  "
                      f"train {curve[-1] if curve else float('nan'):.3f}", flush=True)
        bi = torch.randint(len(A), (batch,), device=device, generator=gen)
        loss = torch.nn.functional.cross_entropy(critic.scores(A[bi], S[bi]), tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, steps // 20) == 0 or step == steps - 1:
            curve.append(float((np.log(batch) - loss.item()) / ln2))
        # Held-out trace: fresh samples periodically turn one run into the whole
        # convergence curve. Costs one extra refill per probe.
        if trace_every and step and step % trace_every == 0:
            with torch.no_grad():
                Ap, Sp = to_t(*refill(batch))
                generated += batch
                lp = torch.nn.functional.cross_entropy(critic.scores(Ap, Sp), tgt)
                trace.append((step, generated,
                              float((np.log(batch) - lp.item()) / ln2)))
            if verbose:
                print(f"      trace step {step:>6}  gen {generated:>10,}  "
                      f"holdout {trace[-1][2]:>7.3f}", flush=True)

    # One refill for the whole evaluation, sliced afterwards, and capped at the training
    # buffer: a single call for `batch * n_eval` asks for 5.3 GB at batch 4096 and fails.
    per_call = max(1, min(batch * n_eval, buffer) // batch) * batch
    with torch.no_grad():
        vals = []
        done = 0
        while done < batch * n_eval:
            want = min(per_call, batch * n_eval - done)
            Ae_all, Se_all = to_t(*refill(want))
            generated += want
            done += want
            for i in range(want // batch):
                sl = slice(i * batch, (i + 1) * batch)
                Ae, Se = Ae_all[sl], Se_all[sl]
                if len(Ae) < batch:
                    break
                l = torch.nn.functional.cross_entropy(
                    critic.scores(Ae, Se), torch.arange(len(Ae), device=device))
                vals.append(float((np.log(len(Ae)) - l.item()) / ln2))

    # Measured on samples generated after training, so no split is needed.
    bits = float(np.mean(vals))
    return dict(bits=bits, sd=float(np.std(vals)), ceiling=float(np.log2(batch)),
                saturation=bits / float(np.log2(batch)), curve=curve,
                generated=generated, train_last=curve[-1] if curve else None,
                trace=trace)
