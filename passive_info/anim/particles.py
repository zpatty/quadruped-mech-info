"""Particles that represent stored energy, updated each frame from the energy in each element.

Each particle stands for a fixed amount of energy and belongs to one element. On every frame,
the code sets the number of particles in each element to that element's energy divided by the
amount per particle:

  * If an element gains energy, it takes particles from the nearest elements that lost energy.
    These particles move to it along the kinematic chain.
  * If the gain is larger than the losses elsewhere, the force has done work at the contact.
    New particles start at the foot, with colours drawn from the force modes in that element.
  * If the losses are larger than the gains, the difference has been dissipated. Those
    particles turn grey and fade out where they are.

The update runs once, in order, over a whole timeline. The renderers can then draw frames in
any order or in parallel from the stored states. The renderer computes positions, because it
knows where each element is on that frame.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

INPUT = -1          #: `src` of a particle that starts at the contact


@dataclass
class FrameParticles:
    """All live particles at one frame. Arrays are aligned, one row per particle."""
    pid: np.ndarray       #: (p,) stable ids, for trails
    src: np.ndarray       #: (p,) element it is leaving, or INPUT
    dst: np.ndarray       #: (p,) element it belongs to
    prog: np.ndarray      #: (p,) 0..1 along the path; 1 once settled
    u: np.ndarray         #: (p,) where on the element it settles, 0..1
    v: np.ndarray         #: (p,) a second free parameter (orbit phase, offset)
    hue: np.ndarray       #: (p, 3) linear RGB
    alpha: np.ndarray     #: (p,) 1 alive, falling to 0 while dying
    dying: np.ndarray     #: (p,) bool


def graph_distances(n: int, edges) -> tuple[np.ndarray, np.ndarray]:
    """All-pairs hop counts and next-hop table for an undirected element graph."""
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    D = np.full((n, n), 99, int)
    nxt = np.full((n, n), -1, int)
    for s in range(n):
        D[s, s] = 0
        nxt[s, s] = s
        q = deque([s])
        while q:
            x = q.popleft()
            for y in adj[x]:
                if D[s, y] == 99:
                    D[s, y] = D[s, x] + 1
                    nxt[s, y] = y if x == s else nxt[s, x]
                    q.append(y)
    return D, nxt


def path(nxt: np.ndarray, a: int, b: int) -> list[int]:
    """Elements from `a` to `b` inclusive."""
    out = [a]
    while out[-1] != b and len(out) < 32:
        out.append(int(nxt[out[-1], b]))
    return out


class Transport:
    """Stateful particle bookkeeping for one timeline."""

    def __init__(self, elements, edges, quantum: float, entry: str = "ground",
                 hues: np.ndarray | None = None, seed: int = 0,
                 hop_time: float = 0.16, fade_time: float = 0.7,
                 max_birth_per_frame: int = 60):
        self.elements = list(elements)
        idx = {e: i for i, e in enumerate(self.elements)}
        self.edges = [(idx[a], idx[b]) for a, b in edges]
        self.D, self.nxt = graph_distances(len(self.elements), self.edges)
        self.entry = idx[entry]
        self.q = float(quantum)
        self.hues = hues
        self.rng = np.random.default_rng(seed)
        self.hop_time, self.fade_time = hop_time, fade_time
        self.max_birth = max_birth_per_frame
        self.p: list[dict] = []
        self._next_id = 0

    # -----------------------------------------------------------------------------------
    def _new(self, dst: int, hue_w: np.ndarray):
        w = np.clip(hue_w[dst], 0.0, None)
        if w.sum() > 0 and self.hues is not None:
            k = self.rng.choice(len(w), p=w / w.sum())
            hue = self.hues[k] * (0.85 + 0.3 * self.rng.random())
        else:
            hue = np.array([1.0, 0.9, 0.8])
        hops = self.D[self.entry, dst] + 1
        self.p.append(dict(pid=self._next_id, src=INPUT, dst=dst, prog=0.0,
                           u=self.rng.random(), v=self.rng.random(), hue=hue, alpha=1.0,
                           dying=False, ttime=self.hop_time * hops))
        self._next_id += 1

    def step(self, energy: np.ndarray, hue_w: np.ndarray, dt: float) -> FrameParticles:
        """Advance one frame against the element energies `energy` (J)."""
        n = len(self.elements)
        for p in self.p:                       # motion first
            if p["dying"]:
                p["alpha"] -= dt / self.fade_time
            elif p["prog"] < 1.0:
                p["prog"] = min(1.0, p["prog"] + dt / p["ttime"])
        self.p = [p for p in self.p if p["alpha"] > 0.0]

        target = np.floor(np.clip(energy, 0.0, None) / self.q + 0.5).astype(int)
        live = [p for p in self.p if not p["dying"]]
        owned = [[] for _ in range(n)]
        for p in live:
            owned[p["dst"]].append(p)
        count = np.array([len(o) for o in owned])
        surplus = {e: count[e] - target[e] for e in range(n) if count[e] > target[e]}
        deficit = {e: target[e] - count[e] for e in range(n) if target[e] > count[e]}

        # particles that have arrived move first; particles still in transit move last
        for e in surplus:
            owned[e].sort(key=lambda p: -p["prog"])

        for e in sorted(deficit, key=lambda e: -deficit[e]):
            need = deficit[e]
            while need > 0 and surplus:
                donor = min(surplus, key=lambda s: (self.D[s, e], -surplus[s]))
                p = owned[donor].pop(0)
                surplus[donor] -= 1
                if surplus[donor] == 0:
                    del surplus[donor]
                hops = self.D[donor, e]
                p.update(src=donor, dst=e, prog=0.0, ttime=self.hop_time * max(hops, 1))
                need -= 1
            births = min(need, self.max_birth)
            for _ in range(births):
                self._new(e, hue_w)

        for e, s in surplus.items():           # dissipated: fade out in place
            for p in owned[e][:s]:
                p["dying"] = True
                p["hue"] = 0.5 * p["hue"] + 0.5 * np.array([0.25, 0.22, 0.20])

        return self.snapshot()

    def snapshot(self) -> FrameParticles:
        P = self.p
        if not P:
            z = np.zeros(0)
            return FrameParticles(z.astype(int), z.astype(int), z.astype(int), z, z, z,
                                  np.zeros((0, 3)), z, z.astype(bool))
        g = lambda k, t=float: np.array([p[k] for p in P], dtype=t)
        return FrameParticles(g("pid", int), g("src", int), g("dst", int), g("prog"),
                              g("u"), g("v"), np.array([p["hue"] for p in P]),
                              np.clip(g("alpha"), 0.0, 1.0), g("dying", bool))
