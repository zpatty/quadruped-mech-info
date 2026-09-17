"""Every `mechinfo` primitive, on mechanisms that are not the Wensing prototype."""
from __future__ import annotations

import json
import pathlib

import numpy as np

OUT = pathlib.Path(__file__).parent / "golden_library.json"

#: Ground the three mechanisms are bolted to. Not the prototype's `K_I`/`ZETA_CONTACT`.
K_I = 3.1e3
ZETA_CONTACT = 0.35


def _mechanisms():
    """`(tag, H, K_struct, J, D_struct, joint_damping)`, none of them the prototype."""
    # Coupled 2-DoF: off-diagonal mass, both coordinates loaded by the foot.
    H2 = np.array([[1.7, -0.29],
                   [-0.29, 0.43]])
    K2 = np.array([[820.0, -820.0],
                   [-820.0, 820.0]])
    J2 = np.array([0.62, -0.11])
    yield "chain2", H2, K2, J2, None, 0.0

    # 3-DoF with a stiffness-free coordinate: K is singular, which is what `sym_sqrt`'s
    # clip and `energy_metric`'s caveat are about.
    H3 = np.diag([2.4, 0.9, 0.16])
    K3 = np.zeros((3, 3))
    K3[1:, 1:] = [[1500.0, -1500.0], [-1500.0, 1500.0]]
    J3 = np.array([1.0, -0.44, 0.0])
    yield "float3", H3, K3, J3, None, 0.0

    # Both damping paths at once -- the branch the prototype never takes together.
    D2 = 4.7 * np.array([[1.0, -1.0], [-1.0, 1.0]])
    yield "damped2", H2, K2, J2, D2, 0.31


def collect() -> dict:
    from mechinfo.mechanics import (assemble, bits_from_gram, energy_metric, ground_damping,
                                    locked_effective_mass, rigid_plant, sea_plant, sym_sqrt,
                                    zoh)

    g: dict[str, float] = {}

    def put(key, M):
        """A matrix as four scalars: enough that any element moving moves one of them."""
        M = np.asarray(M, float)
        g[f"{key}.sum"] = float(np.sum(M))
        g[f"{key}.abssum"] = float(np.sum(np.abs(M)))
        g[f"{key}.absmax"] = float(np.max(np.abs(M)))
        g[f"{key}.trace"] = float(np.trace(M)) if M.ndim == 2 and M.shape[0] == M.shape[1]\
            else float(np.sum(M * M))

    # ---- the mechanism primitives --------------------------------------------------
    for tag, H, K_struct, J, D_struct, jd in _mechanisms():
        # `locked_effective_mass` on BOTH calling shapes. They must agree exactly, and a
        # regression that broke one would otherwise show up only on one arm of the package.
        m_e = locked_effective_mass(H, J)
        g[f"{tag}/m_e"] = float(m_e)
        g[f"{tag}/m_e_row"] = float(locked_effective_mass(H, J.reshape(1, -1)))
        g[f"{tag}/b_i"] = float(ground_damping(K_I, m_e, ZETA_CONTACT))

        A, B_F, K, D = assemble(H, K_struct, J, K_I, m_e, ZETA_CONTACT,
                                joint_damping=jd, D_struct=D_struct)
        for name, M in (("A", A), ("B_F", B_F), ("K", K), ("D", D)):
            put(f"{tag}/{name}", M)

        # Eigenvalues of the open mechanism: a checksum on `A` that a sum cannot fake,
        # sorted so the ordering `eig` happens to return is not what is pinned.
        lam = np.linalg.eigvals(A)
        order = np.lexsort((lam.imag, lam.real))
        for i, z in enumerate(lam[order]):
            g[f"{tag}/eig{i}.re"] = float(z.real)
            g[f"{tag}/eig{i}.im"] = float(z.imag)

        # The energy metric, both variants, and its square root.
        E = energy_metric(K, H)
        put(f"{tag}/E", E)
        put(f"{tag}/E_half", sym_sqrt(E))
        Ew = energy_metric(K, H, w=2.0 * np.pi * 37.0)
        put(f"{tag}/E_kin", Ew)
        put(f"{tag}/E_kin_half", sym_sqrt(Ew))

        # Discretisation at two rates an order apart, so an `expm` scaling bug that is
        # inert at one step size cannot hide.
        for dt in (1.0 / 500.0, 1.0 / 20000.0):
            Ad, Bd = zoh(A, B_F, dt)
            put(f"{tag}/zoh{1/dt:.0f}/Ad", Ad)
            put(f"{tag}/zoh{1/dt:.0f}/Bd", Bd)

        # A Gram built from the discretised input map, through the channel formula.
        Ad, Bd = zoh(A, B_F, 1.0 / 2000.0)
        G = (Bd.T @ Bd) * 1e6
        g[f"{tag}/gram_bits"] = bits_from_gram(np.atleast_2d(G))

    # ---- the two transmission templates, at parameters unlike the prototype's -------
    # These are topology, not the prototype: every number below is passed in.
    for tag, plant in (("rigid_c", rigid_plant(2.3, 0.17, 4.1e-4)),
                       ("rigid_f", rigid_plant(2.3, 0.17, 4.1e-4, m_b=6.8)),
                       ("sea_c", sea_plant(2.7e3, 2.3, 0.17, 4.1e-4, 0.08)),
                       ("sea_f", sea_plant(2.7e3, 2.3, 0.17, 4.1e-4, 0.08, m_b=6.8))):
        H, K_struct, J, D_struct = plant
        for name, M in (("H", H), ("K_struct", K_struct), ("J", J)):
            put(f"plant/{tag}/{name}", M)
        put(f"plant/{tag}/D_struct", np.zeros((1, 1)) if D_struct is None else D_struct)
        g[f"plant/{tag}/m_e"] = float(locked_effective_mass(H, J))

    # The stiff limit is the claim `locked_effective_mass` exists to make, so pin it:
    # as `k_s -> inf` a series-elastic mechanism's locked reference must equal the rigid
    # one's. Held here against the rigid plant's own `H`, which is what `embodied.py`
    # passes as `Hp`. If this row ever moves alone, the locked-reference argument broke.
    Hr, _, Jr, _ = rigid_plant(2.3, 0.17, 4.1e-4)
    g["plant/locked_reference"] = float(locked_effective_mass(Hr, Jr))
    # The committed table shows the defect, which is why these rows are worth reading:
    # `plant/rigid_c/m_e` is 2.3142 and `plant/sea_c/m_e` is 2.3000 exactly -- the rotor's
    # 4.1e-4 kg m^2 is simply absent from the SEA's own effective mass, because `J` does not
    # touch the rotor coordinate. Feeding that 2.3000 to `ground_damping` is what gave the
    # compliant designs a softer ground than the rigid leg they become in the stiff limit.
    # `plant/locked_reference` is the number every design must actually be handed.

    # ---- the estimators, seeded ----------------------------------------------------
    from mechinfo.estimators import (gaussian_mutual_information, ksg_chain_bits,
                                     ksg_mutual_information)

    rng = np.random.default_rng(11)
    for d_y, snr in ((3, 2.0), (8, 0.5)):
        M = np.zeros((d_y, 2))
        M[0, 0], M[1, 1] = snr, 0.5 * snr
        g[f"est/d{d_y}/exact"] = gaussian_mutual_information(M, 1.0, 1.0)
        a = rng.normal(size=(4000, 2))
        y = a @ M.T + rng.normal(size=(4000, d_y))
        for k in (4, 8):
            g[f"est/d{d_y}/ksg_k{k}"] = ksg_mutual_information(a, y, k=k)
        g[f"est/d{d_y}/ksg_raw"] = ksg_mutual_information(a, y, k=4, normalise=False)

    Mc = np.zeros((6, 6))
    for i in range(6):
        Mc[i, i] = 1.0 / (1.0 + i)
    spec = ksg_chain_bits(Mc, 0.05, 1.0, group=2, n_samples=2000, n_blocks=1, seed=5)
    for i, b in enumerate(spec):
        g[f"est/chain/band{i}"] = float(b)

    return g


def main() -> None:
    g = collect()
    OUT.write_text(json.dumps(g, indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT.name}: {len(g)} scalars")


if __name__ == "__main__":
    main()
