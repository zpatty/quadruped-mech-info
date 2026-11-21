# Energy-based whitening: theory note

`energy_whitening.pdf` is a companion note to the paper's Section III-A (Eq. 9). It covers:

- what energy whitening does, and why;
- proofs of its properties: coordinate invariance, the maximum-entropy/Gibbs noise law, the
  port-Hamiltonian structure, the stiff-SEA constraint limit, SNR asymptotics, consistent
  Shapley noise, a closed form for MC_W, mesh independence, and floating bases;
- how it relates to the papers cited in the paper and to related work;
- a numerical check of every claim on this repository's models.

Part II (Sections 9–14) asks whether energy is the *right* choice:
- a uniqueness theorem: energy is the only coordinate-free, local, scale-free,
  time-symmetric noise metric that is invariant under free motion;
- the proof that under energy noise free motion never creates information;
- a comparison of the alternatives (per-coordinate, kinetic, frequency-weighted/action,
  data-driven);
- a thermodynamically consistent Langevin body, with noise injected through the dampers:
  full-state information is capped by the contact damping for every design, and morphology
  matters by routing information to the sensors;
- a closing list of decisions for discussion.

Section 1 explains the idea without proofs.

| File | What it is |
|---|---|
| `energy_whitening.tex`, `refs.bib` | The note. Appendix B records which citations were read in full text. |
| `checks.py` | One function per result. Run it to print every number quoted in the note. |
| `make_figures.py` | Redraws `figures/*.pdf` from `checks.py`. |
| `../../tests/test_energy_whitening.py` | Pytest versions of the checks (about 15 s). |

No library code changes; the golden tables are untouched.

## Rebuilding

```bash
uv run python docs/whitening/checks.py
uv run python docs/whitening/make_figures.py
uv run pytest -q tests/test_energy_whitening.py
cd docs/whitening && latexmk -pdf energy_whitening.tex
```

Building the PDF needs a TeX distribution with `latexmk`, `natbib` and `cleveref`. On Debian or
Ubuntu, install `texlive-latex-extra texlive-science latexmk`.
