# Energy-based whitening: theory note

`energy_whitening.pdf` is a companion note to the paper's Section III-A (Eq. 9). It covers:

- what energy whitening does, and why;
- proofs of its properties: coordinate invariance, the maximum-entropy/Gibbs noise law, the
  port-Hamiltonian structure, the stiff-SEA constraint limit, SNR asymptotics, consistent
  Shapley noise, a closed form for MC_W, mesh independence, and floating bases;
- how it relates to the papers cited in the paper and to related work;
- a numerical check of every claim on this repository's models.

| File | What it is |
|---|---|
| `energy_whitening.tex`, `refs.bib` | The note. Appendix B records which citations were read in full text. |
| `checks.py` | One function per result. Run it to print every number quoted in the note. |
| `make_figures.py` | Redraws `figures/*.pdf` from `checks.py`. |
| `../../tests/test_energy_whitening.py` | Pytest versions of the checks (about 3 s). |

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
