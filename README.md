# slip-passive-info

How much a passive mechanism tells a robot about the forces acting on it. The code compares
actuator transmissions (a rigid proprioceptive drive against series-elastic drives of several
stiffnesses) by the information their sensors carry about contact force.

## Layout

| Package | What it is |
|---|---|
| `mechinfo` | The library: information carried by any mechanism you can write down. It never imports `passive_info`, and `tests/test_boundary.py` checks that. |
| `passive_info` | The paper: the Wensing prototype, the Go2 bench, the figures, and the cached runs in `passive_info/results/`. |

Both names are placeholders.

## Running it

The environment is managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync                        # numpy, scipy, matplotlib, mujoco
uv run pytest -q               # three tests skip without the `sim` extra
uv sync --extra sim            # torch + warp, for the GPU runs
```

The figures redraw from the committed JSON in `passive_info/results/`:

```bash
uv run python -m passive_info.figures          # add --slow for the method-comparison plate
uv run python -m passive_info.sim_figures
```

The GPU runs that produced that JSON:

```bash
uv run slip-passive <k_s | prop> [steps] [band_hz]
uv run slip-controlled <k_s | prop> [steps] [band_hz] [batch]
```

## Regression tables

`tests/golden_analytic.json` and `tests/golden_library.json` pin 1046 and 264 scalars. If you
change something on purpose, regenerate them and read the diff:

```bash
uv run python -m tests.golden
uv run python -m tests.golden_library
git diff tests/
```
