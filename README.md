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

## Animations

`passive_info.anim` renders videos of the result. A contact force is applied to a leg, and the
video shows the energy in each part of the leg and the information about the force in the
motion of the leg, for each frequency of the force. Every value on screen comes from the same
linear models and energy readout as the bit counts above. The final spectrum equals
`information.profile_bits`.

| Style | Content |
|---|---|
| `go2` | the Go2 from MuJoCo Menagerie, restyled, with the front-right leg driven by `embodied.go2_*` |
| `flow` | the 2-DoF prototype from `systems.*`, drawn as a schematic over a flow field |

Each style has two cuts:
- `loop` repeats without a cut. It shows the rigid design, then two series-elastic designs.
  A few short labels explain the marks. Use `--no-text` to render it without labels.
- `talk` shows the rigid design beside each series-elastic design under the same force, with labels.

```bash
sudo apt-get install -y libegl1 libgl1-mesa-dri ffmpeg   # headless GL; or libosmesa6 + MUJOCO_GL=osmesa
uv sync --extra anim
uv run python -m passive_info.anim go2 --cut loop          # add --preview for 960x540
uv run python -m passive_info.anim go2 --cut talk
uv run python -m passive_info.anim flow --cut loop
uv run python -m passive_info.anim go2 --stills 0.55       # PNGs instead of video
```

The videos go to `passive_info/anim/out/` as MP4 and WebM. Two quantities are scaled for
display, and the talk cut states both:
- Joint motion is multiplied by `--gain`, default 250. The real deflections are less than a
  milliradian.
- Time is slowed down by `--slow`, default 55 for the loop and 90 for the talk cut.

Between designs, the display moves the leg back to its rest pose. The model does not do this
by itself, because the clamped leg has directions with zero stiffness.

## Regression tables

`tests/golden_analytic.json` and `tests/golden_library.json` pin 1046 and 264 scalars. If you
change something on purpose, regenerate them and read the diff:

```bash
uv run python -m tests.golden
uv run python -m tests.golden_library
git diff tests/
```
