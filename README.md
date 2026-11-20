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

`passive_info.anim` renders videos of the result. The main one, `go2 --cut loop`, shows the
rigid Go2 leg with its motors off while a contact force is applied to the foot:
- **Glow:** each part glows with the bits of information about the force in the motion of
  that part.
- **Dots:** new information, moving from the contact into the leg.
- **Force plot:** the applied force, and the band of force profiles that are consistent with
  the motion of the leg so far. The band narrows from left to right as the bits accumulate.

Every value on screen comes from the same linear model and energy readout as the bit counts
above. The total at the end of the contact equals `information.profile_bits`. Nothing in the
video estimates the force: the band is the posterior range from that same Gaussian model.

Other cuts compare designs:
- `--cut compare` shows the rigid design, then two series-elastic designs, with energy glow
  and a spectrum ring.
- `--cut talk` shows the rigid design beside each series-elastic design.
- The `flow` style draws the 2-DoF prototype from `systems.*` as a schematic.

```bash
sudo apt-get install -y libegl1 libgl1-mesa-dri ffmpeg   # headless GL; or libosmesa6 + MUJOCO_GL=osmesa
uv sync --extra anim
uv run python -m passive_info.anim go2 --cut loop          # add --preview for 960x540
uv run python -m passive_info.anim go2 --cut compare
uv run python -m passive_info.anim go2 --cut talk
uv run python -m passive_info.anim go2 --stills 0.5        # PNGs instead of video
```

The videos go to `passive_info/anim/out/` as MP4 and WebM. Two quantities are scaled for
display, and every video states both:
- Joint motion is multiplied by `--gain`, default 250. The real deflections are less than a
  milliradian.
- Time is slowed down by `--slow`, default 80 for the loop.

The loop uses a force with the shape of a contact: a half-sine times a small ripple. The bit
counts and the width of the band do not depend on this choice. The motion of the leg and the
centre of the band do.

Between contacts, the display moves the leg back to its rest pose. The model does not do this
by itself, because the clamped leg has directions with zero stiffness.

## Regression tables

`tests/golden_analytic.json` and `tests/golden_library.json` pin 1046 and 264 scalars. If you
change something on purpose, regenerate them and read the diff:

```bash
uv run python -m tests.golden
uv run python -m tests.golden_library
git diff tests/
```
