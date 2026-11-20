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

`passive_info.anim` renders videos of the result. The main one, `go2 --cut loop`, is a square
1080×1080 loop for the website. It shows the rigid Go2 leg with its motors off while a
contact force is applied to the foot:
- **Glow:** each part glows with the bits of information about the force in the motion of
  that part.
- **Dots:** new information, moving from the contact into the leg.
- **Counter and ring:** the total bits in the motion of the whole leg. The ring fills as the
  total rises to 134.5 bits at the end of the contact.
- **Part labels:** the bits for the foot contact and the knee motor rotor.

Every value on screen comes from the same linear model and energy readout as the bit counts
above. The total at the end of the contact equals `information.profile_bits`.

Specs for the website:
- **Shape.** The website shows the video in a round frame of about 300 px, so all the action
  and text stay inside the middle 70% of the frame.
- **Loop.** 9.8 s at 30 fps, seamless, no audio.
- **Poster.** Frame 0 is the poster image. The loop starts near the end of the contact,
  so frame 0 shows the leg with most of the information in.
- **Encodes.** For a 720×720 encode, scale the 1080 source (see below).

Text for the website caption, which is not in the frame:

> Information about a contact force in the motion of a robot leg. Glow: information about the
> force in the motion of each part. Dots: new information, moving from the contact into the
> leg. Motion scaled up 250 times; time slowed down 80 times.

Other cuts compare designs:
- `--cut compare` shows the rigid design, then two series-elastic designs, with energy glow
  and a spectrum ring.
- `--cut talk` shows the rigid design beside each series-elastic design.
- The `flow` style draws the 2-DoF prototype from `systems.*` as a schematic.

```bash
sudo apt-get install -y libegl1 libgl1-mesa-dri ffmpeg   # headless GL; or libosmesa6 + MUJOCO_GL=osmesa
uv sync --extra anim
uv run python -m passive_info.anim go2 --cut loop          # 1080x1080; add --preview for 540x540
uv run python -m passive_info.anim go2 --cut compare
uv run python -m passive_info.anim go2 --cut talk
ffmpeg -i passive_info/anim/out/go2_loop.mp4 -vf scale=720:720:flags=lanczos -an go2_loop_720.mp4
```

The videos go to `passive_info/anim/out/` as MP4 and WebM.

**Scaling.** Joint motion is multiplied by `--gain`, default 250, because the real
deflections are less than a milliradian. Time is slowed down by `--slow`, default 80 for the
loop.

**Force shape.** The loop uses a force with the shape of a contact: a half-sine times a small
ripple. The bit counts do not depend on this choice. The motion of the leg does.

**Rest pose.** Between contacts, the display moves the leg back to its rest pose. The model
does not do this by itself, because the clamped leg has directions with zero stiffness.

## Regression tables

`tests/golden_analytic.json` and `tests/golden_library.json` pin 1046 and 264 scalars. If you
change something on purpose, regenerate them and read the diff:

```bash
uv run python -m tests.golden
uv run python -m tests.golden_library
git diff tests/
```
