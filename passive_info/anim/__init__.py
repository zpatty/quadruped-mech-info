"""Stylized animations of the passive-information results. See `python -m passive_info.anim -h`."""
import os

# MuJoCo picks its GL backend at import, and `passive_info.embodied` imports it, so this has
# to run before anything else in the package. EGL renders headless; set MUJOCO_GL=osmesa to
# fall back to pure software where EGL is unavailable.
os.environ.setdefault("MUJOCO_GL", "egl")
