"""Load a mechanism from an MJCF file. `RobotSpec` carries the robot's naming."""
from __future__ import annotations

import os
import pathlib
import subprocess
from dataclasses import dataclass

import numpy as np

MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"


def menagerie_root(robot: str = "unitree_go2") -> pathlib.Path:
    """Path to a mujoco_menagerie checkout, sparse-cloning `robot` if there is none."""
    env = os.environ.get("MUJOCO_MENAGERIE")
    candidates = [pathlib.Path(env)] if env else []
    candidates += [pathlib.Path("mujoco_menagerie").resolve(),
                   pathlib.Path.home() / ".cache" / "mujoco_menagerie"]
    for c in candidates:
        if (c / robot / "scene.xml").exists():
            return c
        # Below here clones over the network, which is why nothing calls this at import time.
    dst = pathlib.Path.home() / ".cache" / "mujoco_menagerie"
    subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                    MENAGERIE_URL, str(dst)], check=True)
    subprocess.run(["git", "sparse-checkout", "set", robot], cwd=dst, check=True)
    return dst


def menagerie_scene(robot: str = "unitree_go2") -> str:
    """The `scene.xml` path for a Menagerie robot, fetching the collection if need be."""
    return str(menagerie_root(robot) / robot / "scene.xml")


@dataclass(frozen=True)
class RobotSpec:
    """Where to find a robot's landmarks. Feet must be named geoms, one per leg."""
    feet: tuple[str, ...]                       #: foot geom names, one per leg
    xml: str = ""                               #: only `compile_model` needs it
    trunk_body: str = "base"                    #: the body whose free joint is constrained
    hip_suffix: str = "_hip"                    #: `<leg>` + this names the leg's root body
    floor_geom: str = "floor"                   #: named alongside the feet for contact tuning
    parts: tuple[str, ...] = ("hip", "thigh", "calf")
    joint_fmt: str = "{leg}_{part}_joint"       #: formatted with `leg=` and `part=`
    actuator_fmt: str = "{leg}_{part}"

    def joint(self, leg: str, part: str) -> str:
        return self.joint_fmt.format(leg=leg, part=part)

    def actuator(self, leg: str, part: str) -> str:
        return self.actuator_fmt.format(leg=leg, part=part)


@dataclass
class LegIndex:
    """One leg's model indices, resolved once by name."""
    foot_geom: int
    foot_body: int
    foot_radius: float
    hip_body: int
    dofs: np.ndarray            # the joint DoFs of this leg
    qpos: np.ndarray            # their qpos addresses
    act: np.ndarray             # their actuator ids


def compile_model(robot: RobotSpec, armature: float, vertical: bool = True,
                  dt: float = 2.5e-4, joint_damping: float | None = None,
                  frictionloss: float | None = None,
                  stance_feet: tuple[str, ...] | None = None,
                  contact="menagerie", contact_stiff: float = 0.002,
                  contact_solimp=(0.9, 0.99, 1e-4, 0.5, 2.0),
                  trunk_scale: float = 1.0, legs: tuple[str, ...] | None = None,
                  spec_hook=None):
    """Compile the scene with the given reflected inertia and trunk constraint."""
    import mujoco

    if not robot.xml:
        raise ValueError("RobotSpec.xml is empty: compile_model needs an MJCF path. "
                         "A spec built only for index_legs may leave it unset.")
    legs = robot.feet if legs is None else legs
    stance_feet = legs if stance_feet is None else stance_feet

    spec = mujoco.MjSpec.from_file(robot.xml)
    for j in spec.joints:
        if j.type != mujoco.mjtJoint.mjJNT_FREE:
            j.armature = armature
            if joint_damping is not None:
                j.damping = [joint_damping, 0.0, 0.0]
            if frictionloss is not None:
                j.frictionloss = frictionloss
    # vertical: True -> vertical slide named trunk_z; False -> keep the free joint;
    # "fixed" -> delete it, welding the trunk to the world (what the closed form assumes).
    if vertical:
        for k in list(spec.keys):
            spec.delete(k)
        for b in spec.bodies:
            if b.name == robot.trunk_body:
                for j in b.joints:
                    if j.type == mujoco.mjtJoint.mjJNT_FREE:
                        if vertical == "fixed":
                            spec.delete(j)
                            break
                        j.type = mujoco.mjtJoint.mjJNT_SLIDE
                        j.axis = [0, 0, 1]
                        j.name = "trunk_z"
    if set(legs) != set(robot.feet):
        n = len(robot.hip_suffix)
        for b in list(spec.bodies):
            if b.name.endswith(robot.hip_suffix) and b.name[:-n] not in legs:
                spec.delete(b)
    if set(stance_feet) != set(legs):
        for g in spec.geoms:
            if g.name in legs and g.name not in stance_feet:
                g.contype, g.conaffinity = 0, 0
    if contact != "menagerie":
        tc = contact_stiff if contact == "stiff" else float(contact)
        if not tc > 0:
            raise ValueError(f"contact time constant must be positive, got {contact!r}")
        for g in spec.geoms:
            if g.name in robot.feet or g.name == robot.floor_geom:
                g.solref = [tc, 1.0]
                g.solimp = list(contact_solimp)
    if trunk_scale != 1.0:
        for b in spec.bodies:
            if b.name == robot.trunk_body:
                b.mass *= trunk_scale
                b.inertia = [v * trunk_scale for v in b.inertia]
    spec.option.timestep = dt
    if spec_hook is not None:
        spec_hook(spec)
    m = spec.compile()
    return m, mujoco.MjData(m)


def index_legs(model, robot: RobotSpec) -> dict[str, LegIndex]:
    """`{leg: LegIndex}` for the legs present in `model`."""
    import mujoco

    out = {}
    for f in robot.feet:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f)
        # A leg whose foot geom is missing is skipped, so a wrong joint_fmt returns {} rather
        # than raising. Check the result if a particular leg is required.
        if gid < 0:          # this leg was deleted; a single-leg rig has only one
            continue
        dofs, qpos, act = [], [], []
        for part in robot.parts:
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, robot.joint(f, part))
            dofs.append(model.jnt_dofadr[j])
            qpos.append(model.jnt_qposadr[j])
            act.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                         robot.actuator(f, part)))
        out[f] = LegIndex(gid, model.geom_bodyid[gid], model.geom_size[gid, 0],
                          mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                            f + robot.hip_suffix),
                          np.array(dofs), np.array(qpos), np.array(act))
    return out


def ground_reaction(model, data, leg) -> np.ndarray:
    """World-frame ground reaction force on `leg`, summed over its contacts."""
    import mujoco

    total = np.zeros(3)
    buf = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        if leg.foot_geom not in (c.geom1, c.geom2):
            continue
        mujoco.mj_contactForce(model, data, i, buf)
        f_world = c.frame.reshape(3, 3).T @ buf[:3]
        total += f_world if c.geom2 == leg.foot_geom else -f_world
    return total
