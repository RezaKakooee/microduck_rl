"""Bake his lying body into static geometry for RL training.

Runs the real scene (he stands, falls into PLANK, then holds himself
straight under BAM for SETTLE_S, as the story does before she starts) and
records every solid geom of his body: its mesh file and its world pose. `rl/human_bridge/microduck_bridge_sideways_env_cfg.py` rebuilds him from
this file as fixed geometry on the "terrain" body, so her training sees the
body he actually settles into, sag and all.

    python -m microduck_lab.tasks.human_bridge.rl_policy.bake
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import mujoco
import numpy as np

from microduck_lab import paths
from microduck_lab.tasks.human_bridge.rl_policy.brother import SETTLE_S, HoldStraight, LieDown
from microduck_lab.tasks.human_bridge.rl_policy.scene import LAYOUT
from microduck_lab.tasks.human_bridge.rl_policy.world import Brain, World

OUT = Path(__file__).resolve().parent / "bridge_pose.json"


def mesh_files():
    """Mesh name (unprefixed) -> (file, scale), from the robot MJCF."""
    spec = mujoco.MjSpec.from_file(paths.model("robot/robot_allcollisions_mouth.xml"))
    meshdir = Path(spec.meshdir)
    if not meshdir.is_absolute():
        meshdir = (Path(paths.model("robot/robot_allcollisions_mouth.xml")).parent / meshdir).resolve()
    # The MJCF leaves meshes unnamed; MuJoCo names each after its file stem.
    return {(m.name or Path(m.file).stem): (str((meshdir / m.file).resolve().relative_to(paths.REPO)),
                                           list(m.scale))
            for m in spec.meshes}


def mesh_offset(file, scale):
    """The frame MuJoCo moves a mesh into when it compiles it: compiled
    vertex v maps to file coordinates as R_off @ v + p_off."""
    spec = mujoco.MjSpec()
    spec.add_mesh(name="m", file=str(paths.REPO / file), scale=scale)
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="m")
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    return d.geom_xpos[0].copy(), d.geom_xmat[0].reshape(3, 3).copy()


def bake(settle_s=SETTLE_S):
    w = World(LAYOUT.design(), cast=("he",))
    he = w.ducks["he"]
    brain = Brain(he, stand_only=True)
    he.stand_on(LAYOUT.gap_start - 0.005, 0.0, LAYOUT.step_z, brain.default_pose)
    w.start()
    for _ in range(75):
        brain.act()
        w.step()
    lie = LieDown(he, he.target, w.t)
    while not lie(w.t):
        w.step()
    hold = HoldStraight(he)
    for _ in range(int(settle_s / 0.02)):
        hold()
        w.step()
    m, d = w.model, w.data
    files = mesh_files()
    geoms = []
    for g in he.solid:
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g]).partition("_")[2]
        file, scale = files[name]
        # geom_xpos/xmat locate the COMPILED mesh, which MuJoCo re-centres on
        # its own frame. Saving them and compiling the file again applies that
        # shift twice (the first bake did: thighs 56 mm off, soles 50 mm too
        # high). Save the authored frame instead: undo the compile offset.
        p_off, R_off = mesh_offset(file, scale)
        R_g = d.geom_xmat[g].reshape(3, 3)
        R_a = R_g @ R_off.T
        p_a = d.geom_xpos[g] - R_a @ p_off
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, R_a.reshape(-1))
        geoms.append(dict(mesh=name, file=file, scale=scale,
                          pos=p_a.round(6).tolist(), quat=q.round(6).tolist(),
                          friction=m.geom_friction[g].round(5).tolist()))
    return dict(layout=dataclasses.asdict(LAYOUT), settle_s=settle_s,
                trunk=d.xpos[he.trunk].round(5).tolist(), up=round(he.up(), 4), geoms=geoms)


def main():
    b = bake()
    OUT.write_text(json.dumps(b, indent=1) + "\n")
    print(f"wrote {OUT}: {len(b['geoms'])} geoms, trunk {b['trunk']}, up {b['up']}")


if __name__ == "__main__":
    main()
