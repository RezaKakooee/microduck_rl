"""Bake his lying body into static geometry for RL training.

Runs the real scene (he stands, falls into PLANK, then holds himself
straight under BAM for SETTLE_S, as the story does before she starts) and
records every solid geom of his body: its mesh file and its world pose. `rl/microduck_bridge_sideways_env_cfg.py` rebuilds him from
this file as fixed geometry on the "terrain" body, so her training sees the
body he actually settles into, sag and all.

    python -m microduck_lab.tasks.human_bridge.scripted_policy.bake
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import mujoco
import numpy as np

from microduck_lab import paths
from microduck_lab.tasks.human_bridge.scripted_policy.brother import SETTLE_S, HoldStraight, LieDown
from microduck_lab.tasks.human_bridge.scripted_policy.scene import LAYOUT
from microduck_lab.tasks.human_bridge.scripted_policy.world import World
from microduck_lab.tasks.human_bridge.scripted_policy.runtime import brain as Brain

OUT = Path(__file__).resolve().parent / "bridge_pose.json"


def mesh_files():
    """Mesh name (unprefixed) -> (file, scale), from the robot MJCF."""
    spec = mujoco.MjSpec.from_file(paths.model("robot_allcollisions_mouth.xml"))
    meshdir = Path(spec.meshdir)
    if not meshdir.is_absolute():
        meshdir = (Path(paths.model("robot_allcollisions_mouth.xml")).parent / meshdir).resolve()
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


def bake(settle_s=SETTLE_S, on_step=None):
    w = World(LAYOUT.design(), cast=("he",))
    he = w.ducks["he"]
    brain = Brain(he, stand_only=True)
    he.stand_on(LAYOUT.gap_start - 0.005, 0.0, LAYOUT.step_z, brain.default_pose)
    w.start()
    for _ in range(75):
        brain.act()
        w.step()
        if on_step is not None:
            on_step(w)
    lie = LieDown(he, he.target, w.t)
    while not lie(w.t):
        w.step()
        if on_step is not None:
            on_step(w)
    hold = HoldStraight(he)
    for _ in range(int(settle_s / 0.02)):
        hold()
        w.step()
        if on_step is not None:
            on_step(w)
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
    import os
    from types import SimpleNamespace
    import imageio.v2 as imageio
    from microduck_lab.tasks.human_bridge.scripted_policy.story import Camera
    if os.environ.get('MUJOCO_GL') != 'egl' or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Bake through the Codex GPU job so the rollout has a video')
    folder = paths.REPO / 'videos/human_bridge/scripted_policy'
    folder.mkdir(parents=True, exist_ok=True)
    for number in range(1, 100000):
        stem = folder / f'bake_try{number}'
        if stem.with_suffix('.mp4').exists():
            continue
        try:
            stem.with_suffix('.lock').touch(exist_ok=False)
            break
        except FileExistsError:
            continue
    else:
        raise RuntimeError('No free bake video number')
    camera = Camera(640, 360)
    writer = imageio.get_writer(str(stem.with_suffix('.mp4')), fps=25, codec='libx264',
                               macro_block_size=2, output_params=['-threads', '1'])
    ticks = 0
    def frame(world):
        nonlocal ticks
        ticks += 1
        if ticks % 2 == 0:
            writer.append_data(camera.frame(SimpleNamespace(world=world, she=world.ducks['he'])))
    try:
        result = bake(on_step=frame)
        OUT.write_text(json.dumps(result, indent=1) + '\n')
        print(f"wrote {OUT}: {len(result['geoms'])} geoms, trunk {result['trunk']}, up {result['up']}", flush=True)
        print(f"Bake video: {stem.with_suffix('.mp4')}", flush=True)
    finally:
        writer.close()
        if camera.renderer is not None:
            camera.renderer.close()


if __name__ == '__main__':
    main()
