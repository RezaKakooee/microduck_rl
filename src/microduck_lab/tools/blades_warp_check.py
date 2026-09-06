#!/usr/bin/env python3
"""GPU physics check for the skate-blade task. Needs a GPU:

    uv run python -m microduck_lab.tools.blades_warp_check

Part 1 (box): does mujoco_warp honour anisotropic pair friction with the same
slot -> world-axis mapping as CPU MuJoCo (slot 0 = world Y, slot 1 = world X),
and per-world pair_friction values? Expected: X ~0.19 m, Y ~5.1 m for
friction [0.02 1.2], and the two worlds swap.

Part 2 (env, --env): build Mjlab-Velocity-Blades-MicroDuck, step it, and show
that pair_friction is per env, that it moves with the projection, and that the
spec_fn print appears.
"""
import sys
import numpy as np, mujoco, warp as wp
import mujoco_warp as mjw
from mujoco_warp._src.types import vec5

XML = """<mujoco><option timestep="0.005"/><worldbody>
  <geom name="floor" type="plane" size="0 0 .01"/>
  <body pos="0 0 0.05"><freejoint/><geom name="box" type="box" size=".05 .05 .05" mass="1"/></body>
</worldbody><contact><pair geom1="box" geom2="floor" condim="4" friction="0.02 1.2 0.005 0.0001 0.0001"/></contact></mujoco>"""
mjm = mujoco.MjModel.from_xml_string(XML); mjd = mujoco.MjData(mjm); mujoco.mj_forward(mjm, mjd)

def run(axis, per_world=None):
    mujoco.mj_resetData(mjm, mjd); mujoco.mj_forward(mjm, mjd); mjd.qvel[axis] = 2.0
    nworld = 1 if per_world is None else len(per_world)
    m = mjw.put_model(mjm)
    if per_world is not None:
        m.pair_friction = wp.array(np.array(per_world, dtype=np.float32).reshape(nworld, 1, 5), dtype=vec5)
    d = mjw.put_data(mjm, mjd, nworld=nworld)
    for _ in range(600): mjw.step(m, d)
    return d.qpos.numpy()[:, axis]

print("GPU single world, pair [0.02 1.2]: slid X =", run(0).round(2), " Y =", run(1).round(2), " (CPU: X 0.19, Y 5.12)")
pw = [[0.02, 1.2, 0.005, 1e-4, 1e-4], [1.2, 0.02, 0.005, 1e-4, 1e-4]]
print("GPU two worlds w0=[0.02 1.2] w1=[1.2 0.02], shove +X: slid =", run(0, pw).round(2), " (expect [~0.19, ~5.1])")
print("GPU two worlds, shove +Y: slid =", run(1, pw).round(2), " (expect [~5.1, ~0.19])")


def env_check(num_envs=8, steps=40):
    """Build the real task, step it, and show the projection is live per env."""
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    import mjlab_microduck.tasks  # noqa: F401  (registers the tasks)

    cfg = load_env_cfg("Mjlab-Velocity-Blades-MicroDuck")
    cfg.scene.num_envs = num_envs
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    env.reset()
    pf = env.sim.model.pair_friction
    print(f"pair_friction shape: {tuple(pf.shape)}  (expected ({num_envs}, 2, 5): per env)")
    st = env._blade_state
    print("mu_along  per env:", st.mu_along.cpu().numpy().round(3).tolist())
    print("mu_across per env:", st.mu_across.cpu().numpy().round(3).tolist())
    act_dim = getattr(env.action_manager, "total_action_dim", 14)
    act = torch.zeros(env.num_envs, act_dim, device=env.device)
    prev = pf.clone()
    changed = 0
    samples = []
    for k in range(steps):
        env.step(act)
        cur = env.sim.model.pair_friction
        if not torch.allclose(cur, prev):
            changed += 1
        prev = cur.clone()
        if k % 10 == 9:
            f = env.scene["feet_ground_contact"].data.force[0].cpu().numpy()
            samples.append((k + 1, cur[0, :, :2].cpu().numpy().round(3).tolist(), f.round(2).tolist()))
    print(f"pair_friction changed on {changed}/{steps} steps (the step-mode projection is live)")
    for k, p, f in samples:
        print(f"  step {k:2d}: env0 [muY, muX] per pair = {p}   foot net force xyz = {f}")
    spin_roll = pf[..., 2:].cpu().numpy()
    print(f"spin/roll untouched: min {spin_roll.min():.5f} max {spin_roll.max():.5f}")
    env.close()


if __name__ == "__main__" and "--env" in sys.argv:
    print("=== env check ===")
    env_check()
