import mujoco
from microduck_lab.tasks.human_bridge.crawl_policy.story import Story
from microduck_lab.tasks.human_bridge.crawl_policy.world import SERVOS
s = Story(verbose=False)
m, d = s.world.model, s.world.data
J = {name: i for i, name in enumerate(SERVOS)}
foot_l = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_ankle_left')
foot_r = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_ankle_right')

for b in [-0.40, -0.20, 0.0, +0.20, +0.40]:
    q = s.she.data.qpos[s.she.qidx].copy()
    # Left hip pitch base is -0.89. If we add b:
    q[J['left_hip_pitch']] = -0.89 + b
    # Right hip pitch base is +0.89. If we subtract b:
    q[J['right_hip_pitch']] = +0.89 - b
    d.qpos[s.she.qidx] = q
    mujoco.mj_forward(m, d)
    z_l = d.xpos[foot_l, 2]
    z_r = d.xpos[foot_r, 2]
    z_trunk = d.xpos[s.she.trunk, 2]
    print(f"b={b:+.2f} -> trunk_z={z_trunk:.3f}, LF_z={z_l:.3f} (reach={z_trunk - z_l:+.3f}m), RF_z={z_r:.3f} (reach={z_trunk - z_r:+.3f}m)")
