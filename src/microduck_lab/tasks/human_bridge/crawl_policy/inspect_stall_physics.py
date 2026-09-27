"""Diagnose exactly what forces and contacts prevent Sister from advancing past x=0.05m."""
import sys
from pathlib import Path
import numpy as np
import mujoco

SRC_DIR = Path(__file__).resolve().parents[4]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from microduck_lab.tasks.human_bridge.crawl_policy.story import Story, rearmost
from microduck_lab.tasks.human_bridge.crawl_policy.world import SERVOS

s = Story(verbose=False)
m, d = s.world.model, s.world.data

# Let her crawl until t = 10s (when she reaches x = +0.05m)
for _ in range(500):
    s.tick()

print(f"\n--- State at t={s.world.t:.2f}s ---")
pos = s.she.pos()
print(f"Sister pos: {np.round(pos, 4)}, rear: {rearmost(s.she):.4f}, roll: {d.xmat[s.she.trunk, 7]:.3f}")

# Group contacts by what parts of Sister are touching what
print("\nActive contacts on Sister:")
geom_forces = {}
for i in range(d.ncon):
    con = d.contact[i]
    g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, con.geom1)
    g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, con.geom2)
    b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[con.geom1])
    b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[con.geom2])
    
    # 6D contact force in contact frame
    c_force = np.zeros(6)
    mujoco.mj_contactForce(m, d, i, c_force)
    fn = c_force[0] # normal force
    ft = np.linalg.norm(c_force[1:3]) # friction force
    
    if 'she' in str(b1):
        target = f"{b1} ({g1}) touching {b2} ({g2})"
        geom_forces[target] = geom_forces.get(target, 0.0) + fn
        print(f"  {target:60s}: normal={fn:6.2f}N, friction={ft:6.2f}N, pos_x={con.pos[0]:+.3f}, pos_z={con.pos[2]:.3f}")
    elif 'she' in str(b2):
        target = f"{b2} ({g2}) touching {b1} ({g1})"
        geom_forces[target] = geom_forces.get(target, 0.0) + fn
        print(f"  {target:60s}: normal={fn:6.2f}N, friction={ft:6.2f}N, pos_x={con.pos[0]:+.3f}, pos_z={con.pos[2]:.3f}")

# Measure net contact horizontal force (Fx) on Sister
total_fx = 0.0
for i in range(d.ncon):
    con = d.contact[i]
    b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[con.geom1])
    b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[con.geom2])
    if 'she' in str(b1) or 'she' in str(b2):
        c_force = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, c_force)
        # contact frame orientation
        # con.frame has normal along row 0, tangent 1 along row 1, tangent 2 along row 2
        R_c = con.frame.reshape(3, 3)
        # force in world frame
        f_world = R_c.T @ c_force[:3]
        if 'she' in str(b2):
            f_world = -f_world # action-reaction
        total_fx += f_world[0]

print(f"\nTotal contact Fx on Sister: {total_fx:+.3f} N (positive = pushing forward, negative = pushing backward)")
