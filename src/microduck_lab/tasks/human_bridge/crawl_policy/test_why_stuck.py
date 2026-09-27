"""Detailed analysis of why Sister gets stuck at the notch / on Brother."""
import sys
from pathlib import Path
import numpy as np
import mujoco

SRC_DIR = Path(__file__).resolve().parents[4]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from microduck_lab.tasks.human_bridge.crawl_policy.story import Story
from microduck_lab.tasks.human_bridge.crawl_policy.scene import LAYOUT as L
from microduck_lab.tasks.human_bridge.crawl_policy.world import SERVOS

story = Story(verbose=True)
she = story.she
he = story.he
m, d = story.world.model, story.world.data

foot_l = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_ankle_left')
foot_r = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_ankle_right')
knee_l = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_leg')
knee_r = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_leg_2')
jaw = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_mouth_jaw')
trunk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'she_trunk_base')

print(f"Ledge end near_edge = {L.near_edge:.3f} m, near_z = {L.near_z:.3f} m")
print(f"Near step = [{L.near_edge:.3f}, {L.gap_start:.3f}], step_z = {L.step_z:.3f} m")
print(f"Far edge = {L.far_edge:.3f} m, far_z = {L.far_z:.3f} m")

# Run story until 12 seconds
for step in range(600):
    story.tick()
    t = story.world.t
    if step % 25 == 0:
        pos = she.pos()
        lf = d.xpos[foot_l]
        rf = d.xpos[foot_r]
        kl = d.xpos[knee_l]
        kr = d.xpos[knee_r]
        jw = d.xpos[jaw]
        
        # Check active contacts on Sister
        contacts = []
        for c_i in range(d.ncon):
            c = d.contact[c_i]
            b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom1]) or ""
            b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom2]) or ""
            g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
            g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
            if b1.startswith('she_') or b2.startswith('she_'):
                f6 = np.zeros(6)
                mujoco.mj_contactForce(m, d, c_i, f6)
                if f6[0] > 0.3:
                    she_part = b1 if b1.startswith('she_') else b2
                    other = f"{b2}/{g2}" if b1.startswith('she_') else f"{b1}/{g1}"
                    contacts.append(f"{she_part}<->{other}({f6[0]:.1f}N,x={c.pos[0]:.2f},z={c.pos[2]:.2f})")
        
        print(f"\nt={t:5.2f}s pos={np.round(pos, 3)} furthest_x={story.furthest_x:.3f}")
        print(f"  LF=({lf[0]:.3f},{lf[1]:.3f},{lf[2]:.3f}) RF=({rf[0]:.3f},{rf[1]:.3f},{rf[2]:.3f}) Jaw=({jw[0]:.3f},{jw[1]:.3f},{jw[2]:.3f})")
        print(f"  Contacts ({len(contacts)}): {', '.join(contacts[:5])}")

print(f"\nFinal: furthest_x = {story.furthest_x:.3f} m, phase = {story.phase}")
