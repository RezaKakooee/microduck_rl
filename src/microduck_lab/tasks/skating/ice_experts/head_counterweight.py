"""head_counterweight: one-leg ice glide, balancing on the support foot's edge.

Finding that shaped this controller (measured with the harness, see notes at
the bottom): with hip_roll at +0.384 the support sole is tilted 22 deg, so the
robot does NOT stand on a flat foot. It pivots on the outer sole edge, which
sits ~29 mm to the RIGHT of the CoM at t=0. Nothing the head can do moves the
CoM more than ~1.5 mm (head_roll: 0.5 mm per 0.4 rad, head_yaw: 1.5 mm), so a
pure head counterweight cannot stop the fall.

What does move the pivot: yawing the support leg and pitching its foot so the
robot rides on a sole CORNER instead of the edge, plus swinging the free leg
and head to the support side. A static pose search puts the pivot a few mm
LEFT of the CoM. The controller commands that pose (two phases, with servo
overdrive so the slow XL330 model moves at max torque), then regulates roll
with the support hip_roll (moves the pivot sideways) and pitch with the
support ankle (moves the pivot fore-aft).
"""
import sys

import numpy as np

from microduck_lab.tasks.skating.ice_experts.harness import evaluate, BALANCED_POSE, CONTROL_DT  # noqa: E402

import mujoco  # noqa: E402

L_HIP_YAW, L_HIP_ROLL, L_HIP_PITCH, L_KNEE, L_ANKLE = 0, 1, 2, 3, 4
NECK_PITCH, HEAD_PITCH, HEAD_YAW, HEAD_ROLL = 5, 6, 7, 8
R_HIP_YAW, R_HIP_ROLL, R_HIP_PITCH, R_KNEE, R_ANKLE = 9, 10, 11, 12, 13

JNT_LO = np.array([-0.436, -0.384, -1.571, -1.571, -1.571, -1.571, -1.571, -2.967, -0.436,
                   -0.524, -0.384, -1.571, -1.571, -1.571])
JNT_HI = np.array([0.524, 0.384, 1.571, 1.571, 1.571, 1.047, 1.571, 2.967, 0.436,
                   0.436, 0.384, 1.571, 1.571, 1.571])

# Pose from the static search (pivot 3 mm left of CoM at upright trunk).
STATIC_POSE = np.array([-0.436, 0.384, 1.084, -1.059, 0.472,
                        1.047, -0.589, -1.423, -0.436,
                        0.436, 0.384, 0.546, 0.806, -1.233])

DEFAULT = dict(
    pose_a=STATIC_POSE,      # commanded from t=0
    pose_b=STATIC_POSE,      # commanded after t_switch
    t_switch=10.0,
    od=0.0,                  # servo overdrive: cmd = pose + od*(pose - q)
    kp_y=0.0, kd_y=0.0,      # hip_roll feedback on lateral pivot error (mm) / roll rate
    kp_x=0.0, kd_x=0.0,      # ankle feedback on fore-aft pivot error (mm) / pitch rate
    rate=100.0,              # max target change per second (rad/s)
)


class _Foot:
    """Lowest point of the support foot mesh (the pivot), in world frame."""

    def __init__(self, model):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_collision")
        mid = model.geom_dataid[gid]
        va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        self.gid = gid
        self.verts = model.mesh_vert[va:va + vn]

    def pivot(self, data):
        R = data.geom_xmat[self.gid].reshape(3, 3)
        w = self.verts @ R.T + data.geom_xpos[self.gid]
        zmin = w[:, 2].min()
        return w[w[:, 2] < zmin + 0.0005].mean(0)


def make_controller(p=None):
    p = {**DEFAULT, **(p or {})}
    pose_a = np.clip(np.asarray(p["pose_a"], dtype=np.float64), JNT_LO, JNT_HI)
    pose_b = np.clip(np.asarray(p["pose_b"], dtype=np.float64), JNT_LO, JNT_HI)
    st = {"foot": None, "cmd": None}

    def ctrl(s):
        if st["foot"] is None:
            st["foot"] = _Foot(s["model"])
            st["cmd"] = s["q"].copy()
        piv = st["foot"].pivot(s["data"])
        e = (piv - s["com"]) * 1000.0             # mm, + = pivot left of / ahead of CoM
        gx, gy = s["gyro"][0], s["gyro"][1]      # trunk angular velocity
        pose = pose_a if s["t"] < p["t_switch"] else pose_b
        des = pose.copy()
        # roll: e[1] > 0 means toppling right -> move pivot right -> less hip_roll
        des[R_HIP_ROLL] = pose[R_HIP_ROLL] - p["kp_y"] * e[1] - p["kd_y"] * gx
        # pitch: e[0] > 0 means pivot ahead of CoM -> toppling backward
        des[R_ANKLE] = pose[R_ANKLE] + p["kp_x"] * e[0] + p["kd_x"] * gy
        des = np.clip(des, JNT_LO, JNT_HI)
        step = p["rate"] * CONTROL_DT
        des = st["cmd"] + np.clip(des - st["cmd"], -step, step)
        st["cmd"] = des
        cmd = des + p["od"] * (des - s["q"])
        return np.clip(cmd, -10.0, 10.0)

    return ctrl


def run(p=None, **kw):
    return evaluate(make_controller(p), **kw)


BEST = {}

if __name__ == "__main__":
    hold = run(BEST, verbose="-v" in sys.argv)
    print(f"HOLD {hold:.2f}")
