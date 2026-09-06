"""Shared MuJoCo setup for the headless duck scripts.

`infer_policy.py` does a specific sequence of things to the model and the policy
before its loop -- override the timestep, clamp actuator force to the servo's
current limit, stand the robot in DEFAULT_POSE, and ask for projected gravity
rather than the raw accelerometer. Miss any one and the policy falls over within
a couple of seconds for reasons that look like physics. This module is that
sequence, in one place, so the scripts here cannot drift from it.
"""

import os

import numpy as np

import mujoco

from microduck_lab import paths
from microduck_lab.sim.upstream import (
    PolicyInference, MICRODUCK_XML, MICRODUCK_BALL_XML, MICRODUCK_ROLLERS_XML,
)

# Kept as a string because callers join onto it; see microduck_lab.paths for
# the Path objects.
REPO = str(paths.REPO)

# infer_policy.py:882 overrides the XML's 0.002 s so decimation 4 lands on the
# 50 Hz the policies were trained at.
CONTROL_TIMESTEP = 0.005
DECIMATION = 4
CONTROL_DT = DECIMATION * CONTROL_TIMESTEP

# The trunk stands at z = 0.12 m; a fall puts it well under. Note this is a
# WALKING fall test -- a roulade or a sit goes below it on purpose.
FALL_HEIGHT = 0.07

# What every alpha policy drives: 14 actions, 61-dim observation.
POLICY_JOINTS = 14

SCENES = {
    "walk": MICRODUCK_XML,
    "ball": MICRODUCK_BALL_XML,
    "rollers": MICRODUCK_ROLLERS_XML,
}


def load_scene(scene="walk", current_limit=1.75, roller_friction=False):
    """Load a scene and apply the model-level overrides infer_policy.py applies."""
    xml = SCENES.get(scene, scene)
    model = mujoco.MjModel.from_xml_path(xml if os.path.isabs(xml) else os.path.join(REPO, xml))
    model.opt.timestep = CONTROL_TIMESTEP

    # The XL330 saturates current at ~1.75 A, so torque saturates at kt * I_max.
    # Clipping the position actuators' force reproduces the saturation the policy
    # trained against.
    if current_limit and current_limit > 0:
        from bam.model import load_model
        kt = load_model(motor_name="xl330", model="m6").kt.value
        limit = kt * current_limit
        model.actuator_forcerange[:, 0] = -limit
        model.actuator_forcerange[:, 1] = limit
        model.actuator_forcelimited[:] = 1

    # Wheel bearing friction is set here rather than in the XML because a
    # non-zero frictionloss in the model breaks training.
    if roller_friction:
        import re
        for j in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if name and re.match(r"^passive_.*", name):
                model.dof_frictionloss[model.jnt_dofadr[j]] = 0.003

    return model, mujoco.MjData(model)


def make_policy(model, data, **onnx_paths):
    """Build a PolicyInference and stand the duck up in its default pose.

    `use_projected_gravity` is forced True: the class defaults it to False but
    main() passes True, and at the default the policy reads the raw
    accelerometer where it expects projected gravity.
    """
    policy = PolicyInference(
        model, data,
        new_cmd_obs=True,
        use_projected_gravity=True,
        **onnx_paths,
    )
    # PolicyInference sizes itself from model.nu. add_mouth.py appends a 15th
    # actuator, which would push the mouth into the observation and make it
    # 64-dim -- the alpha policies are 61-dim / 14-action. The mouth is last, so
    # trimming to the first 14 restores the contract; the mouth is then ours to
    # drive directly, which is what robotd does on the real robot.
    if policy.n_joints > POLICY_JOINTS:
        policy.n_joints = POLICY_JOINTS
        policy.joint_qpos_indices = policy.joint_qpos_indices[:POLICY_JOINTS]
        policy.joint_qvel_indices = policy.joint_qvel_indices[:POLICY_JOINTS]
        policy.last_action = np.zeros(POLICY_JOINTS, dtype=np.float32)

    policy.vel_max_x, policy.vel_min_x = 0.3, -0.3
    policy.vel_max_y, policy.vel_min_y = 0.2, -0.2
    policy.vel_max_ang = 1.5

    freejoint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    adr = int(model.jnt_qposadr[freejoint])
    data.qpos[adr + 0] = 0.0
    data.qpos[adr + 1] = 0.0
    data.qpos[adr + 2] = 0.125
    data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
    for i, qpos_idx in enumerate(policy.joint_qpos_indices):
        data.qpos[qpos_idx] = policy.default_pose[i]
    data.ctrl[:POLICY_JOINTS] = policy.default_pose
    mujoco.mj_forward(model, data)
    return policy, adr


def trunk_yaw(data, adr):
    """Yaw of the trunk, from its freejoint quaternion."""
    qw, qx, qy, qz = data.qpos[adr + 3:adr + 7]
    return float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))


class Recorder:
    """Offscreen renderer with a camera that follows the trunk.

    Needs a GL stack MuJoCo can open: MUJOCO_GL=egl on a GPU. The scenes
    define no camera of their own, and the default free camera sits still while
    a walking duck leaves the frame.
    """

    def __init__(self, model, fps=30, width=640, height=480, distance=0.9,
                 azimuth=130.0, elevation=-15.0):
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        # Capture every `every` control steps. At 50 Hz control and 30 fps that
        # rounds to every 2nd step = 25 fps, so the file must be written at the
        # rate frames were actually captured, or every clip plays 1.2x fast.
        self.every = max(1, int(round(1.0 / (fps * CONTROL_DT))))
        self.fps = 1.0 / (self.every * CONTROL_DT)
        self.frames = []
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        self.cam.distance = distance
        self.cam.azimuth = azimuth
        self.cam.elevation = elevation

    def maybe_capture(self, step, data):
        if step % self.every == 0:
            self.renderer.update_scene(data, camera=self.cam)
            self.frames.append(self.renderer.render())

    def write(self, path):
        if not self.frames:
            return 0
        import imageio.v3 as iio
        iio.imwrite(path, np.stack(self.frames), fps=self.fps)
        return len(self.frames)
