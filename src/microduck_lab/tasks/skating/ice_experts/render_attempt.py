"""Render the one-legged ice-glide attempt from behind, so the roll is visible."""
import os, sys
import numpy as np
from microduck_lab.sim import duck_sim
from microduck_lab.tasks.skating.ice_experts.harness import make, state, CONTROL_DT, DECIMATION, R_HIP_ROLL, R_ANKLE
import mujoco

out = sys.argv[1] if len(sys.argv) > 1 else "videos/ice_experts/ice_onelegged_attempt.mp4"
# MuJoCo azimuth: 90 = camera on the duck's LEFT (+y), 270 = on its RIGHT, 180 = behind.
azimuth = float(sys.argv[2]) if len(sys.argv) > 2 else 180.0
model, data, policy, adr, target = make(ice_mu=0.1, speed=0.4)
rec = duck_sim.Recorder(model, distance=0.6, azimuth=azimuth, elevation=-5.0)
prev = {"roll": 0.0, "pitch": 0.0}
for i in range(int(5.0 / CONTROL_DT)):
    s = state(model, data, policy, adr, target, i * CONTROL_DT, "left")
    c = s["target"].copy()
    c[R_HIP_ROLL] += 1.5 * (s["roll"] - prev["roll"]) / CONTROL_DT
    c[R_ANKLE] += 4.0 * s["pitch"] + 0.3 * (s["pitch"] - prev["pitch"]) / CONTROL_DT
    prev["roll"], prev["pitch"] = s["roll"], s["pitch"]
    data.ctrl[:14] = c
    for _ in range(DECIMATION):
        mujoco.mj_step(model, data)
    rec.maybe_capture(i, data)
print("frames", rec.write(out), "->", out)
