"""lqr_pendulum: model-based linear state feedback for the one-leg ice glide.

Model. About the support contact the robot is an inverted pendulum in roll
and in pitch: m = 0.74 kg, CoM height h ~ 0.155 m above the contact, so
w0^2 = g/h ~ 63 s^-2. On ice (mu = 0.1) the ground cannot supply lateral
force, so the only real inputs are ones that MOVE the contact under the
CoM or move mass:
  roll : support hip_roll (idx 10) swings the foot laterally, ~0.10 m/rad;
         head_roll (idx 8) is a (tiny) reaction-wheel term.
  pitch: support ankle (idx 13) shifts the pressure point along the 54 mm
         sole; support hip_pitch (idx 11) moves the CoM fore-aft, ~0.09 m/rad.
For each axis the linearised plant is x' = A x + B u with
  A = [[0, 1], [w0^2, 0]],  B = [0, -w0^2 * L / h]   (L = lever, m/rad)
and the LQR gain K = R^-1 B^T P from the continuous Riccati equation seeds
the gain matrix. The full gain matrix (plus a few feed-forward joint offsets
and a slew-rate limit that keeps the XL330s out of current saturation) is
then tuned numerically against harness.evaluate().

Honest physics note (measured in this file's development, see the report):
with hip_roll = +0.384 and a level trunk the sole is tilted ~20 deg because
the ankle has no roll joint, so the foot rests on its OUTER edge at
y = -27 mm while the CoM is at y = +2 mm. The pose is not statically
balanced, and hip_roll is already at its joint limit in the direction that
would bring the foot under the CoM. No feedback law can hold it; the
controller below only slows the tip-over.

Run:   uv run python src/microduck_lab/tasks/skating/ice_experts/lqr_pendulum.py            (best config, prints HOLD <s>)
       uv run python src/microduck_lab/tasks/skating/ice_experts/lqr_pendulum.py --search N (random search, N rollouts)
       uv run python src/microduck_lab/tasks/skating/ice_experts/lqr_pendulum.py --debug    (verbose trace + gyro sign check)
"""
import sys

import mujoco
import numpy as np

from microduck_lab.tasks.skating.ice_experts.harness import evaluate, BALANCED_POSE, CONTROL_DT  # noqa: E402

# ---------------------------------------------------------------- joints
L_HIP_ROLL, L_HIP_PITCH, L_KNEE = 1, 2, 3
NECK_PITCH, HEAD_YAW, HEAD_ROLL = 5, 7, 8
R_HIP_YAW, R_HIP_ROLL, R_HIP_PITCH, R_KNEE, R_ANKLE = 9, 10, 11, 12, 13
JOINT_LO = np.array([-0.436, -0.384, -1.571, -1.571, -1.571, -1.571, -1.571, -2.967, -0.436,
                     -0.524, -0.384, -1.571, -1.571, -1.571])
JOINT_HI = np.array([0.524, 0.384, 1.571, 1.571, 1.571, 1.047, 1.571, 2.967, 0.436,
                     0.436, 0.384, 1.571, 1.571, 1.571])

# ---------------------------------------------------------------- pendulum model
M_TOTAL = 0.737          # kg (model.body_mass.sum())
H_COM = 0.155            # m, CoM above the contact (measured: com z 0.155, contact z ~0)
G = 9.81
W0SQ = G / H_COM         # ~63 s^-2
LEVER_ROLL = 0.103       # m/rad: support hip_roll moves the foot laterally (measured Jacobian)
LEVER_PITCH_HIP = 0.091  # m/rad: support hip_pitch moves CoM fore-aft (measured Jacobian)
LEVER_ANKLE = 0.023      # m/rad: ankle moves CoM fore-aft (measured Jacobian)


def lqr_gain(lever, q_ang=1.0, q_rate=0.1, r=1.0):
    """Continuous LQR for x'' = w0^2 (x - lever*u/h). Returns K (1x2): u = -K x."""
    from scipy.linalg import solve_continuous_are
    A = np.array([[0.0, 1.0], [W0SQ, 0.0]])
    B = np.array([[0.0], [-W0SQ * lever / H_COM]])
    Q = np.diag([q_ang, q_rate])
    R = np.array([[r]])
    P = solve_continuous_are(A, B, Q, R)
    return (np.linalg.solve(R, B.T @ P)).ravel()


# ---------------------------------------------------------------- parameter vector
# Feedback gains (u = K x) and feed-forward offsets, all in one flat vector so the
# numeric search can move them together.
PARAM_NAMES = [
    "k10_roll", "k10_rollrate", "k10_comy",          # support hip_roll
    "k8_roll", "k8_rollrate",                         # head_roll
    "k13_pitch", "k13_pitchrate", "k13_comx",         # support ankle
    "k11_pitch", "k11_pitchrate", "k11_comx",         # support hip_pitch
    "ff_knee", "ff_hip_pitch", "ff_head_yaw", "ff_free_hip_pitch", "ff_free_knee",
    "ff_hip_yaw", "ff_free_hip_roll", "ff_neck_pitch", "ff_ankle",
    "slew",                                           # rad/s, command slew limit
]
N_PARAMS = len(PARAM_NAMES)


def default_params():
    kr = lqr_gain(LEVER_ROLL, q_ang=1.0, q_rate=0.05, r=1.0)
    kp = lqr_gain(LEVER_ANKLE, q_ang=1.0, q_rate=0.05, r=1.0)
    p = np.zeros(N_PARAMS)
    d = dict(zip(PARAM_NAMES, range(N_PARAMS)))
    # sign: positive roll = trunk top toward +y (falling to the free-leg side).
    # Recovering needs the foot to move +y, i.e. hip_roll UP; LQR sign is u=-Kx with B<0 -> K<0.
    p[d["k10_roll"]], p[d["k10_rollrate"]] = -kr[0], -kr[1]
    p[d["k10_comy"]] = 5.0            # rad per m of CoM lateral error
    p[d["k13_pitch"]], p[d["k13_pitchrate"]] = -kp[0], -kp[1]
    p[d["k13_comx"]] = 5.0
    p[d["k11_pitch"]], p[d["k11_pitchrate"]] = 0.5, 0.05
    p[d["slew"]] = 10.0
    return p


BEST_PARAMS = None  # filled in below after search (see bottom of file)
BEST_KW = dict(hip_roll_shift=0.384, speed=0.4)


def make_controller(params):
    d = dict(zip(PARAM_NAMES, params))
    ff = {
        R_KNEE: d["ff_knee"], R_HIP_PITCH: d["ff_hip_pitch"], HEAD_YAW: d["ff_head_yaw"],
        L_HIP_PITCH: d["ff_free_hip_pitch"], L_KNEE: d["ff_free_knee"], R_HIP_YAW: d["ff_hip_yaw"],
        L_HIP_ROLL: d["ff_free_hip_roll"], NECK_PITCH: d["ff_neck_pitch"], R_ANKLE: d["ff_ankle"],
    }
    slew = max(0.5, d["slew"]) * CONTROL_DT
    mem = {"prev": None, "roll": 0.0, "pitch": 0.0, "illegal_t": None, "floor": None, "ok": None}

    def ctrl(s):
        base = s["target"]
        # Honest-contact watchdog: the harness only checks the free-foot SITE height, so a
        # tilted free foot (or a knee) can rest on the floor and still "count". We refuse
        # that: record the first time anything but the support foot touches the floor.
        model, data = s["model"], s["data"]
        if mem["floor"] is None:
            mem["floor"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
            mem["ok"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_collision")
        if mem["illegal_t"] is None:
            for c in range(data.ncon):
                g1, g2 = data.contact[c].geom1, data.contact[c].geom2
                if mem["floor"] in (g1, g2) and mem["ok"] not in (g1, g2):
                    mem["illegal_t"] = s["t"]
                    break
        roll, pitch = s["roll"], s["pitch"]
        # rates: gyro is the trunk angular velocity in the trunk frame; measured roll
        # is -rotation about x, pitch is +rotation about y (checked in --debug).
        roll_rate = -float(s["gyro"][0])
        pitch_rate = float(s["gyro"][1])
        e = s["com_rel_support"]
        com_y, com_x = float(e[1]), float(e[0])

        want = base.copy()
        for j, v in ff.items():
            want[j] += v
        want[R_HIP_ROLL] += d["k10_roll"] * roll + d["k10_rollrate"] * roll_rate + d["k10_comy"] * com_y
        want[HEAD_ROLL] += d["k8_roll"] * roll + d["k8_rollrate"] * roll_rate
        want[R_ANKLE] += d["k13_pitch"] * pitch + d["k13_pitchrate"] * pitch_rate + d["k13_comx"] * com_x
        want[R_HIP_PITCH] += d["k11_pitch"] * pitch + d["k11_pitchrate"] * pitch_rate + d["k11_comx"] * com_x
        want = np.clip(want, JOINT_LO, JOINT_HI)

        prev = mem["prev"] if mem["prev"] is not None else base.copy()
        cmd = prev + np.clip(want - prev, -slew, slew)
        mem["prev"] = cmd
        return cmd

    ctrl.mem = mem
    return ctrl


def run(params, verbose=False, honest=True, **kw):
    """Hold time. With honest=True the hold is cut at the first moment any body part
    other than the support foot touches the floor (stricter than the harness)."""
    kwargs = dict(BEST_KW)
    kwargs.update(kw)
    ctrl = make_controller(params)
    held = evaluate(ctrl, seconds=10.0, ice_mu=0.1, free="left", verbose=verbose, **kwargs)
    bad = ctrl.mem["illegal_t"]
    if honest and bad is not None:
        if verbose:
            print(f"  (illegal floor contact by a non-support body at {bad:.2f}s -> honest hold {min(held, bad):.2f}s)")
        held = min(held, max(0.0, bad - CONTROL_DT))
    return held


# ---------------------------------------------------------------- search
def search(n_rollouts=300, seed=0, start=None, start_kw=None, log=print):
    """(1+lambda)-style Gaussian random search on the parameter vector and hip_roll_shift."""
    rng = np.random.default_rng(seed)
    best_p = default_params() if start is None else np.array(start, dtype=float)
    best_kw = dict(BEST_KW if start_kw is None else start_kw)
    best = run(best_p, **best_kw)
    log(f"start hold={best:.2f}")
    scale = np.array([
        3.0, 0.3, 5.0, 3.0, 0.3, 3.0, 0.3, 5.0, 3.0, 0.3, 5.0,
        0.4, 0.4, 1.0, 0.3, 0.4, 0.3, 0.3, 0.4, 0.3,
        5.0,
    ])
    step = 1.0
    since_improve = 0
    for i in range(n_rollouts):
        cand = best_p + step * scale * rng.standard_normal(N_PARAMS) * (rng.random(N_PARAMS) < 0.4)
        cand[-1] = np.clip(cand[-1], 1.0, 40.0)
        kw = dict(best_kw)
        if rng.random() < 0.3:
            kw["hip_roll_shift"] = float(np.clip(best_kw["hip_roll_shift"] + 0.05 * rng.standard_normal(), 0.25, 0.384))
        h = run(cand, **kw)
        if h > best + 1e-9:
            best, best_p, best_kw = h, cand, kw
            since_improve = 0
            log(f"[{i}] hold={best:.2f} shift={kw['hip_roll_shift']:.3f} step={step:.2f}")
        else:
            since_improve += 1
            if since_improve >= 25:
                step *= 0.7
                since_improve = 0
    log("best params:", np.array2string(best_p, precision=4, separator=", "))
    log("best kw:", best_kw, "hold:", best)
    return best, best_p, best_kw


# ---------------------------------------------------------------- best configuration found
# From the numeric search with the honest objective (600 + 200 rollouts, seeds 2-4).
# Honest hold 0.22 s vs 0.18 s for the zero controller. The search with the raw
# harness objective found "holds" up to 9.98 s, but every one of them was the robot
# squatting to 72 mm and resting its tilted free foot on the floor (the free-foot SITE
# stays 33 mm up, so the harness does not see it). Those are not glides; rejected.
BEST_PARAMS = np.array([
    3.3117, 0.4555, 5.9406,        # hip_roll: roll, roll_rate, com_y
    -0.3507, 0.0,                  # head_roll: roll, roll_rate
    10.9591, 1.7135, 5.0,          # ankle: pitch, pitch_rate, com_x
    4.8662, 0.1627, -3.503,        # hip_pitch: pitch, pitch_rate, com_x
    0.569, 0.0711, -1.7506, 0.3887, 0.4044, 0.44, 0.0, 0.0868, 0.0,   # feed-forward offsets
    10.0,                          # slew limit rad/s
])

if __name__ == "__main__":
    if "--debug" in sys.argv:
        # Gyro sign check: compare gyro to finite-difference of measured roll/pitch.
        prev = {"r": None, "p": None}

        def dbg(s):
            if prev["r"] is not None:
                fr = (s["roll"] - prev["r"]) / CONTROL_DT
                fp = (s["pitch"] - prev["p"]) / CONTROL_DT
                print(f"t={s['t']:.2f} droll_fd={fr:+.3f} gyro={s['gyro']} dpitch_fd={fp:+.3f}")
            prev["r"], prev["p"] = s["roll"], s["pitch"]
            return s["target"].copy()
        evaluate(dbg)
        print("default params:", dict(zip(PARAM_NAMES, np.round(default_params(), 3))))
        run(BEST_PARAMS, verbose=True)
    elif "--search" in sys.argv:
        n = int(sys.argv[sys.argv.index("--search") + 1]) if len(sys.argv) > sys.argv.index("--search") + 1 else 300
        seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0
        search(n, seed=seed)
    else:
        raw = run(BEST_PARAMS, honest=False)
        hold = run(BEST_PARAMS)
        print(f"harness score (site-based criteria): {raw:.2f} s")
        print(f"honest hold (also requires no floor contact except the support foot): {hold:.2f} s")
        print(f"HOLD {hold:.2f}")
