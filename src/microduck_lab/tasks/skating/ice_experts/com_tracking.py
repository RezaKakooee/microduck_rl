"""Whole-body centre-of-mass tracking for the one-legged ice glide.

Each control step:
  1. find the real support point: the mean of the floor contacts on the support
     foot (the foot has no ankle-roll joint, so with hip_roll at its limit the
     sole is tilted ~22 deg and rests on its outer edge, ~15 mm from the site);
  2. CoM error e = com - support_point (x, y) minus a tuned offset, and its
     rate (finite difference);
  3. numerical Jacobian dCoM/dq for a chosen joint set by finite differences
     with mujoco.mj_kinematics on a scratch MjData (qpos copy);
  4. damped least squares:  dq = W J^T (J W J^T + lam I)^-1 (-Kp e - Kd de),
     rate-limited, integrated into the joint targets, clipped to jnt_range and
     to the harness' own limits (free hip_pitch > 0.6, etc.).

Run:  uv run python src/microduck_lab/tasks/skating/ice_experts/com_tracking.py           -> HOLD <s>
      uv run python src/microduck_lab/tasks/skating/ice_experts/com_tracking.py --search N  (random search)
"""

import json
import time

import numpy as np
import mujoco

from microduck_lab.tasks.skating.ice_experts.harness import evaluate, BALANCED_POSE, CONTROL_DT  # noqa: F401

# joint indices
L_HY, L_HR, L_HP, L_KN, L_AN = 0, 1, 2, 3, 4
NECK_P, HEAD_P, HEAD_Y, HEAD_R = 5, 6, 7, 8
R_HY, R_HR, R_HP, R_KN, R_AN = 9, 10, 11, 12, 13

# joints the Jacobian controller may move (support hip_roll first: -95 mm/rad lateral)
JAC_JOINTS = [R_HR, R_AN, R_HP, R_KN, HEAD_Y, HEAD_R, L_HY, R_HY, NECK_P, L_HR]

# --------------------------------------------------------------------------
# best configuration found by search (updated by --search; see bottom)
BEST_PARAMS = {
    "kp": 1.0,
    "kd": 0.1,
    "lam": 0.00010896541078547639,
    "rate": 0.05,
    "w": [
        1.967101556954169,
        0.17183392193909477,
        0.2944138631432196,
        0.290115726363847,
        3.0,
        0.5859105544372039,
        3.0,
        1.3180670550452513,
        0.04219628925222879,
        0.8276629315801308
    ],
    "x_des": -0.039877106048141875,
    "y_des": -0.011158422747701655,
    "pre": {
        "head_y": 0,
        "l_hy": 0,
        "r_hy": 0,
        "l_hp": 0,
        "l_kn": 0,
        "l_hr": 0,
        "neck_p": 0,
        "r_an": -0.95,
        "r_kn": 1.0,
        "r_hp": 0.34
    },
    "roll_kp": 0,
    "roll_kd": 0,
    "speed": 0.4,
    "hip_roll_shift": 0.384
}
# --------------------------------------------------------------------------


class ComTracker:
    def __init__(self, p):
        self.p = p
        self.scratch = None
        self.model_id = None
        self.cmd = None
        self.e_prev = None
        self.sup_prev = None
        self.lo = None
        self.hi = None
        self.n_roll_prev = 0.0

    # ---- helpers ---------------------------------------------------------
    def _setup(self, s):
        model = s["model"]
        self.model_id = id(model)
        self.scratch = mujoco.MjData(model)
        self.mass = model.body_mass.copy()
        self.mtot = self.mass.sum()
        self.sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
        self.foot_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_collision")
        self.floor_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        # qpos address of each policy joint, and joint ranges
        self.qadr = []
        self.lo = np.zeros(14)
        self.hi = np.zeros(14)
        jids = {int(model.jnt_qposadr[j]): j for j in range(model.njnt)}
        # policy joints are the hinge joints in order (free joint first)
        hinge = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        for i, j in enumerate(hinge[:14]):
            self.qadr.append(int(model.jnt_qposadr[j]))
            self.lo[i], self.hi[i] = model.jnt_range[j]
        # harness limits
        self.lo[L_HP] = max(self.lo[L_HP], 0.65)
        # initial command = balanced pose + presets
        c = s["target"].copy()
        pre = self.p["pre"]
        c[HEAD_Y] += pre.get("head_y", 0.0)
        c[L_HY] += pre.get("l_hy", 0.0)
        c[R_HY] += pre.get("r_hy", 0.0)
        c[L_HP] += pre.get("l_hp", 0.0)
        c[L_KN] += pre.get("l_kn", 0.0)
        c[L_HR] += pre.get("l_hr", 0.0)
        c[NECK_P] += pre.get("neck_p", 0.0)
        c[R_AN] += pre.get("r_an", 0.0)
        c[R_KN] += pre.get("r_kn", 0.0)
        c[R_HP] += pre.get("r_hp", 0.0)
        self.cmd = np.clip(c, self.lo, self.hi)
        self.e_prev = None

    def _com(self, d):
        return np.sum(self.mass[:, None] * d.xipos, axis=0) / self.mtot

    def _support_point(self, s):
        """Mean of floor/support-foot contact points; fall back to the site."""
        d = s["data"]
        pts = []
        for k in range(d.ncon):
            c = d.contact[k]
            g = {c.geom1, c.geom2}
            if self.foot_geom in g and self.floor_geom in g:
                pts.append(np.array(c.pos))
        if pts:
            sp = np.mean(pts, axis=0)
            self.sup_prev = sp - s["support_foot"]   # remember offset from site
            return sp
        if self.sup_prev is not None:
            return s["support_foot"] + self.sup_prev
        return s["support_foot"]

    def _jacobian(self, s):
        """dCoM_rel_support/dq (2 x n) by finite differences, on a scratch data."""
        model, data = s["model"], s["data"]
        d2 = self.scratch
        d2.qpos[:] = data.qpos
        mujoco.mj_kinematics(model, d2)
        base = self._com(d2) - d2.site_xpos[self.sid]
        eps = 1e-3
        J = np.zeros((2, len(JAC_JOINTS)))
        for k, j in enumerate(JAC_JOINTS):
            qi = self.qadr[j]
            q0 = d2.qpos[qi]
            d2.qpos[qi] = q0 + eps
            mujoco.mj_kinematics(model, d2)
            c = self._com(d2) - d2.site_xpos[self.sid]
            J[:, k] = (c[:2] - base[:2]) / eps
            d2.qpos[qi] = q0
        return J

    # ---- controller --------------------------------------------------------
    def __call__(self, s):
        if self.scratch is None or self.model_id != id(s["model"]):
            self._setup(s)
        p = self.p
        sp = self._support_point(s)
        e = (s["com"] - sp)[:2] - np.array([p["x_des"], p["y_des"]])
        if self.e_prev is None:
            de = np.zeros(2)
        else:
            de = (e - self.e_prev) / CONTROL_DT
        self.e_prev = e.copy()

        want = -(p["kp"] * e + p["kd"] * de) * CONTROL_DT   # desired CoM step this tick (m)
        J = self._jacobian(s)
        w = np.array(p["w"], dtype=float)
        lo = self.lo[JAC_JOINTS] - self.cmd[JAC_JOINTS]   # room below / above the current command
        hi = self.hi[JAC_JOINTS] - self.cmd[JAC_JOINTS]
        # saturation-aware damped least squares: joints pushed into a limit are
        # dropped (weight 0) and the rest re-solved, so the pinned support
        # hip_roll does not swallow the whole correction.
        dq = np.zeros(len(JAC_JOINTS))
        for _ in range(len(JAC_JOINTS)):
            W = np.diag(w)
            A = J @ W @ J.T + p["lam"] * np.eye(2)
            dq = W @ J.T @ np.linalg.solve(A, want)
            dq = np.clip(dq, -p["rate"], p["rate"])
            sat = ((dq > 1e-6) & (dq > hi + 1e-9)) | ((dq < -1e-6) & (dq < lo - 1e-9))
            sat &= w > 0
            if not sat.any():
                break
            w = np.where(sat, 0.0, w)
        dq = np.clip(dq, lo, hi)

        c = self.cmd.copy()
        for k, j in enumerate(JAC_JOINTS):
            c[j] += dq[k]
        # extra trunk-roll feedback on the support hip roll (rolls the trunk over the foot)
        c[R_HR] += p["roll_kp"] * s["roll"] + p["roll_kd"] * s["gyro"][0]
        c = np.clip(c, self.lo, self.hi)
        self.cmd = c
        return c


def run(params, seconds=10.0, verbose=False):
    """Harness score (seconds held by the shared criteria)."""
    return evaluate(ComTracker(params), seconds=seconds, ice_mu=0.1, speed=params["speed"],
                    free="left", hip_roll_shift=params["hip_roll_shift"], verbose=verbose)


TILT_MAX = np.radians(50.0)


def run_strict(params, seconds=10.0, verbose=False):
    """(harness_hold, strict_hold).  strict_hold stops counting as soon as
    anything but the support foot touches the floor, or the trunk tilts more
    than 50 deg.  The harness alone can be gamed by lying down with the trunk
    propped just above 70 mm; the strict number cannot."""
    ct = ComTracker(params)
    first_bad = [None]

    def wrapped(s):
        c = ct(s)
        if first_bad[0] is None:
            d, m = s["data"], s["model"]
            other = False
            for k in range(d.ncon):
                g = {int(d.contact[k].geom1), int(d.contact[k].geom2)}
                if ct.floor_geom in g and ct.foot_geom not in g:
                    other = True
                    break
            tilt = max(abs(s["roll"]), abs(s["pitch"]))
            if other or tilt > TILT_MAX:
                first_bad[0] = s["t"]
        return c

    h = evaluate(wrapped, seconds=seconds, ice_mu=0.1, speed=params["speed"],
                 free="left", hip_roll_shift=params["hip_roll_shift"], verbose=verbose)
    strict = h if first_bad[0] is None else min(h, max(0.0, first_bad[0] - CONTROL_DT))
    return h, strict


# --------------------------------------------------------------------------
# search
def _perturb(p, rng, scale):
    q = json.loads(json.dumps(p))
    def lg(v, s, lo, hi):
        return float(np.clip(v * np.exp(rng.normal(0, s)), lo, hi))
    q["kp"] = lg(q["kp"], scale, 0.1, 100)
    q["kd"] = lg(q["kd"], scale, 0.0, 20)
    q["lam"] = lg(q["lam"], scale, 1e-7, 1e-1)
    q["rate"] = lg(q["rate"], scale, 0.01, 1.0)
    q["w"] = [lg(w, scale, 0.0, 3.0) for w in q["w"]]
    q["x_des"] = float(np.clip(q["x_des"] + rng.normal(0, 0.01 * scale), -0.04, 0.02))
    q["y_des"] = float(np.clip(q["y_des"] + rng.normal(0, 0.01 * scale), -0.04, 0.04))
    lim = {"head_y": 2.9, "l_hy": 0.43, "r_hy": 0.43, "l_hp": 0.45, "l_kn": 1.2, "l_hr": 0.7, "neck_p": 0.8,
           "r_an": 1.1, "r_kn": 1.4, "r_hp": 1.0}
    for k in lim:
        q["pre"][k] = float(np.clip(q["pre"].get(k, 0.0) + rng.normal(0, 0.3 * scale), -lim[k], lim[k]))
    q["roll_kp"] = float(np.clip(q["roll_kp"] + rng.normal(0, 1.0 * scale), -10, 10))
    q["roll_kd"] = float(np.clip(q["roll_kd"] + rng.normal(0, 0.2 * scale), -3, 3))
    q["speed"] = float(np.clip(q["speed"] + rng.normal(0, 0.1 * scale), 0.0, 1.0))
    q["hip_roll_shift"] = float(np.clip(q["hip_roll_shift"] + rng.normal(0, 0.03 * scale), 0.25, 0.384))
    return q


def search(n, seed=0, start=None, fixed_speed=None, fixed_shift=None):
    rng = np.random.default_rng(seed)
    best = json.loads(json.dumps(start or BEST_PARAMS))
    if fixed_speed is not None:
        best["speed"] = fixed_speed
    if fixed_shift is not None:
        best["hip_roll_shift"] = fixed_shift
    best_h = run_strict(best)[1]
    print(f"start strict hold {best_h:.2f}")
    scale = 1.0
    since = 0
    for i in range(n):
        cand = _perturb(best, rng, scale)
        if fixed_speed is not None:
            cand["speed"] = fixed_speed
        if fixed_shift is not None:
            cand["hip_roll_shift"] = fixed_shift
        hh, h = run_strict(cand)
        if h > best_h + 1e-9:
            best, best_h, since = cand, h, 0
            print(f"[{i}] strict {h:.2f} (harness {hh:.2f})  <- new best  {json.dumps(best)}")
        else:
            since += 1
            if since % 40 == 0:
                scale = max(0.3, scale * 0.7)
    print("BEST", best_h, json.dumps(best))
    return best, best_h


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--search":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        fs = float(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != "-" else None
        fsh = float(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] != "-" else None
        start = json.load(open(sys.argv[6])) if len(sys.argv) > 6 else None
        t0 = time.time()
        search(n, seed, start=start, fixed_speed=fs, fixed_shift=fsh)
        print(f"search took {time.time()-t0:.0f}s")
    else:
        h, strict = run_strict(BEST_PARAMS, verbose=True)
        print(f"strict hold (support foot only on floor, trunk tilt < 50 deg): {strict:.2f}s")
        print(f"HOLD {h:.2f}")
