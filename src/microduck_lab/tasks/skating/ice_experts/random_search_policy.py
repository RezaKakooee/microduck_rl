"""random_search_policy: linear feedback policy tuned by a numpy evolution strategy.

Policy:  ctrl = BALANCED_POSE + slew(clip(W @ features))   on a subset of joints.

Features (10, normalised):
    [1, roll, roll_rate(fd), pitch, pitch_rate(fd), com_rel_support x, com_rel_support y,
     trunk_vel y, gyro x, gyro y]
Joints (9): free hip_yaw, free hip_roll, head_yaw, head_roll, support hip_yaw,
    support hip_roll, support hip_pitch, support knee, support ankle.
So W is 9 x 10 = 90 gains. Column 0 of W is a constant pose offset (feed-forward).

Search: (mu, lambda) evolution strategy with rank-weighted recombination and an
adaptive step size (CMA-ES-like, diagonal only, numpy only).  Objective = harness
hold time (+ a tiny tie-break term so flat plateaus still give a gradient).

Run:   uv run python src/microduck_lab/tasks/skating/ice_experts/random_search_policy.py            -> HOLD <s>
       uv run python src/microduck_lab/tasks/skating/ice_experts/random_search_policy.py --search N [--seed S] [--workers K]
       uv run python src/microduck_lab/tasks/skating/ice_experts/random_search_policy.py -v          (verbose trace)
"""
import json
import sys

import numpy as np

from microduck_lab.tasks.skating.ice_experts.harness import evaluate, BALANCED_POSE, CONTROL_DT  # noqa: E402

L_HIP_YAW, L_HIP_ROLL, L_HIP_PITCH, L_KNEE, L_ANKLE = 0, 1, 2, 3, 4
NECK_PITCH, HEAD_PITCH, HEAD_YAW, HEAD_ROLL = 5, 6, 7, 8
R_HIP_YAW, R_HIP_ROLL, R_HIP_PITCH, R_KNEE, R_ANKLE = 9, 10, 11, 12, 13

JOINTS = [L_HIP_YAW, L_HIP_ROLL, HEAD_YAW, HEAD_ROLL, R_HIP_YAW, R_HIP_ROLL, R_HIP_PITCH, R_KNEE, R_ANKLE]
JOINT_NAMES = ["L_hy", "L_hr", "head_yaw", "head_roll", "R_hy", "R_hr", "R_hp", "R_kn", "R_an"]
FEAT_NAMES = ["1", "roll", "roll_rate", "pitch", "pitch_rate", "com_x", "com_y", "vy", "gyro_x", "gyro_y"]
N_J, N_F = len(JOINTS), len(FEAT_NAMES)

# feature scales: feature / scale ~ O(1)
FEAT_SCALE = np.array([1.0, 0.3, 3.0, 0.3, 3.0, 0.03, 0.03, 0.2, 3.0, 3.0])
# hard joint ranges (rad) so offsets never ask for impossible poses
JNT_LO = np.array([-0.436, -0.384, -1.571, -1.571, -1.571, -1.571, -1.571, -2.967, -0.436,
                   -0.524, -0.384, -1.571, -1.571, -1.571])
JNT_HI = np.array([0.524, 0.384, 1.571, 1.571, 1.571, 1.047, 1.571, 2.967, 0.436,
                   0.436, 0.384, 1.571, 1.571, 1.571])

DEFAULT_CFG = dict(
    a_max=2.0,       # |action| clip (rad) per joint, before slew
    slew=30.0,       # max change of the commanded offset (rad/s)
    overdrive=1.0,   # extra multiplier on the clipped range vs joint range (>1 = command past the limit)
    hip_roll_shift=0.384,
    speed=0.4,
)


def make_controller(W, cfg=None):
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    W = np.asarray(W, dtype=np.float64).reshape(N_J, N_F)
    st = {"prev_roll": None, "prev_pitch": None, "off": np.zeros(N_J), "roll_abs": 0.0, "n": 0}
    lo = JNT_LO * cfg["overdrive"]
    hi = JNT_HI * cfg["overdrive"]

    def ctrl(s):
        roll, pitch = s["roll"], s["pitch"]
        if st["prev_roll"] is None:
            rr, pr = 0.0, 0.0
        else:
            rr = (roll - st["prev_roll"]) / CONTROL_DT
            pr = (pitch - st["prev_pitch"]) / CONTROL_DT
        st["prev_roll"], st["prev_pitch"] = roll, pitch
        e = s["com_rel_support"]
        f = np.array([1.0, roll, rr, pitch, pr, e[0], e[1], s["trunk_vel"][1], s["gyro"][0], s["gyro"][1]])
        f = f / FEAT_SCALE
        a = np.clip(W @ f, -cfg["a_max"], cfg["a_max"])
        step = cfg["slew"] * CONTROL_DT
        st["off"] = st["off"] + np.clip(a - st["off"], -step, step)
        c = s["target"].copy()
        c[JOINTS] = np.clip(c[JOINTS] + st["off"], lo[JOINTS], hi[JOINTS])
        st["roll_abs"] += abs(roll)
        st["n"] += 1
        return c

    ctrl.state = st
    return ctrl


def run(W, cfg=None, verbose=False, shaped=False):
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    c = make_controller(W, cfg)
    hold = evaluate(c, seconds=10.0, ice_mu=0.1, speed=cfg["speed"], free="left",
                    hip_roll_shift=cfg["hip_roll_shift"], verbose=verbose)
    if not shaped:
        return hold
    st = c.state
    mean_roll = st["roll_abs"] / max(st["n"], 1)
    # tie-break only: never worth more than one control step (0.02 s)
    return hold + 0.015 * float(np.exp(-mean_roll / 0.3))


def _worker(args):
    W, cfg = args
    return run(W, cfg, shaped=True)


def search(n_rollouts=300, seed=0, workers=1, start=None, start_cfg=None, sigma0=0.3, pop=12,
           log_path=None):
    rng = np.random.default_rng(seed)
    dim = N_J * N_F
    mean = np.zeros(dim) if start is None else np.asarray(start, dtype=np.float64).reshape(dim)
    cfg = {**DEFAULT_CFG, **(start_cfg or {})}
    sigma = sigma0
    # per-coordinate scale: column 0 (bias) in rad, gains in rad per unit normalised feature
    coord = np.ones((N_J, N_F))
    coord[:, 0] = 1.0
    coord = coord.reshape(dim)
    mu = pop // 2
    wts = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    wts /= wts.sum()
    best_W, best_cfg, best = mean.copy(), dict(cfg), run(mean, cfg, shaped=True)
    curve = [(0, float(best))]
    print(f"start shaped={best:.3f}")
    pool = None
    if workers > 1:
        import multiprocessing as mp
        pool = mp.get_context("fork").Pool(workers)
    n_done = 1
    gen = 0
    stall = 0
    while n_done < n_rollouts:
        gen += 1
        eps = rng.standard_normal((pop, dim)) * coord
        cands = [mean + sigma * e for e in eps]
        cfgs = []
        for _ in range(pop):
            c2 = dict(cfg)
            if rng.random() < 0.3:
                c2["hip_roll_shift"] = float(np.clip(cfg["hip_roll_shift"] + 0.03 * rng.standard_normal(), 0.25, 0.384))
            if rng.random() < 0.2:
                c2["speed"] = float(np.clip(cfg["speed"] + 0.1 * rng.standard_normal(), 0.0, 1.0))
            cfgs.append(c2)
        jobs = list(zip(cands, cfgs))
        scores = pool.map(_worker, jobs) if pool else [_worker(j) for j in jobs]
        n_done += pop
        scores = np.asarray(scores)
        order = np.argsort(-scores)
        top = order[:mu]
        new_mean = sum(w * cands[i] for w, i in zip(wts, top))
        # cfg follows the best candidate of the generation if it improved on the current best
        improved = scores[order[0]] > best + 1e-9
        if improved:
            best, best_W, best_cfg = float(scores[order[0]]), cands[order[0]].copy(), dict(cfgs[order[0]])
            cfg = dict(best_cfg)
            stall = 0
            sigma *= 1.15
        else:
            stall += 1
            sigma *= 0.9
        mean = 0.5 * mean + 0.5 * new_mean if not improved else 0.3 * mean + 0.7 * best_W
        sigma = float(np.clip(sigma, 0.02, 1.0))
        curve.append((n_done, best))
        print(f"gen {gen:3d} rollouts {n_done:4d} gen-best {scores[order[0]]:.3f} mean {scores.mean():.3f} "
              f"best {best:.3f} sigma {sigma:.3f} shift {cfg['hip_roll_shift']:.3f} speed {cfg['speed']:.2f}", flush=True)
        if log_path:
            with open(log_path, "w") as fh:
                json.dump({"best": best, "W": best_W.reshape(N_J, N_F).tolist(), "cfg": best_cfg, "curve": curve}, fh)
    if pool:
        pool.close()
    hold = run(best_W, best_cfg)
    print("best hold (harness):", hold, "cfg:", best_cfg)
    print("W = np.array(" + json.dumps(np.round(best_W.reshape(N_J, N_F), 4).tolist()) + ")")
    return best_W, best_cfg, hold, curve


# --------------------------------------------------------------------------- best found
BEST_W = np.zeros((N_J, N_F))
BEST_CFG = dict(DEFAULT_CFG)

if __name__ == "__main__":
    argv = sys.argv[1:]

    def opt(name, default, typ=float):
        return typ(argv[argv.index(name) + 1]) if name in argv else default

    if "--search" in argv:
        n = opt("--search", 300, int)
        seed = opt("--seed", 0, int)
        workers = opt("--workers", 1, int)
        sigma0 = opt("--sigma", 0.3)
        pop = opt("--pop", 12, int)
        log_path = argv[argv.index("--log") + 1] if "--log" in argv else None
        start = BEST_W if "--warm" in argv else None
        start_cfg = BEST_CFG if "--warm" in argv else None
        search(n, seed=seed, workers=workers, start=start, start_cfg=start_cfg, sigma0=sigma0, pop=pop,
               log_path=log_path)
    else:
        hold = run(BEST_W, BEST_CFG, verbose="-v" in argv)
        print(f"HOLD {hold:.2f}")
