"""Warm-start checkpoint for `Mjlab-Velocity-SidewaysBeam-MicroDuck`.

From a sideways-walking checkpoint (sideways v4 iter 1000), it writes one that
the beam task can resume from:

* The actor is copied as is (same 61D obs, same 14 actions), with its
  normaliser (count 98M, frozen in practice: the actor stays the v4 walker).
* The critic reads new privileged inputs appended at the end of its obs
  (`mdp_beam.beam_privileged`, 18 values). Its first layer gets zero columns
  for them, so at iteration 0 it predicts exactly what it did before and
  learns to use them.
* Critic normaliser. rsl_rl keeps ONE sample count for all inputs, and v4's is
  98M (1000 iterations x 4096 envs x 24 steps). Each step moves the running
  mean by 4096 / count, so with 98M the 18 new inputs would sit at whatever
  they start with for the whole run. So: their mean / var are set from
  `--stats` (measured on warm-start rollouts in the beam task; without it:
  mean 0 / var 1), and the count is lowered to `--critic-count` (default 2M,
  about 20 iterations at 4096 envs), so all critic inputs keep following the
  beam task as the terrain curriculum moves. The old inputs start from v4's
  values.
* The env step counter (`infos.env_state.common_step_counter`, 24024 in v4
  iter 1000) is removed. mjlab restores it on resume, and every step-staged
  curriculum would then start at its late stage.
* The iteration is set to 0 and the Adam moments are cleared (their shapes
  no longer match the critic).

    .venv/bin/python -m microduck_lab.rl.human_bridge.rl_policy.warmstart_beam \\
        logs/rsl_rl/velocity_sideways/2026-09-26_02-26-51_sideways-v4/model_1000.pt \\
        logs/rsl_rl/velocity_sideways_beam/warmstart_v4_1000_s4/model_0.pt \\
        --stats local_storage/hb_dev/rl_policy/s4_fix/beam_obs_stats.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

BEAM_CRITIC_EXTRA = 18
CRITIC_COUNT = 2_000_000


def widen_critic(sd: dict, extra: int, mean=None, var=None, count: int | None = None) -> dict:
    """Append `extra` inputs to a critic state dict: zero weights; normaliser
    mean / var for them from `mean` / `var` (default 0 / 1); count set to
    `count` if given (lower = the running stats keep moving)."""
    if extra <= 0:
        return sd
    w = sd["mlp.0.weight"]
    sd["mlp.0.weight"] = torch.cat([w, torch.zeros(w.shape[0], extra, dtype=w.dtype)], dim=1)
    ref = sd["obs_normalizer._mean"]
    m = torch.zeros(1, extra, dtype=ref.dtype) if mean is None else torch.as_tensor(mean, dtype=ref.dtype).reshape(1, extra)
    v = torch.ones(1, extra, dtype=ref.dtype) if var is None else torch.as_tensor(var, dtype=ref.dtype).reshape(1, extra)
    sd["obs_normalizer._mean"] = torch.cat([sd["obs_normalizer._mean"], m], dim=1)
    sd["obs_normalizer._var"] = torch.cat([sd["obs_normalizer._var"], v], dim=1)
    sd["obs_normalizer._std"] = torch.cat([sd["obs_normalizer._std"], torch.sqrt(v)], dim=1)
    if count is not None and "obs_normalizer.count" in sd:
        sd["obs_normalizer.count"] = torch.tensor(int(count), dtype=sd["obs_normalizer.count"].dtype)
    return sd


def make_warmstart(src, dst, extra: int = BEAM_CRITIC_EXTRA, stats: dict | None = None,
                   critic_count: int | None = CRITIC_COUNT) -> dict:
    ck = torch.load(src, map_location="cpu", weights_only=False)
    mean = var = None
    if stats is not None:
        mean, var = stats["mean"], stats["var"]
        assert len(mean) == len(var) == extra, (len(mean), len(var), extra)
    ck["critic_state_dict"] = widen_critic(ck["critic_state_dict"], extra, mean, var, critic_count)
    infos = dict(ck.get("infos") or {})
    infos.pop("env_state", None)
    ck["infos"] = infos
    ck["iter"] = 0
    opt = ck["optimizer_state_dict"]
    opt["state"] = {}
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, dst)
    return ck


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--extra", type=int, default=BEAM_CRITIC_EXTRA)
    ap.add_argument("--stats", default=None, help="JSON with 'mean' and 'var' lists for the new inputs")
    ap.add_argument("--critic-count", type=int, default=CRITIC_COUNT,
                    help="critic normaliser sample count (v4: 98M); -1 keeps the source value")
    a = ap.parse_args()
    stats = json.loads(Path(a.stats).read_text()) if a.stats else None
    ck = make_warmstart(a.src, a.dst, a.extra, stats, None if a.critic_count < 0 else a.critic_count)
    sd = ck["critic_state_dict"]
    print(f"wrote {a.dst}: critic inputs {sd['mlp.0.weight'].shape[1]}, "
          f"critic count {int(sd['obs_normalizer.count'])}, "
          f"actor count {int(ck['actor_state_dict'].get('obs_normalizer.count', -1))}, "
          f"infos {ck['infos']}, iter {ck['iter']}")


if __name__ == "__main__":
    main()
