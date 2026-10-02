"""Warm-start a checkpoint whose input slots change meaning.

`microduck_bridge_sideways_env_cfg` reuses the six body-pose command slots
(always ~0 before) for bridge state. Two things in the old checkpoint would
turn the new values into garbage at iteration 0:

* the observation normaliser learned mean ~0 and a tiny std for those slots,
  and its sample count is in the millions, so it would barely move: new
  values of order 1 would be blown up by the old std. Reset to identity.
* the first-layer weights reading those slots were fit to ~0 inputs. Zeroed,
  the policy acts exactly as before and learns to use the slots.

    python -m microduck_lab.rl.human_bridge.warmstart IN.pt OUT.pt --actor 55 61 --critic 70 76
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def open_slots(src, dst, actor, critic):
    ck = torch.load(src, map_location="cpu", weights_only=False)
    for key, (a, b) in (("actor_state_dict", actor), ("critic_state_dict", critic)):
        sd = ck[key]
        sd["obs_normalizer._mean"][:, a:b] = 0.0
        sd["obs_normalizer._var"][:, a:b] = 1.0
        sd["obs_normalizer._std"][:, a:b] = 1.0
        sd["mlp.0.weight"][:, a:b] = 0.0
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--actor", type=int, nargs=2, default=(55, 61))
    ap.add_argument("--critic", type=int, nargs=2, default=(70, 76))
    a = ap.parse_args()
    open_slots(a.src, a.dst, tuple(a.actor), tuple(a.critic))
    print(f"wrote {a.dst}")


if __name__ == "__main__":
    main()
