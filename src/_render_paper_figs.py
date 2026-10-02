"""Render static trajectory PNGs of the BEST checkpoint for the paper.

Picks successful, LOS-blocked (detour-required) episodes so the figures show
the agent walking around walls to the RF source, not straight-line hits.
"""
from dataclasses import fields
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

from adwa_env import ADWANavigationEnv, EnvConfig
from train_adwa_ppo import RecurrentActorCritic

OUT = Path("/home/robust_lab/jw_workspace/SICM/paper/figs")
OUT.mkdir(parents=True, exist_ok=True)
CKPT = "checkpoints/adwa_gru_ppo_short75_backup.pt"
# (building, is_holdout) picks to showcase
WANT = [("Ribera", False), ("Sisters1", False), ("Sands2", False),
        ("Eastville", True), ("Mosquito", True), ("Scioto2", True)]
SEEDS = 60

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
valid = {f.name for f in fields(EnvConfig)}
cfg = EnvConfig(**{k: v for k, v in ckpt["env_config"].items() if k in valid})
policy = RecurrentActorCritic(ckpt["observation_size"], memory_size=int(ckpt.get("memory_size", 16)))
policy.load_state_dict(ckpt["model"]); policy.eval()
print(f"ckpt={CKPT} train={ckpt.get('train_success')} holdout={ckpt.get('holdout_success')}", flush=True)

env = ADWANavigationEnv(cfg, seed=0)
holdout = set(env.holdout_buildings)


def rollout():
    obs = env.reset(split="all")
    hidden = policy.blank_memory(); start = True; done = False
    traj = [(float(env.position[0]), float(env.position[1]))]
    info = {"reached": False}
    while not done:
        d, hidden = policy.greedy_direction(torch.as_tensor(obs), hidden, start)
        obs, _, done, info = env.step(d)
        traj.append((float(env.position[0]), float(env.position[1])))
        start = False
    return traj, info


for name, is_ho in WANT:
    if name not in env.map_pool:
        print(f"skip {name}: not in pool"); continue
    env._sample_building = (lambda split, n=name: n)
    best = None  # (priority, traj)
    for s in range(SEEDS):
        env.rng = np.random.default_rng(20_000 + s)
        traj, info = rollout()
        start_cell = env._cell(np.asarray(env._start_position))
        blocked = env._line_has_wall(start_cell, tuple(int(v) for v in env.source))
        # want: reached + blocked start (detour) + reasonably long path
        prio = (int(info["reached"]), int(blocked), len(traj))
        if best is None or prio > best[0]:
            best = (prio, traj, info)
        if info["reached"] and blocked and len(traj) >= 12:
            best = (prio, traj, info)
            break
    _, traj, info = best
    fig = env.render(traj)
    tag = "holdout" if name in holdout else "train"
    path = OUT / f"traj_{name}_{tag}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"{name:<12}[{tag}] reached={info['reached']} steps={info['steps']} -> {path.name}", flush=True)

print("PAPER_FIGS_DONE", flush=True)
