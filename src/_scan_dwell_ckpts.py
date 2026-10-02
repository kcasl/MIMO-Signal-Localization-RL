"""Greedy-evaluate every dwell-run checkpoint on a quick multi-building probe
to see if any is usable. Uses each checkpoint's OWN env_config (its stage).
"""
import glob, os
from dataclasses import fields

import numpy as np
import torch

from adwa_env import ADWANavigationEnv, EnvConfig
from train_adwa_ppo import RecurrentActorCritic

# dwell-run files (created today after 22:00) — exclude the terminal backups
CANDIDATES = [
    "checkpoints/adwa_curriculum_stop.pt",       # end of stop stage (dwell)
    "checkpoints/dwell_stopbest_backup.pt",      # best-holdout during stop (dwell)
    "checkpoints/adwa_curriculum_step0200499.pt",# short stage (dwell)
]
SEEDS = 8
PROBE = None  # set later to a fixed building list


def eval_ckpt(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    valid = {f.name for f in fields(EnvConfig)}
    cfg = EnvConfig(**{k: v for k, v in ck["env_config"].items() if k in valid})
    pol = RecurrentActorCritic(ck["observation_size"], memory_size=int(ck.get("memory_size", 16)))
    pol.load_state_dict(ck["model"]); pol.eval()
    env = ADWANavigationEnv(cfg, seed=0)
    holdout = set(env.holdout_buildings)
    buildings = sorted(env.map_pool)

    def rollout():
        obs = env.reset(split="all"); h = pol.blank_memory(); st = True; done = False
        info = {"reached": False}
        while not done:
            d, h = pol.greedy_direction(torch.as_tensor(obs), h, st)
            obs, _, done, info = env.step(d); st = False
        return info

    tr_s = tr_n = ho_s = ho_n = 0
    for name in buildings:
        env._sample_building = (lambda split, n=name: n)
        s = 0
        for k in range(SEEDS):
            env.rng = np.random.default_rng(70_000 + k)
            s += int(rollout()["reached"])
        if name in holdout: ho_s += s; ho_n += SEEDS
        else: tr_s += s; tr_n += SEEDS
    return (ck.get("phase"), ck.get("total_steps"),
            cfg.max_start_source_distance_m, cfg.require_detour,
            100*tr_s/tr_n, 100*ho_s/ho_n)


for path in CANDIDATES:
    if not os.path.exists(path):
        print(f"{os.path.basename(path):<32} MISSING"); continue
    ph, st, mx, det, tr, ho = eval_ckpt(path)
    print(f"{os.path.basename(path):<32} phase={ph:<14} steps={st:<7} "
          f"stage=->{mx}m/det={det}  greedy train={tr:.0f}% holdout={ho:.0f}%", flush=True)
print("SCAN_DONE", flush=True)
