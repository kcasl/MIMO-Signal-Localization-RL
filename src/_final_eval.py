"""Robust paper evaluation of the BEST checkpoint (terminal-STOP short75).

Greedy, 24 seeds per building over all 17 buildings. Prints per-building and
aggregate train/holdout success, and saves a success-by-building bar chart.
"""
from dataclasses import fields
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from adwa_env import ADWANavigationEnv, EnvConfig
from train_adwa_ppo import RecurrentActorCritic

FIGS = Path("/home/robust_lab/jw_workspace/SICM/paper/figs")
FIGS.mkdir(parents=True, exist_ok=True)
CKPT = "checkpoints/adwa_gru_ppo_short75_backup.pt"
SEEDS = 24

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
valid = {f.name for f in fields(EnvConfig)}
cfg = EnvConfig(**{k: v for k, v in ckpt["env_config"].items() if k in valid})
policy = RecurrentActorCritic(ckpt["observation_size"], memory_size=int(ckpt.get("memory_size", 16)))
policy.load_state_dict(ckpt["model"]); policy.eval()
print(f"ckpt={CKPT} stage={cfg.min_start_source_distance_m}-{cfg.max_start_source_distance_m} "
      f"detour={cfg.require_detour} seeds/bldg={SEEDS}", flush=True)

env = ADWANavigationEnv(cfg, seed=0)
holdout = set(env.holdout_buildings)


def rollout():
    obs = env.reset(split="all"); hidden = policy.blank_memory(); start = True; done = False
    info = {"reached": False}
    while not done:
        d, hidden = policy.greedy_direction(torch.as_tensor(obs), hidden, start)
        obs, _, done, info = env.step(d); start = False
    return info


names, rates, tags = [], [], []
tr_s = tr_n = ho_s = ho_n = 0
for name in sorted(env.map_pool):
    env._sample_building = (lambda split, n=name: n)
    s = 0
    for k in range(SEEDS):
        env.rng = np.random.default_rng(50_000 + k)
        s += int(rollout()["reached"])
    tag = "holdout" if name in holdout else "train"
    print(f"  {name:<12}{tag:<8} {s}/{SEEDS} ({100*s/SEEDS:.0f}%)", flush=True)
    names.append(name); rates.append(100 * s / SEEDS); tags.append(tag)
    if tag == "train": tr_s += s; tr_n += SEEDS
    else: ho_s += s; ho_n += SEEDS

print(f"\nTERMINAL short75: train {tr_s}/{tr_n} = {100*tr_s/tr_n:.1f}%  "
      f"holdout {ho_s}/{ho_n} = {100*ho_s/ho_n:.1f}%", flush=True)

# ---- bar chart: success by building (train blue, holdout orange) ----
order = sorted(range(len(names)), key=lambda i: (tags[i], -rates[i]))
n = [names[i] for i in order]; r = [rates[i] for i in order]
col = ["#e08214" if tags[i] == "holdout" else "#2166ac" for i in order]
fig, ax = plt.subplots(figsize=(9, 3.2))
ax.bar(range(len(n)), r, color=col)
ax.axhline(100 * tr_s / tr_n, ls="--", lw=1, color="#2166ac")
ax.set_xticks(range(len(n))); ax.set_xticklabels(n, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("greedy success (%)"); ax.set_ylim(0, 100)
handles = [plt.Rectangle((0, 0), 1, 1, color="#2166ac"),
           plt.Rectangle((0, 0), 1, 1, color="#e08214")]
ax.legend(handles, ["train building", "holdout building"], loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(FIGS / "success_by_building.png", dpi=150)
print(f"saved {FIGS/'success_by_building.png'}", flush=True)
print("FINAL_EVAL_DONE", flush=True)
