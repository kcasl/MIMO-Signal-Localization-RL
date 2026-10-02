"""Render the best checkpoint greedily on EVERY ADWA building (train + holdout)."""
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from adwa_env import ADWANavigationEnv, EnvConfig
from train_adwa_ppo import RecurrentActorCritic

OUT = Path("/home/robust_lab/jw_workspace/SICM/adwa_eval/all_maps")
OUT.mkdir(parents=True, exist_ok=True)
CKPT = "checkpoints/adwa_gru_ppo_short75_backup.pt"
SEEDS_PER_MAP = 12

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
valid = {f.name for f in fields(EnvConfig)}
cfg = EnvConfig(**{k: v for k, v in ckpt["env_config"].items() if k in valid})
policy = RecurrentActorCritic(ckpt["observation_size"], memory_size=int(ckpt.get("memory_size", 16)))
policy.load_state_dict(ckpt["model"]); policy.eval()
print(f"ckpt={CKPT} phase={ckpt.get('phase')} train={ckpt.get('train_success')} "
      f"holdout={ckpt.get('holdout_success')} stage={cfg.min_start_source_distance_m}-"
      f"{cfg.max_start_source_distance_m} detour={cfg.require_detour}", flush=True)

env = ADWANavigationEnv(cfg, seed=0)
buildings = sorted(env.map_pool)
holdout = set(env.holdout_buildings)


def rollout():
    obs = env.reset(split="all")
    hidden = policy.blank_memory(); start = True; done = False
    traj = [(float(env.position[0]), float(env.position[1]))]
    info = {"reached": False}
    min_euc = 1e9
    while not done:
        min_euc = min(min_euc, env._source_distance_m())
        direction, hidden = policy.greedy_direction(torch.as_tensor(obs), hidden, start)
        obs, _, done, info = env.step(direction)
        traj.append((float(env.position[0]), float(env.position[1])))
        start = False
    return traj, info, min_euc


summary = []
for name in buildings:
    env._sample_building = (lambda split, n=name: n)  # force this building each reset
    best = None            # (priority_tuple, traj, info, seed, min_euc)
    n_succ = 0
    for s in range(SEEDS_PER_MAP):
        env.rng = np.random.default_rng(10_000 + s)
        traj, info, min_euc = rollout()
        n_succ += int(info["reached"])
        # prefer: reached, then LOS-blocked start (real detour), then fewer steps
        start_cell = env._cell(np.asarray(env._start_position))
        blocked = env._line_has_wall(start_cell, tuple(int(v) for v in env.source))
        prio = (int(info["reached"]), int(blocked), -info["steps"], -min_euc)
        if best is None or prio > best[0]:
            best = (prio, traj, info, s, min_euc)
    _, traj, info, seed, min_euc = best
    # decimate long trajectories to keep GIF light
    step = max(1, len(traj) // 160)
    traj_d = traj[::step] + [traj[-1]]
    path = OUT / f"{name}{'_holdout' if name in holdout else ''}.gif"
    env.render_animation(traj_d).save(path, writer="pillow", fps=8)
    tag = "holdout" if name in holdout else "train"
    print(f"{name:<12}[{tag}] succ {n_succ}/{SEEDS_PER_MAP}  best: reached={info['reached']} "
          f"steps={info['steps']} min_euc={min_euc:.2f}m -> {path.name}", flush=True)
    summary.append((name, tag, n_succ, SEEDS_PER_MAP, int(info["reached"])))

print("\n=== per-map greedy success ===", flush=True)
tr = [r for r in summary if r[1] == "train"]
ho = [r for r in summary if r[1] == "holdout"]
for r in summary:
    print(f"  {r[0]:<12}{r[1]:<8} {r[2]}/{r[3]}", flush=True)
print(f"train mean {sum(r[2] for r in tr)}/{sum(r[3] for r in tr)}  "
      f"holdout mean {sum(r[2] for r in ho)}/{sum(r[3] for r in ho)}", flush=True)
print("RENDER_ALL_DONE", flush=True)
