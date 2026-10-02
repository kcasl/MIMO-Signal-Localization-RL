"""Wait for a SHORT-stage checkpoint from the running dwell trainer, then eval
it greedily on all 17 buildings (12 seeds each). Writes a table to stdout.
"""
import glob, os, time
from dataclasses import fields

import numpy as np
import torch

from adwa_env import ADWANavigationEnv, EnvConfig
from train_adwa_ppo import RecurrentActorCritic

SEEDS = 12


def is_short(d):
    c = d.get("env_config", {})
    return (c.get("max_start_source_distance_m") == 7.0
            and c.get("require_detour") is True)


def newest_short():
    best = None
    for f in glob.glob("checkpoints/*.pt"):
        try:
            d = torch.load(f, map_location="cpu", weights_only=False)
        except Exception:
            continue
        if is_short(d) and (d.get("total_steps") or 0) > 164000:
            key = os.path.getmtime(f)
            if best is None or key > best[0]:
                best = (key, f, d)
    return best


print("watching for a short-stage checkpoint...", flush=True)
found = None
for _ in range(3600):  # up to ~60 min
    found = newest_short()
    if found:
        break
    time.sleep(10)
if not found:
    print("no short checkpoint appeared in time", flush=True)
    raise SystemExit

_, path, ckpt = found
# snapshot it so training cannot overwrite the file we evaluate
snap = "checkpoints/dwell_short_eval.pt"
torch.save(ckpt, snap)
print(f"evaluating {path} (steps={ckpt.get('total_steps')} phase={ckpt.get('phase')})", flush=True)

valid = {f.name for f in fields(EnvConfig)}
cfg = EnvConfig(**{k: v for k, v in ckpt["env_config"].items() if k in valid})
policy = RecurrentActorCritic(ckpt["observation_size"], memory_size=int(ckpt.get("memory_size", 16)))
policy.load_state_dict(ckpt["model"]); policy.eval()

env = ADWANavigationEnv(cfg, seed=0)
holdout = set(env.holdout_buildings)


def rollout():
    obs = env.reset(split="all"); hidden = policy.blank_memory(); start = True; done = False
    info = {"reached": False}
    while not done:
        d, hidden = policy.greedy_direction(torch.as_tensor(obs), hidden, start)
        obs, _, done, info = env.step(d); start = False
    return info


tr_s = tr_n = ho_s = ho_n = 0
for name in sorted(env.map_pool):
    env._sample_building = (lambda split, n=name: n)
    s = 0
    for k in range(SEEDS):
        env.rng = np.random.default_rng(30_000 + k)
        s += int(rollout()["reached"])
    tag = "holdout" if name in holdout else "train"
    print(f"  {name:<12}{tag:<8} {s}/{SEEDS}", flush=True)
    if tag == "train": tr_s += s; tr_n += SEEDS
    else: ho_s += s; ho_n += SEEDS

print(f"\nDWELL short: train {tr_s}/{tr_n} = {100*tr_s/tr_n:.0f}%  "
      f"holdout {ho_s}/{ho_n} = {100*ho_s/ho_n:.0f}%", flush=True)
print("DWELL_EVAL_DONE", flush=True)
