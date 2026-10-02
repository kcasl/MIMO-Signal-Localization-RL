"""DAgger-based curriculum PPO on ADWA buildings.

Pipeline (one process, same policy, CSI+pose+prev-action obs only):

  for stage in (stop, short, full):
      1. DAgger aggregations on that spawn band.
         Rollout mixes expert with probability β (1 → 0). Labels are always
         the privileged geodesic heading + STOP-inside-1 m, never the map.
      2. Gate: greedy train-eval. If it clears the stage threshold, or
         aggregations are exhausted, freeze the imitation snapshot.
      3. Critic warmup on on-policy returns (DAgger did not train V).
      4. PPO with a BC regularizer on the aggregated DAgger buffer so the
         collision penalty cannot wipe the clone.
      5. Advance spawn difficulty (distance + require_detour).

Example: python train_adwa_curriculum.py --timesteps 2000000
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from adwa_env import ADWANavigationEnv, EnvConfig
from train_adwa_ppo import RecurrentActorCritic, evaluate, gae, von_mises_entropy


@dataclass(frozen=True)
class Stage:
    name: str
    min_m: float
    max_m: float
    require_detour: bool
    aggregations: int
    collect_episodes: int
    greedy_gate: float
    ppo_steps: int
    bc_coef: float
    stop_pos_weight: float = 15.0


# Easy: same-room approach + STOP. Then short corridors. Then the 5–12 m task.
# Fewer aggregations, but each fits the clone to convergence (see train_dagger),
# which is what the undertrained 6-step version was missing.
STAGES = (
    Stage("stop", 2.5, 4.0, False, aggregations=6, collect_episodes=64,
          greedy_gate=0.40, ppo_steps=150_000, bc_coef=0.40),
    Stage("short", 4.0, 7.0, True, aggregations=8, collect_episodes=80,
          greedy_gate=0.30, ppo_steps=250_000, bc_coef=0.25),
    Stage("full", 5.0, 12.0, True, aggregations=10, collect_episodes=96,
          greedy_gate=0.20, ppo_steps=400_000, bc_coef=0.15),
)


class DaggerBuffer:
    """Episode-wise DAgger aggregate. Labels are expert (θ, stop), not executed actions."""

    def __init__(self, max_episodes: int = 500):
        self.episodes: deque[tuple[np.ndarray, np.ndarray, np.ndarray]] = deque(maxlen=max_episodes)

    def add(self, observations, thetas, stops) -> None:
        if len(observations) == 0:
            return
        self.episodes.append((
            np.asarray(observations, dtype=np.float32),
            np.asarray(thetas, dtype=np.float32),
            np.asarray(stops, dtype=np.float32),
        ))

    def __len__(self) -> int:
        return len(self.episodes)

    def sample_episodes(self, rng: np.random.Generator, max_steps: int = 512, start_at: int = 0):
        """Contiguous shuffled episodes as one sequence, with per-episode start flags.

        ``start_at`` lets the caller sweep the whole buffer across minibatches
        instead of always drawing from the front of a single permutation.
        """
        if not self.episodes:
            raise RuntimeError("DAgger buffer is empty")
        order = rng.permutation(len(self.episodes))
        obs, thetas, stops, starts = [], [], [], []
        total = 0
        n = len(order)
        for k in range(n):
            idx = int(order[(start_at + k) % n])
            o, t, s = self.episodes[idx]
            starts.append(np.concatenate(
                ([1.0], np.zeros(len(o) - 1, dtype=np.float32))
            ))
            obs.append(o)
            thetas.append(t)
            stops.append(s)
            total += len(o)
            if total >= max_steps:
                break
        return (
            np.concatenate(obs),
            np.concatenate(thetas),
            np.concatenate(stops),
            np.concatenate(starts),
        )


def stage_config(base: EnvConfig, stage: Stage) -> EnvConfig:
    return replace(
        base,
        min_start_source_distance_m=stage.min_m,
        max_start_source_distance_m=stage.max_m,
        require_detour=stage.require_detour,
    )


def expert_theta_stop(action: np.ndarray) -> tuple[float, float]:
    theta = float(math.atan2(float(action[1]), float(action[0])))
    stop = 1.0 if float(action[2]) > 0.5 else 0.0
    return theta, stop


def dagger_loss(policy, obs, expert_theta, expert_stop, starts, stop_pos_weight: float):
    mus, stop_logits, _ = policy.forward_sequence(obs, starts)
    dist = policy._dist(mus)
    heading_nll = -dist.log_prob(expert_theta).mean()
    weight = torch.where(expert_stop > 0.5, stop_pos_weight, 1.0)
    stop_bce = F.binary_cross_entropy_with_logits(
        stop_logits, expert_stop, weight=weight, reduction="mean")
    return heading_nll + stop_bce, heading_nll, stop_bce


def collect_dagger_episodes(policy, env, device, rng, beta: float, n_episodes: int,
                            buffer: DaggerBuffer, stop_dwell: int = 4):
    """β-mix of expert/policy actions; store expert labels on visited observations.

    When the expert would STOP (inside 1 m), we keep the ``stop=1`` label but
    execute a small toward-source move for up to ``stop_dwell`` frames before
    actually stopping. That harvests several STOP-positive observations per
    episode instead of one, so the STOP head can cross its −2 bias. This is
    pure DAgger: labels are the expert's, executed actions may differ.
    """
    successes = 0
    n_steps = 0
    reached = []
    for _ in range(n_episodes):
        observation = env.reset(split="train")
        hidden = policy.blank_memory(device)
        episode_start = True
        done = False
        obs_ep, th_ep, st_ep = [], [], []
        info = {"reached": False}
        dwell = 0
        while not done:
            expert = env.expert_action()
            theta_e, stop_e = expert_theta_stop(expert)
            obs_t = torch.as_tensor(observation, device=device)
            if float(rng.random()) < beta:
                direction = expert
                _, hidden = policy.greedy_direction(obs_t, hidden, episode_start)
            else:
                direction, _, _, hidden, _, _ = policy.act(obs_t, hidden, episode_start)
            # Dwell near the source to gather more STOP-positive labels.
            if stop_e > 0.5 and dwell < stop_dwell:
                direction = np.array([expert[0], expert[1], 0.0], dtype=np.float32)
                dwell += 1
            obs_ep.append(np.asarray(observation, dtype=np.float32))
            th_ep.append(theta_e)
            st_ep.append(stop_e)
            observation, _, done, info = env.step(direction)
            episode_start = False
            n_steps += 1
        buffer.add(obs_ep, th_ep, st_ep)
        reached.append(int(info["reached"]))
        successes += int(info["reached"])
    return successes / n_episodes, n_steps, reached


def expert_success_rate(env: ADWANavigationEnv, episodes: int = 8) -> float:
    """Sanity check that the privileged teacher actually solves the current stage."""
    ok = 0
    for _ in range(episodes):
        env.reset(split="train")
        done = False
        info = {"reached": False}
        while not done:
            _, _, done, info = env.step(env.expert_action())
        ok += int(info["reached"])
    return ok / episodes


@torch.no_grad()
def clone_diagnostics(policy, env: ADWANavigationEnv, device, episodes: int = 8) -> dict:
    """Greedy-policy heading error vs expert and STOP-head recall/false-positive.

    This is the honest clone-quality signal the misleading expert-driven
    mix-success hid: it runs the policy greedily and compares against the
    privileged expert on every visited state.
    """
    policy.eval()
    head_errs = []
    stop_tp = stop_pos = stop_fp = stop_neg = 0
    for _ in range(episodes):
        observation = env.reset(split="train")
        hidden = policy.blank_memory(device)
        episode_start = True
        done = False
        while not done:
            expert = env.expert_action()
            theta_e, stop_e = expert_theta_stop(expert)
            obs_t = torch.as_tensor(observation, device=device)
            mem, mask = policy._reset_if_needed(hidden[0], hidden[1], episode_start)
            fused = policy.encode_observation(obs_t)
            state, mem, mask = policy._attend(fused, mem, mask)
            mu = float(policy.actor_mu(state[0]).squeeze(-1))
            logit = float(policy.actor_stop(state[0]).squeeze(-1))
            hidden = (mem, mask)
            err = abs(math.atan2(math.sin(mu - theta_e), math.cos(mu - theta_e)))
            head_errs.append(err)
            greedy_stop = logit > 0.0
            if stop_e > 0.5:
                stop_pos += 1
                stop_tp += int(greedy_stop)
            else:
                stop_neg += 1
                stop_fp += int(greedy_stop)
            # Follow the greedy policy so the trajectory reflects the clone.
            direction = np.array([math.cos(mu), math.sin(mu),
                                  1.0 if greedy_stop else 0.0], dtype=np.float32)
            observation, _, done, _ = env.step(direction)
            episode_start = False
    policy.train()
    return {
        "head_err_deg": float(np.degrees(np.mean(head_errs))) if head_errs else 0.0,
        "stop_recall": stop_tp / stop_pos if stop_pos else 0.0,
        "stop_fp": stop_fp / stop_neg if stop_neg else 0.0,
    }


def train_dagger(policy, optimizer, buffer, rng, device, max_steps: int,
                 stop_pos_weight: float, batch_steps: int = 384,
                 patience: int = 40, min_delta: float = 1e-3):
    """Fit the clone to (near) convergence, not a fixed handful of steps.

    The earlier 6-step version left heading error ~150° and the STOP logit
    pinned at its −2 init. Here we run minibatch SGD over the aggregated
    buffer until the loss stops improving (patience) or ``max_steps`` is hit.
    """
    policy.train()
    best = float("inf")
    since_best = 0
    last = last_h = last_s = 0.0
    ran = 0
    for step in range(max_steps):
        obs_np, th_np, st_np, start_np = buffer.sample_episodes(
            rng, max_steps=batch_steps, start_at=step)
        obs = torch.as_tensor(obs_np, device=device)
        theta = torch.as_tensor(th_np, device=device)
        stop = torch.as_tensor(st_np, device=device)
        starts = torch.as_tensor(start_np, device=device)
        loss, h_nll, s_bce = dagger_loss(policy, obs, theta, stop, starts, stop_pos_weight)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()
        last, last_h, last_s = float(loss.item()), float(h_nll.item()), float(s_bce.item())
        ran = step + 1
        if last < best - min_delta:
            best = last
            since_best = 0
        else:
            since_best += 1
            if since_best >= patience:
                break
    return {"loss": last, "heading_nll": last_h, "stop_bce": last_s, "steps": ran}


def fit_critic(policy, env, optimizer, device, rollout: int, gamma: float, lam: float):
    """Value-only pass so PPO does not start with V≈0 against −150 returns."""
    policy.train()
    observation = env.reset(split="train")
    hidden = policy.blank_memory(device)
    episode_start = True
    observations, rewards, values, starts, dones = [], [], [], [], []
    window_hidden = (hidden[0].detach().clone(), hidden[1].detach().clone())
    for _ in range(rollout):
        direction, _, value, hidden, _, _ = policy.act(
            torch.as_tensor(observation, device=device), hidden, episode_start)
        next_observation, reward, done, _ = env.step(direction)
        observations.append(observation)
        rewards.append(reward)
        values.append(value.cpu())
        starts.append(float(episode_start))
        dones.append(float(done))
        observation, episode_start = next_observation, done
        if done:
            observation = env.reset(split="train")
    obs = torch.as_tensor(np.asarray(observations), dtype=torch.float32, device=device)
    rew = torch.as_tensor(rewards, dtype=torch.float32)
    old_values = torch.stack(values)
    starts_t = torch.as_tensor(starts, device=device)
    dones_t = torch.as_tensor(dones)
    with torch.no_grad():
        bootstrap = (torch.tensor(0.0) if episode_start
                     else policy.value(torch.as_tensor(observation, device=device), hidden, False).cpu())
        advantages = gae(rew, old_values, dones_t, bootstrap, gamma=gamma, lam=lam)
        returns = (advantages + old_values).to(device)
    _, _, predicted = policy.forward_sequence(obs, starts_t, initial_hidden=window_hidden)
    value_loss = 0.5 * (predicted - returns).pow(2).mean()
    optimizer.zero_grad()
    value_loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
    optimizer.step()
    return float(value_loss.item())


def ppo_update(policy, optimizer, env, device, rng, buffer, args, bc_coef: float,
               stop_pos_weight: float, observation, hidden, episode_start, remaining: int):
    window_hidden = (hidden[0].detach().clone(), hidden[1].detach().clone())
    observations, thetas, stops, log_probs, rewards, values, starts, dones = (
        [], [], [], [], [], [], [], [])
    recent_success, recent_collide = [], []
    completed = successes = 0
    n_steps = min(args.rollout, remaining)
    for _ in range(n_steps):
        direction, log_prob, value, hidden, theta, stop = policy.act(
            torch.as_tensor(observation, device=device), hidden, episode_start)
        next_observation, reward, done, info = env.step(direction)
        observations.append(observation)
        thetas.append(theta.detach().cpu())
        stops.append(stop.detach().cpu())
        log_probs.append(log_prob.cpu())
        rewards.append(reward)
        values.append(value.cpu())
        starts.append(float(episode_start))
        dones.append(float(done))
        recent_collide.append(int(info["collision"]))
        observation, episode_start = next_observation, done
        if done:
            completed += 1
            successes += int(info["reached"])
            recent_success.append(int(info["reached"]))
            observation = env.reset(split="train")
    obs = torch.as_tensor(np.asarray(observations), dtype=torch.float32, device=device)
    act = torch.stack(thetas).to(device)
    act_stop = torch.stack(stops).to(device)
    old_lp = torch.stack(log_probs).to(device)
    rew = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    old_values = torch.stack(values).to(device)
    starts_t = torch.as_tensor(starts, device=device)
    dones_t = torch.as_tensor(dones, device=device)
    with torch.no_grad():
        bootstrap = (torch.tensor(0.0) if episode_start
                     else policy.value(torch.as_tensor(observation, device=device), hidden, False).cpu())
        advantages = gae(rew.cpu(), old_values.cpu(), dones_t.cpu(), bootstrap,
                         gamma=args.gamma, lam=args.gae_lambda).to(device)
        returns = advantages + old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    last_loss = last_entropy = 0.0
    for _ in range(args.epochs):
        mus, stop_mus, predicted_values = policy.forward_sequence(
            obs, starts_t, initial_hidden=window_hidden)
        dist = policy._dist(mus)
        stop_dist = policy._stop_dist(stop_mus)
        new_lp = dist.log_prob(act) + stop_dist.log_prob(act_stop)
        ratio = torch.exp(new_lp - old_lp)
        clipped = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip)
        policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
        value_loss = 0.5 * (predicted_values - returns).pow(2).mean()
        entropy = (von_mises_entropy(policy.concentration().expand_as(mus)).mean()
                   + stop_dist.entropy().mean())
        loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy
        if bc_coef > 0.0 and len(buffer) > 0:
            b_obs, b_th, b_st, b_start = buffer.sample_episodes(rng, max_steps=512)
            bc, _, _ = dagger_loss(
                policy,
                torch.as_tensor(b_obs, device=device),
                torch.as_tensor(b_th, device=device),
                torch.as_tensor(b_st, device=device),
                torch.as_tensor(b_start, device=device),
                stop_pos_weight,
            )
            loss = loss + bc_coef * bc
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()
        last_loss = float(loss.item())
        last_entropy = float(entropy.item())
    stats = {
        "completed": completed,
        "successes": successes,
        "recent_success": recent_success,
        "recent_collide": recent_collide,
        "loss": last_loss,
        "entropy": last_entropy,
        "steps": n_steps,
    }
    return observation, hidden, episode_start, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--rollout", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--dagger-max-steps", type=int, default=800,
                        help="Max minibatch SGD steps per aggregation; stops early on plateau.")
    parser.add_argument("--dagger-batch-steps", type=int, default=128)
    parser.add_argument("--dagger-patience", type=int, default=120)
    parser.add_argument("--dagger-min-delta", type=float, default=5e-4)
    parser.add_argument("--stop-dwell", type=int, default=4)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--dagger-lr", type=float, default=1.0e-3)
    parser.add_argument("--ppo-lr", type=float, default=1.0e-4)
    parser.add_argument("--eval-interval", type=int, default=16384)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--checkpoint-interval", type=int, default=50_000)
    parser.add_argument("--memory-size", type=int, default=16)
    parser.add_argument("--critic-warmup", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_cfg = EnvConfig()
    env = ADWANavigationEnv(stage_config(base_cfg, STAGES[0]), args.seed)
    print(f"ADWA train buildings ({len(env.train_buildings)}): {', '.join(env.train_buildings)}")
    print(f"ADWA holdout/test buildings ({len(env.holdout_buildings)}): {', '.join(env.holdout_buildings)}")
    sample = env.map_pool[env.train_buildings[0]]
    print(f"occupancy: native PNG pixels, no downsample "
          f"(example {env.train_buildings[0]} {sample['world'].shape[1]}x{sample['world'].shape[0]} "
          f"@ {sample['resolution_m']} m/px)")
    print(f"obs={env.observation_size} (MIMO+pose+action encoders, memory={args.memory_size}, no lidar)")
    print("curriculum: DAgger → critic-warmup → PPO+BC per spawn stage; expert is privileged BFS+STOP")
    for stage in STAGES:
        print(f"  {stage.name}: geodesic {stage.min_m:.1f}–{stage.max_m:.1f} m "
              f"detour={stage.require_detour} gate={stage.greedy_gate:.0%} "
              f"dagger={stage.aggregations}x{stage.collect_episodes}ep ppo={stage.ppo_steps}")

    policy = RecurrentActorCritic(env.observation_size, memory_size=args.memory_size).to(device)
    Path("checkpoints").mkdir(exist_ok=True)
    buffer = DaggerBuffer()
    total_steps = completed = 0
    recent_successes: deque[int] = deque(maxlen=50)
    recent_collisions: deque[int] = deque(maxlen=1024)
    best_holdout_success = -1.0
    next_checkpoint = args.checkpoint_interval
    next_eval = args.eval_interval

    def checkpoint_payload(**extra):
        data = {
            "model": policy.state_dict(),
            "observation_size": env.observation_size,
            "env_config": asdict(env.config),
            "train_buildings": env.train_buildings,
            "holdout_buildings": env.holdout_buildings,
            "total_steps": total_steps,
            "episodes": completed,
            "memory_size": policy.memory_size,
        }
        data.update(extra)
        return data

    def maybe_eval(tag: str, force: bool = False):
        nonlocal best_holdout_success, next_eval
        if not force and total_steps < next_eval:
            return "", 0.0, 0.0
        print(f"eval at steps={total_steps} [{tag}] "
              f"({args.eval_episodes} train + {args.eval_episodes} holdout, greedy)...",
              flush=True)
        eval_env = ADWANavigationEnv(env.config, seed=args.seed + total_steps)
        train_s, train_r = evaluate(policy, eval_env, device, args.eval_episodes, split="train")
        hold_s, hold_r = evaluate(policy, eval_env, device, args.eval_episodes, split="holdout")
        text = (f" train-eval={train_s:.1%} ({train_r:6.2f})"
                f" holdout-eval={hold_s:.1%} ({hold_r:6.2f})")
        if hold_s >= best_holdout_success:
            best_holdout_success = hold_s
            torch.save(checkpoint_payload(holdout_success=hold_s, train_success=train_s, phase=tag),
                       "checkpoints/adwa_gru_ppo.pt")
        next_eval = total_steps + args.eval_interval
        return text, train_s, hold_s

    def maybe_snapshot():
        nonlocal next_checkpoint
        if total_steps < next_checkpoint:
            return
        snap = Path("checkpoints") / f"adwa_curriculum_step{total_steps:07d}.pt"
        torch.save(checkpoint_payload(), snap)
        print(f"saved {snap}", flush=True)
        next_checkpoint += args.checkpoint_interval

    def log_line(phase: str, extra: str = ""):
        n_ep = max(len(recent_successes), 1)
        n_col = max(len(recent_collisions), 1)
        recent = sum(recent_successes) / n_ep
        collide = sum(recent_collisions) / n_col
        print(f"steps={total_steps:>7} episodes={completed:>4} phase={phase} "
              f"recent-success={sum(recent_successes)}/{len(recent_successes)} ({recent:.1%}) "
              f"collide={sum(recent_collisions)}/{len(recent_collisions)} ({collide:.1%})"
              f"{extra} kappa={float(policy.concentration().detach()):.2f}", flush=True)

    for stage in STAGES:
        if total_steps >= args.timesteps:
            break
        cfg = stage_config(base_cfg, stage)
        env.apply_curriculum(cfg)
        print(f"=== stage={stage.name} geodesic {stage.min_m:.1f}–{stage.max_m:.1f} m "
              f"detour={stage.require_detour} ===", flush=True)

        expert_s = expert_success_rate(env, episodes=args.eval_episodes)
        print(f"stage={stage.name} expert-only train success={expert_s:.1%} "
              f"(teacher uses map BFS; policy still RF-only)", flush=True)

        optimizer = torch.optim.Adam(policy.parameters(), lr=args.dagger_lr)
        gated = False
        for agg in range(stage.aggregations):
            if total_steps >= args.timesteps:
                break
            beta = 1.0 if stage.aggregations == 1 else 1.0 - agg / (stage.aggregations - 1)
            mix_success, n_steps, reached = collect_dagger_episodes(
                policy, env, device, rng, beta, stage.collect_episodes, buffer,
                stop_dwell=args.stop_dwell)
            total_steps += n_steps
            completed += stage.collect_episodes
            recent_successes.extend(reached)
            fit = train_dagger(
                policy, optimizer, buffer, rng, device, args.dagger_max_steps,
                stage.stop_pos_weight, batch_steps=args.dagger_batch_steps,
                patience=args.dagger_patience, min_delta=args.dagger_min_delta)
            clone = clone_diagnostics(policy, env, device, episodes=args.eval_episodes)
            eval_text, train_s, _ = maybe_eval(f"dagger-{stage.name}-{agg}", force=True)
            log_line(f"dagger/{stage.name}/agg{agg} beta={beta:.2f}",
                     extra=(f"{eval_text} dagger[loss={fit['loss']:.3f} h={fit['heading_nll']:.3f}"
                            f" s={fit['stop_bce']:.3f} steps={fit['steps']}]"
                            f" clone[head_err={clone['head_err_deg']:.0f}deg"
                            f" stop_recall={clone['stop_recall']:.0%}"
                            f" stop_fp={clone['stop_fp']:.0%}] expert-mix={mix_success:.1%}"))
            maybe_snapshot()
            if train_s >= stage.greedy_gate:
                print(f"stage={stage.name} passed DAgger gate {train_s:.1%} >= {stage.greedy_gate:.0%}",
                      flush=True)
                gated = True
                break
        if not gated:
            print(f"stage={stage.name} DAgger gate missed; continuing to PPO with current clone",
                  flush=True)

        optimizer = torch.optim.Adam(policy.parameters(), lr=args.ppo_lr)
        for _ in range(args.critic_warmup):
            if total_steps >= args.timesteps:
                break
            vloss = fit_critic(policy, env, optimizer, device, args.rollout, args.gamma, args.gae_lambda)
            total_steps += args.rollout
            log_line(f"critic/{stage.name}", extra=f" value-loss={vloss:.3f}")

        observation = env.reset(split="train")
        hidden = policy.blank_memory(device)
        episode_start = True
        ppo_done = 0
        ppo_budget = min(stage.ppo_steps, max(0, args.timesteps - total_steps))
        while ppo_done < ppo_budget and total_steps < args.timesteps:
            observation, hidden, episode_start, stats = ppo_update(
                policy, optimizer, env, device, rng, buffer, args, stage.bc_coef,
                stage.stop_pos_weight, observation, hidden, episode_start,
                remaining=min(args.rollout, ppo_budget - ppo_done, args.timesteps - total_steps),
            )
            total_steps += stats["steps"]
            ppo_done += stats["steps"]
            completed += stats["completed"]
            recent_successes.extend(stats["recent_success"])
            recent_collisions.extend(stats["recent_collide"])
            eval_text, _, _ = maybe_eval(f"ppo-{stage.name}")
            log_line(f"ppo/{stage.name}", extra=(
                f"{eval_text} loss={stats['loss']:.3f} entropy={stats['entropy']:.3f}"))
            maybe_snapshot()

        torch.save(checkpoint_payload(phase=f"end-{stage.name}"),
                   Path("checkpoints") / f"adwa_curriculum_{stage.name}.pt")

    final = Path("checkpoints") / f"adwa_curriculum_step{total_steps:07d}.pt"
    torch.save(checkpoint_payload(phase="final"), final)
    if best_holdout_success < 0:
        torch.save(checkpoint_payload(), "checkpoints/adwa_gru_ppo.pt")
    print(f"saved {final}", flush=True)
    print(f"Best holdout-building success={max(best_holdout_success, 0.0):.1%}; "
          f"saved checkpoints/adwa_gru_ppo.pt")


if __name__ == "__main__":
    main()
