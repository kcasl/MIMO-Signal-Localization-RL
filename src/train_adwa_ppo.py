"""Train a GRU-PPO policy on ADWA-benchmark buildings.

Continuous Von Mises heading (theta), CSI history + SAVN-CE/MAGNet pose +
previous action, MAGNet-style geodesic-progress reward. Occupancy maps are
the native ADWA PNGs (1 cm/pixel), not a downsampled pixel grid. Holdout is
by building (see adwa_env.HOLDOUT_BUILDINGS).

Example: python train_adwa_ppo.py --timesteps 2000000
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli, VonMises
from torch.nn import functional as F

from encoders import (
    MIMOChannelEncoder, PoseEncoder, ActionEncoder, SceneMemoryEncoder,
    split_observation,
)
from adwa_env import ADWANavigationEnv, EnvConfig


def von_mises_entropy(kappa: torch.Tensor) -> torch.Tensor:
    """Closed-form Von Mises entropy: log(2π I0(κ)) − κ I1(κ)/I0(κ)."""
    i0 = torch.special.i0(kappa)
    i1 = torch.special.i1(kappa)
    return torch.log(2.0 * math.pi * i0) - kappa * (i1 / i0.clamp_min(1e-12))


class RecurrentActorCritic(nn.Module):
    """MAGNet-style modality encoders + sliding encoder memory + Von Mises actor."""

    def __init__(self, observation_size: int, hidden_size: int = 128, memory_size: int = 16):
        super().__init__()
        self.mimo_encoder = MIMOChannelEncoder(embedding_size=128)
        self.pose_encoder = PoseEncoder(embedding_size=16)
        self.action_encoder = ActionEncoder(embedding_size=16)
        self.feature_size = (self.mimo_encoder.embedding_size
                             + self.pose_encoder.embedding_size
                             + self.action_encoder.embedding_size)
        self.memory_encoder = SceneMemoryEncoder(self.feature_size, embedding_size=hidden_size)
        self.actor_mu = nn.Linear(hidden_size, 1)
        self.actor_stop = nn.Linear(hidden_size, 1)
        nn.init.constant_(self.actor_stop.bias, -2.0)
        self.critic = nn.Linear(hidden_size, 1)
        self.log_concentration = nn.Parameter(torch.tensor(2.8))
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.observation_size = observation_size

    def concentration(self) -> torch.Tensor:
        return F.softplus(self.log_concentration) + 0.1

    def _dist(self, mu: torch.Tensor) -> VonMises:
        kappa = self.concentration().expand_as(mu)
        return VonMises(mu, kappa)

    def _stop_dist(self, logit: torch.Tensor) -> Bernoulli:
        return Bernoulli(logits=logit)

    def _action_vector(self, theta: torch.Tensor, stop: torch.Tensor) -> torch.Tensor:
        return torch.stack([torch.cos(theta), torch.sin(theta), stop.float()])

    def encode_observation(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        mimo, prev, pose = split_observation(observation)
        return torch.cat(
            (self.mimo_encoder(mimo), self.pose_encoder(pose), self.action_encoder(prev)),
            dim=-1,
        )

    def blank_memory(self, device=None):
        device = device or next(self.parameters()).device
        memory = torch.zeros(self.memory_size, 1, self.feature_size, device=device)
        mask = torch.zeros(1, self.memory_size, device=device)
        return memory, mask

    def _attend(self, fused: torch.Tensor, memory: torch.Tensor, mask: torch.Tensor):
        state = self.memory_encoder(fused, memory, mask)
        memory = torch.cat([memory[1:], fused.unsqueeze(0)], dim=0)
        mask = torch.cat(
            [mask[:, 1:], torch.ones(mask.shape[0], 1, device=mask.device)],
            dim=1,
        )
        return state, memory, mask

    def _reset_if_needed(self, memory, mask, episode_start):
        if not episode_start:
            return memory, mask
        return memory * 0, mask * 0

    def forward_sequence(self, observations, episode_starts, initial_hidden=None):
        if initial_hidden is None:
            memory, mask = self.blank_memory(observations.device)
        else:
            memory, mask = initial_hidden[0].clone(), initial_hidden[1].clone()
        fused = self.encode_observation(observations)
        mus, stop_logits, values = [], [], []
        for t in range(observations.shape[0]):
            memory, mask = self._reset_if_needed(memory, mask, bool(episode_starts[t].item()))
            state, memory, mask = self._attend(fused[t:t + 1], memory, mask)
            mus.append(self.actor_mu(state[0]).squeeze(-1))
            stop_logits.append(self.actor_stop(state[0]).squeeze(-1))
            values.append(self.critic(state[0]).squeeze())
        return torch.stack(mus), torch.stack(stop_logits), torch.stack(values)

    @torch.no_grad()
    def act(self, observation, hidden, episode_start):
        memory, mask = self._reset_if_needed(hidden[0], hidden[1], episode_start)
        fused = self.encode_observation(observation)
        state, memory, mask = self._attend(fused, memory, mask)
        mu = self.actor_mu(state[0]).squeeze(-1)
        stop_logit = self.actor_stop(state[0]).squeeze(-1)
        dist = self._dist(mu)
        stop_dist = self._stop_dist(stop_logit)
        theta = dist.sample()
        stop = stop_dist.sample()
        direction = self._action_vector(theta, stop)
        value = self.critic(state[0]).squeeze()
        log_prob = dist.log_prob(theta) + stop_dist.log_prob(stop)
        return direction.cpu().numpy(), log_prob, value, (memory, mask), theta, stop

    @torch.no_grad()
    def greedy_direction(self, observation, hidden, episode_start):
        memory, mask = self._reset_if_needed(hidden[0], hidden[1], episode_start)
        fused = self.encode_observation(observation)
        state, memory, mask = self._attend(fused, memory, mask)
        theta = self.actor_mu(state[0]).squeeze(-1)
        stop_logit = self.actor_stop(state[0]).squeeze(-1)
        stop = (stop_logit > 0.0).float()
        direction = self._action_vector(theta, stop)
        return direction.cpu().numpy(), (memory, mask)

    @torch.no_grad()
    def value(self, observation, hidden, episode_start):
        memory, mask = hidden[0].clone(), hidden[1].clone()
        memory, mask = self._reset_if_needed(memory, mask, episode_start)
        fused = self.encode_observation(observation)
        state, _, _ = self._attend(fused, memory, mask)
        return self.critic(state[0]).squeeze()


def gae(rewards, values, dones, bootstrap, gamma=0.99, lam=0.95):
    advantages, carry = torch.zeros_like(rewards), torch.tensor(0.0)
    for t in reversed(range(len(rewards))):
        next_value = bootstrap if t == len(rewards) - 1 else values[t + 1]
        not_done = 1.0 - dones[t]
        carry = rewards[t] + gamma * next_value * not_done - values[t] + gamma * lam * not_done * carry
        advantages[t] = carry
    return advantages


@torch.no_grad()
def evaluate(policy: RecurrentActorCritic, env: ADWANavigationEnv, device: torch.device,
             episodes: int = 40, split: str = "train") -> tuple[float, float]:
    policy.eval()
    successes, reward_sum = 0, 0.0
    for _ in range(episodes):
        observation, done, episode_start = env.reset(split=split), False, True
        hidden = policy.blank_memory(device)
        while not done:
            direction, hidden = policy.greedy_direction(
                torch.as_tensor(observation, device=device), hidden, episode_start
            )
            observation, reward, done, info = env.step(direction)
            reward_sum += reward
            episode_start = done
        successes += int(info["reached"])
    policy.train()
    return successes / episodes, reward_sum / episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--rollout", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--eval-interval", type=int, default=16384,
                        help="Greedy eval every N env steps. Native 1cm maps make each "
                             "step ~30ms, so 40+40 episodes of 300 steps would stall "
                             "training for ~10 minutes per eval.")
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--checkpoint-interval", type=int, default=50_000,
                        help="Numbered snapshot every N env steps (~20 over 1e6).")
    parser.add_argument("--memory-size", type=int, default=16,
                        help="Sliding encoder-memory length (DAgger-style, MAGNet SMT bank).")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = ADWANavigationEnv(EnvConfig(), args.seed)
    print(f"ADWA train buildings ({len(env.train_buildings)}): {', '.join(env.train_buildings)}")
    print(f"ADWA holdout/test buildings ({len(env.holdout_buildings)}): {', '.join(env.holdout_buildings)}")
    sample = env.map_pool[env.train_buildings[0]]
    print(f"occupancy: native PNG pixels, no downsample "
          f"(example {env.train_buildings[0]} {sample['world'].shape[1]}x{sample['world'].shape[0]} "
          f"@ {sample['resolution_m']} m/px)")
    print(f"obs={env.observation_size} (MIMO+pose+action encoders, memory={args.memory_size}, no lidar)")
    print(f"spawn geodesic {env.config.min_start_source_distance_m:.0f}–"
          f"{env.config.max_start_source_distance_m:.0f} m, "
          f"no source in ±{env.config.forward_cone_deg:.0f}° of heading, "
          f"wall or path-detour required")
    max_ep_steps = env.config.max_steps
    n_ckpt = max(args.timesteps // args.checkpoint_interval, 1)
    print(f"budget={args.timesteps} env-steps (no episode cap; max {max_ep_steps} steps/ep "
          f"→ roughly {args.timesteps // max_ep_steps}–{args.timesteps // 50} episodes); "
          f"numbered checkpoints every {args.checkpoint_interval} steps (~{n_ckpt} files)")
    policy = RecurrentActorCritic(env.observation_size, memory_size=args.memory_size).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    Path("checkpoints").mkdir(exist_ok=True)

    observation = env.reset(split="train")
    hidden = policy.blank_memory(device)
    episode_start = True
    total_steps = completed = successes = 0
    recent_successes: deque[int] = deque(maxlen=50)
    recent_collisions: deque[int] = deque(maxlen=1024)
    best_holdout_success = -1.0
    next_checkpoint = args.checkpoint_interval

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

    while total_steps < args.timesteps:
        window_hidden = (hidden[0].detach().clone(), hidden[1].detach().clone())
        observations, thetas, stops, log_probs, rewards, values, starts, dones = (
            [], [], [], [], [], [], [], [])
        for _ in range(min(args.rollout, args.timesteps - total_steps)):
            direction, log_prob, value, hidden, theta, stop = policy.act(
                torch.as_tensor(observation, device=device), hidden, episode_start
            )
            next_observation, reward, done, info = env.step(direction)
            observations.append(observation)
            thetas.append(theta.detach().cpu())
            stops.append(stop.detach().cpu())
            log_probs.append(log_prob.cpu())
            rewards.append(reward)
            values.append(value.cpu())
            starts.append(float(episode_start))
            dones.append(float(done))
            total_steps += 1
            recent_collisions.append(int(info["collision"]))
            observation, episode_start = next_observation, done
            if done:
                completed += 1
                successes += int(info["reached"])
                recent_successes.append(int(info["reached"]))
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
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

        deterministic = ""
        if total_steps % args.eval_interval < args.rollout:
            print(f"eval at steps={total_steps} "
                  f"({args.eval_episodes} train + {args.eval_episodes} holdout, greedy)...",
                  flush=True)
            eval_env = ADWANavigationEnv(seed=args.seed + total_steps)
            train_s, train_r = evaluate(policy, eval_env, device, args.eval_episodes, split="train")
            hold_s, hold_r = evaluate(policy, eval_env, device, args.eval_episodes, split="holdout")
            deterministic = (f" train-eval={train_s:.1%} ({train_r:6.2f})"
                             f" holdout-eval={hold_s:.1%} ({hold_r:6.2f})")
            if hold_s >= best_holdout_success:
                best_holdout_success = hold_s
                torch.save(checkpoint_payload(holdout_success=hold_s, train_success=train_s),
                           "checkpoints/adwa_gru_ppo.pt")
        if total_steps >= next_checkpoint:
            snap = Path("checkpoints") / f"adwa_gru_ppo_step{total_steps:07d}.pt"
            torch.save(checkpoint_payload(), snap)
            print(f"saved {snap}", flush=True)
            next_checkpoint += args.checkpoint_interval
        n_ep = max(len(recent_successes), 1)
        n_col = max(len(recent_collisions), 1)
        recent = sum(recent_successes) / n_ep
        collide = sum(recent_collisions) / n_col
        print(f"steps={total_steps:>7} episodes={completed:>4} "
              f"recent-success={sum(recent_successes)}/{len(recent_successes)} ({recent:.1%}) "
              f"collide={sum(recent_collisions)}/{len(recent_collisions)} ({collide:.1%})"
              f"{deterministic} loss={loss.item():.3f} "
              f"entropy={entropy.item():.3f} kappa={float(policy.concentration().detach()):.2f}")

    final = Path("checkpoints") / f"adwa_gru_ppo_step{total_steps:07d}.pt"
    if not final.exists():
        torch.save(checkpoint_payload(), final)
        print(f"saved {final}", flush=True)
    if best_holdout_success < 0:
        torch.save(checkpoint_payload(), "checkpoints/adwa_gru_ppo.pt")
    print(f"Best holdout-building success={max(best_holdout_success, 0.0):.1%}; saved checkpoints/adwa_gru_ppo.pt")


if __name__ == "__main__":
    main()
