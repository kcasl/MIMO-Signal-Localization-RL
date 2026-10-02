# MIMOPathFinder

RF-only source-seeking: an agent must reach a hidden RF source in an unknown
indoor environment using only a history of MIMO CSI, a SAVN-CE/MAGNet-style
pose relative to the episode start, and the previous heading — never the
occupancy map, global coordinates, or the source location. A GRU actor-critic
outputs a continuous heading angle θ from a Von Mises distribution; the only
trainer is PPO (`train_adwa_ppo.py`).

## Project layout

- **`simple_2D/`** — first-phase codebase. A procedurally generated 2D grid
  world used to develop the observation/reward design and training
  methodology. See `simple_2D/README.md`.

- **Top level (this directory)** — current codebase, built on the
  **ADWA IROS-2020 benchmark**: 17 real building floorplans (ROS
  map_server-style occupancy grids + recorded navigation tasks), extracted
  under `adwa_benchmark/`. Occupancy is used at native PNG resolution
  (1 cm/pixel); it is not max-pooled into a coarser pixel grid.

  - `rf_physics.py` — grid-size-agnostic MIMO channel model, BFS/A* pathfinding.
  - `adwa_maps.py` — loads a building occupancy PNG at yaml resolution.
  - `adwa_env.py` — `ADWANavigationEnv`: CSI + pose + previous action in,
    continuous heading out; MAGNet-style geodesic-progress reward; buildings
    split train/holdout (`adwa_env.HOLDOUT_BUILDINGS`).
  - `train_adwa_ppo.py` — **trainer**: recurrent PPO, continuous θ.
  - `evaluate_adwa_policy.py` — renders one greedy PPO rollout (PNG + GIF)
    on a holdout building by default.

```bash
poetry lock
poetry install
poetry run python train_adwa_ppo.py --timesteps 1000000
poetry run python evaluate_adwa_policy.py
```

Training saves `checkpoints/adwa_gru_ppo.pt` (best holdout-building success
under greedy evaluation).

## Model architecture

### Observation (input)

Every environment step, `adwa_env._feature()` turns the current 4×4 MIMO
channel plus MAGNet-style pose/previous action into a 44-dim vector:

| Component | Source | Dims |
|---|---|---|
| `amp` | `log10(|H|)`, flattened | 16 |
| `phase` | `angle(H)/pi`, flattened | 16 |
| `singular` | `log10(svd(H))` | 4 |
| `mean_power` | `log10(mean(|H|^2))` | 1 |
| `rf_paths` | K-factor, RMS delay spread, path count, dominant ego-AoA, 8-bin AoA power | 13 |
| `lidar` | 8-beam ego rangefinder, range / 2.4 m | 8 |
| `proximity` | min lidar (nearest wall) | 1 |
| `last_collision` | 1 if the previous step hit a wall | 1 |
| `last_direction` | unit vector of the previous executed heading | 2 |
| `pose` | MAGNet format: `(x/20, y/20, cos ψ, sin ψ, t/500)` relative to episode start | 5 |

`H` is a 4×4 narrowband matrix: transmissive direct path plus up to 2-bounce
specular wall reflections. Pose is the agent's displacement and heading **in
the episode-start frame**, not source GPS and not map cells. Lidar/proximity
are onboard local range only. A sliding window of `history_length=4` frames
is concatenated:

```
observation_size = history_length * 67 = 268
```

### Network (`train_adwa_ppo.RecurrentActorCritic`)

```
observation (268)
     │
     ▼
Linear(268 → 192) + Tanh          encoder
     │
     ▼
GRU(192 → 192), 1 layer           gru  (hidden persists across the episode)
     │
     ├─ Linear(192 → 1)           actor μ_θ
     │         + learnable κ      Von Mises(μ_θ, κ)  → sample θ at train time
     └─ Linear(192 → 1)           critic V(s)
     │
     ▼
(cos θ, sin θ)                    unit heading
```

`(cos θ, sin θ)` is always unit-norm. A Von Mises on θ is wrap-safe across
`±π` (unlike a raw Gaussian on the angle). Evaluation uses the mean heading
`μ_θ` with no sampling.

### Reward (training signal for PPO)

Privileged BFS geodesic (map/source used only here, never in the observation),
MAGNet-style plus a collision term because this env has no STOP action:

```
r_t = -0.01
    + 1.0 * (d^{BFS}_{t-1} - d^{BFS}_t) * resolution_m
    + 10.0 if reached
    -  0.5 if collision
```

No Euclidean-to-source term, no teacher-alignment bonus, no extra timeout
penalty (slack already taxes long episodes).
