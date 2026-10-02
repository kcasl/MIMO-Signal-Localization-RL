"""RF-only navigation environment over real ADWA-benchmark building floorplans.

Successor to simple_2D/mimo_rl_env.py: same POMDP design (RF-CSI history plus
SAVN-CE/MAGNet-style pose and previous action in; continuous heading out;
map/coordinates/source never observed) and MAGNet-style geodesic-progress
reward, but the map pool is now real building floorplans (see adwa_maps.py)
split train/holdout BY BUILDING.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import pi

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from adwa_maps import list_buildings, load_map
from rf_physics import (
    COLLISION_DIM, CSI_BASE_DIM, N_RX, N_TX, RF_PATH_DIM,
    bfs_distance_field, bresenham, neighbours, path_rf_features, simulate_mimo_link,
    _range_m,
)

# MAGNet PoseEncoder defaults: metres / 20, episode step / 500.
POSE_DIST_SCALE = 20.0
POSE_TIME_SCALE = 500.0
POSE_DIM = 5
PREV_ACTION_DIM = 2
# amp+phase+svd+power — keep name CSI_DIM for the H-derived block.
CSI_DIM = 2 * N_RX * N_TX + N_RX + 1
assert CSI_DIM == CSI_BASE_DIM
FEATURE_DIM = CSI_DIM + RF_PATH_DIM + COLLISION_DIM + PREV_ACTION_DIM + POSE_DIM

SLACK_REWARD = -0.01
SUCCESS_REWARD = 20.0
DISTANCE_REWARD_SCALE = 1.0
COLLISION_PENALTY = 0.5
GRAZE_PENALTY = 0.15
INTENSITY_REWARD_SCALE = 1.5
MOVE_REWARD_SCALE = 0.3
OPEN_REWARD_SCALE = 0.2
# STOP is NOT terminal unless the agent is within source_radius: a wrong STOP
# just wastes the step (no move) and the agent may move again afterwards. To
# stop it from "camping" in place, consecutive STOPs incur an escalating dwell
# penalty base + scale*(consecutive-1), capped so returns stay bounded.
STOP_DWELL_BASE = 0.3
STOP_DWELL_SCALE = 0.3
STOP_DWELL_CAP = 3.0
CLEAR_RAY_MAX_M = 2.0
CLEAR_RAY_DEG = (-20.0, -10.0, 0.0, 10.0, 20.0)

# Fixed regardless of any env instance's own `seed`, so every env -- the one
# collecting rollouts and the ones evaluate() builds separately -- sees the
# identical building split.
HOLDOUT_BUILDINGS = ("Eastville", "Mosquito", "Sisters2", "Scioto2")


@dataclass(frozen=True)
class EnvConfig:
    max_steps: int = 300
    history_length: int = 4
    source_radius_m: float = 1.0
    step_size_m: float = 0.3
    min_start_source_distance_m: float = 5.0
    max_start_source_distance_m: float = 12.0
    forward_cone_deg: float = 50.0
    require_detour: bool = True


_MAP_POOL: dict[str, dict] | None = None


def _largest_connected_component(world: np.ndarray) -> np.ndarray:
    """Free pixels reachable from each other as an (N, 2) int array (x, y).

    Sampling start/source only from the largest component guarantees every
    episode is actually solvable even if the floorplan has disconnected pockets.
    """
    ny, nx = world.shape
    seen = world.copy()
    best: list[tuple[int, int]] = []
    for y in range(ny):
        row = seen[y]
        for x in range(nx):
            if row[x]:
                continue
            cells: list[tuple[int, int]] = []
            frontier = deque([(x, y)])
            seen[y, x] = True
            while frontier:
                cx, cy = frontier.popleft()
                cells.append((cx, cy))
                for nx_, ny_ in neighbours((cx, cy), world):
                    if not seen[ny_, nx_]:
                        seen[ny_, nx_] = True
                        frontier.append((nx_, ny_))
            if len(cells) > len(best):
                best = cells
    if not best:
        return np.zeros((0, 2), dtype=np.int32)
    return np.asarray(best, dtype=np.int32)


def _structure_band(world: np.ndarray, cells: np.ndarray, resolution_m: float,
                    near_m: float = 0.2, far_m: float = 5.0,
                    coarse_m: float = 0.05) -> np.ndarray:
    """Keep free cells whose distance-to-wall is in [near_m, far_m].

    Drops the empty outdoor field that dominates a raw occupancy PNG, so
    start/source are sampled from rooms and corridors rather than the
    unbounded white padding.
    """
    if len(cells) == 0:
        return cells
    factor = max(1, int(round(coarse_m / resolution_m)))
    h, w = world.shape
    hc, wc = max(1, h // factor), max(1, w // factor)
    coarse = world[:hc * factor, :wc * factor].reshape(hc, factor, wc, factor).any(axis=(1, 3))
    dist = np.full((hc, wc), np.inf, dtype=np.float32)
    frontier = deque()
    ys, xs = np.nonzero(coarse)
    for y, x in zip(ys.tolist(), xs.tolist()):
        dist[y, x] = 0.0
        frontier.append((x, y))
    while frontier:
        x, y = frontier.popleft()
        d = dist[y, x]
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < wc and 0 <= ny < hc and dist[ny, nx] > d + 1.0:
                dist[ny, nx] = d + 1.0
                frontier.append((nx, ny))
    d_m = dist * (resolution_m * factor)
    cx = np.clip(cells[:, 0] // factor, 0, wc - 1)
    cy = np.clip(cells[:, 1] // factor, 0, hc - 1)
    band = (d_m[cy, cx] >= near_m) & (d_m[cy, cx] <= far_m)
    picked = cells[band]
    return picked if len(picked) > 200 else cells


def _build_map_pool() -> dict[str, dict]:
    pool = {}
    for name in list_buildings():
        building = load_map(name)
        free_cells = _structure_band(
            building.world, _largest_connected_component(building.world),
            building.resolution_m,
        )
        pool[name] = {
            "world": building.world,
            "free_cells": free_cells,
            "resolution_m": building.resolution_m,
            "origin_m": building.origin_m,
        }
        print(f"  loaded {name}: {building.world.shape[1]}x{building.world.shape[0]} px "
              f"@ {building.resolution_m*100:.0f} cm/px, spawn={len(free_cells)}")
    return pool


def _get_map_pool() -> dict[str, dict]:
    global _MAP_POOL
    if _MAP_POOL is None:
        print("Loading ADWA occupancy maps at native PNG resolution (no downsample)")
        _MAP_POOL = _build_map_pool()
    return _MAP_POOL


class ADWANavigationEnv:
    """POMDP: RF history + pose + prev action in, a continuous 2D heading out.

    Same interface as simple_2D.mimo_rl_env.MIMORFNavigationEnv, so the
    PPO training loop in train_adwa_ppo.py is a near-verbatim reuse of
    simple_2D/train_recurrent_ppo.py, just pointed at this environment.
    """

    def __init__(self, config: EnvConfig | None = None, seed: int | None = None):
        self.config = config or EnvConfig()
        self.rng = np.random.default_rng(seed)
        self.map_pool = _get_map_pool()
        all_buildings = sorted(self.map_pool)
        self.holdout_buildings = [b for b in all_buildings if b in HOLDOUT_BUILDINGS]
        self.train_buildings = [b for b in all_buildings if b not in HOLDOUT_BUILDINGS]
        if not self.train_buildings:
            raise RuntimeError("No ADWA training buildings found under adwa_benchmark/")
        if not self.holdout_buildings:
            raise RuntimeError(
                f"Holdout buildings {HOLDOUT_BUILDINGS} not found in adwa_benchmark/; "
                f"available={all_buildings}"
            )

        self.building_name = self.train_buildings[0]
        self._select_building(self.building_name)
        self.position, self.heading = np.array(self.free_cells[0], dtype=float), 0.0
        self._start_position = self.position.copy()
        self._start_heading = self.heading
        self.source = self.free_cells[0]
        self.last_direction = np.array([1.0, 0.0], dtype=np.float32)
        self._last_collision = 0.0
        self._intensity = 0.0
        self.steps, self._remaining_distance = 0, 0
        self.legacy_obs = False

    def _select_building(self, name: str) -> None:
        m = self.map_pool[name]
        self.building_name = name
        self.world = m["world"]
        self.free_cells = m["free_cells"]
        self.resolution_m = float(m["resolution_m"])

    def _sample_building(self, split: str) -> str:
        pool = self.train_buildings if split == "train" else (
            self.holdout_buildings if split == "holdout" else sorted(self.map_pool))
        return pool[int(self.rng.integers(len(pool)))]

    @property
    def observation_size(self) -> int:
        return (CSI_DIM + PREV_ACTION_DIM + POSE_DIM) if self.legacy_obs else FEATURE_DIM

    def _sample_position(self) -> tuple[int, int]:
        i = int(self.rng.integers(len(self.free_cells)))
        return int(self.free_cells[i, 0]), int(self.free_cells[i, 1])

    def _line_has_wall(self, a, b) -> bool:
        sa, sb = self._cell(a), self._cell(b)
        for x, y in bresenham(sa, sb):
            if (x, y) == sa:
                continue
            if self.world[y, x]:
                return True
        return False

    def _requires_detour(self, start, source, field) -> bool:
        """Reject a clear same-room dash; 4-connected BFS equals Manhattan in open space."""
        if self._line_has_wall(start, source):
            return True
        geo = int(field[start[1], start[0]])
        manhattan = abs(int(start[0]) - int(source[0])) + abs(int(start[1]) - int(source[1]))
        extra = max(20, int(round(1.0 / self.resolution_m)))
        return geo >= manhattan + extra

    def _heading_faces_source(self, start, heading: float, source) -> bool:
        delta = np.asarray(source, dtype=float) - np.asarray(start, dtype=float)
        dist = float(np.linalg.norm(delta))
        if dist < 1e-8:
            return True
        forward = np.array((np.cos(heading), np.sin(heading)))
        return float(forward @ (delta / dist)) >= np.cos(np.deg2rad(self.config.forward_cone_deg))

    def _sample_heading_not_facing(self, start, source) -> float:
        for _ in range(40):
            heading = float(self.rng.uniform(-pi, pi))
            if not self._heading_faces_source(start, heading, source):
                return heading
        delta = np.asarray(source, dtype=float) - np.asarray(start, dtype=float)
        return float(np.arctan2(-delta[1], -delta[0]))

    def _sample_start_source_pair(self):
        """Source first, then a start on the geodesic ring that is not a free dash."""
        dmin = self.config.min_start_source_distance_m / self.resolution_m
        dmax = self.config.max_start_source_distance_m / self.resolution_m
        cells = self.free_cells
        field = None
        for _ in range(40):
            source = self._sample_position()
            field = bfs_distance_field(self.world, source)
            dists = field[cells[:, 1], cells[:, 0]]
            ok = (dists >= dmin) & (dists <= dmax)
            if not np.any(ok):
                continue
            candidates = np.flatnonzero(ok)
            pick = self.rng.choice(candidates, size=min(80, len(candidates)), replace=False)
            for idx in pick:
                start = (int(cells[idx, 0]), int(cells[idx, 1]))
                if (not self.config.require_detour) or self._requires_detour(start, source, field):
                    return start, source, field
        source = self._sample_position()
        field = bfs_distance_field(self.world, source)
        dists = field[cells[:, 1], cells[:, 0]]
        ok = dists >= dmin
        idx = int(self.rng.choice(np.flatnonzero(ok))) if np.any(ok) else 0
        start = (int(cells[idx, 0]), int(cells[idx, 1]))
        return start, source, field

    def _pose_vector(self) -> np.ndarray:
        """SAVN-CE/MAGNet pose: (x, y, heading) relative to episode start, plus time."""
        meters = self.resolution_m
        dx = (float(self.position[0]) - float(self._start_position[0])) * meters
        dy = (float(self.position[1]) - float(self._start_position[1])) * meters
        c, s = np.cos(self._start_heading), np.sin(self._start_heading)
        x_rel = c * dx + s * dy
        y_rel = -s * dx + c * dy
        heading_rel = self.heading - self._start_heading
        return np.array((
            x_rel / POSE_DIST_SCALE,
            y_rel / POSE_DIST_SCALE,
            np.cos(heading_rel),
            np.sin(heading_rel),
            float(self.steps) / POSE_TIME_SCALE,
        ), dtype=np.float32)

    def _feature(self) -> np.ndarray:
        if self.legacy_obs:
            from rf_physics import _direct_channel
            h = _direct_channel(self.position, self.source, self.world, self.heading,
                                resolution_m=self.resolution_m)
            amp = np.log10(np.abs(h) + 1e-15).ravel()
            phase = (np.angle(h) / pi).ravel()
            singular = np.log10(np.linalg.svd(h, compute_uv=False) + 1e-15)
            mean_power = np.array([np.log10(np.mean(np.abs(h) ** 2) + 1e-30)])
            return np.concatenate((amp, phase, singular, mean_power,
                                   self.last_direction, self._pose_vector())).astype(np.float32)
        h, paths = simulate_mimo_link(
            self.position, self.source, self.world, rx_heading=self.heading,
            resolution_m=self.resolution_m,
        )
        amp = np.log10(np.abs(h) + 1e-15).ravel()
        phase = (np.angle(h) / pi).ravel()
        singular = np.log10(np.linalg.svd(h, compute_uv=False) + 1e-15)
        mean_power = np.array([np.log10(np.mean(np.abs(h) ** 2) + 1e-30)])
        self._intensity = float(mean_power[0])
        return np.concatenate((
            amp, phase, singular, mean_power,
            path_rf_features(paths, self.heading),
            np.array((self._last_collision,), dtype=np.float32),
            self.last_direction, self._pose_vector(),
        )).astype(np.float32)

    def _observation(self) -> np.ndarray:
        return self._feature()

    def _cell(self, point) -> tuple[int, int]:
        ny, nx = self.world.shape
        x = int(np.clip(round(float(point[0])), 0, nx - 1))
        y = int(np.clip(round(float(point[1])), 0, ny - 1))
        return x, y

    def _oracle_path_distance(self, point) -> int:
        x, y = self._cell(point)
        return int(self.distance_field[y, x])

    def apply_curriculum(self, config: EnvConfig) -> None:
        """Swap spawn difficulty without reloading occupancy maps."""
        self.config = config

    def _source_distance_m(self) -> float:
        delta = np.asarray(self.source, dtype=float) - self.position
        return float(np.linalg.norm(delta)) * self.resolution_m

    def _geodesic_next_cell(self, cell: tuple[int, int]) -> tuple[int, int] | None:
        """Privileged 4-neighbour step that decreases BFS distance. Not an observation."""
        cx, cy = cell
        here = int(self.distance_field[cy, cx])
        if here <= 0:
            return None
        best_d = here
        choices: list[tuple[int, int]] = []
        for nx, ny in neighbours((cx, cy), self.world):
            d = int(self.distance_field[ny, nx])
            if d < 0 or d >= here:
                continue
            if d < best_d:
                best_d = d
                choices = [(nx, ny)]
            elif d == best_d:
                choices.append((nx, ny))
        if not choices:
            return None
        hx, hy = float(np.cos(self.heading)), float(np.sin(self.heading))
        return max(choices, key=lambda p: (p[0] - cx) * hx + (p[1] - cy) * hy)

    def expert_action(self) -> np.ndarray:
        """Privileged teacher: geodesic heading + STOP inside 1 m.

        Look-ahead follows the BFS chain, then keeps the farthest cell whose
        straight line is free. A raw 0.3 m chord through a wall would point at
        the source and pin the teacher to the obstacle (same failure mode as
        CSI-only PPO). Never written into the policy observation.
        """
        dist_m = self._source_distance_m()
        if dist_m <= self.config.source_radius_m:
            delta = np.asarray(self.source, dtype=float) - self.position
            nrm = float(np.linalg.norm(delta))
            if nrm < 1e-8:
                vec = np.array((np.cos(self.heading), np.sin(self.heading)))
            else:
                vec = delta / nrm
            return np.array((vec[0], vec[1], 1.0), dtype=np.float32)

        look = max(1, int(round(self.config.step_size_m / self.resolution_m)))
        x, y = self._cell(self.position)
        chain = [(x, y)]
        for _ in range(look):
            nxt = self._geodesic_next_cell((x, y))
            if nxt is None:
                break
            x, y = nxt
            chain.append(nxt)
        chosen = chain[0]
        for cell in chain[1:]:
            if self._line_has_wall(self.position, cell):
                break
            chosen = cell
        if chosen == chain[0] and len(chain) > 1:
            chosen = chain[1]
        delta = np.array(chosen, dtype=float) - self.position
        nrm = float(np.linalg.norm(delta))
        if nrm < 1e-8:
            vec = np.array((np.cos(self.heading), np.sin(self.heading)))
        else:
            vec = delta / nrm
        return np.array((vec[0], vec[1], 0.0), dtype=np.float32)

    def _advance_from(self, start: np.ndarray, unit: np.ndarray, max_cells: float):
        """Walk up to ``max_cells`` along ``unit``. Return (pos, hit, normal, moved)."""
        end = start + max_cells * unit
        sx, sy = self._cell(start)
        last = np.asarray(start, dtype=float)
        for x, y in bresenham((sx, sy), self._cell(end)):
            if (x, y) == (sx, sy):
                continue
            if self.world[y, x]:
                normal = last - np.array((float(x), float(y)))
                nrm = float(np.linalg.norm(normal))
                normal = normal / nrm if nrm > 1e-8 else None
                moved = float(np.linalg.norm(last - start))
                return last, True, normal, moved
            last = np.array((float(x), float(y)))
        return end, False, None, float(np.linalg.norm(end - start))

    def _cone_clearance_m(self, heading: float) -> float:
        """Privileged forward rays for reward only; never written into the observation."""
        clears = [
            _range_m(self.position, heading + deg * pi / 180.0, self.world,
                     CLEAR_RAY_MAX_M, self.resolution_m)
            for deg in CLEAR_RAY_DEG
        ]
        return float(min(clears))

    def _commanded_move(self, unit: np.ndarray) -> tuple[np.ndarray, bool]:
        """Partial step, then slide along the wall instead of aborting the episode.

        Contact normal comes from the occupancy hit, not from a rangefinder.
        Remaining travel is projected onto the wall tangent so the agent can
        keep moving in a corridor after a glancing collision.
        """
        step_cells = self.config.step_size_m / self.resolution_m
        pos, hit, normal, moved = self._advance_from(self.position, unit, step_cells)
        if not hit:
            return pos, False
        remaining = step_cells - moved
        if remaining < 1.0 or normal is None:
            return pos, True
        t1 = np.array((-normal[1], normal[0]))
        tn = float(np.linalg.norm(t1))
        if tn < 1e-8:
            return pos, True
        t_hat = t1 / tn
        here = self._oracle_path_distance(pos)
        best_pos, best_score = pos, (here, 0.0)
        for tangent in (t_hat, -t_hat):
            cand, _, _, slid = self._advance_from(pos, tangent, remaining)
            score = (self._oracle_path_distance(cand), -slid)
            if score < best_score:
                best_pos, best_score = cand, score
        return best_pos, True

    def reset(self, *, seed: int | None = None, split: str = "train") -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._select_building(self._sample_building(split))
        start, source, field = self._sample_start_source_pair()
        self.source = source
        self.distance_field = field
        self.position = np.array(start, dtype=float)
        self.heading = self._sample_heading_not_facing(start, source)
        self.steps = 0
        self._start_position = self.position.copy()
        self._start_heading = self.heading
        self.last_direction = np.array([np.cos(self.heading), np.sin(self.heading)], dtype=np.float32)
        self._last_collision = 0.0
        self._stop_dwell = 0
        self._remaining_distance = self._oracle_path_distance(self.position)
        obs = self._observation()
        return obs

    def step(self, direction: np.ndarray):
        """Heading (cos, sin) plus optional stop flag as a 3rd component > 0.5.

        Forward always travels a fixed 0.3 m. Success requires STOP inside 1 m
        of the RF source; walking through the radius does not count.
        """
        direction = np.asarray(direction, dtype=float).reshape(-1)
        stop = bool(direction.shape[0] >= 3 and float(direction[2]) > 0.5)
        heading_vec = direction[:2]
        norm = float(np.linalg.norm(heading_vec))
        unit = heading_vec / norm if norm > 1e-8 else np.array([np.cos(self.heading), np.sin(self.heading)])

        old_distance = self._remaining_distance
        old_position = self.position.copy()
        old_intensity = self._intensity
        self.heading = float(np.arctan2(unit[1], unit[0]))
        clipped = False
        collision = False
        moved = 0.0
        if not stop:
            clearance_m = self._cone_clearance_m(self.heading)
            self.position, clipped = self._commanded_move(unit)
            self._remaining_distance = self._oracle_path_distance(self.position)
            moved = float(np.linalg.norm(self.position - old_position))
            step_cells = self.config.step_size_m / self.resolution_m
            collision = bool(clipped and moved < 0.3 * step_cells)
        self.steps += 1
        src = np.asarray(self.source, dtype=float)
        dist_m = float(np.linalg.norm(self.position - src)) * self.resolution_m
        reached = bool(stop and dist_m <= self.config.source_radius_m)
        # Only a STOP within source_radius ends the episode ("I am at the
        # source"). A STOP elsewhere is a non-terminal wait: the step is spent
        # in place and the agent is free to move again next step.
        terminated = reached or self.steps >= self.config.max_steps
        progress_m = (old_distance - self._remaining_distance) * self.resolution_m
        moved_m = moved * self.resolution_m
        travel_frac = float(np.clip(moved_m / self.config.step_size_m, 0.0, 1.0))
        reward = SLACK_REWARD + DISTANCE_REWARD_SCALE * progress_m
        if stop:
            if reached:
                reward += SUCCESS_REWARD
            else:
                # Non-terminal wrong STOP: escalating dwell penalty for staying
                # put. Consecutive STOPs cost more, capped so a single stray
                # STOP (then moving on) is cheap but camping is not.
                self._stop_dwell += 1
                dwell_pen = STOP_DWELL_BASE + STOP_DWELL_SCALE * (self._stop_dwell - 1)
                reward -= min(dwell_pen, STOP_DWELL_CAP)
        else:
            self._stop_dwell = 0
            openness = float(np.clip(clearance_m / CLEAR_RAY_MAX_M, 0.0, 1.0))
            reward += MOVE_REWARD_SCALE * travel_frac
            if not collision:
                reward += OPEN_REWARD_SCALE * travel_frac * openness
            if collision:
                reward -= COLLISION_PENALTY
            elif clipped:
                reward -= GRAZE_PENALTY
        self.last_direction = (np.zeros(2, dtype=np.float32) if stop
                               else unit.astype(np.float32))
        self._last_collision = 1.0 if collision else (0.35 if clipped else 0.0)
        obs = self._observation()
        if not stop:
            reward += INTENSITY_REWARD_SCALE * (self._intensity - old_intensity)
        return obs, float(reward), terminated, {
            "reached": reached, "collision": collision, "stop": stop,
            "steps": self.steps, "building": self.building_name,
        }

    def render(self, trajectory: list[tuple[float, float]] | None = None):
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.imshow(self.world, origin="upper", cmap="gray_r", interpolation="nearest")
        if trajectory:
            x, y = zip(*trajectory)
            ax.plot(x, y, color="deepskyblue", lw=2, label="policy trajectory")
        ax.scatter(*self.source, marker="*", s=190, c="crimson", label="RF source", zorder=3)
        ax.scatter(*self.position, s=70, c="lime", edgecolors="black", label="agent", zorder=3)
        ny, nx = self.world.shape
        ax.set(xlim=(0, nx - 1), ylim=(ny - 1, 0), aspect="equal",
               title=f"RF-only navigation -- {self.building_name}")
        ax.legend(loc="lower right")
        return fig

    def render_animation(self, trajectory: list[tuple[float, float]], interval: int = 80) -> FuncAnimation:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.imshow(self.world, origin="upper", cmap="gray_r", interpolation="nearest")
        ax.scatter(*self.source, marker="*", s=190, c="crimson", label="RF source", zorder=3)
        trail, = ax.plot([], [], color="deepskyblue", lw=2, label="policy trajectory")
        agent = ax.scatter([], [], s=90, c="lime", edgecolors="black", label="agent", zorder=4)
        step_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top",
                            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
        ny, nx = self.world.shape
        ax.set(xlim=(0, nx - 1), ylim=(ny - 1, 0), aspect="equal",
               title=f"RF-only navigation -- {self.building_name}")
        ax.legend(loc="lower right")

        def update(frame):
            xs, ys = zip(*trajectory[:frame + 1])
            trail.set_data(xs, ys)
            agent.set_offsets([trajectory[frame]])
            step_text.set_text(f"step {frame}/{len(trajectory) - 1}")
            return trail, agent, step_text

        return FuncAnimation(fig, update, frames=len(trajectory), interval=interval, repeat=False)
