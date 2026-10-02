"""Grid-size-agnostic RF/MIMO physics and pathfinding utilities.

Successor to simple_2D/RF_source_seeking_2D.py's channel model, generalised
to operate on any occupancy grid shape (world.shape) instead of a single
hardcoded (NX, NY) — needed because each ADWA benchmark building has its own
map size.

Grid convention: a point is (x, y); world[y, x] is True for a wall.
"""
from __future__ import annotations

import heapq
from collections import deque
from math import pi

import numpy as np

# Ideal narrowband MIMO configuration -- unchanged from the simple_2D phase.
N_TX = 4
N_RX = 4
CARRIER_HZ = 2.4e9
LIGHT_SPEED = 299_792_458.0
WAVELENGTH = LIGHT_SPEED / CARRIER_HZ
ELEMENT_SPACING = WAVELENGTH / 2
TX_POWER_W = 0.1             # Total transmit power: 20 dBm
NOISE_POWER_W = 1e-12        # Receiver noise power: -90 dBm

# Specular-ray tracing walks occupancy pixels; amplitudes/phase/capture use
# metres via ``resolution_m`` so a 1 cm/px ADWA map and a 1 m toy grid match.
N_SPECULAR_RAYS = 96
MAX_BOUNCES = 2
REFLECTION_GAIN = 0.6
CAPTURE_RADIUS_M = 0.36
MAX_PATH_DIST_M = 36.0
WALL_LOSS_DB = 18.0  # per wall *crossing*, not per occupied pixel
AOA_BINS = 8
LIDAR_BEAMS = 8
FRONT_LIDAR_DEG = (-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0)
LIDAR_MAX_M = 4.0
RF_PATH_DIM = 1 + 1 + 1 + 2 + AOA_BINS  # K-factor, delay spread, n_paths, aoa, hist
FRONT_LIDAR_DIM = len(FRONT_LIDAR_DEG)
LIDAR_DIM = LIDAR_BEAMS + FRONT_LIDAR_DIM
CLEARANCE_DIM = 1
IMMINENT_DIM = 1
PROXIMITY_DIM = 1
COLLISION_DIM = 1
CSI_BASE_DIM = 2 * N_RX * N_TX + N_RX + 1  # amp + phase + svd + mean power


def neighbours(point: tuple[int, int], world: np.ndarray):
    ny, nx = world.shape
    x, y = point
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        px, py = x + dx, y + dy
        if 0 <= px < nx and 0 <= py < ny and not world[py, px]:
            yield px, py


def astar(start: tuple[int, int], goal: tuple[int, int], world: np.ndarray):
    """Return a shortest collision-free four-neighbour path."""
    frontier = [(0, start)]
    came_from = {start: None}
    cost = {start: 0}

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break
        for nxt in neighbours(current, world):
            new_cost = cost[current] + 1
            if nxt not in cost or new_cost < cost[nxt]:
                cost[nxt] = new_cost
                priority = new_cost + abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1])
                heapq.heappush(frontier, (priority, nxt))
                came_from[nxt] = current
    else:
        raise RuntimeError("The goal is unreachable")

    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from[current]
    return path[::-1]


def bfs_distance_field(world: np.ndarray, source: tuple[int, int]) -> np.ndarray:
    """Wall-aware shortest step-count from every free cell to ``source``.

    A single BFS flood fill over the four-neighbour free-cell graph, so a
    cell's value is the length of an actually reachable route around walls,
    never Euclidean/straight-line distance. ``world``/``source`` are fixed
    for the life of an episode's map, so callers should compute this once
    per map and reuse it instead of re-running BFS on every step.
    """
    dist = np.full(world.shape, -1, dtype=np.int32)
    sx, sy = source
    dist[sy, sx] = 0
    frontier = deque([source])
    while frontier:
        current = frontier.popleft()
        cx, cy = current
        for nx, ny in neighbours(current, world):
            if dist[ny, nx] == -1:
                dist[ny, nx] = dist[cy, cx] + 1
                frontier.append((nx, ny))
    return dist


def bresenham(a: tuple[int, int], b: tuple[int, int]):
    """Yield cells sampled along the straight line from a to b."""
    x0, y0 = a
    x1, y1 = b
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    error = dx + dy
    while True:
        yield x0, y0
        if (x0, y0) == (x1, y1):
            return
        twice_error = 2 * error
        if twice_error >= dy:
            error += dy
            x0 += sx
        if twice_error <= dx:
            error += dx
            y0 += sy


def count_path_corners(path: list[tuple[int, int]]) -> int:
    """Number of direction changes along a four-neighbour path."""
    if len(path) < 3:
        return 0
    corners = 0
    prev = (path[1][0] - path[0][0], path[1][1] - path[0][1])
    for (x0, y0), (x1, y1) in zip(path[1:-1], path[2:]):
        cur = (x1 - x0, y1 - y0)
        corners += cur != prev
        prev = cur
    return corners


def ula_positions(center: tuple[float, float], n_elements: int, heading: float,
                  resolution_m: float = 1.0):
    """Element locations in occupancy-pixel coordinates for an ideal ULA."""
    perpendicular = (-np.sin(heading), np.cos(heading))
    offsets = (np.arange(n_elements) - (n_elements - 1) / 2) * ELEMENT_SPACING / resolution_m
    return np.array([
        (center[0] + offset * perpendicular[0], center[1] + offset * perpendicular[1])
        for offset in offsets
    ])


def _count_wall_crossings(a, b, world) -> int:
    """Number of occupied-region entries along the pixel line a→b."""
    ny, nx = world.shape
    hits = 0
    prev = False
    for x, y in bresenham((int(round(a[0])), int(round(a[1]))),
                          (int(round(b[0])), int(round(b[1])))):
        wall = not (0 <= x < nx and 0 <= y < ny) or bool(world[y, x])
        if wall and not prev:
            hits += 1
        prev = wall
    return hits


def ula_steering(n_elements: int, heading: float, arrival_bearing: float) -> np.ndarray:
    """Far-field ULA steering vector. ``arrival_bearing`` is the world angle
    of the direction the wave is travelling *from* (AoA) or *toward* (AoD)."""
    left = np.array((-np.sin(heading), np.cos(heading)))
    incoming = np.array((np.cos(arrival_bearing), np.sin(arrival_bearing)))
    sin_az = float(incoming @ left)
    m = np.arange(n_elements) - (n_elements - 1) / 2
    return np.exp(-1j * 2 * pi * ELEMENT_SPACING / WAVELENGTH * m * sin_az)


def _first_wall_hit(x: float, y: float, dx: float, dy: float, world: np.ndarray,
                    max_dist: float):
    """DDA to the first wall face. Returns (hx, hy, nx, ny, dist) or None."""
    ny, nx = world.shape
    length = float(np.hypot(dx, dy))
    if length < 1e-15 or max_dist <= 1e-9:
        return None
    dx, dy = dx / length, dy / length
    map_x, map_y = int(np.floor(x)), int(np.floor(y))
    delta_x = abs(1.0 / dx) if abs(dx) > 1e-15 else 1e30
    delta_y = abs(1.0 / dy) if abs(dy) > 1e-15 else 1e30
    step_x = 1 if dx > 0 else -1
    step_y = 1 if dy > 0 else -1
    side_x = ((map_x + 1 - x) if dx > 0 else (x - map_x)) * delta_x
    side_y = ((map_y + 1 - y) if dy > 0 else (y - map_y)) * delta_y
    while True:
        if side_x < side_y:
            dist = side_x
            side_x += delta_x
            map_x += step_x
            side = 0
        else:
            dist = side_y
            side_y += delta_y
            map_y += step_y
            side = 1
        if dist > max_dist:
            return None
        hit_x, hit_y = x + dx * dist, y + dy * dist
        if not (0 <= map_x < nx and 0 <= map_y < ny) or world[map_y, map_x]:
            normal = (-float(step_x), 0.0) if side == 0 else (0.0, -float(step_y))
            return hit_x, hit_y, normal[0], normal[1], float(dist)


def _capture_distance(ox, oy, dx, dy, travel, sx, sy, radius) -> float | None:
    t = float(np.clip((sx - ox) * dx + (sy - oy) * dy, 0.0, travel))
    qx, qy = ox + t * dx, oy + t * dy
    if np.hypot(sx - qx, sy - qy) <= radius:
        return t
    return None


def _direct_path(robot, source, world, resolution_m: float) -> dict:
    dist_m = max(float(np.hypot(source[0] - robot[0], source[1] - robot[1])) * resolution_m, 1e-3)
    wall_hits = _count_wall_crossings(source, robot, world)
    amp = (WAVELENGTH / (4 * pi * dist_m)) * 10 ** (-WALL_LOSS_DB * wall_hits / 20)
    aoa = float(np.arctan2(source[1] - robot[1], source[0] - robot[0]))
    aod = float(np.arctan2(robot[1] - source[1], robot[0] - source[0]))
    return {"dist": dist_m, "amp": float(amp), "aoa": aoa, "aod": aod,
            "bounces": 0, "kind": "direct", "walls": int(wall_hits)}


def _clear_los(a, b, world) -> bool:
    ny, nx = world.shape
    cells = list(bresenham((int(round(a[0])), int(round(a[1]))),
                           (int(round(b[0])), int(round(b[1])))))
    # Skip endpoints: they may sit on the bounce face or the source cell.
    for x, y in cells[1:-1] if len(cells) > 2 else cells:
        if 0 <= x < nx and 0 <= y < ny and world[y, x]:
            return False
    return True


def _merge_paths(paths: list[dict]) -> list[dict]:
    """Keep the shortest path per (bounce count, coarse AoA)."""
    best: dict[tuple, dict] = {}
    bin_w = 2 * pi / N_SPECULAR_RAYS
    for p in paths:
        key = (int(p["bounces"]), int(np.round(p["aoa"] / bin_w)))
        if key not in best or p["dist"] < best[key]["dist"]:
            best[key] = p
    return list(best.values())


def _specular_paths(robot, source, world, resolution_m: float) -> list[dict]:
    """Launch rays at the robot, specular-bounce on walls, capture the source.

    Reciprocity: a reverse ray that hits the source is a valid TX→RX path.
    Bounce-0 (LOS) captures are skipped; the transmissive direct path covers that.
    """
    sx, sy = float(source[0]), float(source[1])
    rx, ry = float(robot[0]), float(robot[1])
    max_cells = MAX_PATH_DIST_M / resolution_m
    radius = CAPTURE_RADIUS_M / resolution_m
    scatter_eps = 0.18 / resolution_m
    found: list[dict] = []
    for k in range(N_SPECULAR_RAYS):
        ang = 2 * pi * k / N_SPECULAR_RAYS
        dx, dy = np.cos(ang), np.sin(ang)
        x, y = rx, ry
        travelled = 0.0
        bounces = 0
        while travelled < max_cells and bounces <= MAX_BOUNCES:
            hit = _first_wall_hit(x, y, dx, dy, world, max_cells - travelled)
            if hit is None:
                break
            hx, hy, nxn, nyn, seg = hit
            cap = _capture_distance(x, y, dx, dy, seg, sx, sy, radius)
            if cap is not None and bounces > 0:
                dist_m = (travelled + cap) * resolution_m
                amp = (WAVELENGTH / (4 * pi * max(dist_m, 1e-3))) * (REFLECTION_GAIN ** bounces)
                aod = float(np.arctan2(-dy, -dx))
                found.append({"dist": float(dist_m), "amp": float(amp), "aoa": float(ang),
                              "aod": aod, "bounces": bounces, "kind": "reflect", "walls": 0})
                break
            # Single-bounce scatter: if this wall point can see the source, the
            # doorway/corner radiates even when the discrete ray misses exact
            # specular capture (helps L-shaped corridors).
            if bounces == 0:
                px, py = hx - dx * scatter_eps, hy - dy * scatter_eps
                ix, iy = int(np.floor(px)), int(np.floor(py))
                ny, nx = world.shape
                if 0 <= ix < nx and 0 <= iy < ny and not world[iy, ix]:
                    if _clear_los((px, py), (sx, sy), world):
                        d1 = travelled + max(seg - scatter_eps, 0.0)
                        d2 = float(np.hypot(sx - px, sy - py))
                        dist_m = (d1 + d2) * resolution_m
                        amp = (WAVELENGTH / (4 * pi * max(dist_m, 1e-3))) * 0.35
                        aod = float(np.arctan2(py - sy, px - sx))
                        found.append({"dist": float(dist_m), "amp": float(amp), "aoa": float(ang),
                                      "aod": aod, "bounces": 1, "kind": "scatter", "walls": 0})
            travelled += seg
            if abs(nxn) >= abs(nyn):
                dx = -dx
            else:
                dy = -dy
            bounces += 1
            x = hx + dx * 1e-3
            y = hy + dy * 1e-3
    return _merge_paths(found)


def collect_paths(robot, source, world, resolution_m: float = 1.0) -> list[dict]:
    return ([_direct_path(robot, source, world, resolution_m)]
            + _specular_paths(robot, source, world, resolution_m))


def _direct_channel(robot, source, world, rx_heading: float,
                    resolution_m: float = 1.0) -> np.ndarray:
    tx = ula_positions(source, N_TX, heading=0.0, resolution_m=resolution_m)
    rx = ula_positions(robot, N_RX, heading=rx_heading, resolution_m=resolution_m)
    h = np.zeros((N_RX, N_TX), dtype=np.complex128)
    for r, r_pos in enumerate(rx):
        for t, t_pos in enumerate(tx):
            distance_m = max(float(np.linalg.norm(r_pos - t_pos)) * resolution_m, 1e-3)
            wall_hits = _count_wall_crossings(t_pos, r_pos, world)
            amplitude = (WAVELENGTH / (4 * pi * distance_m)) * 10 ** (-WALL_LOSS_DB * wall_hits / 20)
            h[r, t] = amplitude * np.exp(-1j * 2 * pi * distance_m / WAVELENGTH)
    return h


def simulate_mimo_link(
    robot: tuple[float, float], source: tuple[float, float], world: np.ndarray,
    rx_heading: float | None = None, resolution_m: float = 1.0,
) -> tuple[np.ndarray, list[dict]]:
    """4x4 H = transmissive direct path + far-field specular reflections."""
    robot = (float(robot[0]), float(robot[1]))
    source = (float(source[0]), float(source[1]))
    heading = (float(np.arctan2(source[1] - robot[1], source[0] - robot[0]))
               if rx_heading is None else float(rx_heading))
    paths = collect_paths(robot, source, world, resolution_m)
    h = _direct_channel(robot, source, world, heading, resolution_m)
    for p in paths:
        if p["kind"] == "direct":
            continue
        a_rx = ula_steering(N_RX, heading, p["aoa"])
        a_tx = ula_steering(N_TX, 0.0, p["aod"])
        alpha = p["amp"] * np.exp(-1j * 2 * pi * p["dist"] / WAVELENGTH)
        h = h + alpha * np.outer(a_rx, np.conj(a_tx))
    return h, paths


def mimo_channel(
    robot: tuple[float, float], source: tuple[float, float], world: np.ndarray,
    rx_heading: float | None = None, resolution_m: float = 1.0,
) -> np.ndarray:
    """Return the 4x4 narrowband baseband channel with LOS + wall reflections."""
    h, _ = simulate_mimo_link(robot, source, world, rx_heading=rx_heading,
                              resolution_m=resolution_m)
    return h


def path_rf_features(paths: list[dict], heading: float) -> np.ndarray:
    """K-factor, RMS delay spread, path count, dominant ego-AoA, 8-bin AoA histogram.

    RF-derived only: reflected energy from around a corner shows up as
    off-boresight AoA bins even when the Euclidean ray is blocked.
    """
    amps = np.array([p["amp"] for p in paths], dtype=float)
    dists = np.array([p["dist"] for p in paths], dtype=float)
    aoas = np.array([p["aoa"] for p in paths], dtype=float)
    power = np.maximum(amps, 0.0) ** 2
    total = float(power.sum()) + 1e-30
    direct = next((p for p in paths if p["kind"] == "direct"), None)
    k_factor = (direct["amp"] ** 2 / total) if direct is not None else 0.0
    mean_d = float((power * dists).sum() / total)
    delay_spread = float(np.sqrt(((power * (dists - mean_d) ** 2).sum()) / total)) / MAX_PATH_DIST_M
    peak = float(power.max()) + 1e-30
    n_paths = float(np.sum(power > 0.05 * peak)) / 8.0
    i_star = int(np.argmax(power))
    aoa_rel = aoas[i_star] - heading
    hist = np.zeros(AOA_BINS, dtype=np.float64)
    for aoa, pwr in zip(aoas, power):
        rel = (aoa - heading + pi) % (2 * pi)
        hist[int(rel / (2 * pi) * AOA_BINS) % AOA_BINS] += pwr
    hist = hist / (hist.sum() + 1e-30)
    return np.concatenate((
        np.array((k_factor, delay_spread, n_paths, np.cos(aoa_rel), np.sin(aoa_rel)), dtype=np.float32),
        hist.astype(np.float32),
    ))


def _range_m(position, angle: float, world: np.ndarray,
             max_range_m: float, resolution_m: float) -> float:
    x, y = float(position[0]), float(position[1])
    max_cells = max_range_m / resolution_m
    hit = _first_wall_hit(x, y, np.cos(angle), np.sin(angle), world, max_cells)
    dist_cells = hit[4] if hit is not None else max_cells
    return float(dist_cells * resolution_m)


def ego_lidar(position, heading: float, world: np.ndarray,
              n_beams: int = LIDAR_BEAMS, max_range_m: float = LIDAR_MAX_M,
              resolution_m: float = 1.0) -> np.ndarray:
    """Surround rangefinder in the robot frame. 1 = clear to max, 0 = wall on the nose."""
    ranges = np.empty(n_beams, dtype=np.float32)
    for i in range(n_beams):
        ang = heading + 2 * pi * i / n_beams
        ranges[i] = np.clip(_range_m(position, ang, world, max_range_m, resolution_m) / max_range_m,
                            0.0, 1.0)
    return ranges


def front_lidar(position, heading: float, world: np.ndarray,
                max_range_m: float = LIDAR_MAX_M, resolution_m: float = 1.0) -> np.ndarray:
    """Dense frontal beams so a 0.3 m step cannot miss a 1 cm wall off-boresight."""
    ranges = np.empty(FRONT_LIDAR_DIM, dtype=np.float32)
    for i, deg in enumerate(FRONT_LIDAR_DEG):
        ang = heading + deg * pi / 180.0
        ranges[i] = np.clip(_range_m(position, ang, world, max_range_m, resolution_m) / max_range_m,
                            0.0, 1.0)
    return ranges


def obstacle_scan(position, heading: float, world: np.ndarray, resolution_m: float,
                  step_size_m: float, last_collision: float) -> np.ndarray:
    """Onboard obstacle block: surround + front lidar, forward clearance, imminent hit.

    No occupancy map. Clearance is metres-to-wall along the current heading.
    Imminent is 1 when that wall is closer than one commanded step.
    """
    surround = ego_lidar(position, heading, world, resolution_m=resolution_m)
    front = front_lidar(position, heading, world, resolution_m=resolution_m)
    forward_m = _range_m(position, heading, world, LIDAR_MAX_M, resolution_m)
    clearance = np.float32(np.clip(forward_m / LIDAR_MAX_M, 0.0, 1.0))
    imminent = np.float32(1.0 if forward_m < 1.25 * step_size_m else 0.0)
    proximity = np.float32(min(float(surround.min()), float(front.min()), float(clearance)))
    return np.concatenate((
        surround, front,
        np.array((clearance, imminent, proximity, np.float32(last_collision)), dtype=np.float32),
    ))


def proximity_feature(lidar: np.ndarray) -> np.ndarray:
    return np.array((float(np.min(lidar)),), dtype=np.float32)


def mimo_metrics(h: np.ndarray) -> tuple[float, float]:
    """Return mean received power (dB, unit transmit power) and channel rank."""
    mean_power_db = 10 * np.log10(np.mean(np.abs(h) ** 2) + 1e-30)
    rank = np.linalg.matrix_rank(h, tol=1e-12)
    return mean_power_db, rank
