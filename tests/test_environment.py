import numpy as np
from swarm_sar.environment import Environment, SimConfig


def _load(sf="default"):
    config = SimConfig(scenario_path=f"scenarios/{sf}.txt")
    return Environment.from_config(config)


def test_load_default_scenario():
    env = _load("default")
    assert env.home == (6, 1)
    assert env.grid[env.home[1], env.home[0]] == 2
    assert env.spawn == env.home


def test_exactly_one_home_each_scenario():
    for sf in ("default", "empty", "maze", "walls"):
        env = _load(sf)
        assert np.sum(env.grid == 2) == 1, f"{sf} has wrong home count"


def test_procedural_determinism():
    g1 = Environment.generate_grid(20, 0.10, seed=42)
    g2 = Environment.generate_grid(20, 0.10, seed=42)
    assert np.array_equal(g1, g2)


def test_procedural_home_and_8_neighbors_not_obstacle():
    for seed in range(10):
        g = Environment.generate_grid(30, 0.25, seed=seed)
        h = g.shape[1] // 2, g.shape[0] // 2
        assert g[h[1], h[0]] != 1
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nx, ny = h[0]+dx, h[1]+dy
                if 0 <= nx < g.shape[1] and 0 <= ny < g.shape[0]:
                    assert g[ny, nx] != 1


def test_connectivity_at_moderate_density():
    for seed in range(5):
        g = Environment.generate_grid(30, 0.20, seed=seed)
        hx, hy = g.shape[1] // 2, g.shape[0] // 2
        reachable = _flood(g, (hx, hy))
        empties = {(x, y) for y, x in zip(*np.where(g == 0))}
        assert empties.issubset(reachable)


def _flood(grid, start):
    from collections import deque
    q = deque([start])
    seen = {start}
    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0]:
                if grid[ny, nx] != 1 and (nx, ny) not in seen:
                    seen.add((nx, ny))
                    q.append((nx, ny))
    return seen
