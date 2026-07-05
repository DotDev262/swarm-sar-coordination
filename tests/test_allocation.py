import numpy as np
from src.swarm_sar.mission_manager import MissionManager, SimConfig, Environment


def _mm(home=(2, 2), w=5, h=5):
    grid = np.zeros((h, w), dtype=np.uint8)
    grid[home[1], home[0]] = 2
    env = Environment(width=w, height=h, grid=grid, home=home, spawn=home)
    return MissionManager(env, SimConfig(n_drones=1, grid_size=w))


def test_assign_returns_something():
    r = _mm().assign_task(0, (0, 0))
    assert r is not None
    x, y = r
    assert 0 <= x < 5 and 0 <= y < 5


def test_none_when_exhausted():
    mm = _mm()
    while True:
        c = mm.assign_task(0, (0, 0))
        if c is None:
            break
        mm.mark_searched({c}, 0)
    assert mm.assign_task(0, (0, 0)) is None


def test_assigned_cell_not_reassigned():
    mm = _mm()
    r1 = mm.assign_task(0, (0, 0))
    assert r1 is not None
    r2 = mm.assign_task(1, (0, 0))
    assert r2 is None or r2 != r1


def test_searchable_excludes_obstacles():
    grid = np.zeros((5, 5), dtype=np.uint8)
    for oy, ox in [(1, 1), (1, 2), (1, 3), (3, 1), (3, 2), (3, 3)]:
        grid[oy, ox] = 1
    grid[2, 2] = 2
    env = Environment(5, 5, grid, (2, 2), (2, 2))
    mm = MissionManager(env, SimConfig(n_drones=1, grid_size=5))
    for oy, ox in [(1,1),(1,2),(1,3),(3,1),(3,2),(3,3)]:
        assert (ox, oy) not in mm.searchable
