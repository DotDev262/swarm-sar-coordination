import numpy as np
from src.swarm_sar.astar import astar


def _grid(h, w, obs=None):
    g = np.zeros((h, w), dtype=np.uint8)
    if obs:
        for x, y in obs:
            g[y, x] = 1
    return g


def test_shortest_path_empty():
    p = astar(_grid(5, 5), set(), (0, 0), (4, 4))
    assert len(p) == 4
    assert p[0] == (1, 1) and p[-1] == (4, 4)


def test_detours_around_wall():
    p = astar(_grid(5, 5, [(0, 2)]), set(), (0, 0), (4, 0))
    assert len(p) > 2
    assert (0, 2) not in p
    assert p[-1] == (4, 0)


def test_no_path_when_goal_blocked():
    assert astar(_grid(3, 3), {(1, 1)}, (0, 0), (1, 1)) == []


def test_no_path_when_isolated():
    assert astar(_grid(3, 3, [(1, 0), (0, 1), (1, 1)]), set(), (0, 0), (2, 2)) == []


def test_start_eq_goal_returns_empty():
    assert astar(_grid(3, 3), set(), (2, 2), (2, 2)) == []


def test_blockers_respected():
    blockers = {(1, 0), (0, 1)}
    p = astar(_grid(5, 5), blockers, (0, 0), (2, 2))
    assert (1, 0) not in p and (0, 1) not in p
