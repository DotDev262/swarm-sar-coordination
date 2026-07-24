import numpy as np
from swarm_sar.mission_manager import MissionManager, SimConfig, Environment


def _mm():
    grid = np.zeros((5, 5), dtype=np.uint8)
    grid[2, 2] = 2
    env = Environment(5, 5, grid, (2, 2), (2, 2))
    return MissionManager(env, SimConfig(n_drones=1, grid_size=5))


def test_coverage_starts_zero():
    assert _mm().coverage() == 0.0


def test_coverage_scales():
    mm = _mm()
    mm.mark_searched({(0, 0), (1, 0)}, 0)
    assert mm.coverage() == 2.0 / len(mm.searchable)


def test_re_marking_no_double_count():
    mm = _mm()
    mm.mark_searched({(0, 0)}, 0)
    mm.mark_searched({(0, 0)}, 0)
    assert mm.coverage() == 1.0 / len(mm.searchable)


def test_failure_does_not_credit_coverage():
    mm = _mm()
    t = mm.assign_task(0, (0, 0))
    assert t is not None
    mm.on_drone_killed(0, t, tick=5)
    assert t not in mm.searched
