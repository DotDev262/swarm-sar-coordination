import numpy as np
from src.swarm_sar.drone import DroneAgent, DroneState
from src.swarm_sar.environment import SimConfig, Snapshot


def _drone(pos=(0, 0), state=DroneState.IDLE, battery=100.0):
    cfg = SimConfig(n_drones=1, grid_size=5, seed=0)
    d = DroneAgent(0, (2, 2), cfg)
    d.pos = pos
    d.state = state
    d.battery = battery
    return d


def _snap():
    return Snapshot(
        grid=tuple(tuple([0] * 5) for _ in range(5)),
        obstacles=frozenset(),
        searched=frozenset(),
        drone_positions={},
        home=(2, 2),
        searchable=frozenset(),
    )


def test_idle_proposes_request_task():
    action = _drone().propose(_snap(), __import__("random").Random(0))
    assert action.request_task is True
    assert action.move_to is None


def test_searching_proposes_move():
    d = _drone(state=DroneState.SEARCHING)
    d.target = (4, 4)
    d.path = [(1, 1), (2, 2), (3, 3), (4, 4)]
    a = d.propose(_snap(), __import__("random").Random(0))
    assert a.move_to == (1, 1)
    assert not a.request_task


def test_searching_drains_battery():
    d = _drone(state=DroneState.SEARCHING, battery=50.0)
    d.target = (1, 0)
    d.path = [(1, 0)]
    snap = _snap()
    rng = __import__("random").Random(0)
    for _ in range(10):
        d.propose(snap, rng)
        d.commit(d.propose(snap, rng), snap)
    assert d.battery <= 40.0


def test_returning_triggers_on_low_battery():
    d = _drone(state=DroneState.SEARCHING, battery=5.0)
    d.pos = (4, 4)
    d.target = (4, 4)
    d.path = []
    snap = _snap()
    rng = __import__("random").Random(0)
    action = d.propose(snap, rng)
    d.commit(action, snap)
    assert d.state == DroneState.RETURNING


def test_at_home_returning_transitions_to_reporting():
    d = _drone(state=DroneState.RETURNING, battery=50.0)
    d.pos = (2, 2)
    d.path = [(2, 2)]
    snap = _snap()
    rng = __import__("random").Random(0)
    d.propose(snap, rng)
    d.commit(d.propose(snap, rng), snap)
    assert d.state == DroneState.REPORTING


def test_reporting_transitions_to_idle():
    d = _drone(state=DroneState.REPORTING, battery=50.0)
    d.reporting_ticks_remaining = 1
    snap = _snap()
    rng = __import__("random").Random(0)
    action = d.propose(snap, rng)
    d.commit(action, snap)
    assert d.state == DroneState.IDLE
    assert d.battery == 100.0
