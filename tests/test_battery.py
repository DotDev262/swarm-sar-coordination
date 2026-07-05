from swarm_sar.drone import DroneAgent, DroneState
from swarm_sar.environment import SimConfig
from swarm_sar.mission_manager import Snapshot


def _drone(state, battery, pos=(0, 0)):
    cfg = SimConfig(n_drones=1, grid_size=5, seed=0)
    d = DroneAgent(0, (2, 2), cfg)
    d.state = state
    d.battery = battery
    d.pos = pos
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


def test_returning_drains():
    d = _drone(DroneState.RETURNING, 50.0)
    snap = _snap()
    rng = __import__("random").Random(0)
    action = d.propose(snap, rng)
    d.commit(action, snap)
    assert d.battery <= 49.0


def test_idle_recharges():
    d = _drone(DroneState.IDLE, 50.0)
    rng = __import__("random").Random(0)
    for _ in range(5):
        action = d.propose(_snap(), rng)
        d.commit(action, _snap())
    assert d.battery > 50.0


def test_reporting_recharges():
    d = _drone(DroneState.REPORTING, 50.0)
    d.reporting_ticks_remaining = 3
    rng = __import__("random").Random(0)
    for _ in range(3):
        action = d.propose(_snap(), rng)
        d.commit(action, _snap())
    assert d.battery > 50.0


def test_crashes_below_zero_on_returning():
    d = _drone(DroneState.RETURNING, 0.5)
    rng = __import__("random").Random(0)
    action = d.propose(_snap(), rng)
    d.commit(action, _snap())
    assert not d.alive
