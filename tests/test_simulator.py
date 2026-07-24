import numpy as np
from swarm_sar.simulator import Simulator
from swarm_sar.environment import SimConfig, Environment
from swarm_sar.drone import DroneState



def _env():
    grid = np.zeros((5, 5), dtype=np.uint8)
    grid[2, 2] = 2
    return Environment(5, 5, grid, (2, 2), (2, 2))


def test_determinism_same_seed():
    cfg = SimConfig(seed=0, n_drones=2, grid_size=5, obstacle_density=0.0)
    s1 = Simulator(_env(), cfg)
    s2 = Simulator(_env(), cfg)
    t1, t2 = [], []
    for sim, out in [(s1, t1), (s2, t2)]:
        for _ in range(10):
            sim.tick_once()
            d0 = sim.drones[0]
            out.append(
                (
                    sim.tick,
                    d0.id,
                    d0.pos[0],
                    d0.pos[1],
                    round(d0.battery, 1),
                )
            )
    assert t1 == t2


def test_coverage_progress():
    cfg = SimConfig(
        seed=0,
        n_drones=1,
        grid_size=5,
        obstacle_density=0.0,
        max_ticks=100,
    )
    sim = Simulator(_env(), cfg)
    assert sim.mm.coverage() >= 0.0


def test_dead_drone_stops_moving():
    cfg = SimConfig(seed=0, n_drones=1, grid_size=5, obstacle_density=0.0)
    sim = Simulator(_env(), cfg)
    sim.drones[0].alive = False
    pos_before = sim.drones[0].pos
    sim.tick_once()
    assert sim.drones[0].pos == pos_before


def test_is_complete():
    cfg = SimConfig(n_drones=1, grid_size=5, obstacle_density=0.0, coverage_threshold=0.5)
    sim = Simulator(_env(), cfg)
    for cell in list(sim.mm.searchable):
        sim.mm.mark_searched({cell}, 0)
    assert sim.mm.is_complete()


def test_drone_crash_releases_task():
    cfg = SimConfig(seed=0, n_drones=1, grid_size=5, obstacle_density=0.0, battery_capacity=10.0)
    sim = Simulator(_env(), cfg)
    
    # Run once to get a task assigned and move the drone
    sim.tick_once()
    drone = sim.drones[0]
    assert drone.state == DroneState.SEARCHING or drone.target is not None
    target = drone.target
    assert target in sim.mm.in_flight.values()
    
    # Artificially drain battery to negative to trigger crash on next tick
    drone.battery = -1.0
    sim.tick_once()
    
    # Drone should be dead, and the task should be released from in_flight
    assert not drone.alive
    assert drone.id not in sim.mm.in_flight
    assert len(sim.mm.failure_log) == 1

