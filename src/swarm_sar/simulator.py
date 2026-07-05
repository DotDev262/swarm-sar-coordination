import random
from src.swarm_sar.environment import Environment
from src.swarm_sar.mission_manager import MissionManager, SimConfig, Snapshot
from src.swarm_sar.drone import DroneAgent, DroneState, Action
from src.swarm_sar.astar import astar


class Simulator:
    def __init__(self, env: Environment, config: SimConfig):
        self.env = env
        self.config = config
        self.mm = MissionManager(env, config)
        self.drones = [
            DroneAgent(i, env.spawn, config) for i in range(config.n_drones)
        ]
        self.tick = 0
        self.rng = random.Random(config.seed)
        hx, hy = self.env.home
        placed = 0
        for radius in range(1, 20):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    nx, ny = hx + dx, hy + dy
                    if (
                        0 <= nx < self.env.width
                        and 0 <= ny < self.env.height
                        and placed < len(self.drones)
                    ):
                        self.drones[placed].pos = (nx, ny)
                        placed += 1
                    if placed >= len(self.drones):
                        break
                if placed >= len(self.drones):
                    break

    def tick_once(self):
        snap = self.mm.snapshot(self.drones)
        alive = [d for d in self.drones if d.alive]

        proposals = [d.propose(snap, self.rng) for d in alive]

        for d, action in zip(alive, proposals):
            if action.mark_cells:
                self.mm.mark_searched(action.mark_cells, d.id)

        for idx, (d, action) in enumerate(zip(alive, proposals)):
            if action.request_task and d.state.value == "IDLE":
                result = self.mm.assign_task(d.id, d.pos)
                if result is not None:
                    d.target = result
                    d.state = DroneState.SEARCHING
                    others = {
                        alive[j].pos
                        for j in range(len(alive))
                        if j != idx
                    }
                    d.path = astar(
                        self.env.grid,
                        others,
                        d.pos,
                        result,
                    )

        for d in self.drones:
            d._moved_this_tick = False

        for d, action in zip(alive, proposals):
            d.commit(action, snap)
            self.mm.record_move(d.id)

        self.tick += 1

    def is_complete(self):
        return self.mm.is_complete()
