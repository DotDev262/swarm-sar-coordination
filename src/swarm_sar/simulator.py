import random
from swarm_sar.environment import Environment
from swarm_sar.mission_manager import MissionManager, SimConfig
from swarm_sar.drone import DroneAgent, DroneState
from swarm_sar.astar import astar


class Simulator:
    """Orchestrates the swarm search and rescue simulation.

    Manages spawn positioning, step-by-step sequential time ticks, agent path planning,
    movement commitments, and checks for mission completion.
    """

    def __init__(self, env: Environment, config: SimConfig):
        """Initializes the Simulator and spawns drones near the home position.

        Args:
            env: The Environment instance containing the grid and spawn layout.
            config: The SimConfig settings for the simulation.
        """
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
        """Advances the simulation by a single tick.

        Performs task assignment, sequential prioritized path planning, action
        generation, searched footprint marking, state commitments, battery updates,
        and death checks.
        """
        import dataclasses
        snap = self.mm.snapshot(self.drones)
        alive = [d for d in self.drones if d.alive]

        # Prioritized sequential path planning (drones with lower index have higher priority)
        proposals = []
        future_positions = {d.id: d.pos for d in self.drones}
        
        for d in alive:
            # Plan using the expected next positions of already processed drones
            current_snap = dataclasses.replace(snap, drone_positions=future_positions.copy())
            action = d.propose(current_snap, self.rng)
            proposals.append(action)
            
            # If the drone plans to move, update its expected next position
            if action.move_to:
                future_positions[d.id] = action.move_to

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
                        future_positions[alive[j].id]
                        for j in range(len(alive))
                        if j != idx
                    } - {result}
                    d.path = astar(
                        self.env.grid,
                        others,
                        d.pos,
                        result,
                    )
                elif d.pos != self.env.home:
                    d.state = DroneState.RETURNING
                    d.target = None
                    d.path = []

        for d in self.drones:
            d._moved_this_tick = False

        for d, action in zip(alive, proposals):
            was_alive = d.alive
            prev_target = d.target
            if action.state_transition == "RETURNING":
                self.mm.in_flight.pop(d.id, None)
            d.commit(action, snap)
            if was_alive and not d.alive:
                self.mm.on_drone_killed(d.id, prev_target, self.tick)
            elif d.state == DroneState.SEARCHING and d.target is not None and d.pos == d.target:
                d.state = DroneState.IDLE
                d.target = None
                d.path = []
                self.mm.in_flight.pop(d.id, None)

        self.tick += 1

    def is_complete(self):
        """Checks if the overall simulation run is complete.

        Returns:
            True if the target is found or target coverage threshold is met.
        """
        return self.mm.is_complete()
