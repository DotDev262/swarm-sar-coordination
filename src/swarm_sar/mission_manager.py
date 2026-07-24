from dataclasses import dataclass
from typing import Optional
import numpy as np
from swarm_sar.environment import Environment, SimConfig


@dataclass(frozen=True)
class Snapshot:
    grid: tuple[tuple[int, ...], ...]
    searched: frozenset[tuple[int, int]]
    drone_positions: dict[int, tuple[int, int]]
    home: tuple[int, int]
    searchable: frozenset[tuple[int, int]]


class MissionManager:
    """Manages the search and rescue mission state and statistics.

    Responsible for task allocation, tracking coverage, generating snapshots,
    recording search progress, logging agent states, and managing drone lifecycle
    callbacks.
    """

    def __init__(self, env: Environment, config: SimConfig):
        """Initializes the MissionManager.

        Args:
            env: The Environment instance representing the grid and spawn areas.
            config: The SimConfig settings for the simulation.
        """
        self.env = env
        self.config = config
        self.home = env.home
        self.searchable: set[tuple[int, int]] = {
            (int(x), int(y)) for y, x in zip(*np.where(env.grid == 0))
        }
        self.searched: set[tuple[int, int]] = set()
        self.searched_by: dict[tuple[int, int], int] = {}
        self.in_flight: dict[int, tuple[int, int]] = {}
        self.per_drone_stats: dict[int, dict] = {
            i: {
                "assignments": 0,
                "cells_marked": 0,
                "distance": 0,
                "time_in_state": {
                    "IDLE": 0,
                    "SEARCHING": 0,
                    "RETURNING": 0,
                    "REPORTING": 0,
                },
            }
            for i in range(config.n_drones)
        }
        self.failure_log: list[tuple[int, int, int]] = []
        self._log_path: Optional[str] = None

    def set_log_path(self, path: str) -> None:
        """Sets the log file path and initializes the CSV header.

        Args:
            path: Absolute or relative file path to the log file.
        """
        self._log_path = path
        cols = [
            "tick", "active_drones", "failed_drones", "coverage_pct",
            "remaining_cells", "avg_battery",
        ]
        for i in range(self.config.n_drones):
            cols += [
                f"drone_{i}_state", f"drone_{i}_pos", f"drone_{i}_target",
                f"drone_{i}_battery",
            ]
        with open(path, "w", newline="") as fh:
            fh.write(",".join(cols) + "\n")

    def snapshot(self, drones: list) -> Snapshot:
        """Creates a snapshot of the current environment and drone states.

        Args:
            drones: The list of DroneAgent instances in the simulation.

        Returns:
            A Snapshot instance containing frozen state data.
        """
        return Snapshot(
            grid=tuple(tuple(r) for r in self.env.grid),
            searched=frozenset(self.searched),
            drone_positions={d.id: d.pos for d in drones},
            home=self.home,
            searchable=frozenset(self.searchable),
        )

    def log_drone_states(self, tick: int, drones: list):
        """Appends the current tick states of all drones to the log file.

        Args:
            tick: The current simulation tick index.
            drones: The list of DroneAgent instances.
        """
        if not self._log_path:
            return
        active = sum(1 for d in drones if d.alive)
        failed = len(self.failure_log)
        cov = self.coverage() * 100
        remaining = len(self.searchable - self.searched)
        avg_bat = (
            sum(d.battery for d in drones if d.alive) / active
            if active > 0
            else 0.0
        )
        fields = [str(tick), str(active), str(failed), f"{cov:.2f}",
                  str(remaining), f"{avg_bat:.2f}"]
        for d in drones:
            state = d.state.name if d.alive else "DEAD"
            pos = f"{int(d.pos[0])};{int(d.pos[1])}"
            target = (
                f"{int(d.target[0])};{int(d.target[1])}"
                if d.target is not None
                else "-"
            )
            fields += [state, pos, target, f"{d.battery:.2f}"]
        with open(self._log_path, "a") as fh:
            fh.write(",".join(fields) + "\n")

    def assign_task(
        self, drone_id: int, drone_pos: tuple[int, int]
    ) -> tuple[int, int] | None:
        """Assigns the next best cell target to an idle drone.

        Uses a distance-to-information-gain ratio heuristic.

        Args:
            drone_id: The ID of the requesting drone.
            drone_pos: The current (x, y) position of the requesting drone.

        Returns:
            The selected target (x, y) cell coordinates, or None if no cells
            are left to search.
        """
        candidates = (
            self.searchable - self.searched - set(self.in_flight.values()) - {drone_pos}
        )
        if not candidates:
            return None

        # Pre-filter to the closest 100 candidates by Chebyshev distance for performance
        def chem_dist(cell):
            return max(abs(cell[0] - drone_pos[0]), abs(cell[1] - drone_pos[1]))
        
        closest_candidates = sorted(candidates, key=chem_dist)[:100]

        r = self.config.sensor_radius
        unsearched = self.searchable - self.searched

        def score(cell):
            cx, cy = cell
            # Count how many unsearched cells would be covered if drone goes here
            gain = 0
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if (cx + dx, cy + dy) in unsearched:
                        gain += 1
            dist = chem_dist(cell)
            return dist / (gain + 0.1)

        best = min(closest_candidates, key=score)
        self.in_flight[drone_id] = best
        self.per_drone_stats[drone_id]["assignments"] += 1
        return best

    def mark_searched(self, cells, drone_id: int) -> None:
        """Marks a set of cells as searched and attributes discovery to a drone.

        Args:
            cells: A set/frozenset of coordinates to mark as searched.
            drone_id: The ID of the drone that discovered these cells.
        """
        new_cells = cells - self.searched
        self.searched |= cells
        for cell in new_cells:
            self.searched_by[cell] = drone_id
        if drone_id in self.per_drone_stats:
            self.per_drone_stats[drone_id]["cells_marked"] += len(new_cells)

    def coverage(self) -> float:
        """Calculates the current search coverage ratio.

        Returns:
            The ratio of searched searchable cells to total searchable cells.
        """
        if not self.searchable:
            return 1.0
        return len(self.searched & self.searchable) / len(self.searchable)

    def is_complete(self) -> bool:
        """Checks if the search mission is complete.

        Returns:
            True if the target is found or target coverage threshold is met.
        """
        target_found = self.env.target is not None and self.env.target in self.searched
        return target_found or self.coverage() >= self.config.coverage_threshold

    def on_drone_killed(
        self, drone_id: int, target: tuple[int, int] | None, tick: int
    ) -> None:
        """Callback triggered when a drone crashes or dies.

        Args:
            drone_id: ID of the crashed drone.
            target: The drone's active target coordinate, if any.
            tick: The current simulation tick.
        """
        self.in_flight.pop(drone_id, None)
        if target is not None and target in self.searchable:
            pass
        self.failure_log.append((tick, drone_id, 0))

    def summary_line(self, tick: int) -> str:
        """Generates the final summary CSV line for the mission run.

        Args:
            tick: The final completion tick of the simulation.

        Returns:
            CSV formatted summary line as a string.
        """
        return (f"{tick},,{len(self.failure_log)},{self.coverage()*100:.2f},"
                f"{len(self.searchable - self.searched)},,0")
