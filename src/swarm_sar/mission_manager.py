import csv
import time
from dataclasses import dataclass
from typing import Optional
import numpy as np
from src.swarm_sar.environment import Environment, SimConfig


@dataclass(frozen=True)
class Snapshot:
    grid: tuple[tuple[int, ...], ...]
    obstacles: frozenset[tuple[int, int]]
    searched: frozenset[tuple[int, int]]
    drone_positions: dict[int, tuple[int, int]]
    home: tuple[int, int]
    searchable: frozenset[tuple[int, int]]


class MissionManager:
    def __init__(self, env: Environment, config: SimConfig):
        self.env = env
        self.config = config
        self.home = env.home
        self.searchable: set[tuple[int, int]] = {
            (x, y) for y, x in zip(*np.where(env.grid == 0))
        }
        self.searched: set[tuple[int, int]] = set()
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
        self._log_buf: list[str] = []
        self._wrote_header = False
        self._log_interval = config.log_interval_ticks

    def set_log_path(self, path: str) -> None:
        self._log_path = path
        cols = [
            "tick", "active_drones", "failed_drones", "coverage_pct",
            "remaining_cells", "avg_battery", "fps",
        ]
        for i in range(self.config.n_drones):
            cols += [
                f"drone_{i}_state", f"drone_{i}_pos", f"drone_{i}_target",
                f"drone_{i}_battery",
            ]
        with open(path, "w", newline="") as fh:
            fh.write(",".join(cols) + "\n")

    def snapshot(self, drones: list) -> Snapshot:
        return Snapshot(
            grid=tuple(tuple(r) for r in self.env.grid),
            obstacles=frozenset(zip(*np.where(self.env.grid == 1))),
            searched=frozenset(self.searched),
            drone_positions={d.id: d.pos for d in drones},
            home=self.home,
            searchable=frozenset(self.searchable),
        )

    def log_drone_states(self, tick: int, drones: list):
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
        line = (
            f"{tick},{active},{failed},{cov:.2f},{remaining},"
            f"{avg_bat:.2f},0\n"
        )
        for d in drones:
            state = d.state.name if d.alive else "DEAD"
            pos = f"({int(d.pos[0])},{int(d.pos[1])})"
            target = (
                f"({int(d.target[0])},{int(d.target[1])})"
                if d.target is not None
                else "-"
            )
            bat = f"{d.battery:.2f}"
            line += f"{state},{pos},{target},{bat},\n"
        with open(self._log_path, "a") as fh:
            fh.write(line)

    def flush_log(self) -> None:
        if self._log_buf:
            with open(self._log_path, "a", newline="") as fh:
                fh.write("\n".join(self._log_buf))
            self._log_buf = []

    def assign_task(
        self, drone_id: int, drone_pos: tuple[int, int]
    ) -> tuple[int, int] | None:
        candidates = (
            self.searchable - self.searched - set(self.in_flight.values()) - {drone_pos}
        )
        if not candidates:
            return None

        def dist(cell):
            return (cell[0] - drone_pos[0]) ** 2 + (
                cell[1] - drone_pos[1]
            ) ** 2

        best = min(candidates, key=dist)
        self.in_flight[drone_id] = best
        if drone_id not in self.per_drone_stats:
            self.per_drone_stats[drone_id] = {
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
        self.per_drone_stats[drone_id]["assignments"] += 1
        return best

    def mark_searched(self, cells, drone_id: int) -> None:
        new_cells = cells - self.searched
        self.searched |= cells
        if drone_id in self.per_drone_stats:
            self.per_drone_stats[drone_id]["cells_marked"] += len(new_cells)

    def coverage(self) -> float:
        if not self.searchable:
            return 1.0
        return len(self.searched & self.searchable) / len(self.searchable)

    def is_complete(self) -> bool:
        return self.coverage() >= self.config.coverage_threshold

    def on_drone_killed(
        self, drone_id: int, target: tuple[int, int] | None, tick: int
    ) -> None:
        self.in_flight.pop(drone_id, None)
        if target is not None and target in self.searchable:
            pass
        self.failure_log.append((tick, drone_id, 0))

    def on_drone_reported(self, drone_id: int) -> None:
        pass

    def record_move(self, drone_id: int) -> None:
        pass

    def update(self, drones: list, tick: int) -> None:
        pass

    def summary_line(self, tick: int) -> str:
        return (f"{tick},,{len(self.failure_log)},{self.coverage()*100:.2f},"
                f"{len(self.searchable - self.searched)},,0")
