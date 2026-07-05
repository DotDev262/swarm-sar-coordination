from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np
import math
from src.swarm_sar.environment import SimConfig, Snapshot
from src.swarm_sar.astar import astar


@dataclass(frozen=True)
class Action:
    move_to: tuple[int, int] | None
    mark_cells: frozenset[tuple[int, int]]
    request_task: bool
    state_transition: str | None


class DroneState(Enum):
    IDLE = "IDLE"
    SEARCHING = "SEARCHING"
    RETURNING = "RETURNING"
    REPORTING = "REPORTING"


@dataclass
class DroneAgent:
    id: int
    home: tuple[int, int]
    config: SimConfig
    pos: tuple[int, int] = field(default_factory=lambda: (0, 0))
    state: DroneState = DroneState.IDLE
    battery: float = 100.0
    alive: bool = True
    target: Optional[tuple[int, int]] = None
    path: list[tuple[int, int]] = field(default_factory=list)
    _moved_this_tick: bool = field(default=False, repr=False)
    reporting_ticks_remaining: int = field(default=0, repr=False)
    crashed: bool = field(default=False, repr=False)
    _reported: bool = field(default=False, repr=False)

    def propose(self, snap: Snapshot, rng):
        if not self.alive:
            return Action(None, frozenset(), False, None)
        if self.state == DroneState.IDLE:
            return Action(None, frozenset(), True, None)
        if self.state == DroneState.SEARCHING:
            if self.check_return_threshold():
                return Action(
                    None,
                    frozenset(),
                    False,
                    "RETURNING",
                )
            move_to = self._next_search_move(snap)
            footprint = self._sensor_footprint(move_to or self.pos)
            self._moved_this_tick = move_to is not None
            return Action(move_to, footprint, False, None)
        if self.state == DroneState.RETURNING:
            move_to = self._next_return_move(snap)
            self._moved_this_tick = move_to is not None
            return Action(move_to, frozenset(), False, None)
        if self.state == DroneState.REPORTING:
            self.reporting_ticks_remaining -= 1
            self._moved_this_tick = False
            if self.reporting_ticks_remaining <= 0:
                return Action(None, frozenset(), False, "IDLE")
            return Action(None, frozenset(), False, None)
        return Action(None, frozenset(), False, None)

    def commit(self, action, snap: Snapshot) -> None:
        if not self.alive:
            return
        was_returning = self.state == DroneState.RETURNING
        if action.state_transition == "RETURNING":
            self.state = DroneState.RETURNING
            self.target = None
            self.path = []
        elif action.state_transition == "IDLE":
            self.state = DroneState.IDLE
            self.target = None
            self.path = []
            self._reported = False
        elif action.state_transition == "REPORTING":
            self.state = DroneState.REPORTING
            self.reporting_ticks_remaining = self.config.reporting_duration_ticks
            self.path = []
        if action.move_to and not _is_blocked_snap(snap, action.move_to):
            self.pos = action.move_to
        if self.path:
            self.path = self.path[1:]
        if self.state in (DroneState.SEARCHING, DroneState.RETURNING):
            if not (was_returning and self.pos == self.home):
                self.battery -= 1.0
        elif action.state_transition == "IDLE":
            self.battery = 100.0
        else:
            self.battery = min(100.0, self.battery + self.config.recharge_rate)
        if was_returning and self.pos == self.home and self.battery > 0:
            self.state = DroneState.REPORTING
        if self.battery < 0 and self.state == DroneState.RETURNING:
            self.crashed = True
            self.alive = False

    def _sensor_footprint(self, center):
        r = self.config.sensor_radius
        cx, cy = center
        w = h = self.config.grid_size
        cells: set = set()
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < w and 0 <= ny < h:
                    cells.add((nx, ny))
        return frozenset(cells)

    def _next_search_move(self, snap):
        if not self.target:
            return None
        if not self.path or self.path[-1] != self.target:
            self.path = list(
                astar(
                    np.asarray(snap.grid),
                    set(snap.drone_positions.values()) - {self.pos},
                    self.pos,
                    self.target,
                )
                or []
            )
        if not self.path:
            return None
        if self.path[0] == self.pos:
            self.path = self.path[1:]
        if not self.path:
            return None
        nxt = self.path[0]
        if nxt == self.pos or _is_blocked_snap(snap, nxt):
            detour = self._find_detour(snap)
            if detour is not None:
                self.path = [detour] + self.path
                return detour
            return None
        return nxt

    def _next_return_move(self, snap):
        goal = self.home
        if not self.path or self.path[-1] != goal:
            blockers = set(snap.drone_positions.values()) - {self.pos}
            grid = np.array(snap.grid) if not hasattr(snap.grid, "shape") else snap.grid
            result = astar(grid, blockers, self.pos, goal)
            self.path = result if result else []
        if not self.path:
            return None
        if self.pos == goal:
            self.path = []
            return None
        nxt = self.path[0]
        if _is_blocked_snap(snap, nxt):
            return None
        return nxt

    def _find_detour(self, snap):
        cx, cy = self.pos
        gx = self.target[0] if self.target else self.home[0]
        gy = self.target[1] if self.target else self.home[1]
        best = None
        best_d2 = None
        for dx, dy in _NEIGHBORS_8:
            nx, ny = cx + dx, cy + dy
            if _is_blocked_snap(snap, (nx, ny)):
                continue
            d2 = (nx - gx) ** 2 + (ny - gy) ** 2
            if best is None or d2 < best_d2:
                best, best_d2 = (nx, ny), d2
        return best if best and best != self.pos else None

    def check_return_threshold(self) -> bool:
        if self.state != DroneState.SEARCHING or not self.target:
            return False
        dx = abs(self.pos[0] - self.home[0])
        dy = abs(self.pos[1] - self.home[1])
        est = (dx + dy) * 1.5 * self.config.battery_safety_margin
        return self.battery <= est


_NEIGHBORS_8 = [
    (dx, dy)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if not (dx == 0 and dy == 0)
]


def _is_blocked_snap(snap, pos):
    x, y = pos
    h, w = len(snap.grid), len(snap.grid[0])
    if not (0 <= x < w and 0 <= y < h):
        return True
    return snap.grid[y][x] == 1 or pos in snap.drone_positions.values()
