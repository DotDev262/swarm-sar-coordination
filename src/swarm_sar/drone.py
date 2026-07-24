from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np
from swarm_sar.environment import SimConfig
from swarm_sar.mission_manager import Snapshot
from swarm_sar.astar import astar


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
    """Represents a drone agent in the Search and Rescue Swarm.

    Maintains position, state machine transitions, battery usage, and path planning
    activities.
    """

    id: int
    home: tuple[int, int]
    config: SimConfig
    pos: tuple[int, int] = field(default_factory=lambda: (0, 0))
    state: DroneState = DroneState.IDLE
    battery: float = 0.0
    alive: bool = True

    def __post_init__(self):
        """Initializes battery to capacity if not explicitly set above zero."""
        if self.battery <= 0:
            self.battery = self.config.battery_capacity
    target: Optional[tuple[int, int]] = None
    path: list[tuple[int, int]] = field(default_factory=list)
    _moved_this_tick: bool = field(default=False, repr=False)
    reporting_ticks_remaining: int = field(default=0, repr=False)
    crashed: bool = field(default=False, repr=False)
    _reported: bool = field(default=False, repr=False)

    def propose(self, snap: Snapshot, rng) -> Action:
        """Proposes an action to execute in the next tick based on current state.

        Args:
            snap: A Snapshot instance representing the current environment.
            rng: A random generator instance.

        Returns:
            An Action instance containing target movement and state transition details.
        """
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

    def commit(self, action: Action, snap: Snapshot) -> None:
        """Commits the proposed action and updates internal battery/crashed states.

        Args:
            action: The Action proposed during the decision phase.
            snap: The Snapshot representing the environment state.
        """
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
            if self.pos == self.home:
                self.battery = self.config.battery_capacity
        else:
            if self.pos == self.home:
                self.battery = min(
                    self.config.battery_capacity,
                    self.battery + self.config.recharge_rate,
                )
        if was_returning and self.pos == self.home and self.battery > 0:
            self.state = DroneState.REPORTING
        if self.battery < 0:
            self.crashed = True
            self.alive = False

    def _sensor_footprint(self, center: tuple[int, int]) -> frozenset[tuple[int, int]]:
        """Calculates coordinates covered by the sensor footprint at a cell.

        Args:
            center: The center (x, y) coordinate.

        Returns:
            A frozenset of grid coordinates covered by the sensor.
        """
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

    def _next_search_move(self, snap: Snapshot) -> Optional[tuple[int, int]]:
        """Plans and returns the next grid move when in SEARCHING state.

        Args:
            snap: The current Snapshot data.

        Returns:
            The next step (x, y) coordinates, or None if blocked.
        """
        if not self.target:
            return None
        # If the path is empty, outdated, or the next step is blocked, recalculate
        if not self.path or self.path[-1] != self.target or _is_blocked_snap(snap, self.path[0]):
            self.path = list(
                astar(
                    np.asarray(snap.grid),
                    set(snap.drone_positions.values()) - {self.pos, self.target},
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
            return None
        return nxt

    def _next_return_move(self, snap: Snapshot) -> Optional[tuple[int, int]]:
        """Plans and returns the next grid move when in RETURNING state.

        Args:
            snap: The current Snapshot data.

        Returns:
            The next step (x, y) coordinates, or None if blocked.
        """
        goal = self.home
        # If the path is empty, outdated, or the next step is blocked, recalculate
        if not self.path or self.path[-1] != goal or (self.path and _is_blocked_snap(snap, self.path[0])):
            blockers = set(snap.drone_positions.values()) - {self.pos, goal}
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

    def check_return_threshold(self) -> bool:
        """Determines if the battery is low enough to require returning to base.

        Returns:
            True if remaining battery capacity falls below the estimated return
            travel distance with safety margins. False otherwise.
        """
        if self.state != DroneState.SEARCHING or not self.target:
            return False
        dx = abs(self.pos[0] - self.home[0])
        dy = abs(self.pos[1] - self.home[1])
        est = (dx + dy) * 1.5 * self.config.battery_safety_margin
        return self.battery <= est


def _is_blocked_snap(snap: Snapshot, pos: tuple[int, int]) -> bool:
    """Checks if a cell is blocked by grid borders, obstacles, or other drones.

    Args:
        snap: The current Snapshot data.
        pos: The (x, y) coordinate target check.

    Returns:
        True if the position is blocked. False otherwise.
    """
    x, y = pos
    h, w = len(snap.grid), len(snap.grid[0])
    if not (0 <= x < w and 0 <= y < h):
        return True
    if snap.grid[y][x] == 1:
        return True
    if pos == snap.home:
        return False
    if pos in snap.drone_positions.values():
        return True
    return False
