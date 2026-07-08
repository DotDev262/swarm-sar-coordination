from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set, Tuple

import numpy as np
import random


@dataclass(frozen=True)
class SimConfig:
    """Configuration settings for the Swarm SAR Coordination simulation."""
    scenario_path: Optional[str] = None
    n_drones: int = 5
    grid_size: int = 20
    obstacle_density: float = 0.10
    seed: int = 0
    sensor_radius: int = 2
    ticks_per_second: int = 10
    max_ticks: int = 10000
    coverage_threshold: float = 1.0
    log_interval_ticks: int = 10

    battery_safety_margin: float = 1.2
    recharge_rate: float = 2.0
    reporting_duration_ticks: int = 3
    battery_capacity: float = 0.0

    def __post_init__(self):
        """Scales battery capacity relative to grid size when no explicit value is set."""
        if self.battery_capacity <= 0:
            # Scale capacity so that drones must return to base and recharge in default runs
            object.__setattr__(self, "battery_capacity",
                                1.5 * (self.grid_size + self.grid_size))


def _ensure_connectivity(grid: np.ndarray, home: Tuple[int, int]) -> np.ndarray:
    """Ensures that all traversable grid cells are connected to the home base.

    Floods the grid from home and marks any unreachable empty cells as obstacles.

    Args:
        grid: A 2D numpy array representing the environment grid.
        home: The (x, y) coordinates of the home base.

    Returns:
        The modified numpy grid.
    """
    h, w = grid.shape
    hx, hy = home

    seen: Set[Tuple[int, int]] = {(hx, hy)}
    q = [(hx, hy)]
    while q:
        x, y = q.pop()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                if grid[ny, nx] == 0 and (nx, ny) not in seen:
                    seen.add((nx, ny))
                    q.append((nx, ny))

    for y in range(h):
        for x in range(w):
            if grid[y, x] == 0 and (x, y) not in seen:
                grid[y, x] = 1

    return grid


@dataclass
class Environment:
    """Representation of the search and rescue grid environment."""
    width: int
    height: int
    grid: np.ndarray
    home: Tuple[int, int]
    spawn: Tuple[int, int]
    target: Optional[Tuple[int, int]] = None

    @classmethod
    def generate_grid(cls, size: int, density: float, seed: int) -> np.ndarray:
        """Generates a randomized grid with obstacles and a home base.

        Args:
            size: Width and height of the square grid.
            density: Ratio of cells that should contain obstacles.
            seed: Random seed for repeatability.

        Returns:
            A 2D numpy array representing the generated grid.
        """
        rng = random.Random(seed)
        grid = np.zeros((size, size), dtype=np.uint8)
        hx = hy = size // 2

        for y in range(size):
            for x in range(size):
                if rng.random() < density:
                    grid[y, x] = 1

        # clear home + 8-neighbourhood
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nx, ny = hx + dx, hy + dy
                if 0 <= nx < size and 0 <= ny < size:
                    grid[ny, nx] = 0

        # ensure connectivity
        _ensure_connectivity(grid, (hx, hy))

        grid[hy, hx] = 2
        return grid

    @classmethod
    def from_config(cls, config: SimConfig) -> "Environment":
        """Constructs an Environment instance from a SimConfig settings object.

        Args:
            config: A SimConfig instance.

        Returns:
            An Environment instance with spawned target cell details.
        """
        if config.scenario_path:
            env = cls._load_scenario(Path(config.scenario_path))
        else:
            grid = cls.generate_grid(config.grid_size, config.obstacle_density, config.seed)
            env = cls._from_grid(grid)
        
        rng = random.Random(config.seed)
        searchable_cells = [
            (int(x), int(y)) for y, x in zip(*np.where(env.grid == 0))
        ]
        if searchable_cells:
            env.target = rng.choice(searchable_cells)
        return env

    @classmethod
    def _from_grid(cls, grid: np.ndarray) -> "Environment":
        """Helper method to construct Environment from pre-existing grid array.

        Args:
            grid: A 2D numpy grid array.

        Returns:
            An Environment instance.
        """
        h, w = grid.shape
        assert np.sum(grid == 2) == 1, "generate_grid must mark exactly one home"
        hy, hx = np.where(grid == 2)
        home = (int(hx[0]), int(hy[0]))
        return cls(width=w, height=h, grid=grid, home=home, spawn=home)

    @classmethod
    def _load_scenario(cls, path: Path) -> "Environment":
        """Loads a grid environment scenario from a text file.

        Args:
            path: Path to the scenario file.

        Returns:
            An Environment instance loaded from file.

        Raises:
            ValueError: If scenario has format issues like missing home base.
        """
        rows = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rows.append(line)

        height = len(rows)
        width = max(len(r) for r in rows)
        grid = np.zeros((height, width), dtype=np.uint8)
        home: Optional[Tuple[int, int]] = None

        for y, row in enumerate(rows):
            for x, ch in enumerate(row):
                if ch == ".":
                    grid[y, x] = 0
                elif ch == "#":
                    grid[y, x] = 1
                elif ch == "H":
                    grid[y, x] = 2
                    if home is not None:
                        raise ValueError(f"Multiple H in {path}")
                    home = (x, y)

        if home is None:
            raise ValueError(f"No 'H' in {path}")

        spawn = home
        for y, row in enumerate(rows):
            for x, ch in enumerate(row):
                if ch == "S":
                    spawn = (x, y)

        return cls(width=width, height=height, grid=grid, home=home, spawn=spawn)
