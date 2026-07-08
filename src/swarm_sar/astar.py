import heapq
from typing import Optional


_NEIGHBORS = [(dx, dy)
              for dy in (-1, 0, 1)
              for dx in (-1, 0, 1)
              if not (dx == 0 and dy == 0)]


def _blocked(grid, blockers, x, y):
    """Checks if a cell is blocked by an obstacle or another agent.

    Args:
        grid: A 2D numpy array representing the environment grid.
        blockers: A set of (x, y) coordinates currently occupied by other agents.
        x: The target x-coordinate.
        y: The target y-coordinate.

    Returns:
        True if the cell is out of bounds, contains an obstacle (1), or is in
        the blockers set. False otherwise.
    """
    if not (0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]):
        return True
    return grid[y, x] == 1 or (x, y) in blockers


def _chebyshev(x1: int, y1: int, x2: int, y2: int) -> float:
    """Calculates the Chebyshev distance between two points.

    Args:
        x1: x-coordinate of the first point.
        y1: y-coordinate of the first point.
        x2: x-coordinate of the second point.
        y2: y-coordinate of the second point.

    Returns:
        The Chebyshev distance as a float.
    """
    return float(max(abs(x1 - x2), abs(y1 - y2)))


def astar(grid, blockers, start, goal) -> list[tuple[int, int]]:
    """Finds the shortest path on a grid using the A* algorithm.

    Diagonal moves are supported and have a cost of 1.0. Prevents diagonal
    corner-cutting through solid obstacles.

    Args:
        grid: A 2D numpy array representing the environment grid.
        blockers: A set of (x, y) coordinates occupied by other agents.
        start: The starting (x, y) coordinate.
        goal: The goal (x, y) coordinate.

    Returns:
        A list of (x, y) coordinate tuples representing the path from start
        to goal (excluding start, including goal), or an empty list if no path
        is found.
    """
    sx, sy = start
    gx, gy = goal
    if not (0 <= gx < grid.shape[1] and 0 <= gy < grid.shape[0]):
        return []
    if grid[gy, gx] == 1 or (gx, gy) in blockers or start == goal:
        return []

    hq: list[tuple[float, int, int]] = [(0.0, sx, sy)]
    came: dict[tuple[int, int], tuple[int, int]] = {}
    gscore: dict[tuple[int, int], float] = {start: 0.0}

    while hq:
        _, cx, cy = heapq.heappop(hq)
        if (cx, cy) == goal:
            path: list[tuple[int, int]] = []
            cur: Optional[tuple[int, int]] = goal
            while cur is not None and cur != start:
                path.append(cur)
                cur = came.get(cur)
            path.reverse()
            return path

        for dx, dy in _NEIGHBORS:
            nx, ny = cx + dx, cy + dy
            if _blocked(grid, blockers, nx, ny):
                continue
            # Prevent diagonal corner-cutting through solid obstacles/walls
            if dx != 0 and dy != 0:
                if grid[cy, nx] == 1 or grid[ny, cx] == 1:
                    continue
            move_cost = 1.0
            tentative = gscore.get((cx, cy), float("inf")) + move_cost
            if tentative < gscore.get((nx, ny), float("inf")):
                gscore[(nx, ny)] = tentative
                came[(nx, ny)] = (cx, cy)
                h = _chebyshev(nx, ny, gx, gy) * 1.001
                heapq.heappush(hq, (tentative + h, nx, ny))
    return []

