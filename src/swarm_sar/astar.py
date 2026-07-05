import heapq
from typing import Optional
import numpy as np


_NEIGHBORS = [(dx, dy)
              for dy in (-1, 0, 1)
              for dx in (-1, 0, 1)
              if not (dx == 0 and dy == 0)]


def _blocked(grid, blockers, x, y):
    if not (0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]):
        return True
    return grid[y, x] == 1 or (x, y) in blockers


def _octile(x1: int, y1: int, x2: int, y2: int) -> float:
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    return max(dx, dy) + 0.4142135623730951 * min(dx, dy)


def astar(grid, blockers, start, goal) -> list[tuple[int, int]]:
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
            move_cost = 1.0 if dx == 0 or dy == 0 else 1.0
            tentative = gscore.get((cx, cy), float("inf")) + move_cost
            if tentative < gscore.get((nx, ny), float("inf")):
                gscore[(nx, ny)] = tentative
                came[(nx, ny)] = (cx, cy)
                h = _octile(nx, ny, gx, gy) * 1.001
                heapq.heappush(hq, (tentative + h, nx, ny))
    return []
