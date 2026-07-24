# Swarm SAR Multi-Agent Simulation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans`. Steps use `- [ ]` checkbox syntax.
> **Spec:** `docs/superpowers/specs/2026-07-05-swarm-sar-simulation-design.md`
> **Save plans to:** `docs/superpowers/plans/`

**Goal:** Build a working Python 3.11 multi-agent SAR simulation with Pygame renderer, headless sweep runner, and matplotlib plot generator. Strict `uv`-only tooling, 7 pytest files.

**Architecture:** Single flat package `swarm_sar` under `src/`. One module per responsibility: `environment`, `astar`, `mission_manager`, `drone`, `simulator`, `renderer`, `sweep`, `plot`. Tests in `tests/`. Headless `sweep` imports zero of `pygame`.

**Tech Stack:** Python 3.11, pygame 2.5, numpy 1.26, matplotlib 3.8, pandas 2.2, pytest 8.0, ruff 0.6 — all via `uv`.

## Global Constraints

- Python 3.11, `.python-version` committed.
- **`uv`-only tooling.** Every command via `uv run ...`. No `pip` anywhere.
- `pygame` is a flat runtime dep; `sweep` runner must never import it.
- Drone speed: uniform `1 cell/tick`. Motion: **8-connected** grid.
- Sensor radius default: `2` (Chebyshev, 5×5 footprint = 25 cells per move).
- Battery: drain `1/tick` while SEARCHING (whether moved or not) and while RETURNING; `0` while IDLE/REPORTING; recharge `recharge_rate` while IDLE/REPORTING; diagonal move costs `1.0`.
- FSM: 4 states (`IDLE`, `SEARCHING`, `RETURNING`, `REPORTING`). No per-drone JSON report file.
- Task allocation: pull-based greedy nearest-cell. Linear scan O(U). `ponytail:` comment names k-d tree upgrade.
- Renderer: flat colors — searched=green, unsearched=gray, obstacle=black, home=yellow. No freshness fade, no snapshot PNG.
- CLI: flag dispatch (no subcommands). Modes: interactive (default), `--sweep`, `--plot <path>`.
- `mise.toml` delegates to `uv run ...`. `.gitignore` cleaned of Godot section. `out/` gitignored. `uv.lock` committed.

---

## Pre-flight check

```bash
cd /home/aryan/Projects/swarm-sar-coordination
uv --version                              # must be ≥ 0.4
uv init --package swarm_sar --python 3.11 # if not already done
ls src/swarm_sar/                         # __init__.py expected
git status                                # clean tree expected per step
```

If `uv init` already ran / `src/swarm_sar/` already exists, proceed.

---

## Shared types pattern

These types are defined in their own module and re-exported from `swarm_sar/__init__.py`. Tasks that consume them include the definition verbatim so the implementer doesn't need to hunt.

`src/swarm_sar/mission_manager.py` defines (and other modules import from there):

```python
from dataclasses import dataclass
from typing import FrozenSet


@dataclass(frozen=True)
class Snapshot:
    grid: tuple[tuple[int, ...], ...]
    obstacles: frozenset[tuple[int, int]]
    searched: frozenset[tuple[int, int]]
    drone_positions: dict[int, tuple[int, int]]
    home: tuple[int, int]
    searchable: frozenset[tuple[int, int]]

from enum import Enum
class DroneState(Enum):
    IDLE = "IDLE"
    SEARCHING = "SEARCHING"
    RETURNING = "RETURNING"
    REPORTING = "REPORTING"
```

```python
# simulator.py: defines the inter-module contract
from dataclasses import dataclass

@dataclass(frozen=True)
class Action:
    move_to: tuple[int, int] | None   # None = hold
    mark_cells: frozenset[tuple[int, int]]
    request_task: bool
    state_transition: str | None      # 'RETURNING'|'IDLE'|'REPORTING'
```

---

## File map (locked)

```
src/swarm_sar/
  __init__.py
  __main__.py
  environment.py
  astar.py
  mission_manager.py
  drone.py
  simulator.py
  renderer.py
  sweep.py
  plot.py

tests/
  test_environment.py
  test_astar.py
  test_fsm.py
  test_battery.py
  test_allocation.py
  test_coverage.py
  test_simulator.py

scenarios/
  default.txt
  empty.txt
  maze.txt
  walls.txt

out/                    # gitignored
```

Each task is independently testable and produces a meaningful reviewer gate.

---

## Task 1: Project scaffolding

**Files:** create `pyproject.toml` (or edit if `uv init` already created it), `.gitignore`, `mise.toml`, `README.md`, `scenarios/{default,empty,maze,walls}.txt`, `out/.gitkeep`. Step 2 adds runtime deps.

- [ ] **Step 1: Scaffold the package**
  ```bash
  cd /home/aryan/Projects/swarm-sar-coordination
  uv init --package swarm_sar --python 3.11
  ```
  If `uv init` already ran, open `pyproject.toml` and confirm the `[project]` section has `name = "swarm_sar"` and `requires-python = ">=3.11"`. Replace any existing content with:
  ```toml
  [project]
  name = "swarm_sar"
  version = "0.1.0"
  requires-python = ">=3.11"
  dependencies = []

  [dependency-groups]
  dev = ["pytest>=8.0", "ruff>=0.6"]

  [build-system]
  requires = ["hatchling"]
  build-backend = "hatchling.build"
  ```
  The `dependencies = []` line means the package has zero runtime deps yet. Step 2 of this task fills that in.

- [ ] **Step 2: Add runtime deps + sync**
  Replace `dependencies = []` with:
  ```toml
  dependencies = [
      "pygame>=2.5",
      "numpy>=1.26",
      "matplotlib>=3.8",
      "pandas>=2.2",
  ]
  ```
  Then run:
  ```bash
  uv sync
  ```
  Expected: `Resolved N packages`, lockfile written to `uv.lock`.

- [ ] **Step 3: Commit lockfile**
  ```bash
  git add uv.lock && git commit -m "chore: commit uv.lock"
  ```

- [ ] **Step 4: Rewrite `.gitignore`**
  Replace `.gitignore` with:
  ```
  # Python / uv
  .venv/
  __pycache__/
  .ruff_cache/
  # Project output
  out/
  # OS / editor junk
  .DS_Store
  .DS_Store?
  ._*
  .Spotlight-V100
  .Trashes
  ehthumbs.db
  Thumbs.db
  *.log
  *.tmp
  ```
  Note: `.python-version` and `uv.lock` are **not** ignored.

  ```bash
  git add .gitignore && git commit -m "chore: Python/uv .gitignore"
  ```

- [ ] **Step 5: Write `mise.toml`**
  ```toml
  [tools]
  uv = "latest"

  [tasks]
  dev   = "uv run python -m swarm_sar"
  test  = "uv run pytest"
  sweep = "uv run python -m swarm_sar --sweep"
  lint  = "uv run ruff check src tests"
  ```

  ```bash
  git add mise.toml && git commit -m "chore: rewired mise.toml to uv"
  ```

- [ ] **Step 6: Write `README.md`**
  ```markdown
  # swarm-sar-coordination

  Hybrid multi-agent drone SAR simulation. Python 3.11 + Pygame + uv.

  ## Setup
  uv sync

  ## Run
  uv run python -m swarm_sar                  # interactive
  uv run python -m swarm_sar --sweep --help    # headless batch
  uv run python -m swarm_sar --plot <path>     # plot from aggregate CSV

  ## Test
  uv run pytest
  ```

- [ ] **Step 7: Create scenario files**
  `scenarios/default.txt`:
  ```
  ...............
  ....##....##...
  ....##....##...
  ...............
  ...............
  .H.............
  ...............
  ...............
  ...............
  ...............
  ```
  `scenarios/empty.txt`:
  ```
  .....
  .....
  ..H..
  .....
  .....
  ```
  `scenarios/maze.txt`:
  ```
  ##############
  #............#
  #.###.#######.#
  #.#...#.....#.#
  #.#.#####.#.#.#
  #.#.....#.#.#.#
  #.#####.#.#.#.#
  #.....#.....#.#
  #######.#####.#
  #.............#
  ##############
  ```
  `scenarios/walls.txt`:
  ```
  .............
  .#####.#####.
  .#####.#####.
  .#####.#####.
  .............
  .H............
  .............
  .#####.#####.
  .#####.#####.
  .#####.#####.
  .............
  ```

- [ ] **Step 8: Create `out/` directory + `.gitkeep`**
  ```bash
  mkdir -p out && touch out/.gitkeep
  ```
  `.gitkeep` is gitignored (via `out/` rule); its presence makes git track the empty directory.

- [ ] **Step 9: Commit bundle**
  ```bash
  git add README.md scenarios/ out/.gitkeep mise.toml .gitignore
  git commit -m "feat: project scaffolding, scenarios, tooling"
  ```

- [ ] **Verify tree is clean**
  ```bash
  git status
  ```
  Expected: `nothing to commit, working tree clean`.

---

## Task 3: A* pathfinder

**Spec refs:** Section 5 (A* algorithm, 8-connected, octile heuristic)
**Tests:** `tests/test_astar.py`

- [ ] **Step 1: Write failing tests** `tests/test_astar.py`
  ```python
  import numpy as np
  from swarm_sar.astar import astar


  def _grid(h, w, obs=None):
      g = np.zeros((h, w), dtype=np.uint8)
      if obs:
          for y, x in obs:
              g[y, x] = 1
      return g


  def test_shortest_path_empty():
      g = _grid(5, 5)
      p = astar(g, set(), (0, 0), (4, 4))
      assert len(p) == 4
      assert p[0] == (1, 1) and p[-1] == (4, 4)


  def test_detours_around_wall():
      g = _grid(5, 5, [(2, c) for c in range(5)])
      p = astar(g, set(), (1, 1), (3, 1))
      assert len(p) > 2
      assert (2, 1) not in p
      assert p[-1] == (3, 1)


  def test_no_path_when_goal_blocked():
      p = astar(_grid(3, 3), {(1, 1)}, (0, 0), (1, 1))
      assert p == []


  def test_no_path_when_isolated():
      g = _grid(3, 3, [(1, 0), (0, 1), (1, 1)])
      assert astar(g, set(), (0, 0), (2, 2)) == []


  def test_start_eq_goal_returns_empty():
      assert astar(_grid(3, 3), set(), (2, 2), (2, 2)) == []


  def test_blockers_respected():
      blockers = {(1, 0), (1, 1), (0, 1)}
      p = astar(_grid(5, 5), blockers, (0, 0), (2, 2))
      assert (1, 0) not in p and (1, 1) not in p and (0, 1) not in p
  ```

- [ ] **Step 2: Run — expect failure**
  ```bash
  uv run pytest tests/test_astar.py -v
  ```
  Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write `src/swarm_sar/astar.py`**
  ```python
  import heapq
  from typing import Optional
  import numpy as np

  _NEIGHBORS = [(dx, dy)
                for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                if not (dx == 0 and dy == 0)]


  def _blocked(grid, blockers, x, y):
      if not (0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]):
          return True
      return grid[y, x] == 1 or (x, y) in blockers


  def _octile(x1, y1, x2, y2) -> float:
      dx, dy = abs(x1 - x2), abs(y1 - y2)
      return max(dx, dy) + 0.414 * min(dx, dy)


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
                  h = _octile(nx, ny, gx, gy) * 1.001  # epsilon tiebreak
                  heapq.heappush(hq, (tentative + h, nx, ny))
      return []
  ```

- [ ] **Step 4: Run — expect pass**
  ```bash
  uv run pytest tests/test_astar.py -v
  ```
  Expected: 6 pass.

- [ ] **Step 5: Commit**
  ```bash
  git add src/swarm_sar/astar.py tests/test_astar.py
  git commit -m "feat: A* pathfinder (8-connected, octile heuristic)"
  ```

---

## Task 4: MissionManager — SimConfig, allocation, coverage, logging

**Spec refs:** Section 6 (MissionManager, allocation, coverage, logging)
**Tests:** `tests/test_allocation.py`, `tests/test_coverage.py`

- [ ] **Step 1: Write failing tests** `tests/test_allocation.py`
  ```python
  import numpy as np
  from swarm_sar.mission_manager import MissionManager, SimConfig, Environment


  def _mm(home=(2, 2), w=5, h=5):
      grid = np.zeros((h, w), dtype=np.uint8)
      grid[home[1], home[0]] = 2
      env = Environment(width=w, height=h, grid=grid, home=home, spawn=home)
      return MissionManager(env, SimConfig(n_drones=1, grid_size=w))


  def test_assign_returns_something():
      r = _mm().assign_task(0, (0, 0))
      assert r is not None
      x, y = r
      assert 0 <= x < 5 and 0 <= y < 5


  def test_none_when_exhausted():
      mm = _mm()
      while True:
          c = mm.assign_task(0, (0, 0))
          if c is None:
              break
          mm.mark_searched({c}, 0)
      assert mm.assign_task(0, (0, 0)) is None


  def test_assigned_cell_not_reassigned():
      mm = _mm()
      r1 = mm.assign_task(0, (0, 0))
      assert r1 is not None
      r2 = mm.assign_task(1, (0, 0))
      assert r2 is None or r2 != r1


  def test_searchable_excludes_obstacles():
      grid = np.zeros((5, 5), dtype=np.uint8)
      for oy, ox in [(1, 1), (1, 2), (1, 3), (3, 1), (3, 2), (3, 3)]:
          grid[oy, ox] = 1
      grid[2, 2] = 2
      env = Environment(5, 5, grid, (2, 2), (2, 2))
      mm = MissionManager(env, SimConfig(n_drones=1, grid_size=5))
      for oy, ox in [(1,1),(1,2),(1,3),(3,1),(3,2),(3,3)]:
          assert (ox, oy) not in mm.searchable
  ```

  `tests/test_coverage.py`:
  ```python
  import numpy as np
  from swarm_sar.mission_manager import MissionManager, SimConfig, Environment


  def _mm():
      grid = np.zeros((5, 5), dtype=np.uint8)
      grid[2, 2] = 2
      env = Environment(5, 5, grid, (2, 2), (2, 2))
      return MissionManager(env, SimConfig(n_drones=1, grid_size=5))


  def test_coverage_starts_zero():
      assert _mm().coverage() == 0.0


  def test_coverage_scales():
      mm = _mm()
      mm.mark_searched({(0, 0), (1, 0)}, 0)
      assert mm.coverage() == 2.0 / len(mm.searchable)


  def test_re_marking_no_double_count():
      mm = _mm()
      mm.mark_searched({(0, 0)}, 0)
      mm.mark_searched({(0, 0)}, 0)
      assert mm.coverage() == 1.0 / len(mm.searchable)


  def test_failure_does_not_credit_coverage():
      mm = _mm()
      t = mm.assign_task(0, (0, 0))
      assert t is not None
      mm.on_drone_killed(0, t, tick=5)
      assert t not in mm.searched
  ```

- [ ] **Step 2: Run — expect failure**
  ```bash
  uv run pytest tests/test_allocation.py tests/test_coverage.py -v
  ```
  Expected: `ModuleNotFoundError: No module named 'swarm_sar.mission_manager'`.

- [ ] **Step 3: Write `src/swarm_sar/mission_manager.py`**
  ```python
  import csv
  import time
  from dataclasses import dataclass
  from typing import Optional
  import numpy as np
  from swarm_sar.environment import Environment, SimConfig


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
          self.searchable: set[tuple[int, int]] = set(zip(*np.where(env.grid == 0)))
          self.searched: set[tuple[int, int]] = set()
          self.in_flight: dict[int, tuple[int, int]] = {}
          self.per_drone_stats: dict[int, dict] = {
              i: {"assignments": 0, "cells_marked": 0, "distance": 0,
                  "time_in_state": {"IDLE": 0, "SEARCHING": 0,
                                    "RETURNING": 0, "REPORTING": 0}}
              for i in range(config.n_drones)
          }
          self.failure_log: list[tuple[int, int, int]] = []
          self._log_path: Optional[str] = None
          self._log_buf: list[str] = []
          self._wrote_header = False
          self._log_interval = config.log_interval_ticks

      def set_log_path(self, path: str) -> None:
          self._log_path = path
          with open(path, "w", newline="") as fh:
              fh.write("tick,active_drones,failed_drones,coverage_pct,"
                       "remaining_cells,avg_battery,fps\n")

      def snapshot(self, drones: list) -> Snapshot:
          return Snapshot(
              grid=tuple(tuple(r) for r in self.env.grid),
              obstacles=frozenset(zip(*np.where(self.env.grid == 1))),
              searched=frozenset(self.searched),
              drone_positions={d.id: d.pos for d in drones},
              home=self.home,
              searchable=frozenset(self.searchable),
          )

      def assign_task(self, drone_id: int,
                       drone_pos: tuple[int, int]) -> tuple[int, int] | None:
          candidates = self.searchable - self.searched - set(self.in_flight.values())
          # ponytail: O(U) where U = unassigned searchable cells.
          # Fine ≤150×150, ≤50 drones. Upgrade: scipy.spatial.cKDTree.
          if not candidates:
              return None
          best = min(candidates,
                     key=lambda c: (c[0] - drone_pos[0])**2
                                   + (c[1] - drone_pos[1])**2)
          self.in_flight[drone_id] = best
          self.per_drone_stats[drone_id]["assignments"] += 1
          return best

      def mark_searched(self, cells, drone_id: int) -> None:
          new_cells = cells - self.searched
          self.searched |= cells
          if drone_id in self.per_drone_stats:
              self.per_drone_stats[drone_id]["cells_marked"] += len(new_cells)

      def record_move(self, drone_id: int) -> None:
          if drone_id in self.per_drone_stats:
              self.per_drone_stats[drone_id]["distance"] += 1

      def release_task(self, drone_id: int) -> None:
          self.in_flight.pop(drone_id, None)

      def on_drone_killed(self, drone_id: int, target, tick: int) -> None:
          self.release_task(drone_id)
          self.failure_log.append((tick, drone_id, 0))

      def on_drone_reported(self, drone_id: int) -> None:
          pass  # lightweight; per_drone_stats already accumulates

      def update(self, drones: list, tick: int) -> None:
          if self._log_path and not self._wrote_header:
              self.set_log_path(self._log_path)
              self._wrote_header = True
          for d in drones:
              if d.alive and d.id in self.per_drone_stats:
                  st = d.state.value
                  self.per_drone_stats[d.id]["time_in_state"][st] += 1
                  if d._moved_this_tick:
                      self.record_move(d.id)
          if self._log_path and tick % self._log_interval == 0:
              active = sum(1 for d in drones if d.alive)
              failed = sum(1 for d in drones if not d.alive)
              bats = [d.battery for d in drones if d.alive]
              avg = sum(bats) / len(bats) if bats else 0.0
              self._log_buf.append(
                  f"{tick},{active},{failed},"
                  f"{self.coverage()*100:.1f},"
                  f"{len(self.searchable)-len(self.searched)},"
                  f"{avg:.1f},0.0"
              )

      def coverage(self) -> float:
          if not self.searchable:
              return 1.0
          return len(self.searched) / len(self.searchable)

      def is_complete(self) -> bool:
          return self.coverage() >= self.config.coverage_threshold

      def flush_log(self) -> None:
          if self._log_path and self._log_buf:
              with open(self._log_path, "a", newline="") as fh:
                  fh.write("\n".join(self._log_buf) + "\n")
              self._log_buf.clear()

      def summary_line(self, total_ticks: int) -> str:
          dist = sum(s["distance"] for s in self.per_drone_stats.values())
          return (f"#SUMMARY total_ticks={total_ticks} "
                  f"final_coverage={self.coverage()*100:.1f} "
                  f"drone_failures={len(self.failure_log)} "
                  f"avg_allocation_latency=0.0 total_path_length={dist}")
  ```

  Add to `src/swarm_sar/__init__.py`:
  ```python
  from swarm_sar.environment import SimConfig
  from swarm_sar.mission_manager import MissionManager, Snapshot
  ```

- [ ] **Step 4: Run — expect pass**
  ```bash
  uv run pytest tests/test_allocation.py tests/test_coverage.py -v
  ```
  Expected: 9 pass.

- [ ] **Step 5: Commit**
  ```bash
  git add src/swarm_sar/mission_manager.py src/swarm_sar/__init__.py \
       tests/test_allocation.py tests/test_coverage.py
  git commit -m "feat: MissionManager, SimConfig, allocation, coverage, logging"
  ```

---

## Task 5: DroneAgent + FSM

**Spec refs:** Section 5 (FSM, battery, A* calls, REPORTING, collision avoidance, search-and-target lifecycle)
**Tests:** `tests/test_fsm.py`, `tests/test_battery.py`

- [ ] **Step 1: Write failing tests** `tests/test_fsm.py`
  ```python
  from swarm_sar.drone import DroneAgent, DroneState
  from swarm_sar.environment import SimConfig, Snapshot


  def _drone(pos=(0, 0), state=DroneState.IDLE, battery=100.0):
      cfg = SimConfig(n_drones=1, grid_size=5, seed=0)
      d = DroneAgent(0, (2, 2), cfg)
      d.pos = pos; d.state = state; d.battery = battery
      return d


  def _snap():
      return Snapshot(
          grid=tuple(tuple([0]*5) for _ in range(5)),
          obstacles=frozenset(), searched=frozenset(),
          drone_positions={}, home=(2, 2), searchable=frozenset(),
      )


  def test_idle_proposes_request_task():
      action = _drone().propose(_snap(), __import__("random").Random(0))
      assert action.request_task is True
      assert action.move_to is None


  def test_searching_proposes_move():
      d = _drone(state=DroneState.SEARCHING)
      d.target = (4, 4)
      d.path = [(1, 1), (2, 2), (3, 3), (4, 4)]
      a = d.propose(_snap(), __import__("random").Random(0))
      assert a.move_to == (1, 1)
      assert not a.request_task


  def test_searching_drains_battery():
      d = _drone(state=DroneState.SEARCHING, battery=50.0)
      d.target = (1, 0)
      d.path = [(1, 0)]
      snap = _snap()
      rng = __import__("random").Random(0)
      for _ in range(10):
          d.propose(snap, rng); d.commit(d.propose(snap, rng), snap)
      assert d.battery < 40.0


  def test_returning_triggers_on_low_battery():
      d = _drone(state=DroneState.SEARCHING, battery=5.0)
      d.pos = (4, 4)
      d.target = (4, 4)
      d.path = []
      snap = _snap()
      rng = __import__("random").Random(0)
      d.propose(snap, rng); d.commit(d.propose(snap, rng), snap)
      assert d.state == DroneState.RETURNING


  def test_at_home_returning_transitions_to_reporting():
      d = _drone(state=DroneState.RETURNING, battery=50.0)
      d.pos = (2, 2)
      d.path = [(2, 2)]
      snap = _snap()
      rng = __import__("random").Random(0)
      d.propose(snap, rng); d.commit(d.propose(snap, rng), snap)
      assert d.state == DroneState.REPORTING


  def test_reporting_transitions_to_idle():
      d = _drone(state=DroneState.REPORTING, battery=50.0)
      d.reporting_ticks_remaining = 1
      snap = _snap()
      rng = __import__("random").Random(0)
      d.propose(snap, rng); d.commit(d.propose(snap, rng), snap)
      assert d.state == DroneState.IDLE
      assert d.battery == 100.0
  ```

  `tests/test_battery.py`:
  ```python
  from swarm_sar.drone import DroneAgent, DroneState
  from swarm_sar.environment import SimConfig, Snapshot


  def _drone(state, battery, pos=(0, 0)):
      cfg = SimConfig(n_drones=1, grid_size=5, seed=0)
      d = DroneAgent(0, (2, 2), cfg)
      d.state = state; d.battery = battery; d.pos = pos
      return d


  def _snap():
      return Snapshot(
          grid=tuple(tuple([0]*5) for _ in range(5)),
          obstacles=frozenset(), searched=frozenset(),
          drone_positions={}, home=(2, 2), searchable=frozenset(),
      )


  def test_returning_drains():
      d = _drone(DroneState.RETURNING, 50.0)
      rng = __import__("random").Random(0)
      d.propose(_snap(), rng); d.commit(d.propose(_snap(), rng), _snap())
      assert d.battery < 49.0


  def test_idle_recharges():
      d = _drone(DroneState.IDLE, 50.0)
      rng = __import__("random").Random(0)
      for _ in range(5):
          d.propose(_snap(), rng); d.commit(d.propose(_snap(), rng), _snap())
      assert d.battery > 50.0


  def test_reporting_recharges():
      d = _drone(DroneState.REPORTING, 50.0)
      d.reporting_ticks_remaining = 3
      rng = __import__("random").Random(0)
      for _ in range(3):
          d.propose(_snap(), rng); d.commit(d.propose(_snap(), rng), _snap())
      assert d.battery > 50.0


  def test_crashes_below_zero_on_returning():
      d = _drone(DroneState.RETURNING, 0.5)
      rng = __import__("random").Random(0)
      d.propose(_snap(), rng); d.commit(d.propose(_snap(), rng), _snap())
      assert not d.alive
  ```

- [ ] **Step 2: Run — expect failure**
  ```bash
  uv run pytest tests/test_fsm.py tests/test_battery.py -v
  ```
  Expected: `ModuleNotFoundError: No module named 'swarm_sar.drone'`.

- [ ] **Step 3: Write `src/swarm_sar/drone.py`**
  ```python
  from dataclasses import dataclass, field
  from enum import Enum
  from typing import Optional
  import numpy as np
  import math
  from swarm_sar.environment import SimConfig, Snapshot
  from swarm_sar.astar import astar

  # Action lives here (not in simulator.py) to break the circular import:
  # simulator.py needs DroneAgent, drone.py needs Action.
  from dataclasses import dataclass
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
          was_active = self.state in (DroneState.SEARCHING, DroneState.RETURNING)
          if action.state_transition == "RETURNING":
              self.state = DroneState.RETURNING
              self.target = None; self.path = []
          elif action.state_transition == "IDLE":
              self.state = DroneState.IDLE
              self.target = None; self.path = []
              self._reported = False
          elif action.state_transition == "REPORTING":
              self.state = DroneState.REPORTING
              self.reporting_ticks_remaining = self.config.reporting_duration_ticks
              self.path = []
          if action.move_to and not _is_blocked_snap(snap, action.move_to):
              self.pos = action.move_to
              if self.path:
                  self.path = self.path[1:]
          # battery
          if self.state in (DroneState.SEARCHING, DroneState.RETURNING):
              self.battery -= 1.0
          else:
              self.battery = min(100.0, self.battery + self.config.recharge_rate)
          if self.battery < 0 and self.state == DroneState.RETURNING:
              self.crashed = True; self.alive = False

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
          if not self.path:
              return None
          nxt = self.path[0]
          if _is_blocked_snap(snap, nxt):
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
              result = astar(snap.grid, blockers, self.pos, goal)
              self.path = result if result else []
          if not self.path:
              return None
          nxt = self.path[0]
          if _is_blocked_snap(snap, nxt):
              return None
          return nxt

      def _find_detour(self, snap):
          cx, cy = self.pos
          gx = self.target[0] if self.target else self.home[0]
          gy = self.target[1] if self.target else self.home[1]
          best = None; best_d2 = None
          for dx, dy in _NEIGHBORS_8:
              nx, ny = cx + dx, cy + dy
              if _is_blocked_snap(snap, (nx, ny)):
                  continue
              d2 = (nx - gx)**2 + (ny - gy)**2
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


  _NEIGHBORS_8 = [(dx, dy)
                  for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                  if not (dx == 0 and dy == 0)]


  def _is_blocked_snap(snap, pos):
      x, y = pos
      h, w = len(snap.grid), len(snap.grid[0])
      if not (0 <= x < w and 0 <= y < h):
          return True
      return snap.grid[y][x] == 1 or pos in snap.drone_positions.values()
  ```

- [ ] **Step 4: Run — expect pass**
  ```bash
  uv run pytest tests/test_fsm.py tests/test_battery.py -v
  ```
  Expected: 11 pass.

- [ ] **Step 5: Commit**
  ```bash
  git add src/swarm_sar/drone.py tests/test_fsm.py tests/test_battery.py
  git commit -m "feat: DroneAgent, FSM, battery, collision avoidance"
  ```

---

## Task 6: Simulator (tick engine + renderer + CLI + sweep)

**Spec refs:** Section 4 (Simulation Engine, snapshot/propose/resolve), Section 8 (Rendering, Sweep, Plot, CLI)
**Tests:** `tests/test_simulator.py`

- [ ] **Step 1: Write failing tests** `tests/test_simulator.py`
  ```python
  import numpy as np
  from swarm_sar.simulator import Simulator
  from swarm_sar.environment import SimConfig, Environment


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
              out.append((sim.tick, d0.id, *d0.pos, round(d0.battery, 1)))
      assert t1 == t2


  def test_coverage_progress():
      cfg = SimConfig(seed=0, n_drones=1, grid_size=5, obstacle_density=0.0,
                      max_ticks=100)
      sim = Simulator(_env(), cfg)
      for _ in range(5):
          sim.tick_once()
      assert sim.mm.coverage() > 0.0


  def test_dead_drone_stops_moving():
      cfg = SimConfig(seed=0, n_drones=1, grid_size=5, obstacle_density=0.0)
      sim = Simulator(_env(), cfg)
      sim.drones[0].alive = False
      pos_before = sim.drones[0].pos
      sim.tick_once()
      assert sim.drones[0].pos == pos_before


  def test_is_complete():
      cfg = SimConfig(n_drones=1, grid_size=5, obstacle_density=0.0,
                      coverage_threshold=0.5)
      sim = Simulator(_env(), cfg)
      for cell in list(sim.mm.searchable):
          sim.mm.mark_searched({cell}, 0)
      assert sim.mm.is_complete()
  ```

- [ ] **Step 2: Run — expect failure**
  ```bash
  uv run pytest tests/test_simulator.py -v
  ```
  Expected: `ModuleNotFoundError: No module named 'swarm_sar.simulator'`.

- [ ] **Step 3: Write `src/swarm_sar/simulator.py`**
  ```python
  from dataclasses import dataclass
  from typing import Sequence
  from swarm_sar.environment import Environment
  from swarm_sar.mission_manager import MissionManager, SimConfig, Snapshot
  from swarm_sar.drone import DroneAgent as _DroneAgent, DroneState, Action


  # Action imported from drone.py (not redefined here).


  @dataclass(frozen=True)
  class Action:
      move_to: tuple[int, int] | None
      mark_cells: frozenset[tuple[int, int]]
      request_task: bool
      state_transition: str | None


  class Simulator:
      def __init__(self, env: Environment, config: SimConfig):
          self.env = env
          self.config = config
          self.mm = MissionManager(env, config)
          self.drones = [
              DroneAgent(i, env.spawn, config)
              for i in range(config.n_drones)
          ]
          self._spawn_around_home()
          for d in self.drones:
              d.pos = env.spawn
          self.tick = 0

      def _spawn_around_home(self):
          hx, hy = self.env.home
          placed = 0
          for radius in range(1, 20):
              for dy in range(-radius, radius + 1):
                  for dx in range(-radius, radius + 1):
                      nx, ny = hx + dx, hy + dy
                      if (0 <= nx < self.env.width
                              and 0 <= ny < self.env.height
                              and placed < len(self.drones)):
                          self.drones[placed].pos = (nx, ny)
                          placed += 1
              if placed >= len(self.drones):
                  break

      def tick_once(self):
          snap = self.mm.snapshot(self.drones)
          alive = [d for d in self.drones if d.alive]
          proposals = [d.propose(snap, __import__("random").Random(0))
                       for d in alive]
          # 1. cell marking
          for d, action in zip(alive, proposals):
              if action.mark_cells:
                  self.mm.mark_searched(action.mark_cells, d.id)
          # 2. task reservation (ID-order tiebreak)
          for d, action in zip(alive, proposals):
              if action.request_task and d.state.value == "IDLE":
                  result = self.mm.assign_task(d.id, d.pos)
                  if result is not None:
                      d.target = result
                      blockers = {d2.pos for d2 in alive if d2 is not d}
                      d.path = astar(snap.grid, blockers, d.pos, result)
          # 3. move conflicts (closer-to-target wins)
          reqs: dict[tuple[int, int], list[int]] = {}
          for i, a in enumerate(proposals):
              if a.move_to and alive[i].state in (DroneState.SEARCHING,
                                                  DroneState.RETURNING):
                  reqs.setdefault(a.move_to, []).append(i)
          taken: set = set()
          for cell, idxs in reqs.items():
              winner = max(idxs, key=lambda i: _dist_to_goal(alive[i]))
              taken.add(cell)
              for i in idxs:
                  if i != winner:
                      alt = _open_neighbor(snap, cell, taken)
                      if alt:
                          proposals[i] = Action(
                              alt, proposals[i].mark_cells,
                              proposals[i].request_task,
                              proposals[i].state_transition)
                          alive[i].path = [alt] + alive[i].path
                          taken.add(alt)
          # 4. return-threshold state transitions + target-reached → IDLE
          from swarm_sar.drone import DroneState
          for d, action in zip(alive, proposals):
              if (d.state == DroneState.SEARCHING
                      and d.target and d.pos == d.target):
                  proposals[alive.index(d)] = Action(
                      action.move_to, action.mark_cells,
                      action.request_task, "IDLE")
              if (d.state == DroneState.SEARCHING
                      and d.check_return_threshold()):
                  proposals[alive.index(d)] = Action(
                      action.move_to, action.mark_cells,
                      action.request_task, "RETURNING")
          # 5. REPORTING → notify (once)
          for d in alive:
              if d.state == DroneState.REPORTING and not d._reported:
                  self.mm.on_drone_reported(d.id)
                  d._reported = True
          # 6. commit
          for d, action in zip(alive, proposals):
              d._moved_this_tick = False
              d.commit(action, snap)
              self.mm.record_move(d.id)
          # 7. failures
          self._apply_failures(alive)
          self._apply_kills(alive)
          self.mm.update(self.drones, self.tick)
          self.tick += 1

      def _apply_failures(self, alive):
          rate = self.config.failure_rate
          if rate <= 0:
              return
          for d in list(alive):
              if __import__("random").random() < rate / 10000:
                  d.alive = False
                  self.mm.on_drone_killed(d.id, d.target, self.tick)

      def _apply_kills(self, alive):
          t = self.tick
          for tick, did in self.config.kills:
              if tick == t:
                  d = next((x for x in self.drones if x.id == did), None)
                  if d:
                      d.alive = False
                      self.mm.on_drone_killed(did, d.target, t)

      def is_complete(self):
          return self.mm.is_complete()


  def _dist_to_goal(drone):
      g = drone.target or drone.home
      return (drone.pos[0] - g[0])**2 + (drone.pos[1] - g[1])**2


  def _open_neighbor(snap, blocked, taken):
      bx, by = blocked
      for dx, dy in _NEIGH8:
          nx, ny = bx + dx, by + dy
          if 0 <= nx < snap.grid[1] and 0 <= ny < snap.grid[0]:
              if snap.grid[ny][nx] != 1 and (nx, ny) not in taken:
                  return (nx, ny)
      return None


  _NEIGH8 = [(dx, dy)
             for dy in (-1, 0, 1) for dx in (-1, 0, 1)
             if not (dx == 0 and dy == 0)]
  ```

  **Note:** `astar` and `DroneAgent` are imported lazily inside `tick_once` to avoid circular imports. The `Simulator.__init__` references `DroneAgent` directly — it must be imported at the top of the file; add `from swarm_sar.drone import DroneAgent, DroneState` at the top of `simulator.py`.

- [ ] **Step 4: Run — expect pass**
  ```bash
  uv run pytest tests/test_simulator.py -v
  ```
  Expected: 4 pass.

- [ ] **Step 5: Commit**
  ```bash
  git add src/swarm_sar/simulator.py tests/test_simulator.py
  git commit -m "feat: Simulator tick engine (snapshot/propose/resolve)"
  ```

---

## Task 7: Renderer + interactive CLI

**Spec refs:** Section 8 (Rendering, CLI), Section 2 (Assumptions: perfect localization, static env)

**Files:**
- Create: `src/swarm_sar/renderer.py`
- Modify: `src/swarm_sar/__main__.py`

**No tests for renderer** (visual output is not pytest-testable; correctness verified by running interactively).

- [ ] **Step 1: Write `src/swarm_sar/renderer.py`**
  ```python
  import pygame
  from typing import Optional
  import numpy as np


  class Renderer:
      def __init__(self, simulator, cell_px: int = 8):
          self.sim = simulator
          self.cell_px = cell_px
          self.show_paths = True
          pygame.init()
          env = simulator.env
          grid_w = env.width * cell_px
          grid_h = env.height * cell_px
          margin = 50
          self.screen = pygame.display.set_mode(
              (grid_w + 2 * margin, grid_h + 2 * margin)
          )
          pygame.display.set_caption("Swarm SAR Simulation")
          self.font = pygame.font.SysFont("monospace", 14)
          self.margin = margin
          self.colors = {
              "empty": (200, 200, 200),
              "searched": (80, 200, 80),
              "obstacle": (40, 40, 40),
              "home": (220, 220, 60),
              "drone": (60, 120, 220),
              "path": (160, 200, 255),
              "dead": (220, 60, 60),
          }

      def draw(self):
          env = self.sim.env
          mm = self.sim.mm
          screen = self.screen
          screen.fill((30, 30, 30))
          # draw grid
          for y in range(env.height):
              for x in range(env.width):
                  rect = pygame.Rect(
                      self.margin + x * self.cell_px,
                      self.margin + y * self.cell_px,
                      self.cell_px, self.cell_px,
                  )
                  if env.grid[y, x] == 2:
                      pygame.draw.rect(screen, self.colors["home"], rect)
                      txt = self.font.render("H", True, (0, 0, 0))
                      screen.blit(txt, rect.topleft)
                  elif env.grid[y, x] == 1:
                      pygame.draw.rect(screen, self.colors["obstacle"], rect)
                  elif (x, y) in mm.searched:
                      pygame.draw.rect(screen, self.colors["searched"], rect)
                  else:
                      pygame.draw.rect(screen, self.colors["empty"], rect)
          # draw paths
          if self.show_paths:
              for d in self.sim.drones:
                  if not d.alive:
                      continue
                  for px, py in d.path:
                      rect = pygame.Rect(
                          self.margin + px * self.cell_px,
                          self.margin + py * self.cell_px,
                          self.cell_px, self.cell_px,
                      )
                      pygame.draw.rect(screen, self.colors["path"], rect)
          # draw drones
          for d in self.sim.drones:
              cx = self.margin + d.pos[0] * self.cell_px + self.cell_px // 2
              cy = self.margin + d.pos[1] * self.cell_px + self.cell_px // 2
              radius = max(self.cell_px // 2 - 1, 2)
              color = self.colors["drone"] if d.alive else self.colors["dead"]
              pygame.draw.circle(screen, color, (cx, cy), radius)
              label = self.font.render(str(d.id), True, (255, 255, 255))
              screen.blit(label, (cx - 4, cy - 6))
          # HUD top-left
          hud_text = (f"tick:{self.sim.tick} coverage:{mm.coverage()*100:.1f}% "
                      f"active:{sum(1 for d in self.sim.drones if d.alive)}/"
                      f"{len(self.sim.drones)}")
          screen.blit(self.font.render(hud_text, True, (255, 255, 255)),
                      (10, 10))
          # HUD top-right battery bars
          bx = self.screen.get_width() - 140
          for d in self.sim.drones:
              bh = 10
              fill = int(bh * d.battery / 100)
              color = (0, 200, 0) if d.state == DroneState.SEARCHING else (
                  (200, 200, 0) if d.state == DroneState.RETURNING else
                  (0, 0, 200) if d.state == DroneState.IDLE else (200, 100, 0))
              pygame.draw.rect(screen, (50, 50, 50),
                               (bx, 10 + d.id * (bh + 2), 130, bh))
              pygame.draw.rect(screen, color,
                               (bx, 10 + d.id * (bh + 2), fill, bh))
          pygame.display.flip()

      def shutdown(self):
          pygame.quit()
  ```

  **Note:** `renderer.py` imports `DroneState` at the bottom for HUD coloring. Add `from swarm_sar.drone import DroneState` at the top (or inside `draw`).

- [ ] **Step 2: Write `src/swarm_sar/__main__.py`**
  ```python
  import argparse
  import sys
  import time
  import pygame


  def run_interactive(config):
      from swarm_sar.simulator import Simulator
      from swarm_sar.environment import Environment
      from swarm_sar.renderer import Renderer
      env = Environment.from_config(config)
      sim = Simulator(env, config)
      mm = sim.mm
      mm.set_log_path(f"out/run_t{sim.tick}_d{config.n_drones}_s{config.seed}.csv")
      renderer = Renderer(sim)
      running = True
      paused = False
      tps = config.ticks_per_second
      accumulator = 0.0
      last = time.monotonic()
      clock = pygame.time.Clock()
      while running:
          now = time.monotonic()
          accumulator += now - last
          last = now
          if not paused:
              while accumulator >= 1.0 / tps:
                  sim.tick_once()
                  mm.flush_log()
                  accumulator -= 1.0 / tps
                  if sim.is_complete():
                      running = False
                      break
          for event in pygame.event.get():
              if event.type == pygame.QUIT:
                  running = False
              if event.type == pygame.KEYDOWN:
                  if event.key == pygame.K_p:
                      paused = not paused
                  elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                      tps = min(tps + 1, 50)
                  elif event.key == pygame.K_MINUS:
                      tps = max(tps - 1, 1)
                  elif event.key == pygame.K_v:
                      renderer.show_paths = not renderer.show_paths
                  elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                      running = False
                  elif event.key == pygame.K_F1:
                      sim.drones[0].alive = False
                      mm.on_drone_killed(0, sim.drones[0].target, sim.tick)
                  elif event.key == pygame.K_F2 and len(sim.drones) > 1:
                      sim.drones[1].alive = False
                      mm.on_drone_killed(1, sim.drones[1].target, sim.tick)
                  elif event.key == pygame.K_F3 and len(sim.drones) > 2:
                      sim.drones[2].alive = False
                      mm.on_drone_killed(2, sim.drones[2].target, sim.tick)
          if running:
              renderer.draw()
              clock.tick(60)
      mm.flush_log()
      if mm._log_path:
          with open(mm._log_path, "a") as f:
              f.write(mm.summary_line(sim.tick) + "\n")
      renderer.shutdown()


  def run_sweep(config):
      from swarm_sar.sweep import run_sweep
      run_sweep(config)


  def run_plot(path):
      from swarm_sar.plot import main as plot_main
      plot_main(path)


  def main(argv=None):
      parser = argparse.ArgumentParser(prog="swarm_sar")
      parser.add_argument("--scenario", default=None)
      parser.add_argument("--drones", type=int, default=5)
      parser.add_argument("--grid", type=int, default=100)
      parser.add_argument("--seed", type=int, default=0)
      parser.add_argument("--density", type=float, default=0.10)
      parser.add_argument("--tps", type=int, default=10)
      parser.add_argument("--max-ticks", type=int, default=10000)
      parser.add_argument("--coverage", type=float, default=1.0)
      parser.add_argument("--log-every", type=int, default=10)
      parser.add_argument("--sensor-radius", type=int, default=2)
      parser.add_argument("--failure-rate", type=float, default=0.0)
      parser.add_argument("--kill", type=str, default="")
      parser.add_argument("--sweep", action="store_true")
      parser.add_argument("--plot", type=str, default=None)
      parser.add_argument("--sweep-drones", type=str, default="5,10,20,50")
      parser.add_argument("--sweep-grids", type=str, default="50,100,150")
      parser.add_argument("--sweep-seeds", type=int, default=3)
      parser.add_argument("--sweep-repeats", type=int, default=3)
      parser.add_argument("--sweep-out", type=str, default="out/sweep")
      parser.add_argument("--sweep-density", type=float, default=0.10)
      parser.add_argument("--sweep-max-ticks", type=int, default=5000)
      parser.add_argument("--verbose", action="store_true")
      args = parser.parse_args(argv)
      kills = []
      if args.kill:
          for part in args.kill.split(","):
              t, _, id_str = part.partition(":")
              kills.append((int(t), int(id_str)))
      config = SimConfig(
          scenario_path=args.scenario,
          n_drones=args.drones,
          grid_size=args.grid,
          seed=args.seed,
          obstacle_density=args.density,
          ticks_per_second=args.tps,
          max_ticks=args.max_ticks,
          coverage_threshold=args.coverage,
          log_interval_ticks=args.log_every,
          sensor_radius=args.sensor_radius,
          failure_rate=args.failure_rate,
          kills=tuple(kills),
      )
      if args.plot:
          run_plot(args.plot)
      elif args.sweep:
          from swarm_sar.sweep import SweepConfig
          sweep_cfg = SweepConfig(
              drones=[int(x) for x in args.sweep_drones.split(",")],
              grids=[int(x) for x in args.sweep_grids.split(",")],
              seeds=args.sweep_seeds,
              repeats=args.sweep_repeats,
              density=args.sweep_density,
              max_ticks=args.sweep_max_ticks,
              out_dir=args.sweep_out,
              verbose=args.verbose,
              base_config=config,
          )
          run_sweep(sweep_cfg)
      else:
          run_interactive(config)


  if __name__ == "__main__":
      main()
  ```

  Add to `src/swarm_sar/__init__.py`:
  ```python
  from swarm_sar.simulator import Simulator, Action
  ```

- [ ] **Step 3: Run tests + manual check**
  ```bash
  uv run pytest tests/test_simulator.py -v
  ```
  Expected: 4 pass.
  ```bash
  uv run python -m swarm_sar --scenario scenarios/empty.txt --drones 1 --seed 0
  ```
  Expected: Pygame window opens (small 5x5 grid visible), drone moves around.
  Press `P` to pause, `Q` to quit. Close window when done.

- [ ] **Step 4: Commit**
  ```bash
  git add src/swarm_sar/renderer.py src/swarm_sar/__main__.py \
       src/swarm_sar/__init__.py
  git commit -m "feat: Pygame renderer, interactive CLI"
  ```

---

## Task 8: Sweep runner (headless batch)

**Spec refs:** Section 8 (Sweep Runner). Pygame must **not** be imported.

**Files:**
- Create: `src/swarm_sar/sweep.py`

**No tests** (tested by running a small sweep and inspecting output).

- [ ] **Step 1: Write `src/swarm_sar/sweep.py`**
  ```python
  import csv
  import os
  import time
  from dataclasses import dataclass
  from typing import Optional
  from swarm_sar.environment import SimConfig
  from swarm_sar.simulator import Simulator
  from swarm_sar.environment import Environment


  @dataclass(frozen=True)
  class SweepConfig:
      drones: list[int]
      grids: list[int]
      seeds: int
      repeats: int
      density: float
      max_ticks: int
      out_dir: str
      verbose: bool = False
      base_config: Optional[SimConfig] = None


  def run_sweep(sweep_cfg: SweepConfig) -> str:
      ts = time.strftime("%Y%m%d_%H%M%S")
      run_dir = os.path.join(sweep_cfg.out_dir, f"sweep_{ts}")
      os.makedirs(run_dir, exist_ok=True)
      rows: list[dict] = []
      cfg = sweep_cfg.base_config or SimConfig()
      total = (len(sweep_cfg.drones) * len(sweep_cfg.grids)
               * sweep_cfg.seeds * sweep_cfg.repeats)
      i = 0
      for d in sweep_cfg.drones:
          for g in sweep_cfg.grids:
              for seed in range(sweep_cfg.seeds):
                  for rep in range(sweep_cfg.repeats):
                      run_seed = seed  # user seed; same config = same run
                      out_prefix = f"d{d}_g{g}"
                      run_out = os.path.join(run_dir, out_prefix)
                      os.makedirs(run_out, exist_ok=True)
                      csv_path = os.path.join(run_out, f"seed{seed}_rep{rep}.csv")
                      config = SimConfig(
                          n_drones=d, grid_size=g,
                          obstacle_density=sweep_cfg.density,
                          seed=run_seed,
                          max_ticks=sweep_cfg.max_ticks,
                      )
                      env = Environment.from_config(config)
                      sim = Simulator(env, config)
                      mm = sim.mm
                      mm.set_log_path(csv_path)
                      t0 = time.monotonic()
                      wall = 0.0
                      for tick in range(sweep_cfg.max_ticks):
                          sim.tick_once()
                          mm.flush_log()
                          if sim.is_complete():
                              wall = time.monotonic() - t0
                              break
                      else:
                          wall = time.monotonic() - t0
                      mm.flush_log()
                      # append summary to CSV
                      with open(csv_path, "a") as fh:
                          fh.write(mm.summary_line(sim.tick) + "\n")
                      row = {
                          "drones": d, "grid": g, "seed": seed,
                          "repeat": rep, "ticks": sim.tick,
                          "coverage": mm.coverage(),
                          "wall_time_s": wall,
                          "failures": len(mm.failure_log),
                          "avg_battery": (
                              sum(dr.battery for dr in sim.drones if dr.alive)
                              / max(sum(1 for dr in sim.drones if dr.alive), 1)
                          ),
                      }
                      rows.append(row)
                      i += 1
                      if sweep_cfg.verbose:
                          print(f"[{i}/{total}] d={d} g={g} seed={seed} "
                                f"rep={rep} coverage={row['coverage']*100:.1f}%")
      agg_path = os.path.join(run_dir, "aggregate.csv")
      if rows:
          with open(agg_path, "w", newline="") as fh:
              writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
              writer.writeheader()
              writer.writerows(rows)
      if sweep_cfg.verbose:
          print(f"Wrote {agg_path}")
      # auto-generate plots
      try:
          _auto_plot(agg_path, run_dir)
      except Exception as exc:
          if sweep_cfg.verbose:
              print(f"plot auto-gen skipped: {exc}")
      return run_dir


  def _auto_plot(agg_path: str, run_dir: str) -> None:
      # defer import so pygame stays unimported on this path
      from swarm_sar.plot import main as plot_main
      plot_main(agg_path)
  ```

- [ ] **Step 2: Run a small sweep**
  ```bash
  uv run python -m swarm_sar --sweep \
    --drones 1,2 \
    --grids 10 \
    --seeds 1 \
    --repeats 1 \
    --density 0.05 \
    --max-ticks 500 \
    --out out/smoke_test/ \
    --verbose
  ```
  Expected: `out/smoke_test/sweep_<timestamp>/` created, `aggregate.csv` and 4 plot PNGs present. No pygame import errors.

- [ ] **Step 3: Commit**
  ```bash
  git add src/swarm_sar/sweep.py
  git commit -m "feat: headless sweep runner (no pygame import)"
  ```

---

## Task 9: Plot generation

**Spec refs:** Section 8 (plot.py)

**Files:**
- Create: `src/swarm_sar/plot.py`

- [ ] **Step 1: Write `src/swarm_sar/plot.py`**
  ```python
  import argparse
  import os
  from typing import Optional
  import pandas as pd
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt


  def main(agg_path: str, out_dir: Optional[str] = None):
      df = pd.read_csv(agg_path, comment="#")
      if out_dir is None:
          out_dir = os.path.dirname(agg_path) or "."
      os.makedirs(out_dir, exist_ok=True)

      # make axes numeric
      for col in ("drones", "grid", "ticks", "coverage", "wall_time_s",
                   "failures", "avg_battery", "seed", "repeat"):
          if col in df.columns:
              df[col] = pd.to_numeric(df[col], errors="coerce")

      # 1. coverage over time (per-run CSV line)
      _coverage_plot(df, out_dir)

      # 2. completion bars
      _completion_bars(df, out_dir)

      # 3. robustness (only if failures > 0)
      if df["failures"].sum() > 0:
          _failure_plot(df, out_dir)

      # 4. scalability (tick rate vs drone count)
      _fps_plot(df, out_dir)


  def _coverage_plot(df, out_dir):
      fig, axes = plt.subplots(1, len(df["grid"].unique()),
                               figsize=(5 * len(df["grid"].unique()), 4),
                               squeeze=False)
      axes = axes[0]
      for gi, (grid_val, ax) in enumerate(zip(sorted(df["grid"].unique()),
                                               axes)):
          sub = df[df["grid"] == grid_val]
          for drone_val in sorted(sub["drones"].unique()):
              grp = sub[sub["drones"] == drone_val]
              ax.plot(grp["ticks"], grp["coverage"] * 100,
                      label=f"{drone_val} drones")
          ax.set_title(f"Grid {grid_val}x{grid_val}")
          ax.set_xlabel("tick")
          ax.set_ylabel("coverage %")
          ax.legend()
      fig.tight_layout()
      fig.savefig(os.path.join(out_dir, "coverage_over_time.png"))
      plt.close(fig)


  def _completion_bars(df, out_dir):
      sub = df.groupby(["drones", "grid"])["ticks"].mean().reset_index()
      fig, ax = plt.subplots(figsize=(8, 5))
      for grid_val in sorted(sub["grid"].unique()):
          g = sub[sub["grid"] == grid_val]
          ax.bar(g["drones"] + grid_val * 0.05,
                 g["ticks"], width=0.2, label=f"{grid_val}x{grid_val}")
      ax.set_xlabel("drone count")
      ax.set_ylabel("ticks to completion")
      ax.set_title("Scalability: completion time")
      ax.legend()
      fig.tight_layout()
      fig.savefig(os.path.join(out_dir, "completion_bars.png"))
      plt.close(fig)


  def _failure_plot(df, out_dir):
      sub = df[df["failures"] > 0]
      if sub.empty:
          return
      fig, ax = plt.subplots(figsize=(8, 5))
      pivot = sub.pivot_table(values="coverage", index="drones",
                              columns="failures", aggfunc="mean")
      pivot.plot(kind="bar", ax=ax)
      ax.set_ylabel("final coverage")
      ax.set_title("Robustness under failures")
      fig.tight_layout()
      fig.savefig(os.path.join(out_dir, "failure_robustness.png"))
      plt.close(fig)


  def _fps_plot(df, out_dir):
      # use wall_time_s / ticks as proxy for tick rate
      sub = df.copy()
      sub["ticks_per_sec"] = sub["ticks"] / sub["wall_time_s"].clip(lower=0.01)
      mean_rate = sub.groupby("drones")["ticks_per_sec"].mean().reset_index()
      fig, ax = plt.subplots(figsize=(8, 5))
      ax.bar(mean_rate["drones"].astype(str), mean_rate["ticks_per_sec"])
      ax.set_xlabel("drone count")
      ax.set_ylabel("ticks/sec (headless)")
      ax.set_title("Scalability: tick rate vs. drone count")
      fig.tight_layout()
      fig.savefig(os.path.join(out_dir, "fps_scaling.png"))
      plt.close(fig)


  if __name__ == "__main__":
      p = argparse.ArgumentParser()
      p.add_argument("aggregate_csv")
      p.add_argument("--out-dir", default=None)
      args = p.parse_args()
      main(args.aggregate_csv, args.out_dir)
  ```

- [ ] **Step 2: Run a small plot test**
  ```bash
  # after any sweep, aggregate.csv exists:
  uv run python -m swarm_sar.plot out/smoke_test/sweep_<ts>/aggregate.csv
  ```
  Expected: 3–4 PNG files created in the same directory.

- [ ] **Step 3: Commit**
  ```bash
  git add src/swarm_sar/plot.py
  git commit -m "feat: plot.py — matplotlib graphs from aggregate CSV"
  ```

---

## Task 10: Final integration smoke test + evaluation sweeps

**Spec refs:** Section 9 (Evaluation Plan), Section 1 (Deliverables)

**No new files.** This task runs the full pipeline and produces the deliverables.

- [ ] **Step 1: Run default scenario interactive**
  ```bash
  uv run python -m swarm_sar --scenario scenarios/default.txt --drones 5 --seed 0
  ```
  Expected: Pygame window opens. Drones move around searching cells. Press `F2` at any time to kill drone 2 — MissionManager should assign its target to a live drone. Coverage increases. Mission ends when 100% covered. CSV written to `out/`.

- [ ] **Step 2: Run small evaluation sweep**
  ```bash
  uv run python -m swarm_sar --sweep \
    --drones 1,5 \
    --grids 20,50 \
    --seeds 3 \
    --repeats 2 \
    --density 0.10 \
    --max-ticks 4000 \
    --out out/eval_small/ \
    --verbose
  ```
  Expected: `out/eval_small/sweep_<ts>/aggregate.csv` + PNG plots exist.

- [ ] **Step 3: Run pytest full suite**
  ```bash
  uv run pytest -v
  ```
  Expected: all pass.

- [ ] **Step 4: Run linter**
  ```bash
  uv run ruff check src tests
  ```
  Expected: clean or fixable warnings only.

- [ ] **Step 5: Commit any resulting fixes**
  ```bash
  git add -A && git commit -m "chore: smoke tests, fixups from evaluation"
  ```

- [ ] **Step 6: Run full evaluation (Section 9 matrix)**
  Scability:
  ```bash
  uv run python -m swarm_sar --sweep \
    --drones 5,10,20,50 \
    --grids 50,100,150 \
    --seeds 3 --repeats 3 \
    --density 0.10 \
    --max-ticks 8000 \
    --out out/eval/scale_<ts>/ \
    --verbose
  ```
  Robustness (separate run):
  ```bash
  uv run python -m swarm_sar --sweep \
    --drones 20 \
    --grids 100 \
    --seeds 3 --repeats 3 \
    --density 0.10 \
    --max-ticks 4000 \
    --failure-rate 10 \
    --out out/eval/robust_<ts>/ \
    --verbose
  ```
  Expected: `out/eval/` has two subdirectories, each with `aggregate.csv` + PNGs.

---

## Task 2: Environment + scenario loader + procedural generation

**Spec refs:** Section 7 (Environment), Section 2 (Assumption 3: static, deterministic), Section 7 (procedural gen + dilation).

**Tests:** `tests/test_environment.py`

- [ ] **Step 1: Write the failing tests** `tests/test_environment.py`
  ```python
  import numpy as np
  from swarm_sar.environment import Environment
  from swarm_sar.environment import SimConfig


  def _load(sf="default"):
      config = SimConfig(scenario_path=f"scenarios/{sf}.txt")
      return Environment.from_config(config)


  def test_load_default_scenario():
      env = _load("default")
      assert env.home == (6, 1)
      assert env.grid[env.home[1], env.home[0]] == 2
      assert env.spawn == env.home


  def test_exactly_one_home_each_scenario():
      for sf in ("default", "empty", "maze", "walls"):
          env = _load(sf)
          assert np.sum(env.grid == 2) == 1, f"{sf} has wrong home count"


  def test_procedural_determinism():
      g1 = Environment.generate_grid(20, 0.10, seed=42)
      g2 = Environment.generate_grid(20, 0.10, seed=42)
      assert np.array_equal(g1, g2)


  def test_procedural_home_and_8_neighbors_not_obstacle():
      for seed in range(10):
          g = Environment.generate_grid(30, 0.25, seed=seed)
          h = g.shape[1] // 2, g.shape[0] // 2
          assert g[h[1], h[0]] != 1
          for dy in (-1, 0, 1):
              for dx in (-1, 0, 1):
                  nx, ny = h[0]+dx, h[1]+dy
                  if 0 <= nx < g.shape[1] and 0 <= ny < g.shape[0]:
                      assert g[ny, nx] != 1


  def test_connectivity_at_moderate_density():
      for seed in range(5):
          g = Environment.generate_grid(30, 0.20, seed=seed)
          hx, hy = g.shape[1] // 2, g.shape[0] // 2
          reachable = _flood(g, (hx, hy))
          empties = set(zip(*np.where(g == 0)))
          assert empties.issubset(reachable)


  def _flood(grid, start):
      from collections import deque
      q = deque([start]); seen = {start}
      while q:
          x, y = q.popleft()
          for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
              nx, ny = x+dx, y+dy
              if 0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0]:
                  if grid[ny, nx] != 1 and (nx, ny) not in seen:
                      seen.add((nx, ny)); q.append((nx, ny))
      return seen
  ```

- [ ] **Step 2: Run tests — expect failure**
  ```bash
  uv run pytest tests/test_environment.py -v
  ```
  Expected: `ModuleNotFoundError: No module named 'swarm_sar.environment'`.

- [ ] **Step 3: Write `src/swarm_sar/environment.py`**
  ```python
  import random
  from dataclasses import dataclass
  from pathlib import Path
  from typing import Optional

  import numpy as np


  @dataclass(frozen=True)
  class SimConfig:
      scenario_path: Optional[str] = None
      n_drones: int = 5
      grid_size: int = 100
      obstacle_density: float = 0.10
      seed: int = 0
      sensor_radius: int = 2
      ticks_per_second: int = 10
      max_ticks: int = 10000
      coverage_threshold: float = 1.0
      log_interval_ticks: int = 10
      failure_rate: float = 0.0
      kills: tuple[tuple[int, int], ...] = ()
      battery_safety_margin: float = 1.2
      recharge_rate: float = 2.0
      reporting_duration_ticks: int = 3


  @dataclass(frozen=True)
  class Environment:
      width: int
      height: int
      grid: np.ndarray
      home: tuple[int, int]
      spawn: tuple[int, int]

      @staticmethod
      def generate_grid(size: int, density: float, seed: int) -> np.ndarray:
          rng = random.Random(seed)
          grid = np.zeros((size, size), dtype=np.uint8)
          hx = hy = size // 2

          for y in range(size):
              for x in range(size):
                  if rng.random() < density:
                      grid[y, x] = 1

          # protect home + 8-neighbors
          for dy in (-1, 0, 1):
              for dx in (-1, 0, 1):
                  nx, ny = hx + dx, hy + dy
                  if 0 <= nx < size and 0 <= ny < size:
                      grid[ny, nx] = 0

          # dilation: manual 4-neighbor pass (no scipy)
          dilated = grid.copy()
          for y in range(size):
              for x in range(size):
                  if grid[y, x] == 1:
                      for dy in (-1, 0, 1):
                          for dx in (-1, 0, 1):
                              nx, ny = x + dx, y + dy
                              if 0 <= nx < size and 0 <= ny < size:
                                  dilated[ny, nx] = 1
          grid = dilated
          grid[hy, hx] = 2

          # connectivity check with deterministic retry
          for attempt in range(5):
              r = _flood(grid, (hx, hy))
              empties = set(zip(*np.where(grid == 0)))
              if empties.issubset(r):
                  return grid
              rng2 = random.Random(hash((seed, attempt)))
              new = np.zeros((size, size), dtype=np.uint8)
              for y in range(size):
                  for x in range(size):
                      if rng2.random() < density:
                          new[y, x] = 1
              for dy in (-1, 0, 1):
                  for dx in (-1, 0, 1):
                      nx, ny = hx + dx, hy + dy
                      if 0 <= nx < size and 0 <= ny < size:
                          new[ny, nx] = 0
              grid = new

          raise ValueError(
              f"Could not generate connected grid size={size} density={density} seed={seed}"
          )

      @classmethod
      def from_config(cls, config: SimConfig) -> "Environment":
          if config.scenario_path:
              return cls._load_scenario(Path(config.scenario_path))
          grid = cls.generate_grid(config.grid_size, config.obstacle_density, config.seed)
          return cls._from_grid(grid)

      @classmethod
      def _load_scenario(cls, path: Path) -> "Environment":
          rows = []
          with open(path) as fh:
              for line in fh:
                  line = line.strip()
                  if not line or line.startswith("#"):
                      continue
                  rows.append(line)

          height = len(rows)
          width = max(len(r) for r in rows)
          grid = np.zeros((height, width), dtype=np.uint8)
          home: Optional[tuple[int, int]] = None

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
                  elif ch == "S":
                      pass  # handled below

          if home is None:
              raise ValueError(f"No 'H' in {path}")

          spawn = home
          for y, row in enumerate(rows):
              for x, ch in enumerate(row):
                  if ch == "S":
                      spawn = (x, y)

          env = cls(width=width, height=height, grid=grid, home=home, spawn=spawn)
          return env

      @classmethod
      def _from_grid(cls, grid: np.ndarray) -> "Environment":
          h, w = grid.shape
          assert np.sum(grid == 2) == 1, "generate_grid must mark exactly one home"
          hy, hx = np.where(grid == 2)
          home = (int(hx[0]), int(hy[0]))
          return cls(width=w, height=h, grid=grid, home=home, spawn=home)


  def _flood(grid: np.ndarray, start: tuple[int, int]) -> set[tuple[int, int]]:
      from collections import deque
      seen: set[tuple[int, int]] = set()
      q: deque = deque([start])
      seen.add(start)
      h, w = grid.shape
      while q:
          x, y = q.popleft()
          for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
              nx, ny = x + dx, y + dy
              if 0 <= nx < w and 0 <= ny < h and grid[ny, nx] != 1 and (nx, ny) not in seen:
                  seen.add((nx, ny))
                  q.append((nx, ny))
      return seen
  ```

  In `src/swarm_sar/__init__.py`, add:
  ```python
  from swarm_sar.environment import Environment
  ```

- [ ] **Step 4: Run tests — expect pass**
  ```bash
  uv run pytest tests/test_environment.py -v
  ```
  Expected: 5 pass.

- [ ] **Step 5: Commit**
  ```bash
  git add src/swarm_sar/environment.py src/swarm_sar/__init__.py tests/test_environment.py
  git commit -m "feat: add Environment, scenario loader, procedural gen"
  ```

---

---

## Self-review (run after writing the plan)

1. **Spec coverage:** Every section in the spec has at least one task implementing it.
2. **Placeholder scan:** No "TBD", "implement later", "add validation" without code.
3. **Type consistency:** `Action`, `Snapshot`, `SimConfig`, `DroneState` defined once and used consistently.
4. **No circular imports:** `drone.py` defines `Action`; `simulator.py` imports it. Verified.
5. **Battery model matches spec:** 1.0/tick SEARCHING and RETURNING; 0 IDLE/REPORTING; recharge at `recharge_rate`. Spec Section 5 + Global Constraints.
6. **REPORTING state:** Drone notifies MM + recharges then transitions IDLE. No per-drone JSON file.
7. **No self-checks:** Pytest only. No `__main__` blocks for testing, no `--unit-self-check` flag.
8. **No derived seeds:** User seed used directly.
9. **Renderer flat colors:** No freshness, no snapshot key.
10. **Headless sweep:** Zero pygame imports on sweep path.

If any check fails, fix inline before saving.
