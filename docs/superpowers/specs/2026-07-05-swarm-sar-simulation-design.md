# Hybrid Multi-Agent Drone-Based SAR Simulation — Design

**Status:** Approved via brainstorming session 2026-07-05
**Author:** Aryan (with opencode brainstorming)
**Stack:** Python 3.11, Pygame, NumPy, pandas, matplotlib — managed by uv
**Out of scope:** Physical drones, hardware, capstone writeup/PDF (graphs produced; writeup is the student's job)

---

## 1. Overview & Scope

A simulation-based multi-agent Search and Rescue (SAR) system. Autonomous drones search a 2D obstacle grid under a **hybrid coordination model**: a centralized `MissionManager` allocates search tasks (pull-based, greedy nearest); each `DroneAgent` autonomously plans paths (A\*), monitors battery, avoids collisions, executes a finite state machine, and reports. The system is evaluated under scaling (drone count, grid size) and stress (drone failures).

### Deliverable boundary

- `uv run python -m swarm_sar` → interactive Pygame window with live stats, pause/resume, manual failure injection (keys).
- `uv run python -m swarm_sar --sweep ...` → headless batch run; dumps one CSV per `(drones × grid × seed × repeat)` plus `aggregate.csv`, then auto-invokes `plot.py` on the aggregate to write comparison plots to `out/plots/`.
- `uv run python -m swarm_sar --plot <aggregate.csv>` → regenerate plots from any sweep's aggregate file.
- **Out of scope:** the capstone writeup / PDF. The project produces graphs; the report is the student's deliverable.

### Scope cuts (YAGNI)

The following items were considered and deliberately excluded (post-review, second pass):

- Variable per-drone speed (uniform `1 cell/tick`).
- Multiple home bases (exactly one).
- Per-drone sensor radii (one global tunable per run).
- Per-event CSV logging (sampled snapshots only).
- `#EVENT` lines / `--log-events` flag (cut — CSV metrics are enough).
- "Run forever" mode (coverage-threshold termination).
- scipy dependency (cut for `kdtree`; cut for `binary_dilation` — replaced with manual NumPy slicing).
- `tqdm` dependency for progress bars (5-line stdlib replacement).
- Dependency extras / optional-install split (pygame stays a flat dep; sweep simply never imports it).
- Per-drone JSON report file (`out/reports_<run_id>.json`) — cut. MissionManager already tracks all data; REPORTING just notifies the MM and recharges. No separate report artifact.
- Renderer snapshot PNG export (`S` key) — cut. Cute feature, not necessary.
- Renderer freshness fading / heatmap on searched cells — cut. Flat `searched = green`, `unsearched = gray`. Simpler draws, no cosmetic state.
- Special "hover battery drain" rule — cut. Battery drains uniformly while in SEARCHING (regardless of movement) and in RETURNING. Replaces the two-rule version with one rule; avoids the infinite-holding-pattern stall bug.
- Module-level `__main__` self-checks and the `--unit-self-check` flag — cut. Pytest is the only gate. (Saves ~4 modules' worth of duplication.)
- Derived-seed calculation (`seed + drones*1000 + ...`) — cut. Use the user-provided seed directly. Different configs already produce different runs; mutating the seed adds nothing.

---

## 2. Simulation Assumptions

The following simplifications are deliberate and define the scope of what this project models. Each is a place where a more realistic system would introduce complexity that does not serve the capstone's AI/coordination focus.

- **Identical drones.** All drones have the same movement speed (`1 cell/tick`), battery capacity (`100`), sensor radius, and recharge rate. No per-drone heterogeneity.
- **Instantaneous, reliable communication.** Drone↔MissionManager messages (task requests, completion reports, failure notifications) arrive in-tick with zero latency and zero loss. No comms-delay modeling, no packet drop, no retry logic.
- **Static environment.** Obstacles do not move, appear, or disappear during a run. The grid is fixed at `Environment` construction.
- **Perfect localization.** Drones know their own `(x, y)` and the home base position exactly. No GPS drift, no pose estimation error.
- **Perfect sensing within footprint.** Any searchable cell inside the drone's Chebyshev sensor radius is detected with probability 1 when the drone is within range of it. No false negatives, no false positives, no sensor noise. Sensing is local to the drone's position, not global — a cell is marked searched only when the drone physically occupies a position within range of it.
- **Exclusive cell occupancy at tick end.** At most one drone may occupy a grid cell at the end of a tick. The `resolve` step enforces this via the closer-to-target rule; the simulator never allows two drones to co-locate in committed state. The home base cell is exempt from this rule: multiple drones may occupy home simultaneously while in REPORTING, since that state is stationary and occurs only at base.
- **No wind, no drift, no energy recovery.** Battery only depletes (motion/active-state drain) or recharges (at base). No environmental energy harvesting, no movement assistance from wind, no terrain-dependent motion cost.

### What these assumptions buy

They let the simulation focus purely on **coordination, path planning, task allocation, and fault tolerance** — the actual capstone-gradeable AI aspects. Removing any one of them would add either a sensor-modeling subsystem (noise, drift), a networking subsystem (latency, drop), or a physics subsystem (terrain cost, wind) — none of which serves the stated objectives.

### Reference points for the writeup

If asked "why no comm delays?": assumption 2 — the project studies *coordination algorithms*, not network reliability. Adding latency would conflate algorithm-inherent failure modes with network-induced ones.
If asked "why perfect sensors?": assumption 5 — sensor noise introduces a *coverage-confidence* problem (probabilistic coverage), which is a different project. This project studies deterministic coverage.
If asked "why identical drones?": assumption 1 — heterogeneous fleets are a known SAR research direction but require a more complex task-allocation strategy (capability-aware assignment). Out of scope for a single capstone.

---

## 3. Project Layout & Tooling

```
swarm-sar-coordination/
├── pyproject.toml              # uv-managed; deps + [dependency-groups.dev]
├── .python-version             # 3.11
├── uv.lock                     # committed for reproducibility
├── mise.toml                   # REWIRED from Godot → uv commands
├── .gitignore                  # REWIRED from Godot → Python/uv
├── README.md                   # how to run
├── src/swarm_sar/
│   ├── __init__.py
│   ├── __main__.py             # argparse; dispatches interactive / sweep / plot
│   ├── simulator.py            # tick engine: snapshot / propose / resolve
│   ├── environment.py          # Grid, obstacles, home base, scenario loader
│   ├── mission_manager.py      # global state, task allocation, stats, logging
│   ├── drone.py                # DroneAgent + FSM
│   ├── astar.py                # 8-connected A*, octile heuristic
│   ├── renderer.py             # Pygame drawing (no AI logic)
│   ├── sweep.py                # headless batch runner
│   └── plot.py                 # matplotlib graph generation
├── tests/
│   ├── test_environment.py
│   ├── test_astar.py
│   ├── test_fsm.py
│   ├── test_allocation.py
│   ├── test_coverage.py
│   ├── test_battery.py
│   └── test_simulator.py       # tick-level integration
├── scenarios/
│   ├── default.txt
│   ├── empty.txt
│   ├── maze.txt
│   └── walls.txt
└── out/                        # gitignored; CSVs + plots land here
```

### uv discipline (strict)

**No `pip` anywhere** — not in scripts, README, or docstrings. Adding a dependency = edit `pyproject.toml` + `uv sync`.

- **Project init:** `uv init --package swarm_sar --python 3.11` produces `pyproject.toml`, `.python-version`, `src/swarm_sar/` skeleton, `uv.lock`.
- **Runtime deps:** `pygame>=2.5`, `numpy>=1.26`, `matplotlib>=3.8`, `pandas>=2.2`.
- **Dev deps:** under `[dependency-groups.dev]`: `pytest>=8.0`, `ruff>=0.6`.
- **Lockfile:** `uv.lock` committed; `uv sync --frozen` in CI.
- **Every command via `uv run`**: `uv run python -m swarm_sar`, `uv run pytest`, `uv run ruff check src tests`. Never `python -m ...` directly.
- **`mise.toml`** delegates to uv:
  ```toml
  [tools]
  uv = "latest"

  [tasks]
  dev   = "uv run python -m swarm_sar"
  test  = "uv run pytest"
  sweep = "uv run python -m swarm_sar --sweep"
  lint  = "uv run ruff check src tests"
  ```
- **`.gitignore`** cleaned of Godot section; keeps `.venv/`, `__pycache__/`, `out/`, `.ruff_cache/`. `uv.lock` is **not** ignored (committed for reproducibility).

### Dependencies (locked)

`pygame` (render), `numpy` (grid as ndarray + vectorized sensor footprint), `matplotlib` (plots), `pandas` (CSV pre-aggregation in `plot.py`). Dev: `pytest`, `ruff`. **No scipy, no tqdm, no doctrine, no plugin system.**

---

## 4. Simulation Engine & Tick Model

### Core loop

The `Simulator` class owns the world and drives one tick at a time. Rendering is decoupled: Pygame draws "the state after the last tick" at the render frame rate, independent of tick cadence.

```python
class Simulator:
    def __init__(self, config: SimConfig):
        self.env = Environment.from_config(config)  # uses scenario_path or procedural
        self.mm = MissionManager(self.env, config)
        self.drones = [DroneAgent(i, self.env.home, config) for i in range(config.n_drones)]
        self.tick = 0
        self.logger = MissionLogger(config.log_path, config.log_interval_ticks)
        self.rng = random.Random(config.seed)

    def tick_once(self):
        snap = self.mm.snapshot(self.drones)              # read-only world view
        alive = [d for d in self.drones if d.alive]
        proposals = [d.propose(snap, self.rng) for d in alive]
        actions = self.resolve(proposals, snap)          # atomic collision + cell-mark
        for d, a in zip(alive, actions):
            d.commit(a, snap)
        self.mm.update(self.drones, self.tick)
        self.logger.sample_if_due(self.tick, self.drones, self.mm)
        self.tick += 1
```

### Tick cadence & interactive loop

Configurable `ticks_per_second` (default 10). The main loop decouples ticks from render frames via an accumulator:

```python
accumulator = 0.0
last = time.monotonic()
while running:
    now = time.monotonic()
    accumulator += now - last
    last = now
    while accumulator >= 1.0 / tps:
        sim.tick_once()
        accumulator -= 1.0 / tps
        if sim.mm.is_complete():
            running = False
            break
    renderer.draw(sim)
    pygame.display.flip()
```

- `P` pause/resume (gates the accumulator).
- `+` / `-` speed up / slow down (adjust `tps` live, clamped to `[1, 50]`).

### Update order within a single tick (the canonical reference)

```
Simulation Tick
   │
   ▼
MissionManager.snapshot(drones)   ── build read-only world view
   │
   ▼
DroneAgent.propose(snap, rng)     ── each live drone returns one Action
   │
   ▼
Simulator.resolve(proposals, snap)
   ├─ 1. Cell marking (sensor footprint) → MissionManager.searched
   ├─ 2. Move conflicts (closer-to-target wins)
   ├─ 3. Task reservation (drone-ID-order tiebreak)
   └─ 4. State transitions (RETURN/REPORTING/IDLE)
   │
   ▼
DroneAgent.commit(action, snap)   ── drones update internal state
   │
   ▼
MissionManager.update(drones, tick)   ── coverage, completion check, per_drone_stats
   │
   ▼
MissionLogger.sample_if_due(tick) ── CSV row every N ticks
   │
   ▼
(tick ++)
   │
   ▼
(interactive only) Renderer.draw(sim)  ── isolated, no sim effect
```

Rendering is shown last and is **outside** the tick. Per-tick work above happens in `tick_once()`; the renderer reads the post-tick state on each `display.flip()`.

### Headless mode

`sweep.py` constructs the `Simulator` directly and calls `tick_once()` in a tight `for tick in range(max_ticks)` loop. **Zero Pygame imports on this path.** `renderer.py` is imported lazily inside `__main__.py`'s interactive path only. Pygame remains a flat dependency in pyproject.

### Snapshot / propose / resolve semantics (Tick Architecture A)

- **Snapshot** is a frozen world view: `grid`, `obstacles`, `searched_set`, `drone_positions: dict[int, tuple[int,int]]`, `home`. Built once per tick, shared by all drones. Drones do not see siblings' uncommitted proposals within the same tick.
- **Propose**: each live drone returns an `Action` (next move, mark-cell, request task, no-op, state-transition request). Pure function of (drone state, snapshot, rng). No shared mutation.
- **Resolve**: a single sequential pass applies proposals atomically.
  1. **Cell marking** (sensor footprint) — applied first to `MissionManager.searched`.
  2. **Move conflicts** — two drones propose the same target cell: **closer-to-target wins** (drone whose Euclidean distance to its own goal is smaller). Loser picks an open 8-neighbor or waits. Deterministic tiebreak.
  3. **Task reservation** — task-assignment channel uses **drone-ID order** as tiebreak when two idle drones request simultaneously. The cell enters `in_flight` immediately on assignment; second drone's request sees it as unavailable.
  4. **State transitions** (RETURN triggered by battery, REPORTING→IDLE) — applied after motion.
- **Commit**: each drone receives its (possibly modified) action and updates internal state.

### Determinism

Same `--seed` + same `--scenario` + same `--drones` → identical tick-by-tick output. The `random` module is seeded once in `Simulator.__init__`; `DroneAgent.propose` accepts the shared `rng` as a parameter. Tests assert exact `(tick, drone_id, x, y, battery)` tuples for short runs.

### Performance ceiling, named

Linear scan per idle drone per tick for task allocation (O(U), U = unassigned searchable cells). Fine for ≤150×150 grid, ≤50 drones. Ponytail comment in `mission_manager.py:assign_task` names the `scipy.spatial.cKDTree` upgrade path. No premature optimization.

---

## 5. Drone FSM, A\*, Collision Avoidance, Battery

### FSM (4 states)

```
IDLE ──[task assigned]──► SEARCHING
SEARCHING ──[target cell reached]──► IDLE
SEARCHING ──[battery ≤ return-threshold]──► RETURNING
RETURNING ──[at home base]──► REPORTING
REPORTING ──[report submitted + battery recharged]──► IDLE
Any state ──[drone killed]──► (removed; task returned to pool)
```

Transitions asserted in `test_fsm.py`.

### State behaviors

- **IDLE.** No movement. If `battery < 100`, recharge at `recharge_rate` per tick (default 2.0 battery/tick). Issue a task request to `MissionManager.assign_task`. On assignment → SEARCHING.
- **SEARCHING.** Follow A\* path to assigned target. On entering a cell, `MissionManager.mark_searched` is called by the simulator's resolve step with the drone's sensor footprint. When `pos == target` → IDLE. If battery hits return-threshold → RETURNING (target discarded, returned to pool via `release_task`).
- **RETURNING.** Path to home computed once via A\* (replan once per tick if blocked). On reaching home → REPORTING.
- **REPORTING.** Stationary at home for `reporting_duration_ticks` (default 3):
  1. Battery recharges at `recharge_rate`.
  2. On entering REPORTING, drone notifies MissionManager (`on_drone_reported(drone_id, summary_stats)`) — a single in-memory call, no per-drone JSON file written. MissionManager already tracks cells searched per drone, assignment counts, etc.; the call just flags "this drone completed a tour and is back at base."
  After `reporting_duration_ticks` → IDLE with battery clamped to 100.

### Why no per-drone JSON report file

MissionManager already accumulates per-drone lifetime stats from `mark_searched` (which receives `drone_id`), `assign_task` (counts assignments), and `on_drone_killed`. Dumping those again as a per-drone JSON at mission end would duplicate state, add a write path, and contribute nothing to the capstone metrics (coverage, robustness, scalability). Cut. The MM's in-memory tracking stays available if the actual evaluation ever needs it; it just doesn't become a separate artifact.

### Battery model

- `battery`: float in `[0, 100]`, starts at 100.
- **Motion drain:** `1.0` per cell moved (diagonal moves also cost `1.0` — uniform for predictable evaluation; real Euclidean diagonal would be √2 but capstone math wants linear).
- **Active-state drain:** `1.0` per tick while in SEARCHING (whether the drone moved or not — waiting on collision avoidance is still searching). Same `1.0` per tick while in RETURNING. This single rule replaces a separate per-state drain table and avoids the infinite-holding-pattern bug where a stuck drone never depletes.
- **Idle / REPORTING drain:** 0 (charging / processing — drone is at base).
- **Recharge:** `recharge_rate` per tick (default 2.0) while IDLE (battery < 100) and throughout REPORTING. Battery is clamped to 100 on transition out of REPORTING.
- **Return threshold:** `battery ≤ est_return_cost * safety_margin`, where `est_return_cost = manhattan(pos, home) * 1.5` and `safety_margin = 1.2` (both in `SimConfig`). When crossed during SEARCHING → RETURNING.
- **Crash rule:** if battery drops below 0 mid-RETURNING, the drone crashes (becomes failed via the normal `on_drone_killed` path). Real consequence, not just injected.

Drain summary: SEARCHING and RETURNING cost `1`/tick uniformly (with an extra `1` per cell if you move more than once per tick — but at 1 cell/tick uniform speed, that's already included). Ponysimplification over a multi-row drain table: one rule for active states, one for rest states.

### A\* implementation (`astar.py`)

- 8-connected neighbors. Obstacle cells = impassable. Other drone positions passed in as a `blockers` set per query.
- Heuristic: **octile distance** — `max(dx, dy) + 0.414 * min(dx, dy)`.
- Path-tiebreak: tiny ε bias to the heuristic (`h * 1.001`) to prefer paths closer to the straight-line direction.
- Signature: `def astar(grid: np.ndarray, blockers: set[tuple[int,int]], start: tuple[int,int], goal: tuple[int,int]) -> list[tuple[int,int]]`. Returns path from start (exclusive) to goal (inclusive), or `[]` if no path.
- Failure is **returnable** — drones handle empty paths by requesting reassignment, not by crashing.
- Pure function — trivially testable.

### Collision avoidance (local, in `DroneAgent.propose`)

1. Look at next cell in current path.
2. Check `snapshot.drone_positions` for any drone there.
3. If blocked: try next-best 8-neighbor that is (a) not an obstacle, (b) not in `snapshot.drone_positions`, (c) closer to goal. If found, replace path with one-step detour; A\* replans from new position next tick. If no detour available, propose no-op (wait).
4. Collision avoidance lives inside the drone; the `resolve` step just enforces the deterministic **closer-to-target wins** rule when two drones nonetheless propose the same cell.

### Failure handling

- **Manual:** `F<digit>` key (e.g. `F3`) kills drone with that id. CLI `--kill 200:1,350:2,400:3` scripts kills at specific ticks.
- **Stochastic:** `--failure-rate <per_drone_per_100_ticks>` (default 0). The expected semantics: `failure_rate = N` means each drone has an `N/100` chance of failing per 100 ticks on average. Per-tick probability is therefore `rate / 10000` — i.e. `random.random() < rate / 10000` per drone per tick (uses the shared sim `rng`).
- **On failure:** `drone.alive = False`, `MissionManager.release_task(drone_id)` returns its target to the searchable pool, `MissionManager.on_drone_killed(drone_id, target)` records the failure. Resolve step skips dead drones (their proposals aren't collected).

---

## 6. MissionManager, Task Allocation, Coverage, Logging

### Owned state

```python
class MissionManager:
    env: Environment
    searchable: set[tuple[int,int]]            # non-obstacle cells; fixed at init
    searched: set[tuple[int,int]]              # grows over time
    in_flight: dict[int, tuple[int,int]]       # drone_id -> target cell currently assigned
    per_drone_stats: dict[int, dict]           # in-memory lifetime counters per drone_id
    failure_log: list[tuple[int, int, int]]    # (tick, drone_id, cause); kept in memory
    log_path: str
    csv_writer: ...
    config: SimConfig
```

`per_drone_stats` tracks `assignments`, `cells_marked`, `distance`, `time_in_state` per drone — accumulated in-memory only. Used by `#SUMMARY` and (optionally) by the renderer HUD. Not written to a separate JSON file.

### Why `searched: set` not a counter

A counter would lose to failure recovery: when a drone dies mid-task, its in-flight target returns to the pool but coverage must not be double-counted. We need set membership to ask "was this cell ever marked searched?". The counter would have been the wrong tool. Set maxes at 22,500 entries (150²) — memory non-issue.

### Lifecycle

- `__init__`: read scenario, compute `searchable`, write CSV header.
- `snapshot(drones) -> Snapshot`: build the read-only view. Fresh each tick.
- `assign_task(drone_id, drone_pos) -> tuple[int,int] | None`: pull-based greedy nearest. Candidate pool = `searchable - searched - set(in_flight.values())`. Linear scan for min Euclidean distance to `drone_pos`. On success, add to `in_flight`. Returns `None` if no candidate → drone stays IDLE next tick.
- `mark_searched(cells: set[tuple[int,int]], drone_id: int)`: union into `searched`; accumulate `per_drone_stats[drone_id]['cells_marked']`.
- `record_move(drone_id, src, dst)`: increment `per_drone_stats[drone_id]['distance']` and tick `time_in_state` bookkeeping. Called from the resolve step.
- `release_task(drone_id)`: drop from `in_flight`. Cell becomes a candidate again next tick if not yet searched.
- `on_drone_killed(drone_id, target, tick)`: release task; record failure in `failure_log`. Resolve step skips dead drones going forward.
- `on_drone_reported(drone_id, summary_stats)`: called by drone when entering REPORTING. Lightweight — notification + stats merge into `per_drone_stats[drone_id]`. No file write.
- `update(drones, tick)`: recompute coverage, check completion, sample logger.
- `coverage() -> float`: `len(searched) / len(searchable)` — O(1) given `len()`.
- `is_complete() -> bool`: `coverage() >= config.coverage_threshold`.

### Greedy nearest — linear scan with named ceiling

```python
def assign_task(self, drone_id, drone_pos):
    candidates = self.searchable - self.searched - set(self.in_flight.values())
    # ponytail: O(U) per call where U = unassigned searchable cells.
    # Fine for ≤150×150 grid, ≤50 drones. Upgrade path: scipy.spatial.cKDTree
    # rebuilt per N ticks if U gets large. Profile before adopting.
    if not candidates:
        return None
    best = min(candidates, key=lambda c: (c[0]-drone_pos[0])**2 + (c[1]-drone_pos[1])**2)
    self.in_flight[drone_id] = best
    return best
```

### Concurrent assignment safety

Within a tick, multiple drones may request tasks. `assign_task` is called sequentially in the resolve step; idle drones processed in **drone-ID order** for the task channel. The cell enters `in_flight` immediately on assignment, so a later drone's request sees the just-assigned cell as unavailable. Deterministic.

### Coverage calculation

`len(searched) / len(searchable) * 100`. Logged each sample tick. No caching needed.

### Logging

- **File:** `out/run_<scenario>_d<drones>_<seed>.csv` (interactive) or `out/sweep_<run_id>/d<d>_g<g>/seed<N>.csv` (sweep).
- **Header + one row per N ticks** (`log_interval_ticks`, default 10):
  ```
  tick,active_drones,failed_drones,coverage_pct,remaining_cells,avg_battery,fps
  ```
- **`fps` column:** actual render FPS for interactive runs; `ticks_per_second_actual` (ticks processed per wall-clock second) for headless sweep runs.
- **`#SUMMARY` line** at mission completion (comment-prefixed so pandas skips it; the sweep runner and `plot.py` parse it via a small custom reader):
  ```
  #SUMMARY total_ticks=N final_coverage=N.N drone_failures=N avg_allocation_latency=N.N total_path_length=N
  ```

### Per-drone stats — in memory only

MissionManager accumulates `per_drone_stats` and `failure_log` while the sim runs. The `#SUMMARY` line uses aggregated values. No per-drone JSON file is written; if the eventual capstone writeup needs per-drone breakdowns, the renderer or an extended logging path can read `per_drone_stats` directly.

### Drops from original logging design

- **`#EVENT` lines / `--log-events` flag**: cut. CSV metrics are enough for evaluation.
- **Separate per-drone JSON report file** (`out/reports_<run_id>.json`): cut. `#SUMMARY` covers it.

### `SimConfig` (single source of truth)

```python
@dataclass(frozen=True)
class SimConfig:
    scenario_path: str | None = None        # if None, procedural generation used
    n_drones: int = 5
    grid_size: int = 100                     # only used if scenario_path is None
    obstacle_density: float = 0.10           # only used if procedural
    seed: int = 0
    sensor_radius: int = 2                   # 5x5 footprint per move = 25 cells
    ticks_per_second: int = 10
    max_ticks: int = 10000                   # safety cap; sweep uses this
    coverage_threshold: float = 1.0          # mission complete when coverage ≥ this
    log_interval_ticks: int = 10
    failure_rate: float = 0.0                # per_drone_per_100_ticks
    kills: tuple[tuple[int, int], ...] = ()  # ((tick, drone_id), ...)
    battery_safety_margin: float = 1.2
    recharge_rate: float = 2.0                # battery per tick
    reporting_duration_ticks: int = 3
```

Defaults chosen so an interactive run with `scenarios/default.txt` is immediately interesting.

---

## 7. Environment, Scenarios, Obstacle Generation

### `Environment` class

```python
class Environment:
    width: int
    height: int
    grid: np.ndarray[uint8]    # shape (H, W); 0=empty, 1=obstacle, 2=home
    home: tuple[int, int]
    spawn: tuple[int, int]     # defaults to home
```

The environment is constructed once and is effectively immutable for the run. No drone state, no searched state — those live in `MissionManager`.

### Scenario file format (plain text)

- One row per grid line, single char per cell.
- `.` empty | `#` obstacle | `H` home base (exactly one required) | `S` spawn (optional; default `H`).
- Full-line comments allowed via `#` at start of line (not trailing).
- Optional metadata line `# GRID 15 10` ignored — parser reads dims from row count and longest-row width; shorter rows padded with `.`.

Example `scenarios/default.txt` (15×10):
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

Text over JSON because a 100×100 grid is 100 lines of single chars — reviewable in a PR diff. A 10k-element nested JSON array would be uneditable.

### Bundled scenarios (`scenarios/`)

- `default.txt` — moderately cluttered; the showcase demo.
- `empty.txt` — no obstacles; control case for stress tests.
- `maze.txt` — heavily obstacle-dense with corridors; tests A\* and avoidance under pressure.
- `walls.txt` — long walls forcing detours.

### CLI override

```
uv run python -m swarm_sar --scenario scenarios/walls.txt
uv run python -m swarm_sar --seed 42 --grid 100 --obstacle-density 0.15
```

`--grid N` + `--obstacle-density d` triggers **procedural generation** (seeded). `--scenario path` loads a file. They're mutually exclusive; if both passed, `--scenario` wins and a warning is printed.

### Procedural generation (`environment.generate_grid(size, density, seed)`)

1. Seed `random` with `seed`.
2. Each cell independently obstacle with probability `density`, **except** `home` and its 8-neighbors (spawn feasibility).
3. **Cluster pass:** dilate each obstacle cell once — 4-line idiom using NumPy slicing instead of `scipy.ndimage.binary_dilation` (no scipy). Produces blob-like obstacles instead of salt noise; more realistic and visually cleaner.
4. **Connectivity check:** flood fill from `home`. If any empty cell is unreachable, regenerate with deterministic retry (`random.seed(hash((seed, attempt)))`). After 5 failed attempts, throw — user should lower density.

### Home base placement

Default: grid center, unless scenario file says otherwise. Drone spawn = home (the REPORTING state expects drones at home).

### Sensor footprint

All cells within Chebyshev distance `sensor_radius` of the drone, skipping obstacle cells and out-of-bounds. **Default `sensor_radius = 2`** (5×5 = 25 cells per move), chosen so that:
- Empty 100×100 grid (10,000 cells) with 5 drones needs ~80 moves per drone to fully cover → ~80 ticks to completion.
- Keeps missions non-trivial; obstacle grids extend this further.

Chebyshev matches the 8-connected grid's natural square footprint.

### Obstacles are not searchable

`searchable` = set of all non-obstacle cells. Obstacles never appear in `searched` and never enter the task pool.

### Edge cases

1. Home cell is "searched" from tick 0 (drones start there; immediately sensor-swept).
2. Cells in the sensor footprint at spawn time are marked searched at tick 0 (after first drone proposal).
3. **Spawn pile-up resolution:** multiple drones start on `home`. Before tick 0, drones are placed in a ring around home (concentric 8-neighbor cells expanding outward). First tick acts normally.

---

## 8. Rendering, Sweep Runner, Plot Generation, CLI

### Renderer (`renderer.py`) — interactive path only

Decoupled from ticks (per Section 4). One `display.flip()` per render frame. **No AI logic in this file.**

- **Window:** 800×600 default; cell pixel size `cell_px` (default 8) scales with grid; grid centered with 50px HUD margin.
- **Cell colors (flat, no fading):**
  - empty + unsearched → light gray
  - obstacle → dark gray / black
  - home base → yellow (drawn larger, with white `H` overlay)
  - searched → solid green (one color, no freshness gradient)
- **Drones:** blue circles with white ID digit overlay. Dead drones drawn as red `×` at last position. Planned path drawn as faint blue dots (toggle with `V`).
- **HUD top-left:** `tick: N | coverage: NN.N% | active: N/N | fps: N`
- **HUD top-right:** per-drone mini-bars — battery level + state abbreviation (`I`/`S`/`R`/`P`).

Renderer has no per-cell "freshness" state; searched cells are simply on or off in `MissionManager.searched`. Keeps the renderer stateless w.r.t. mission history.

### Key bindings (interactive mode, polled in `__main__.py`)

Dispatched per frame via `pygame.event.get()` in `__main__.py` to the correct consumer (sim config, renderer flag, or drone-kill). Keys do not interrupt the sim.

| Key | Action |
|-----|--------|
| `P` | pause/resume |
| `+` / `-` | speed up / slow down (clamp tps to `[1, 50]`) |
| `V` | toggle path preview |
| `F<digit>` | kill drone with that id (e.g. `F3` kills drone 3) |
| `Q` / `ESC` | quit |

Dropped keys (post-review): `S` (PNG snapshot) — cut; not necessary.

### Sweep runner (`sweep.py`) — headless batch

**Pygame stays unimported.** Import surface: `simulator`, `environment`, `mission_manager`, `drone`, `astar` only.

Sweep CLI:
```
uv run python -m swarm_sar --sweep \
  --drones 5,10,20,50 \
  --grids 50,100,150 \
  --seeds 3 \
  --repeats 3 \
  --density 0.10 \
  --max-ticks 5000 \
  --out out/sweep_YYYYMMDD/
```

Per `(drones, grid, seed, repeat)`:
1. Generate grid procedurally (or load scenario if `--scenario` overrides).
2. Construct `Simulator(config)` with `coverage_threshold=1.0`, `max_ticks`, `failure_rate`.
3. Tight loop: `sim.tick_once()` until `mm.is_complete()` or `max_ticks`. No rendering, no event pump, no sleep. Wall-clock timed.
4. Write `out/sweep_<run>/d<d>_g<g>/seed<N>.csv` (per-tick sampled stats). A single `#SUMMARY` line is appended to this CSV at run end — no separate JSON file.

After all runs: emit `out/sweep_<run>/aggregate.csv` — one row per `(drones, grid, seed, repeat)` with key completion metrics (ticks to complete, final coverage, total wall time, avg battery at completion, num failures). This is the master file `plot.py` reads.

`--verbose` prints a progress line per run; default prints one stdlib 5-line progress bar (no `tqdm` dep).

### Plot generation (`plot.py`)

`uv run python -m swarm_sar --plot <aggregate.csv>` reads an `aggregate.csv` and writes:

- `coverage_over_time.png` — line per `(drones, grid)`, x=tick, y=coverage%. One subplot per grid size.
- `completion_bars.png` — bars: x=drone count, y=ticks-to-completion, grouped by grid size.
- `failure_robustness.png` — only generated if `--failure-rate` used; y = final coverage at `max_ticks` with kills, grouped by killed-fraction.
- `fps_scaling.png` — y = avg tick rate achieved in headless mode, x = drone count. Shows linear degradation.

Written to the same `out/...` directory the sweep targeted. `plot.py` also callable on any `aggregate.csv` standalone.

### CLI dispatcher (`__main__.py`)

No subcommands — flag dispatch (capstone-appropriate, avoids ceremony):

- (default) → interactive Pygame
- `--sweep ...` → run sweep batch (auto-invokes `plot.py` on the aggregate at the end)
- `--plot <path>` → generate plots from existing CSV
- `--version` → print version

Argparse throughout. Stdlib only for CLI surface.

### Determinism across sweep configs

Sweep runs use the user-provided `--seed` directly. Different configs (drone count, grid size, density) already produce different runs because the environment and number of agents differ — no derived-seed scheme needed. Each `Simulator` constructs its own `random.Random(seed)` instance; the seed is reused only across the `--repeats` axis (which is the point of repeats — same config, same seed, deterministic re-run as a sanity check).

### Testing gates

Pytest is the only formal gate. No module-level `__main__` self-checks and no `--unit-self-check` CLI flag — pytest covers the same ground with less duplication. Run `uv run pytest` before each commit; CI runs the same.

---

## 9. Testing Strategy

### Module-level pytest tests (`tests/`)

| File | Targets |
|------|---------|
| `test_environment.py` | Scenario parse round-trips; procedural gen is deterministic; obstacle dilation produces blobs; connectivity retry logic |
| `test_astar.py` | Path correctness on empty / maze / blocked grids; octile optimality; blockers respected |
| `test_fsm.py` | All 4 transitions; illegal transitions rejected; state-specific side effects (battery drain, recharge) |
| `test_allocation.py` | Nearest-cell selection; concurrent assignment with ID tiebreak; failure recovery releases tasks |
| `test_coverage.py` | Sensor footprint marking; obstacles excluded from searchable; coverage math |
| `test_battery.py` | Drain on motion; uniform SEARCHING drain; return threshold triggers RETURNING; crash on `battery < 0` mid-RETURN |
| `test_simulator.py` | Snapshot/propose/resolve determinism: same seed → identical 50-tick trace; collision tiebreaks; cell-marking order |

### Manual verification step

`uv run pytest` is the acceptance gate before each commit. No second framework, no module-level `__main__` self-checks.

---

## 10. Evaluation Plan

Repeated per `(drones, grid_size)` cell, averaged across `--seeds` and `--repeats`:

| Metric | Source | Plot |
|--------|--------|------|
| Coverage efficiency | per-tick `coverage_pct` from CSV | `coverage_over_time.png` |
| Mission completion time | `total_ticks` from `#SUMMARY` | `completion_bars.png` |
| Robustness under failure | `final_coverage` at `max_ticks` across `--failure-rate` sweep | `failure_robustness.png` |
| Scalability | `ticks_per_second_actual` (headless FPS) vs drone count | `fps_scaling.png` |

Scalability sweep matrix:
- Drones: 5, 10, 20, 50
- Grids: 50×50, 100×100, 150×150
- Seeds: 3
- Repeats: 3

Robustness sweep matrix (separate invocation):
- Drones: 20 (fixed)
- Grid: 100×100 (fixed)
- `--failure-rate`: 0, 1, 5, 10, 25 (% per drone per 100 ticks → killed fraction)
- Seeds: 3, repeats: 3

All sweep output lands in `out/`; plots in `out/plots/`.

---

## 11. Algorithmic Complexity

| Operation | Worst case | Where | Notes |
|-----------|-----------|-------|-------|
| A\* path search | `O(E log V)` | `astar.py` | `V` = cells in grid, `E` ≤ `8V` (8-connected). Called per task assignment, and on replan. Binary heap priority queue. |
| Task allocation (greedy nearest) | `O(U)` | `MissionManager.assign_task` | `U` = unassigned searchable cells. Linear scan. Ponytail ceiling noted; upgrade path is `scipy.spatial.cKDTree` if profiled hot. |
| Coverage update (one tick) | `O(k)` | `MissionManager.mark_searched` | `k` = cells in sensor footprint (`(2r+1)² - obstacles ≤ 49` at default `r=2`). Set union is amortized `O(k)`. |
| Snapshot construction | `O(N)` | `MissionManager.snapshot` | `N` = drone count. Builds the read-only view each tick. |
| Resolve pass | `O(N)` | `Simulator.resolve` | Per-tick collision / reservation work scales with active drone count. |

Per-tick total: `O(U×idle_drones + V log V + N + k·moved_drones)`. The dominating term in the worst case is `U×idle_drones` for allocation; in practice only a few drones are idle simultaneously and most ticks have zero allocation calls.

---

## 12. Implementation Phasing (informational, plan comes later)

The implementation plan (produced by writing-plans skill) will phase this work. The intended cuts, in order, so you can see what's coming:

1. **Python scaffolding.** uv init, pyproject.toml, mise.toml + .gitignore fix, src/ skeleton, README, scenarios/ files.
2. **`environment.py` + scenario loader + procedural gen (with dilation).** `test_environment.py` passes.
3. **`astar.py`.** `test_astar.py` passes.
4. **`drone.py` FSM skeleton (no movement yet, just state transitions on synthetic inputs).** `test_fsm.py` + `test_battery.py` pass.
5. **`mission_manager.py` (searchable set, assign_task, coverage, per_drone_stats, on_drone_reported).** `test_allocation.py` + `test_coverage.py` pass.
6. **`simulator.py` (snapshot/propose/resolve, but with trivial drone behavior).** `test_simulator.py` determinism test passes.
7. **Wire drone movement into the simulator (A\* drives SEARCHING; uniform battery drain).** Single-drone end-to-end finishes a small grid.
8. **Multi-drone + collision avoidance + failure handling.** 5-drone run completes default scenario.
9. **`renderer.py` + interactive `__main__.py`** (flat colors, no freshness, no snapshot key). Visual sanity check.
10. **`sweep.py` + `plot.py`.** Run a small sweep, produce plots. Sweep auto-invokes plot at the end.
11. **Full evaluation sweeps.** 50-drone × 150×150 runs, failure sweeps; produce final graphs.

Drops from earlier phasing (post-review): no per-module self-check step, no JSON report writer, no freshness state in renderer, no `--log-events` wiring, no `--unit-self-check` flag plumbing.

---

## 13. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| A\* implementation bugs | High | Unit test on known mazes; octile-optimality check; tiebreak epsilon tested |
| FSM transition bugs | Medium | Log transitions in `per_drone_stats` + assert in `test_fsm.py`; verify state-specific side effects |
| Performance collapse at 50 drones × 150² | Medium | Linear-scan ceiling named; `max_ticks` cap protects sweep; profile before adopting k-d tree |
| CSV log data loss | Low | Append-mode writes per row; flush every sample; `#SUMMARY` at end |
| Spawn pile-up | Low | Ring placement before tick 0 (Section 7 edge case 3) |
| Stochastic test flakiness | Low | All randomness via shared `random.Random(seed)`; no module-level RNG state |
| Holding-pattern stalls (drones block forever) | Medium | Uniform SEARCHING-time drain means stuck drones still deplete → trigger RETURNING → free up the cell. No "wait forever" equilibrium. |
