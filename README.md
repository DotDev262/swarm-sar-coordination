# swarm-sar-coordination

A hybrid multi-agent **S**earch-**A**nd-**R**escue (SAR) drone simulation.
A centralized `MissionManager` allocates unsearched grid cells to autonomous
`DroneAgent`s that pathfind with A*, manage battery drain/recharge, and cycle
through a finite state machine. Interactive mode renders with Pygame; headless
mode, sweep mode, and plot mode run without any GUI.

Python 3.11 · Pygame · NumPy · matplotlib · uv · pytest

---

## Table of contents

1. [Setup](#setup)
2. [Architecture](#architecture)
3. [Run modes](#run-modes)
   1. [Interactive (Pygame window)](#interactive-pygame-window)
   2. [Headless (no Pygame)](#headless-no-pygame)
   3. [Sweep (headless, multi-config)](#sweep-headless-multi-config)
   4. [Plot (matplotlib)](#plot-matplotlib)
4. [Flag reference](#flag-reference)
5. [Keyboard controls (interactive)](#keyboard-controls-interactive)
6. [Scenarios](#scenarios)
7. [Output files](#output-files)
8. [Battery model](#battery-model)
9. [State machine](#state-machine)
10. [Determinism](#determinism)
11. [Testing & lint](#testing--lint)
12. [Project layout](#project-layout)
13. [Extending](#extending)

---

## Setup

```bash
uv sync          # create .venv and install all deps from pyproject.toml
```

Requires Python ≥ 3.11 and `uv`. No system Pygame/SDL install needed — `uv
sync` pulls `pygame>=2.5`, `numpy>=1.26`, `matplotlib>=3.8`, `pandas>=2.2`,
plus the dev group `pytest>=8.0` and `ruff>=0.6`.

If you use [mise](https://mise.jdx.dev/) (`mise.toml` is included), the same
commands are aliased:

```bash
mise run dev      # = uv run python -m swarm_sar
mise run test     # = uv run pytest
mise run sweep    # = uv run python -m swarm_sar --sweep
mise run lint     # = uv run ruff check src tests
```

---

## Architecture

```
            ┌─────────────────┐
            │   SimConfig      │  frozen dataclass, all tunables (env.py)
            └────────┬─────────┘
                     │
   ┌─────────────────┼─────────────────┐
   ▼                 ▼                 ▼
┌──────┐       ┌──────────┐       ┌──────────┐
│ Env  │◀──────│Simulator │──────▶│  MMgr    │
│ grid │       │ tick loop│       │ tasks    │
│ home │       │ propose  │       │ coverage │
│ spawn│       │ resolve  │       │ fail log │
└──────┘       │ commit   │       └──────────┘
   ▲           └────┬─────┘             ▲
   │                │                    │
   │                ▼                    │
   │           ┌──────────┐              │
   │           │ Drone[]  │──────────────┘
   │           │ FSM      │
   │           │ battery  │
   │           │ A* path  │
   │           └────┬─────┘
   │                │
   └────────────────┘
        astar (8-conn, octile heuristic)
```

**Per-tick flow** (in `Simulator.tick_once`):

1. **Snapshot** — `MissionManager.snapshot()` captures the grid, current drone
   positions, searched set, and obstacles into an immutable `Snapshot`.
2. **Propose** — every alive drone returns an `Action` (move-to cell, cells to
   mark, task-request flag, optional state transition) based on the snapshot.
3. **Resolve** — searched cells from proposals are committed to the mission
   manager. IDLE drones requesting a task are assigned the nearest unsearched
   cell (Euclidean distance), exclusive of cells already in flight and their
   own current cell; their A* path is computed against the other drones'
   positions as blockers.
4. **Commit** — each drone applies its action: pos updates, FSM state
   transitions, battery drain/recharge. After commit, a drone that reached its
   target cell transitions SEARCHING → RETURNING and releases its in-flight
   slot so other drones can claim nearby cells.
5. The mission manager checks `coverage() ≥ coverage_threshold` and stops.

---

## Run modes

Exactly one mode runs per invocation, dispatched in this order:
`--plot` → `--sweep` → `--headless`/`--no-ui` → interactive (default).

### Interactive (Pygame window)

```bash
uv run python -m swarm_sar
uv run python -m swarm_sar --scenario scenarios/maze.txt --drones 8 --seed 42
uv run python -m swarm_sar --grid 60 --drones 10 --density 0.15 --tps 20
```

Opens a Pygame window. Grid cells are drawn flat: green = searched, gray =
unsearched empty, black = obstacle, yellow = home base. Drones render as blue
circles (red if dead) with their numeric id; their planned path is drawn in
light blue (toggle with `V`). A HUD shows `tick`, `coverage %`, `active/total`
drones; a per-drone battery bar sits at the right edge.

The loop is **decoupled**: `--tps` controls simulation ticks/sec; the renderer
runs at up to 60 FPS via `pygame.time.Clock`. Pausing/resuming is key-driven,
not sim-driven. A CSV run log is written every tick to
`out/run_d<N>_s<S>.csv`.

### Headless (no Pygame)

```bash
uv run python -m swarm_sar --headless --grid 50 --drones 10 --seed 0 --coverage 1.0
uv run python -m swarm_sar --no-ui --max-ticks 2000 --grid 100
```

Runs the full simulation with no window; Pygame is **not imported** for this
path. Writes a per-tick CSV (`out/headless_d<N>_s<S>.csv`) and prints a
summary on completion: ticks run, coverage %, cells searched, failed drones.

Ideal for CI, SSH hosts without a display, batch experiments, and capturing
run logs for analysis.

### Sweep (headless, multi-config)

```bash
uv run python -m swarm_sar --sweep \
  --sweep-drones 5,10,20,50 \
  --sweep-grids 50,100,150 \
  --sweep-seeds 3 --sweep-repeats 3 \
  --sweep-out out/sweep --verbose
```

Runs a cartesian product of `drones × grids × seeds × repeats`, each as an
independent headless simulation capped by `--sweep-max-ticks`. Each run writes
a per-run JSON (`out/sweep/sweep_d<N>_g<G>_s<S>_r<R>.json`) and a combined
`out/sweep/sweep_summary.json`. Pygame is not imported.

`--verbose` prints each run's config before it executes. Non-verbose runs are
silent except for the final summary line. The `--coverage`, `--sensor-radius`,
`--failure-rate`, `--tps`, `--seed` (used as the base), and `--battery` flags
apply to every run in the sweep; `--grid`, `--drones`, and `--density` are
overridden by the sweep-specific `--sweep-*` counterparts.

### Plot (matplotlib)

```bash
uv run python -m swarm_sar --plot out/sweep/sweep_d10_g100_s0_r0.json
# writes out/sweep/sweep_d10_g100_s0_r0.png
```

Reads a sweep run JSON, draws a coverage-vs-drones bar chart, and writes a PNG
next to the input (same stem, `.png` extension). Matplotlib is imported lazily
inside `plot()`, so it does not slow startup of other modes.

---

## Flag reference

All flags optional. `--headless` and `--no-ui` are aliases (both map to the same
arg). Sweep is exclusive of headless/interactive. Plot is exclusive of all.

### Core simulation flags (interactive, headless, sweep)

| Flag | Type | Default | Applies to | Description |
|------|------|---------|------------|-------------|
| `--scenario` | path | none | all | Load a `.txt` scenario (see [Scenarios](#scenarios)). Overrides procedural generation; `--grid` and `--density` are ignored. |
| `--drones` | int | `5` | all | Number of `DroneAgent`s. Must be ≥ 1. Spawned in outward rings around home. |
| `--grid` | int | `100` | interactive, headless | Side length N of the N×N procedural grid (only when `--scenario` is not given). |
| `--seed` | int | `0` | all | RNG seed. Same seed + same config → byte-identical run (see [Determinism](#determinism)). |
| `--density` | float | `0.10` | interactive, headless | Obstacle fraction in `[0.0, 1.0)` for procedural grids. Unreachable cells are converted to obstacles (connectivity guaranteed). |
| `--tps` | int | `10` | interactive | Simulation ticks per second. Decoupled from render FPS (60). Clamped to `[1, 50]` by `+`/`-` keys at runtime. |
| `--max-ticks` | int | `10000` | interactive, headless | Hard stop after this many ticks, even if coverage threshold isn't met. |
| `--coverage` | float | `1.0` | all | Stop when `coverage ≥ threshold`. `1.0` = search every reachable cell. `0.5` = stop at 50%. |
| `--log-every` | int | `10` | all | *(Reserved.)* Currently the CSV log is written every tick regardless. Kept for forward-compat with a future sampling logger. |
| `--sensor-radius` | int | `2` | interactive, headless | Chebyshev sensor footprint radius. `r=2` → 5×5 (25 cells) marked searched per drone step. Bigger radius = faster coverage, less granularity. |
| `--failure-rate` | float | `0.0` | all | *(Reserved.)* Per-tick probability of a random drone failure. Currently inert; reserved for the failure-injection extension. |
| `--battery` | float | `0.0` | all | Battery capacity per drone. `0` = auto-scale to `4 × (2 × grid_size)` (e.g. grid 100 → cap 800), enough to cross the grid and return with margin. Override for fixed-capacity experiments. |
| `--headless` | flag | off | — | Run without Pygame (see [Headless](#headless-no-pygame)). |
| `--no-ui` | flag | off | — | Alias for `--headless`. |

### Mode-selection flags

| Flag | Description |
|------|-------------|
| `--sweep` | Run the parameter sweep (see [Sweep](#sweep-headless-multi-config)). Takes precedence over `--headless`. |
| `--plot PATH` | Read `PATH` (a sweep JSON), write a PNG. Takes precedence over all other modes. |
| `--verbose` | Print per-run progress lines during `--sweep`. No effect in other modes. |

### Sweep-only flags (ignored outside `--sweep`)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--sweep-drones` | csv | `"5,10,20,50"` | Comma-separated drone counts to sweep. |
| `--sweep-grids` | csv | `"50,100,150"` | Comma-separated grid sizes to sweep. |
| `--sweep-seeds` | int | `3` | Number of distinct seeds per (drones, grid) cell. |
| `--sweep-repeats` | int | `3` | Repeats per seed. Repeat `r` offsets the seed by `r × 1000`. |
| `--sweep-out` | path | `out/sweep` | Output directory for per-run JSON + summary. Created if missing. |
| `--sweep-max-ticks` | int | `5000` | Per-run tick cap inside the sweep (independent of `--max-ticks`). |
| `--sweep-density` | float | `0.10` | Obstacle density applied to all sweep runs. |

### Plot

The plot mode takes a single positional `PATH` (the JSON to read) and writes
`<PATH>.png`. No other flags apply.

---

## Keyboard controls (interactive)

| Key | Action |
|-----|--------|
| `P` | Pause/resume the simulation. Tick loop stops; renderer keeps drawing. |
| `+` or `=` | Speed up: `tps = min(tps + 1, 50)`. |
| `-` | Slow down: `tps = max(tps - 1, 1)`. |
| `V` | Toggle drawing of planned A* paths (on by default). |
| `Q` or `Esc` | Quit immediately; flushes the CSV summary line and shuts down Pygame. |
| Window close (×) | Same as `Q`. |

---

## Scenarios

Scenario files are plain text in `scenarios/`:

```
.  empty cell (searchable)
#  obstacle (not searchable, blocks movement)
H  home base (exactly one per scenario)
```

Shipped scenarios:
- `scenarios/empty.txt` — 5×5 open grid, centered home.
- `scenarios/default.txt` — tiny 2×7 starter grid.
- `scenarios/maze.txt` — walled maze with corridors.
- `scenarios/walls.txt` — `H` in the corner, one row of empty.

When `--scenario` is given, `--grid` and `--density` are ignored — the file
fully defines the grid and home. Drone spawn is still auto-computed around
home. Unknown characters in the file are treated as empty (`.`).

Load with:
```bash
uv run python -m swarm_sar --scenario scenarios/maze.txt --drones 8
```

---

## Output files

All output goes under `out/` (gitignored). The directory is created on demand.

| Path | Mode | Contents |
|------|------|----------|
| `out/run_d<N>_s<S>.csv` | interactive | Per-tick run log. One row per tick with aggregate fields + 4 per-drone field groups. |
| `out/headless_d<N>_s<S>.csv` | headless | Same format as above. |
| `out/sweep/sweep_d<N>_g<G>_s<S>_r<R>.json` | sweep | Per-run metrics: coverage %, ticks, searched/total cells, active/failed drones, avg battery. |
| `out/sweep/sweep_summary.json` | sweep | Array of all per-run JSON objects. |
| `<input>.png` | plot | Bar chart of coverage vs drone count, written next to the input JSON. |

**CSV schema** (interactive + headless): one header row, then one row per tick:

```
tick,active_drones,failed_drones,coverage_pct,remaining_cells,avg_battery,fps,
  drone_0_state,drone_0_pos,drone_0_target,drone_0_battery,
  drone_1_state,drone_1_pos,drone_1_target,drone_1_battery,
  ... (N drones)
```

- `drone_X_pos` and `drone_X_target` use `x;y` format (semicolons inside one
  CSV field so pandas `pd.read_csv` parses cleanly with default separators).
- `drone_X_target` is `-` when the drone has no target (IDLE / REPORTING).
- A final `summary_line` row is appended after the run completes.

**Sweep JSON schema** (per-run):

```json
{
  "seed": 0, "n_drones": 10, "grid_size": 100,
  "obstacle_density": 0.1, "final_tick": 14803,
  "coverage_pct": 100.0, "searched_cells": 9000,
  "total_searchable": 9000, "active_drones": 5,
  "failed_drones": 0, "avg_battery": 761.4,
  "repeat": 0
}
```

---

## Battery model

Each drone has a single `battery` value, initialized to `config.battery_capacity`
at spawn.

| State | Battery change per tick |
|-------|-------------------------|
| SEARCHING | `-1.0` (drains while moving) |
| RETURNING | `-1.0` (drains while moving home) — *except* the tick it steps onto home (no drain that tick) |
| REPORTING | `+recharge_rate` (tops up while waiting out `reporting_duration_ticks`) |
| IDLE | `+recharge_rate` (tops up, capped at `battery_capacity`) |
| IDLE transition (from REPORTING) | Instant reset to `battery_capacity` |

**Return threshold** (when a SEARCHING drone turns back): a drone returns home
when its remaining battery ≤ `(manhattan_to_home × 1.5 × battery_safety_margin)`.
This uses a 1.5× diagonal-distance estimate so drones don't strand one cell
short. `battery_safety_margin` defaults to `1.2` (only settable via `SimConfig`,
not exposed as a CLI flag).

**Auto-scale**: when `--battery` is `0` (the default), capacity is set to
`4 × (grid_size + grid_size)` — enough that a drone can cross the grid and
return with a healthy margin. Grid 100 → cap 800. Override with `--battery 500`
or `SimConfig(battery_capacity=500)` for fixed-capacity experiments. A drone
that hits battery `< 0` while RETURNING crashes (`alive = False`).

---

## State machine

```
        assign_task()              reaches target cell
        ┌─────────────┐           ┌─────────────────┐
        ▼             │           ▼                 │
  ┌─────────┐    ┌─────────┐  ┌──────────┐    ┌──────────┐
  │  IDLE   │───▶│ SEARCH  │─▶│ RETURNING│───▶│ REPORTING│
  │         │    │  ING    │  │          │    │          │
  └─────────┘    └─────────┘  └──────────┘    └──────────┘
        ▲                         │                │
        │                         │   steps onto    │
        └─────────────────────────┘    home (battery>0) │
          (REPORTING countdown done, after report)   │
                                                       │
                          battery < 0 while RETURNING → DEAD
```

- **IDLE** → proposes `request_task=True`; simulator assigns nearest unsearched
  cell; drone flips to SEARCHING.
- **SEARCHING** → A*-steps toward target; marks its `(2r+1)²` sensor footprint
  each tick; transitions to RETURNING either on low battery (threshold above)
  or on reaching the target cell (handled in `Simulator.tick_once`).
- **RETURNING** → A*-steps toward home; home base is **exempt from exclusive
  occupancy** so multiple returning drones can stack on home without deadlocking.
- **REPORTING** → counts down `reporting_duration_ticks` (default 3), recharging
  each tick; on completion flips to IDLE with a full battery reset.

A drone that crashes (battery < 0 in RETURNING) is marked dead and removed
from the alive list; its target cell is released back to the searchable pool.

---

## Determinism

Given the same `SimConfig` (seed included), a run is byte-reproducible:

- The only RNG in the system is `random.Random(config.seed)`, held by the
  `Simulator`. It is currently **not consumed** by any deterministic logic
  (mission allocation, A* tiebreaks, detour selection are all deterministic);
  it is reserved for the future `--failure-rate` injection.
- Mission allocation uses `min(candidates, key=dist)` — ties resolve to the
  lowest-tuple cell, which is stable across runs.
- Drone spawn order is a fixed outward-ring spiral around home.
- Set iteration order in Python is insertion-ordered for `dict` but
  unordered for `set`; `searchable` is a `set`, so allocation among
  equidistant cells depends on set iteration. This is stable *within* a Python
  minor version but not formally guaranteed. If cross-version reproducibility
  matters, sort candidates before `min`.

To reproduce a run:
```bash
uv run python -m swarm_sar --headless --grid 50 --drones 10 --seed 7 --coverage 1.0
# delete out/headless_d10_s7.csv, run again → identical bytes
```

---

## Testing & lint

```bash
uv run pytest                       # 33 unit/integration tests
uv run ruff check src/swarm_sar/ tests/
```

Tests live in `tests/test_*.py` and cover: environment generation &
connectivity, A* correctness, mission allocation, coverage accounting,
FSM transitions, battery drain/recharge/crash, and simulator integration
(drones reach 100% coverage on a small grid). No test framework beyond pytest;
no fixtures; no `__main__` self-checks. Linting via `ruff` with default rules.

---

## Project layout

```
swarm-sar-coordination/
├── pyproject.toml              # uv + hatchling build, deps, scripts
├── mise.toml                   # delegates all tasks to `uv run`
├── scenarios/                  # .txt scenario files
│   ├── default.txt
│   ├── empty.txt
│   ├── maze.txt
│   └── walls.txt
├── src/swarm_sar/
│   ├── __init__.py
│   ├── __main__.py             # argparse CLI, run_interactive/_headless/_sweep
│   ├── environment.py          # SimConfig, Environment, scenario loader
│   ├── astar.py                # 8-connected A* pathfinder (octile heuristic)
│   ├── mission_manager.py      # Snapshot, task allocation, coverage, logging
│   ├── drone.py                # DroneAgent FSM, battery, sensor footprint
│   ├── simulator.py            # Snapshot/propose/resolve/commit tick engine
│   ├── renderer.py             # Pygame renderer (HUD, paths, battery bars)
│   ├── sweep.py                # SweepConfig, run_sweep (no Pygame import)
│   └── plot.py                 # Matplotlib bar chart from sweep JSON
├── tests/                      # test_*.py (33 tests)
└── out/                        # gitignored; CSV + JSON + PNG output
```

---

## Extending

Common extension points (all the source is short — read it):

- **Failure injection**: implement `--failure-rate`. In `Simulator.tick_once`,
  draw `rng.random() < config.failure_rate` per alive drone and call
  `mm.on_drone_killed(d.id, d.target, self.tick)` + mark `d.alive = False`.
- **Region-based allocation**: replace `MissionManager.assign_task`'s
  nearest-cell `min` with a region partition (north/east/south/west) and have
  each drone stay in its assigned region. The `Snapshot` plumbing already
  carries everything you need.
- **Smarter collision avoidance**: `_is_blocked_snap` in `drone.py` is the
  single chokepoint for occupancy checks. Right now it exempts only `home`; add
  exemptions (e.g. allow stacking at reporting cells) there.
- **Freshness heatmap**: `mark_searched` currently stores a flat set; switch to
  `dict[cell -> tick_last_marked]` and have the renderer blend color by age.
- **Real-time CSV**: `log_drone_states` writes every tick. To sample, gate on
  `tick % config.log_interval_ticks == 0` (the `--log-every` flag is already
  threaded and reserved).
