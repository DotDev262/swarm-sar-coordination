# swarm-sar-coordination

Hybrid multi-agent drone Search-and-Rescue (SAR) simulation. Python 3.11 + Pygame + uv.

Centralized `MissionManager` allocates unsearched cells to autonomous `DroneAgent`s
that navigate an 8-connected grid with A*, manage battery drain/recharge, and cycle
through an IDLE→SEARCHING→RETURNING→REPORTING FSM.

## Setup

```bash
uv sync
```

## Run

### Interactive (Pygame window)

```bash
uv run python -m swarm_sar
uv run python -m swarm_sar --scenario scenarios/maze.txt --drones 8 --seed 42
```

Keys: `P` pause, `+`/`-` speed, `V` toggle paths, `Q`/`Esc` quit.

### Headless (no Pygame)

Runs the simulation without opening a window; writes a per-tick CSV run log
(`out/headless_d<N>_s<S>.csv`) and prints a summary.

```bash
uv run python -m swarm_sar --headless --grid 50 --drones 10 --seed 0 --coverage 1.0
uv run python -m swarm_sar --no-ui --max-ticks 2000
```

### Sweep (headless, multi-config)

Runs a grid of (drones × sizes × seeds × repeats), writes per-run JSON + a
summary to `--sweep-out`. Must not import Pygame.

```bash
uv run python -m swarm_sar --sweep \
  --sweep-drones 5,10,20,50 \
  --sweep-grids 50,100,150 \
  --sweep-seeds 3 --sweep-repeats 3 \
  --sweep-out out/sweep --verbose
```

### Plot (matplotlib)

```bash
uv run python -m swarm_sar --plot out/sweep/sweep_d10_g100_s0_r0.json
# writes <path>.png
```

## Common flags

| Flag | Default | Description |
|------|---------|-------------|
| `--scenario` | none | Path to a scenario `.txt` (overrides procedural grid) |
| `--drones` | 5 | Number of drones |
| `--grid` | 100 | Grid size (NxN) for procedural generation |
| `--seed` | 0 | RNG seed (deterministic) |
| `--density` | 0.10 | Obstacle density (procedural) |
| `--tps` | 10 | Ticks per second (interactive) |
| `--max-ticks` | 10000 | Stop after N ticks |
| `--coverage` | 1.0 | Stop when coverage ≥ threshold (0–1) |
| `--sensor-radius` | 2 | Sensor footprint radius (Chebyshev) |
| `--battery` | 0 | Battery capacity per drone (0 = auto-scale to `4×(2×grid)`, enough to cross the grid and return) |
| `--headless` / `--no-ui` | off | Run without Pygame |
| `--sweep` | off | Headless parameter sweep |
| `--plot PATH` | none | Plot a sweep run JSON to PNG |

## Test

```bash
uv run pytest
uv run ruff check src/swarm_sar/ tests/
```

## Scenarios

Scenario files live in `scenarios/` (`default.txt`, `empty.txt`, `maze.txt`,
`walls.txt`). Format: rows of `.` (empty), `#` (obstacle), `H` (home base).
One home per scenario.

## Output

- `out/run_d<N>_s<S>.csv` — interactive run log (one row per tick, per-drone state)
- `out/headless_d<N>_s<S>.csv` — headless run log
- `out/sweep/sweep_d*.json` — per-run sweep results
- `out/sweep/sweep_summary.json` — all sweep results
