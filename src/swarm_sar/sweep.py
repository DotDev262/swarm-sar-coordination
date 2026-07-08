import json
import os
from dataclasses import dataclass

from swarm_sar.environment import SimConfig
from swarm_sar.simulator import Simulator


@dataclass
class SweepConfig:
    """Configuration for a parameter sweep across drones, grids, seeds, and repeats."""

    drones: list
    grids: list
    seeds: int
    repeats: int
    density: float
    max_ticks: int
    out_dir: str
    verbose: bool
    base_config: object


def _run_one(cfg: SimConfig) -> dict:
    """Runs a single simulation to completion and returns result metrics.

    Args:
        cfg: SimConfig for this individual run.

    Returns:
        Dict with seed, drone/grid config, final tick, coverage, and stats.
    """
    from swarm_sar.environment import Environment
    env = Environment.from_config(cfg)
    sim = Simulator(env, cfg)
    while not sim.is_complete() and sim.tick < cfg.max_ticks:
        sim.tick_once()
    active = sum(1 for d in sim.drones if d.alive)
    avg_bat = (
        sum(d.battery for d in sim.drones if d.alive) / active
        if active > 0
        else 0.0
    )
    return {
        "seed": cfg.seed,
        "n_drones": cfg.n_drones,
        "grid_size": cfg.grid_size,
        "obstacle_density": cfg.obstacle_density,
        "final_tick": sim.tick,
        "coverage_pct": round(sim.mm.coverage() * 100, 2),
        "searched_cells": len(sim.mm.searched),
        "total_searchable": len(sim.mm.searchable),
        "active_drones": active,
        "failed_drones": len(sim.mm.failure_log),
        "avg_battery": round(avg_bat, 2),
    }


def _cfg_from_sweep(sc: SweepConfig, n_drones: int, grid: int, seed: int) -> SimConfig:
    """Builds a SimConfig for a single sweep cell from the sweep configuration.

    Args:
        sc: The SweepConfig for the overall batch.
        n_drones: Number of drones for this run.
        grid: Grid size for this run.
        seed: Random seed for this run.

    Returns:
        A SimConfig instance ready for simulation.
    """
    base = sc.base_config
    return SimConfig(
        scenario_path=getattr(base, "scenario", None),
        n_drones=n_drones,
        grid_size=grid,
        seed=seed,
        obstacle_density=sc.density,
        ticks_per_second=getattr(base, "tps", 10),
        max_ticks=sc.max_ticks,
        coverage_threshold=getattr(base, "coverage", 1.0),
        log_interval_ticks=getattr(base, "log_every", 10),
        sensor_radius=getattr(base, "sensor_radius", 2),
        battery_capacity=getattr(base, "battery", 0.0),
    )


def run_sweep(sc: SweepConfig) -> list:
    """Executes the full parameter sweep, writing per-run and summary JSON files.

    Args:
        sc: SweepConfig defining the parameter grid.

    Returns:
        List of result dicts from every run in the sweep.
    """
    os.makedirs(sc.out_dir, exist_ok=True)
    results = []
    for n in sc.drones:
        for g in sc.grids:
            for seed in range(sc.seeds):
                for rep in range(sc.repeats):
                    run_seed = seed + rep * 1000
                    cfg = _cfg_from_sweep(sc, n, g, run_seed)
                    if sc.verbose:
                        print(f"sweep drones={n} grid={g} seed={run_seed}")
                    res = _run_one(cfg)
                    res["repeat"] = rep
                    fname = os.path.join(
                        sc.out_dir,
                        f"sweep_d{n}_g{g}_s{run_seed}_r{rep}.json",
                    )
                    with open(fname, "w") as fh:
                        json.dump(res, fh, indent=2)
                    results.append(res)
    summary = os.path.join(sc.out_dir, "sweep_summary.json")
    with open(summary, "w") as fh:
        json.dump(results, fh, indent=2)
    if sc.verbose:
        print(f"sweep complete: {len(results)} runs -> {summary}")
    return results
