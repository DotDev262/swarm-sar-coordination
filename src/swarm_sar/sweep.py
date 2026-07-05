from typing import List, Optional
from src.swarm_sar.environment import Environment, SimConfig
from src.swarm_sar.simulator import Simulator
from src.swarm_sar.mission_manager import MissionManager


def run_sweep(config: SimConfig) -> List[dict]:
    """
    Run a sweep simulation with the given configuration.

    Args:
        config: Simulation configuration

    Returns:
        List of results dictionaries containing simulation metrics
    """
    env = Environment.from_config(config)
    sim = Simulator(env, config)
    mm = sim.mm

    results = []

    while not sim.is_complete() and sim.tick < config.max_ticks:
        sim.tick_once()
        mm.flush_log()

    summary = {
        "seed": config.seed,
        "n_drones": config.n_drones,
        "grid_size": config.grid_size,
        "coverage": sim.mm.coverage(),
        "coverage_pct": sim.mm.coverage() * 100,
        final_tick=sim.tick,
        "searched_cells": len(sim.mm.searched),
        "total_cells": len(sim.mm.searchable),
        "active_drones": sum(1 for d in sim.drones if d.alive),
        failed_drones=len(mm.failure_log),
        "avg_battery": sum(d.battery for d in sim.drones if d.alive) / max(1, sum(1 for d in sim.drones if d.alive)),
    }
    results.append(summary)

    return results


def run_sweep_wrapper(
    drones: List[int],
    grids: List[int],
    seeds: int = 3,
    repeats: int = 3,
    density: float = 0.10,
    max_ticks: int = 5000,
    out_dir: str = "out/sweep",
    verbose: bool = False,
    base_config: Optional[object] = None,
):
    """
    Wrapper function for running sweeps with multiple configurations.

    Args:
        drones: List of drone counts to test
        grids: List of grid sizes to test
        seeds: Number of random seeds to test
        repeats: Number of repeats per configuration
        density: Obstacle density
        max_ticks: Maximum ticks per simulation
        out_dir: Output directory
        verbose: Enable verbose output
        base_config: Base configuration object
    """
    import os
    import json

    if base_config is None:

        class BaseConfig:
            pass

        base_config = BaseConfig()
        base_config.ticks_per_second = 10
        base_config.log_interval_ticks = 10
        base_config.sensor_radius = 2
        base_config.failure_rate = 0.0
        base_config.recharge_rate = 1.0
        base_config.battery_safety_margin = 1.0
        base_config.reporting_duration_ticks = 2

    os.makedirs(out_dir, exist_ok=True)
    results = []

    for drone_count in drones:
        for grid_size in grids:
            for seed in range(seeds):
                for repeat in range(repeats):
                    config = SimConfig(
                        scenario_path=None,
                        n_drones=drone_count,
                        grid_size=grid_size,
                        seed=seed + repeat * 1000,
                        obstacle_density=density,
                        ticks_per_second=base_config.ticks_per_second,
                        max_ticks=max_ticks,
                        coverage_threshold=base_config.coverage,
                        log_interval_ticks=base_config.log_interval_ticks,
                        sensor_radius=base_config.sensor_radius,
                        failure_rate=base_config.failure_rate,
                        kills=base_config.kills,
                        recharge_rate=base_config.recharge_rate,
                        battery_safety_margin=base_config.battery_safety_margin,
                        reporting_duration_ticks=base_config.reporting_duration_ticks,
                    )

                    if verbose:
                        print(f"Running: drones={drone_count}, grid={grid_size}, seed={seed}, repeat={repeat}")

                    sweep_results = run_sweep(config)

                    for result in sweep_results:
                        result["drone_count"] = drone_count
                        result["grid_size"] = grid_size
                        result["repeat"] = repeat

                        filename = f"{out_dir}/sweep_d{drone_count}_g{grid_size}_s{seed}_r{repeat}.json"

                        with open(filename, "w") as f:
                            json.dump(result, f, indent=2)

                        results.append(result)

                        if verbose:
                            print(f"Saved: {filename}")

    summary_path = f"{out_dir}/sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    return results