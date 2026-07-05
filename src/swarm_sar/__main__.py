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
    mm.set_log_path(f"out/run_d{config.n_drones}_s{config.seed}.csv")
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
                elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
        if running:
            renderer.draw()
        clock.tick(60)
    mm.flush_log()
    if mm._log_path:
        with open(mm._log_path, "a") as fh:
            fh.write(mm.summary_line(sim.tick) + "\n")
    renderer.shutdown()


def run_sweep(args):
    from swarm_sar.sweep import run_sweep, SweepConfig
    sweep_cfg = SweepConfig(
        drones=[int(x) for x in args.sweep_drones.split(",")],
        grids=[int(x) for x in args.sweep_grids.split(",")],
        seeds=args.sweep_seeds,
        repeats=args.sweep_repeats,
        density=args.sweep_density,
        max_ticks=args.sweep_max_ticks,
        out_dir=args.sweep_out,
        verbose=args.verbose,
        base_config=args,
    )
    run_sweep(sweep_cfg)


def run_headless(config):
    from swarm_sar.simulator import Simulator
    from swarm_sar.environment import Environment
    
    env = Environment.from_config(config)
    sim = Simulator(env, config)
    mm = sim.mm
    mm.set_log_path(f"out/headless_d{config.n_drones}_s{config.seed}.csv")
    
    while not sim.is_complete() and sim.tick < config.max_ticks:
        sim.tick_once()
        mm.log_drone_states(sim.tick, sim.drones)
        mm.flush_log()
    
    if mm._log_path:
        with open(mm._log_path, "a") as fh:
            fh.write(mm.summary_line(sim.tick) + "\n")
    print(f"Headless simulation completed")
    print(f"  Ticks: {sim.tick}")
    print(f"  Coverage: {sim.mm.coverage()*100:.2f}%")
    print(f"  Searched cells: {len(sim.mm.searched)}")
    print(f"  Failed drones: {len(sim.mm.failure_log)}")


def main(argv=None):
    from swarm_sar.environment import SimConfig
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
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--sweep-drones", type=str, default="5,10,20,50")
    parser.add_argument("--sweep-grids", type=str, default="50,100,150")
    parser.add_argument("--sweep-seeds", type=int, default=3)
    parser.add_argument("--sweep-repeats", type=int, default=3)
    parser.add_argument("--sweep-out", type=str, default="out/sweep")
    parser.add_argument("--sweep-max-ticks", type=int, default=5000)
    parser.add_argument("--sweep-density", type=float, default=0.10)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-ui", dest="headless", action="store_true")
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
        from swarm_sar.plot import main as plot_main
        plot_main(args.plot)
    elif args.sweep:
        run_sweep(args)
    elif args.headless:
        run_headless(config)
    else:
        run_interactive(config)


if __name__ == "__main__":
    main()
