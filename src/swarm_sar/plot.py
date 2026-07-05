import json
from typing import List, Dict, Any
import matplotlib.pyplot as plt


def plot_coverage(filepath: str, output_file: Optional[str] = None):
    """Plot coverage over time from a sweep results file.

    Args:
        filepath: Path to sweep results JSON file
        output_file: Output path for the plot, if None, shows directly
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    plt.figure(figsize=(12, 6))
    
    if isinstance(data, dict):
        data = [data]
    
    colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray"]
    
    for idx, run in enumerate(data):
        if "coverage_pct" in run:
            plt.plot(
                [idx] * len(data) if len(data) == 1 else range(len(data)),
                data[idx].get("coverage_per_tick", [0]),
                color=colors[idx % len(colors)],
                marker="o",
                label=f"Run {idx + 1}",
            )

    plt.title("Coverage Over Time")
    plt.xlabel("Tick")
    plt.ylabel("Coverage Percentage")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if output_file:
        plt.savefig(output_file, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_multi_sweep_results(
    directory: str = "out/sweep",
    pattern: str = "sweep_*.json",
    output_file: str = "coverage_over_time.png",
):
    """Plot coverage curves from multiple sweep runs.

    Args:
        directory: Directory containing sweep JSON files
        pattern: Pattern for JSON files to include
        output_file: Output path for the plot
    """
    import glob
    from pathlib import Path

    all_data = []
    data_paths = glob.glob(f"{directory}/{pattern}")
    
    for path in data_paths:
        try:
            with open(path, "r") as f:
                data = json.load(f)
                all_data.append((path, data))
        except Exception as e:
            print(f"Failed to load {path}: {e}")

    if not all_data:
        print(f"No sweep data found in {directory}")
        return

    plt.figure(figsize=(14, 8))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    
    run_indices = sorted(range(len(all_data)))
    num_series = len(run_indices)
    
    for run_idx in run_indices:
        path, data = all_data[run_idx]
        color = colors[run_idx % len(colors)]
        
        if "coverage_per_tick" in data:
            ticks = list(range(len(data["coverage_per_tick"])))
            coverage = data["coverage_per_tick"]
            plt.plot(ticks, coverage, color=color, linewidth=2, alpha=0.7, label=f"{Path(path).stem}")
        elif "coverage_pct" in data:
            if isinstance(data, dict):
                ticks = [0]
                coverage = [data.get("coverage_pct", 0)]
                plt.plot(
                    ticks,
                    coverage,
                    color=color,
                    linewidth=2,
                    alpha=0.7,
                    marker="o",
                    label=f"{Path(path).stem}",
                )
            else:
                print(f"Unknown data format in {path}: {type(data)}")

    plt.title("Coverage Over Time - Multi-Run Sweep")
    plt.xlabel("Tick")
    plt.ylabel("Coverage Percentage")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_file, dpi=150)
    print(f"Plot saved to {output_file}")


def compare_sweep(configurations: List[Dict[str, Any]], output_file: str = "comparison.png"):
    """Compare multiple sweep results.

    Args:
        configurations: List of configuration dictionaries with "path" and "label"
        output_file: Output path for the comparison plot
    """
    plt.figure(figsize=(12, 8))
    
    colors = ["blue", "orange", "green", "red", "purple", "brown", "pink"]
    
    for idx, config in enumerate(configurations):
        path = config.get("path")
        label = config.get("label", f"Config {idx + 1}")
        
        if not path:
            print(f"Missing path in configuration {idx}")
            continue
        
        try:
            with open(path, "r") as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                plt.plot(
                    [0],
                    [data.get("coverage_pct", 0)],
                    color=colors[idx % len(colors)],
                    linewidth=3,
                    marker="o",
                    markersize=8,
                    label=label,
                )
        except Exception as e:
            print(f"Failed to load {path}: {e}")

    plt.title("Sweep Configuration Comparison")
    plt.xlabel("Configuration Index")
    plt.ylabel("Final Coverage Percentage")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_file, dpi=150)
    print(f"Comparison plot saved to {output_file}")


def plot_drone_comparison(sweep_dir: str = "out/sweep", output_file: str = "drone_comparison.png"):
    """Plot drone comparison metrics.

    Args:
        sweep_dir: Directory containing sweep results
        output_file: Output path for the plot
    """
    import glob
    from pathlib import Path

    data = []
    files = glob.glob(f"{sweep_dir}/sweep_*.json")
    
    for file_path in files:
        with open(file_path, "r") as f:
            result = json.load(f)
            if isinstance(result, dict):
                result["file"] = Path(file_path).name
                data.append(result)
    
    if not data:
        print(f"No sweep results found in {sweep_dir}")
        return

    drone_counts = sorted(list(set(d.get("n_drones", 0) for d in data)))
    grid_sizes = sorted(list(set(d.get("grid_size", 0) for d in data)))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, param, title in zip(axes, ["n_drones", "grid_size"], ["Drone Count", "Grid Size"]):
        for idx, dc_val in enumerate(drone_counts if param == "n_drones" else grid_sizes):
            filtered = [d for d in data if d.get(param) == dc_val]
            
            if filtered:
                coverages = [d.get("coverage_pct", 0) for d in filtered]
                
                ax.bar(
                    idx + param,
                    sum(coverages) / len(coverages),
                    color="skyblue",
                    alpha=0.7,
                    label=f"{param} = {dc_val}",
                )
                ax.errorbar(
                    idx + param,
                    sum(coverages) / len(coverages),
                    yerr=[max(coverages) - min(coverages) for _ in coverages],
                    color="red",
                    capsize=3,
                    alpha=0.5,
                )
    
    fig.suptitle("Sweep Results Analysis")
    axes[0].set_ylabel("Average Coverage Percentage")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=150)
    print(f"Drone comparison plot saved to {output_file}")


def main(filepath: str = None):
    """Main entry point for the plotting script.

    Args:
        filepath: Path to sweep results file to plot
    """
    if filepath is None:
        print("Usage: python -m swarm_sar.plot <sweep_results_file> [output_file]")
        print("Example: python -m swarm_sar.plot out/sweep/sweep_d5_g100_s0_r0.json")
        return

    import sys

    if len(sys.argv) > 1 and not sys.argv[1].startswith("--") and sys.argv[1] != filepath:
        output_file = sys.argv[1]
    else:
        output_file = None

    plot_coverage(filepath, output_file)


if __name__ == "__main__":
    main()