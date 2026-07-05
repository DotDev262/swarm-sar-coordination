import json
from typing import Optional


def plot_coverage(filepath: str, output_file: Optional[str] = None):
    import matplotlib
    matplotlib.use("Agg") if output_file else None
    import matplotlib.pyplot as plt

    with open(filepath, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]

    fig, ax = plt.subplots(figsize=(8, 5))
    for run in data:
        cov = run.get("coverage_pct", 0)
        ax.bar([run.get("n_drones", 0)], [cov], color="tab:blue")
    ax.set_xlabel("drones")
    ax.set_ylabel("coverage %")
    ax.set_title("Sweep coverage")
    fig.tight_layout()
    if output_file:
        fig.savefig(output_file, dpi=150)
        print(f"plot -> {output_file}")
    else:
        plt.show()
    plt.close(fig)


def main(filepath: str):
    plot_coverage(filepath, filepath.replace(".json", ".png"))
