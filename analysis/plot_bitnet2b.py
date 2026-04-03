"""Publication-quality plots for BitNet 2B4T analysis.

Generates 4 figures from the layer_analysis.json output:
1. Frustration profile: layer-by-layer frustration vs random baseline
2. Vacancy profile: per-layer zero fraction
3. Weight distribution: {-1, 0, +1} fractions per layer
4. Summary dashboard: all key metrics in one figure

All figures saved as PNG at 300 DPI, single-column width (3.5").

Usage:
    python -m analysis.plot_bitnet2b [--input results/bitnet2b/layer_analysis.json]
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — works without a display
import matplotlib.pyplot as plt
import numpy as np

# Try SciencePlots, fall back gracefully
try:
    import scienceplots  # noqa: F401 — registers styles with matplotlib
    plt.style.use(["science", "no-latex"])
except Exception:
    plt.style.use("seaborn-v0_8-paper")
    print("Note: SciencePlots not available, using seaborn style")

DPI = 300
WIDTH = 7.0  # inches (double-column for readability)
COLORS = {
    "trained": "#2196F3",
    "random": "#FF9800",
    "neg1": "#E53935",
    "zero": "#9E9E9E",
    "pos1": "#43A047",
}


def load_results(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _clean_layer_name(name: str) -> str:
    """Shorten long HuggingFace layer names for axis labels."""
    # e.g. "model.layers.0.self_attn.q_proj.weight" -> "L0.attn.q"
    parts = name.replace("model.layers.", "L").replace(".weight", "")
    parts = parts.replace("self_attn.", "attn.").replace("_proj", "")
    parts = parts.replace("mlp.", "mlp.")
    return parts


def plot_frustration_profile(results: list[dict], out_dir: Path) -> Path:
    """Figure 1: Frustration index per layer, trained vs random baseline."""
    fig, ax = plt.subplots(figsize=(WIDTH, 4))

    x = np.arange(len(results))
    trained = [r["frustration_index"] for r in results]
    random_bl = [r["random_baseline_frustration"] for r in results]

    ax.bar(x - 0.15, trained, width=0.3, color=COLORS["trained"],
           label="Trained BitNet 2B4T", alpha=0.9)
    ax.bar(x + 0.15, random_bl, width=0.3, color=COLORS["random"],
           label="Random ternary (matched sparsity)", alpha=0.9)

    ax.set_xlabel("Layer index")
    ax.set_ylabel("Frustration index")
    ax.set_title("BitNet 2B4T: Frustration Profile by Layer")
    ax.legend(fontsize=8)

    # Only show every Nth tick if many layers
    if len(results) > 30:
        tick_step = max(1, len(results) // 20)
        ax.set_xticks(x[::tick_step])
        ax.set_xticklabels(x[::tick_step], fontsize=6)
    else:
        ax.set_xticks(x)
        labels = [_clean_layer_name(r["name"]) for r in results]
        ax.set_xticklabels(labels, rotation=90, fontsize=5)

    ax.set_ylim(0, 0.55)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3, label="Max frustration")

    fig.tight_layout()
    path = out_dir / "bitnet2b_frustration_profile.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_frustration_difference(results: list[dict], out_dir: Path) -> Path:
    """Frustration difference (trained - random). Negative = learned structure."""
    fig, ax = plt.subplots(figsize=(WIDTH, 3.5))

    x = np.arange(len(results))
    diff = [r["frustration_vs_random"] for r in results]

    colors = ["#43A047" if d < 0 else "#E53935" for d in diff]
    ax.bar(x, diff, color=colors, alpha=0.85)

    ax.set_xlabel("Layer index")
    ax.set_ylabel("Frustration (trained - random)")
    ax.set_title("BitNet 2B4T: Frustration vs Random Baseline")
    ax.axhline(y=0, color="black", linewidth=0.8)

    # Annotate
    n_below = sum(1 for d in diff if d < 0)
    ax.text(
        0.02, 0.95,
        f"{n_below}/{len(diff)} layers less frustrated than random",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    if len(results) > 30:
        tick_step = max(1, len(results) // 20)
        ax.set_xticks(x[::tick_step])
    fig.tight_layout()
    path = out_dir / "bitnet2b_frustration_difference.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_vacancy_profile(results: list[dict], out_dir: Path) -> Path:
    """Figure 2: Vacancy fraction (% zeros) per layer."""
    fig, ax = plt.subplots(figsize=(WIDTH, 3.5))

    x = np.arange(len(results))
    vacancy = [r["vacancy_fraction"] * 100 for r in results]

    ax.bar(x, vacancy, color=COLORS["zero"], alpha=0.85)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Vacancy fraction (%)")
    ax.set_title("BitNet 2B4T: Zero-Weight Fraction by Layer")

    mean_v = np.mean(vacancy)
    ax.axhline(y=mean_v, color="red", linestyle="--", alpha=0.6,
               label=f"Mean: {mean_v:.1f}%")
    ax.legend(fontsize=8)

    if len(results) > 30:
        tick_step = max(1, len(results) // 20)
        ax.set_xticks(x[::tick_step])

    fig.tight_layout()
    path = out_dir / "bitnet2b_vacancy_profile.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_weight_distribution(results: list[dict], out_dir: Path) -> Path:
    """Figure 3: Stacked bar — {-1, 0, +1} fractions per layer."""
    fig, ax = plt.subplots(figsize=(WIDTH, 3.5))

    x = np.arange(len(results))
    neg1 = [r["frac_neg1"] * 100 for r in results]
    zero = [r["frac_zero"] * 100 for r in results]
    pos1 = [r["frac_pos1"] * 100 for r in results]

    ax.bar(x, neg1, color=COLORS["neg1"], label="-1", alpha=0.9)
    ax.bar(x, zero, bottom=neg1, color=COLORS["zero"], label="0", alpha=0.9)
    bottom2 = [n + z for n, z in zip(neg1, zero)]
    ax.bar(x, pos1, bottom=bottom2, color=COLORS["pos1"], label="+1", alpha=0.9)

    ax.set_xlabel("Layer index")
    ax.set_ylabel("Weight fraction (%)")
    ax.set_title("BitNet 2B4T: Weight Distribution per Layer")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, 105)

    if len(results) > 30:
        tick_step = max(1, len(results) // 20)
        ax.set_xticks(x[::tick_step])

    fig.tight_layout()
    path = out_dir / "bitnet2b_weight_distribution.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_summary_dashboard(results: list[dict], out_dir: Path) -> Path:
    """Figure 4: 2x2 dashboard combining all key metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 7))

    x = np.arange(len(results))
    tick_step = max(1, len(results) // 15)

    # Top-left: frustration profile
    ax = axes[0, 0]
    trained = [r["frustration_index"] for r in results]
    random_bl = [r["random_baseline_frustration"] for r in results]
    ax.plot(x, trained, "o-", color=COLORS["trained"], markersize=2, linewidth=1, label="Trained")
    ax.plot(x, random_bl, "s-", color=COLORS["random"], markersize=2, linewidth=1, label="Random")
    ax.set_ylabel("Frustration index")
    ax.set_title("Frustration profile", fontsize=9)
    ax.legend(fontsize=6)
    ax.set_xticks(x[::tick_step])

    # Top-right: frustration difference
    ax = axes[0, 1]
    diff = [r["frustration_vs_random"] for r in results]
    colors = ["#43A047" if d < 0 else "#E53935" for d in diff]
    ax.bar(x, diff, color=colors, alpha=0.8)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_ylabel("Trained - Random")
    ax.set_title("Frustration difference", fontsize=9)
    ax.set_xticks(x[::tick_step])

    # Bottom-left: vacancy
    ax = axes[1, 0]
    vacancy = [r["vacancy_fraction"] * 100 for r in results]
    ax.bar(x, vacancy, color=COLORS["zero"], alpha=0.8)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Zero fraction (%)")
    ax.set_title("Vacancy profile", fontsize=9)
    ax.set_xticks(x[::tick_step])

    # Bottom-right: weight distribution
    ax = axes[1, 1]
    neg1 = [r["frac_neg1"] for r in results]
    pos1 = [r["frac_pos1"] for r in results]
    neg1 = [r["frac_neg1"] * 100 for r in results]
    pos1 = [r["frac_pos1"] * 100 for r in results]
    zero = [r["frac_zero"] * 100 for r in results]
    ax.plot(x, neg1, "v-", color=COLORS["neg1"], markersize=2, linewidth=1, label="-1")
    ax.plot(x, pos1, "^-", color=COLORS["pos1"], markersize=2, linewidth=1, label="+1")
    ax.plot(x, zero, "s-", color=COLORS["zero"], markersize=2, linewidth=1, label="0")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Fraction (%)")
    ax.set_title("Weight fractions", fontsize=9)
    ax.legend(fontsize=6)
    ax.set_xticks(x[::tick_step])

    fig.suptitle("BitNet b1.58 2B4T — Spin Glass Analysis", fontsize=11, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "bitnet2b_summary_dashboard.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def print_key_findings(results: list[dict]) -> None:
    """Print a summary of key findings to the terminal."""
    n = len(results)
    frustrations = [r["frustration_index"] for r in results]
    diffs = [r["frustration_vs_random"] for r in results]
    vacancies = [r["vacancy_fraction"] for r in results]

    print("\n" + "=" * 60)
    print("KEY FINDINGS — BitNet 2B4T Spin Glass Analysis")
    print("=" * 60)
    print(f"Layers analyzed: {n}")
    print(f"")
    print(f"FRUSTRATION:")
    print(f"  Mean frustration:     {np.mean(frustrations):.4f}")
    print(f"  Std frustration:      {np.std(frustrations):.4f}")
    print(f"  Min:                  {np.min(frustrations):.4f}")
    print(f"  Max:                  {np.max(frustrations):.4f}")
    print(f"  Layers < random:      {sum(1 for d in diffs if d < 0)}/{n}")
    print(f"  Mean diff vs random:  {np.mean(diffs):.4f}")
    print(f"")
    print(f"VACANCY (zero fraction):")
    print(f"  Mean vacancy:         {np.mean(vacancies)*100:.1f}%")
    print(f"  Std vacancy:          {np.std(vacancies)*100:.1f}%")
    print(f"  Min:                  {np.min(vacancies)*100:.1f}%")
    print(f"  Max:                  {np.max(vacancies)*100:.1f}%")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot BitNet 2B4T analysis results")
    parser.add_argument("--input", type=str, default="results/bitnet2b/layer_analysis.json")
    parser.add_argument("--output-dir", type=str, default="figures")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {args.input}")
    results = load_results(args.input)
    print(f"Found {len(results)} layers\n")

    print("Generating figures...")
    plot_frustration_profile(results, out_dir)
    plot_frustration_difference(results, out_dir)
    plot_vacancy_profile(results, out_dir)
    plot_weight_distribution(results, out_dir)
    plot_summary_dashboard(results, out_dir)

    print_key_findings(results)

    print(f"\nAll figures saved to {out_dir}/")
    print("Open them with any image viewer, or screenshot to post online.")


if __name__ == "__main__":
    main()
