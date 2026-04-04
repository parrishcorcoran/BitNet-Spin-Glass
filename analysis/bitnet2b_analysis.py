"""Phase 0: Analyze Microsoft's BitNet 2B4T pretrained model.

Computes per-layer:
- Vacancy fraction (% zeros)
- Weight distribution {-1, 0, +1} fractions
- Signed-graph frustration index
- Comparison to random ternary baseline with matched sparsity

Usage:
    python -m analysis.bitnet2b_analysis [--output-dir results/bitnet2b]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from analysis.frustration import compute_frustration_index


def load_bitnet_model() -> dict[str, torch.Tensor]:
    """Load BitNet 2B4T and return named weight tensors."""
    from transformers import AutoModelForCausalLM

    print("Loading microsoft/bitnet-b1.58-2B-4T-bf16 ...")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/bitnet-b1.58-2B-4T-bf16",
        dtype=torch.bfloat16,
    )
    # Extract all 2D weight matrices
    weights: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.ndim == 2:
            weights[name] = param.detach().float()
    print(f"Extracted {len(weights)} weight matrices")
    del model  # free memory
    return weights


def analyze_weight_distribution(w: torch.Tensor) -> dict:
    """Compute ternary distribution stats for a weight matrix."""
    w_flat = w.flatten()
    total = w_flat.numel()

    # Round to nearest integer for ternary classification
    w_int = w_flat.round().clamp(-1, 1)

    n_neg = (w_int == -1).sum().item()
    n_zero = (w_int == 0).sum().item()
    n_pos = (w_int == 1).sum().item()
    n_other = total - n_neg - n_zero - n_pos  # values that don't round to {-1,0,1}

    return {
        "total_weights": total,
        "frac_neg1": n_neg / total,
        "frac_zero": n_zero / total,
        "frac_pos1": n_pos / total,
        "frac_other": n_other / total,
        "vacancy_fraction": n_zero / total,
        "mean": w_flat.mean().item(),
        "std": w_flat.std().item(),
    }


def analyze_layer(
    name: str,
    w: torch.Tensor,
    gauge_trials: int = 5000,
) -> dict:
    """Full analysis for one layer."""
    dist = analyze_weight_distribution(w)

    # Frustration on ternary-rounded weights
    w_ternary = w.round().clamp(-1, 1)

    # Only compute frustration for reasonably sized matrices
    rows, cols = w_ternary.shape
    if rows <= 2048 and cols <= 2048:
        frust = compute_frustration_index(w_ternary.numpy(), gauge_trials=gauge_trials)
    else:
        # For very large matrices, use fewer trials
        frust = compute_frustration_index(w_ternary.numpy(), gauge_trials=1000)

    # Random baseline with matched sparsity
    sparsity = dist["vacancy_fraction"]
    random_frust = _random_ternary_frustration(rows, cols, sparsity, gauge_trials=min(gauge_trials, 1000))

    return {
        "name": name,
        "shape": [rows, cols],
        **dist,
        "frustration_index": frust,
        "random_baseline_frustration": random_frust,
        "frustration_vs_random": frust - random_frust,
    }


def _random_ternary_frustration(
    rows: int, cols: int, sparsity: float, gauge_trials: int = 1000, n_samples: int = 5,
) -> float:
    """Average frustration of random ternary matrices with given sparsity."""
    frustrations = []
    for i in range(n_samples):
        rng = np.random.default_rng(seed=i)
        # Generate random {-1, 0, +1} with specified zero fraction
        w = rng.choice([-1, 0, 1], size=(rows, cols), p=[
            (1 - sparsity) / 2, sparsity, (1 - sparsity) / 2
        ]).astype(np.float32)
        f = compute_frustration_index(w, gauge_trials=gauge_trials)
        frustrations.append(f)
    return float(np.mean(frustrations))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze BitNet 2B4T weights")
    parser.add_argument("--output-dir", type=str, default="results/bitnet2b")
    parser.add_argument("--gauge-trials", type=int, default=5000)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weights = load_bitnet_model()

    results = []
    for name, w in tqdm(weights.items(), desc="Analyzing layers"):
        result = analyze_layer(name, w, gauge_trials=args.gauge_trials)
        results.append(result)
        print(f"  {name}: shape={result['shape']}, "
              f"vacancy={result['vacancy_fraction']:.3f}, "
              f"frustration={result['frustration_index']:.4f}")

    # Save results
    with open(out_dir / "layer_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_dir / 'layer_analysis.json'}")
    print(f"Analyzed {len(results)} layers")


if __name__ == "__main__":
    main()
