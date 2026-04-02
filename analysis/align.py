"""Weight matching (Hungarian algorithm) for permutation alignment.

CRITICAL: Replicas MUST be aligned before computing overlaps.
Without alignment, all P(q) measurements are artifacts.

Uses cost matrix C_ij = -|dot(w_a_row_i, w_b_row_j)|.
For ternary weights, dot products are integers → many ties.
Reports sensitivity to tie-breaking across multiple random seeds.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def align_replicas(
    weights_a: list[np.ndarray],
    weights_b: list[np.ndarray],
    tie_breaking_seed: int = 0,
) -> list[np.ndarray]:
    """Align replica B's weights to replica A using Hungarian matching.

    Parameters
    ----------
    weights_a : list[np.ndarray]
        Weight matrices from replica A, one per layer.
    weights_b : list[np.ndarray]
        Weight matrices from replica B, one per layer.
    tie_breaking_seed : int
        Seed for random perturbation to break ties in cost matrix.

    Returns
    -------
    list[np.ndarray]
        Permuted weight matrices for replica B, aligned to A.
    """
    assert len(weights_a) == len(weights_b), "Replicas must have same number of layers"

    rng = np.random.default_rng(tie_breaking_seed)
    n_layers = len(weights_a)
    permuted_b = []

    # Process hidden layers (skip output layer — no permutation needed)
    for layer_idx in range(n_layers - 1):
        wa = weights_a[layer_idx]  # (out_features, in_features)
        wb = weights_b[layer_idx]

        assert wa.shape == wb.shape, (
            f"Layer {layer_idx}: shape mismatch {wa.shape} vs {wb.shape}"
        )

        # Cost matrix: C_ij = -|dot(wa_row_i, wb_row_j)|
        # We want to maximize |dot|, so negate for minimization
        cost = -np.abs(wa @ wb.T)

        # Add tiny random noise to break ties (important for ternary weights)
        noise = rng.uniform(0, 1e-10, size=cost.shape)
        cost = cost + noise

        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost)

        # Permute rows of wb to match wa
        perm = np.zeros_like(col_ind)
        perm[row_ind] = col_ind
        wb_permuted = wb[perm]

        # Also permute columns of the NEXT layer's weight matrix
        # (because permuting outputs of layer L means permuting inputs of layer L+1)
        if layer_idx + 1 < n_layers:
            weights_b[layer_idx + 1] = weights_b[layer_idx + 1][:, perm]

        permuted_b.append(wb_permuted)

    # Output layer (already had its columns permuted by the last hidden layer)
    permuted_b.append(weights_b[-1])

    return permuted_b


def align_and_report_sensitivity(
    weights_a: list[np.ndarray],
    weights_b: list[np.ndarray],
    n_seeds: int = 5,
) -> dict:
    """Run alignment with multiple tie-breaking seeds and report sensitivity.

    Parameters
    ----------
    weights_a : list of np.ndarray
    weights_b : list of np.ndarray
    n_seeds : int
        Number of different tie-breaking seeds to test.

    Returns
    -------
    dict with:
        - aligned_weights: list of aligned weight matrices (from seed 0)
        - overlaps_per_seed: list of global overlap values
        - overlap_mean, overlap_std: summary stats
    """
    overlaps = []
    first_aligned = None

    for seed in range(n_seeds):
        # Deep copy weights_b since alignment modifies it in-place
        wb_copy = [w.copy() for w in weights_b]
        aligned = align_replicas(weights_a, wb_copy, tie_breaking_seed=seed)

        if first_aligned is None:
            first_aligned = aligned

        # Compute global overlap
        q = _global_overlap(weights_a, aligned)
        overlaps.append(q)

    return {
        "aligned_weights": first_aligned,
        "overlaps_per_seed": overlaps,
        "overlap_mean": float(np.mean(overlaps)),
        "overlap_std": float(np.std(overlaps)),
    }


def _global_overlap(wa_list: list[np.ndarray], wb_list: list[np.ndarray]) -> float:
    """Compute global overlap q = (1/N) sum(w_a * w_b)."""
    numerator = 0.0
    total_weights = 0
    for wa, wb in zip(wa_list, wb_list):
        numerator += np.sum(wa * wb)
        total_weights += wa.size
    return float(numerator / total_weights)
