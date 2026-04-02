"""Measurement 1: Overlap distribution P(q).

Computes pairwise overlap between aligned replicas, both globally and per-layer.
Produces histograms with configurable bins.

PREREQUISITE: Replicas must be aligned via align.py FIRST.
"""

import numpy as np
from itertools import combinations

from src.config import OVERLAP_HISTOGRAM_BINS


def compute_pairwise_overlaps(
    aligned_weights: list[list[np.ndarray]],
) -> dict:
    """Compute all pairwise overlaps for a set of aligned replicas.

    Parameters
    ----------
    aligned_weights : list of list of np.ndarray
        aligned_weights[replica_idx][layer_idx] = weight matrix.
        All replicas must already be aligned to the SAME reference.

    Returns
    -------
    dict with:
        - global_overlaps: 1D array of all pairwise global q values
        - per_layer_overlaps: dict[layer_idx -> 1D array of pairwise q values]
        - n_pairs: number of pairs
        - n_replicas: number of replicas
    """
    n_replicas = len(aligned_weights)
    n_layers = len(aligned_weights[0])

    global_overlaps = []
    per_layer_overlaps: dict[int, list[float]] = {i: [] for i in range(n_layers)}

    for a, b in combinations(range(n_replicas), 2):
        wa = aligned_weights[a]
        wb = aligned_weights[b]

        # Global overlap
        total_dot = 0.0
        total_n = 0
        for layer_idx in range(n_layers):
            dot = np.sum(wa[layer_idx] * wb[layer_idx])
            n = wa[layer_idx].size

            # Per-layer overlap
            per_layer_overlaps[layer_idx].append(dot / n)

            total_dot += dot
            total_n += n

        global_overlaps.append(total_dot / total_n)

    return {
        "global_overlaps": np.array(global_overlaps),
        "per_layer_overlaps": {k: np.array(v) for k, v in per_layer_overlaps.items()},
        "n_pairs": len(global_overlaps),
        "n_replicas": n_replicas,
    }


def compute_overlap_histogram(
    overlaps: np.ndarray,
    bins: int = OVERLAP_HISTOGRAM_BINS,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalized histogram of overlap values.

    Returns (counts, bin_edges) where counts are probability densities.
    """
    counts, bin_edges = np.histogram(overlaps, bins=bins, density=True)
    return counts, bin_edges
