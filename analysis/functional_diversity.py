"""Measurement 10: Functional diversity check.

After alignment, compare test accuracy across replicas.
If all replicas have identical accuracy AND P(q) ~ 1, they've all found the
same solution modulo permutation → Entezari conjecture holds, no genuine RSB.

This is still a publishable finding for ternary networks.
"""

import numpy as np


def compute_functional_diversity(
    test_accuracies: list[float],
    global_overlaps: np.ndarray,
) -> dict:
    """Assess functional diversity across replicas.

    Parameters
    ----------
    test_accuracies : list of float
        Test accuracy for each replica.
    global_overlaps : np.ndarray
        Pairwise global overlaps (after alignment).

    Returns
    -------
    dict with:
        - acc_mean, acc_std: accuracy statistics
        - acc_range: max - min accuracy
        - overlap_mean, overlap_std: overlap statistics
        - all_same_solution: bool, True if acc_std < 0.001 and mean overlap > 0.95
        - entezari_holds: bool, heuristic check
    """
    accs = np.array(test_accuracies)

    acc_mean = float(np.mean(accs))
    acc_std = float(np.std(accs))
    acc_range = float(np.max(accs) - np.min(accs))

    overlap_mean = float(np.mean(global_overlaps))
    overlap_std = float(np.std(global_overlaps))

    # Heuristic: all replicas found the same solution
    all_same = acc_std < 0.001 and overlap_mean > 0.95
    entezari = overlap_mean > 0.9

    return {
        "acc_mean": acc_mean,
        "acc_std": acc_std,
        "acc_range": acc_range,
        "overlap_mean": overlap_mean,
        "overlap_std": overlap_std,
        "all_same_solution": bool(all_same),
        "entezari_holds": bool(entezari),
    }
