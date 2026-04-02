"""Measurement 6: Spin glass susceptibility.

chi_SG = N * <q^2>

where q is the aligned overlap between replica pairs.
Peak in chi_SG indicates proximity to a phase transition.
"""

import numpy as np


def compute_susceptibility(overlaps: np.ndarray, n_weights: int) -> float:
    """Compute spin glass susceptibility from pairwise overlaps.

    Parameters
    ----------
    overlaps : np.ndarray
        Array of pairwise overlap values (already aligned).
    n_weights : int
        Total number of weights N in the model.

    Returns
    -------
    float
        chi_SG = N * <q^2>
    """
    return float(n_weights * np.mean(overlaps ** 2))


def compute_susceptibility_vs_epoch(
    overlaps_per_epoch: dict[int, np.ndarray],
    n_weights: int,
) -> tuple[list[int], list[float]]:
    """Compute chi_SG at each epoch.

    Parameters
    ----------
    overlaps_per_epoch : dict[epoch -> array of pairwise overlaps]
    n_weights : int
        Total number of weights.

    Returns
    -------
    epochs : list of int
    chi_sg_values : list of float
    """
    epochs = sorted(overlaps_per_epoch.keys())
    chi_values = [
        compute_susceptibility(overlaps_per_epoch[e], n_weights)
        for e in epochs
    ]
    return epochs, chi_values
