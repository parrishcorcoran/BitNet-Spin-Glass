"""Measurement 7: Binder cumulant for finite-size scaling.

g = (1/2) * (3 - <q^4> / <q^2>^2)

Crossing point of g vs epoch for different model sizes = phase transition location.
Gold standard for identifying phase transitions in finite-size systems.
"""

import numpy as np


def compute_binder_cumulant(overlaps: np.ndarray) -> float:
    """Compute Binder cumulant from pairwise overlaps.

    Parameters
    ----------
    overlaps : np.ndarray
        Array of pairwise overlap values (already aligned).

    Returns
    -------
    float
        Binder cumulant g. For Gaussian: g=0. For delta-function: g=1.
    """
    q2_mean = np.mean(overlaps ** 2)
    q4_mean = np.mean(overlaps ** 4)

    if q2_mean == 0:
        return 0.0

    return float(0.5 * (3.0 - q4_mean / (q2_mean ** 2)))


def compute_binder_vs_epoch(
    overlaps_per_epoch: dict[int, np.ndarray],
) -> tuple[list[int], list[float]]:
    """Compute Binder cumulant at each checkpoint epoch.

    Parameters
    ----------
    overlaps_per_epoch : dict[epoch -> 1D array of pairwise overlaps]

    Returns
    -------
    epochs : list of int
    g_values : list of float
    """
    epochs = sorted(overlaps_per_epoch.keys())
    g_values = [compute_binder_cumulant(overlaps_per_epoch[e]) for e in epochs]
    return epochs, g_values
