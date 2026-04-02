"""Measurement 5: Vacancy fraction (per-layer, per-epoch).

For ternary models: fraction of weights that are exactly zero.
For FP models: fraction of weights that round to zero.
"""

import numpy as np


def compute_vacancy_fraction(weight_matrix: np.ndarray, threshold: float = 0.0) -> float:
    """Fraction of weights that are zero (or below threshold).

    Parameters
    ----------
    weight_matrix : np.ndarray
        Weight matrix.
    threshold : float
        Count weights with |w| <= threshold as vacant. Default 0.0 (exact zero).

    Returns
    -------
    float
        Vacancy fraction in [0, 1].
    """
    total = weight_matrix.size
    if total == 0:
        return 0.0
    vacant = np.sum(np.abs(weight_matrix) <= threshold)
    return float(vacant / total)


def compute_vacancy_profile(
    weight_matrices: list[np.ndarray],
    threshold: float = 0.0,
) -> list[float]:
    """Compute vacancy fraction for each layer.

    Parameters
    ----------
    weight_matrices : list of np.ndarray
        One matrix per layer.
    threshold : float
        Vacancy threshold.

    Returns
    -------
    list of float
        Vacancy fraction per layer.
    """
    return [compute_vacancy_fraction(w, threshold) for w in weight_matrices]
