"""Measurement 8: Ultrametricity test.

For all triples of aligned replicas, sort {q_ab, q_ac, q_bc}.
Ultrametric property: the two smallest overlaps should be equal.
"""

import numpy as np
from itertools import combinations


def test_ultrametricity(
    pairwise_overlaps: dict[tuple[int, int], float],
    n_replicas: int,
    tolerance: float = 0.05,
) -> dict:
    """Test ultrametric property on all replica triples.

    Parameters
    ----------
    pairwise_overlaps : dict[(i, j) -> float]
        Overlap between replica pair (i, j) with i < j.
    n_replicas : int
        Total number of replicas.
    tolerance : float
        Relative tolerance for "two smallest are equal".

    Returns
    -------
    dict with:
        - ultrametric_fraction: fraction of triples satisfying ultrametricity
        - gap_distribution: array of |q_min - q_mid| / |q_max - q_min + 1e-10|
        - n_triples: total triples tested
    """
    gaps = []
    ultrametric_count = 0
    n_triples = 0

    for a, b, c in combinations(range(n_replicas), 3):
        q_ab = pairwise_overlaps.get((min(a, b), max(a, b)), 0.0)
        q_ac = pairwise_overlaps.get((min(a, c), max(a, c)), 0.0)
        q_bc = pairwise_overlaps.get((min(b, c), max(b, c)), 0.0)

        q_sorted = sorted([q_ab, q_ac, q_bc])
        q_min, q_mid, q_max = q_sorted

        gap = abs(q_mid - q_min)
        spread = abs(q_max - q_min) + 1e-10
        relative_gap = gap / spread

        gaps.append(relative_gap)

        if relative_gap < tolerance:
            ultrametric_count += 1
        n_triples += 1

    if n_triples == 0:
        return {
            "ultrametric_fraction": 0.0,
            "gap_distribution": np.array([]),
            "n_triples": 0,
        }

    return {
        "ultrametric_fraction": ultrametric_count / n_triples,
        "gap_distribution": np.array(gaps),
        "n_triples": n_triples,
    }
