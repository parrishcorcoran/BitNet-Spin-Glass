"""Measurement 9: Effect size metrics.

KS test will always be significant with 4950 pairs. Report:
- Wasserstein distance (Earth mover's distance)
- Cohen's d on the mean overlap
- Jensen-Shannon divergence between P(q) histograms
- Benjamini-Hochberg FDR correction across all tests
"""

import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon


def wasserstein_distance(overlaps_a: np.ndarray, overlaps_b: np.ndarray) -> float:
    """Earth mover's distance between two overlap distributions."""
    return float(stats.wasserstein_distance(overlaps_a, overlaps_b))


def cohens_d(overlaps_a: np.ndarray, overlaps_b: np.ndarray) -> float:
    """Cohen's d effect size between two overlap distributions."""
    n_a, n_b = len(overlaps_a), len(overlaps_b)
    mean_a, mean_b = np.mean(overlaps_a), np.mean(overlaps_b)
    var_a, var_b = np.var(overlaps_a, ddof=1), np.var(overlaps_b, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))

    if pooled_std == 0:
        return 0.0

    return float((mean_a - mean_b) / pooled_std)


def jensen_shannon_divergence(
    overlaps_a: np.ndarray,
    overlaps_b: np.ndarray,
    bins: int = 100,
) -> float:
    """Jensen-Shannon divergence between two overlap histograms."""
    # Shared bin edges
    all_vals = np.concatenate([overlaps_a, overlaps_b])
    bin_edges = np.linspace(all_vals.min() - 1e-6, all_vals.max() + 1e-6, bins + 1)

    hist_a, _ = np.histogram(overlaps_a, bins=bin_edges, density=True)
    hist_b, _ = np.histogram(overlaps_b, bins=bin_edges, density=True)

    # Add small epsilon to avoid log(0)
    hist_a = hist_a + 1e-12
    hist_b = hist_b + 1e-12

    # Normalize to probability distributions
    hist_a = hist_a / hist_a.sum()
    hist_b = hist_b / hist_b.sum()

    return float(jensenshannon(hist_a, hist_b) ** 2)  # squared = divergence


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    p_values : np.ndarray
        Array of p-values.
    alpha : float
        Desired FDR level.

    Returns
    -------
    np.ndarray of bool
        True = significant after correction.
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    # BH threshold: p_(k) <= (k/n) * alpha
    thresholds = np.arange(1, n + 1) / n * alpha
    significant_sorted = sorted_p <= thresholds

    # Find largest k where p_(k) <= threshold
    if not np.any(significant_sorted):
        return np.zeros(n, dtype=bool)

    max_k = np.max(np.where(significant_sorted)[0])
    significant_sorted[:max_k + 1] = True
    significant_sorted[max_k + 1:] = False

    # Map back to original order
    result = np.zeros(n, dtype=bool)
    result[sorted_indices] = significant_sorted
    return result


def full_comparison(
    overlaps_fp: np.ndarray,
    overlaps_ternary: np.ndarray,
) -> dict:
    """Run all effect size metrics comparing FP vs ternary overlaps.

    Returns dict with all metrics.
    """
    ks_stat, ks_p = stats.ks_2samp(overlaps_fp, overlaps_ternary)

    return {
        "wasserstein": wasserstein_distance(overlaps_fp, overlaps_ternary),
        "cohens_d": cohens_d(overlaps_fp, overlaps_ternary),
        "jsd": jensen_shannon_divergence(overlaps_fp, overlaps_ternary),
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_p),
        "mean_fp": float(np.mean(overlaps_fp)),
        "mean_ternary": float(np.mean(overlaps_ternary)),
        "std_fp": float(np.std(overlaps_fp)),
        "std_ternary": float(np.std(overlaps_ternary)),
    }
