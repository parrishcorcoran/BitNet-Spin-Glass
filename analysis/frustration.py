"""Measurement 4: Signed-graph frustration index.

Uses random gauge search + greedy local optimization.
NEVER computes via W^TW eigenvalues (always positive semidefinite = useless).

For ternary weights: minimum 5000 gauge trials for publication quality.

Fully vectorized with NumPy — no Python loops over rows/columns.
"""

import numpy as np
from multiprocessing import Pool, cpu_count


def compute_frustration_index(
    weight_matrix: np.ndarray,
    gauge_trials: int = 5000,
    greedy_iterations: int = 50,
    seed: int = 0,
    n_workers: int | None = None,
) -> float:
    """Compute frustration index for a weight matrix via random gauge search.

    The frustration index is the minimum fraction of edges that cannot be
    simultaneously satisfied by any gauge assignment.

    Parameters
    ----------
    weight_matrix : np.ndarray, shape (m, n)
        The weight matrix. We use sign(W_ij) for nonzero entries.
    gauge_trials : int
        Number of random starting configurations.
    greedy_iterations : int
        Max steps of greedy local search per trial.
    seed : int
        Random seed.
    n_workers : int or None
        Number of parallel workers. None = use all cores.

    Returns
    -------
    float
        Frustration index in [0, 0.5]. Lower = less frustrated.
    """
    W = weight_matrix.astype(np.float32)
    m, n = W.shape

    # Extract sign matrix for nonzero weights
    nonzero_mask = W != 0
    total_edges = int(nonzero_mask.sum())
    if total_edges == 0:
        return 0.0

    sign_matrix = np.sign(W)

    # For small matrices, run single-threaded
    if m * n < 10000 or gauge_trials < 100:
        return _run_trials_vectorized(
            sign_matrix, nonzero_mask, total_edges,
            gauge_trials, greedy_iterations, seed,
        )

    # For large matrices, parallelize across cores
    if n_workers is None:
        n_workers = min(cpu_count(), 16)

    trials_per_worker = gauge_trials // n_workers
    remainder = gauge_trials % n_workers

    args = []
    for i in range(n_workers):
        t = trials_per_worker + (1 if i < remainder else 0)
        if t > 0:
            args.append((sign_matrix, nonzero_mask, total_edges,
                         t, greedy_iterations, seed + i * 10000))

    with Pool(n_workers) as pool:
        results = pool.starmap(_run_trials_vectorized, args)

    best_frust = min(results)
    return best_frust


def _run_trials_vectorized(
    sign_matrix: np.ndarray,
    nonzero_mask: np.ndarray,
    total_edges: int,
    gauge_trials: int,
    greedy_iterations: int,
    seed: int,
) -> float:
    """Run gauge trials with fully vectorized greedy optimization."""
    m, n = sign_matrix.shape
    rng = np.random.default_rng(seed)
    best_frustrated = total_edges

    # Precompute: sign_matrix masked (zeros where no edge)
    # For row optimization: we need sign_matrix * s_col for all rows at once
    # For col optimization: we need sign_matrix.T * s_row for all cols at once

    for _ in range(gauge_trials):
        # Random gauge assignment
        s_row = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=m)
        s_col = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=n)

        # Greedy local search — fully vectorized
        for _ in range(greedy_iterations):
            improved = False

            # === Optimize ALL row gauges at once ===
            # For each row i, compute: sum of sign_matrix[i,:] * s_col where nonzero
            # If this sum is positive, s_row[i] should be +1; if negative, -1
            # score[i] = sum_j( sign_matrix[i,j] * s_col[j] ) for nonzero edges
            row_scores = (sign_matrix * s_col[np.newaxis, :]).sum(axis=1)  # (m,)

            # Optimal s_row: sign of row_scores
            new_s_row = np.where(row_scores >= 0, np.float32(1.0), np.float32(-1.0))
            if not np.array_equal(new_s_row, s_row):
                improved = True
                s_row = new_s_row

            # === Optimize ALL column gauges at once ===
            col_scores = (sign_matrix * s_row[:, np.newaxis]).sum(axis=0)  # (n,)

            new_s_col = np.where(col_scores >= 0, np.float32(1.0), np.float32(-1.0))
            if not np.array_equal(new_s_col, s_col):
                improved = True
                s_col = new_s_col

            if not improved:
                break

        # Count frustrated edges
        gauge_product = s_row[:, np.newaxis] * s_col[np.newaxis, :]
        satisfied = sign_matrix * gauge_product
        frustrated = int(np.sum((satisfied < 0) & nonzero_mask))
        best_frustrated = min(best_frustrated, frustrated)

    return best_frustrated / total_edges
