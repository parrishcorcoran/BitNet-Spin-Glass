"""Measurement 4: Signed-graph frustration index.

Uses random gauge search + greedy local optimization.
NEVER computes via W^TW eigenvalues (always positive semidefinite = useless).

For ternary weights: minimum 5000 gauge trials for publication quality.
"""

import numpy as np


def compute_frustration_index(
    weight_matrix: np.ndarray,
    gauge_trials: int = 5000,
    greedy_iterations: int = 50,
    seed: int = 0,
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

    sign_matrix = np.sign(W)  # {-1, 0, +1}

    rng = np.random.default_rng(seed)
    best_frustrated = total_edges  # worst case

    for _ in range(gauge_trials):
        # Random gauge assignment: s_row in {-1, +1}^m, s_col in {-1, +1}^n
        s_row = rng.choice([-1, 1], size=m).astype(np.float32)
        s_col = rng.choice([-1, 1], size=n).astype(np.float32)

        # Greedy local search
        for _ in range(greedy_iterations):
            improved = False

            # Optimize row gauges
            for i in range(m):
                # Count satisfied edges for s_row[i] = +1 vs -1
                contrib = sign_matrix[i, :] * s_col  # (n,)
                positive_satisfied = int(np.sum((contrib > 0) & nonzero_mask[i, :]))
                negative_satisfied = int(np.sum((contrib < 0) & nonzero_mask[i, :]))

                if s_row[i] > 0 and negative_satisfied > positive_satisfied:
                    s_row[i] = -1
                    improved = True
                elif s_row[i] < 0 and positive_satisfied > negative_satisfied:
                    s_row[i] = 1
                    improved = True

            # Optimize column gauges
            for j in range(n):
                contrib = sign_matrix[:, j] * s_row  # (m,)
                positive_satisfied = int(np.sum((contrib > 0) & nonzero_mask[:, j]))
                negative_satisfied = int(np.sum((contrib < 0) & nonzero_mask[:, j]))

                if s_col[j] > 0 and negative_satisfied > positive_satisfied:
                    s_col[j] = -1
                    improved = True
                elif s_col[j] < 0 and positive_satisfied > negative_satisfied:
                    s_col[j] = 1
                    improved = True

            if not improved:
                break

        # Count frustrated edges
        gauge_product = np.outer(s_row, s_col)
        satisfied = sign_matrix * gauge_product  # positive = satisfied
        frustrated = int(np.sum((satisfied < 0) & nonzero_mask))
        best_frustrated = min(best_frustrated, frustrated)

    return best_frustrated / total_edges
