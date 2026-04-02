"""Measurement 3: Two-time autocorrelation C(t_w + t, t_w).

C(t_w, t) = (1/N) * sum( w_i(t_w + t) * w_i(t_w) )

If curves for different t_w don't collapse, that's aging = glass behavior.
"""

import numpy as np


def compute_autocorrelation(
    weight_snapshots: list[np.ndarray],
    t_w: int,
    max_t: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute C(t_w + t, t_w) for a range of t values.

    Parameters
    ----------
    weight_snapshots : list of np.ndarray
        weight_snapshots[epoch] = flattened weight vector at that epoch.
    t_w : int
        Waiting time (epoch index).
    max_t : int or None
        Maximum lag. If None, uses all available epochs after t_w.

    Returns
    -------
    t_values : np.ndarray
        Array of t (lag) values.
    c_values : np.ndarray
        Corresponding C(t_w + t, t_w) values.
    """
    total_epochs = len(weight_snapshots)
    if max_t is None:
        max_t = total_epochs - t_w - 1

    max_t = min(max_t, total_epochs - t_w - 1)
    if max_t <= 0:
        return np.array([]), np.array([])

    w_tw = weight_snapshots[t_w].flatten()
    n = w_tw.size

    t_values = np.arange(1, max_t + 1)
    c_values = np.empty(len(t_values))

    for i, t in enumerate(t_values):
        w_tw_t = weight_snapshots[t_w + t].flatten()
        c_values[i] = np.dot(w_tw, w_tw_t) / n

    return t_values, c_values


def compute_autocorrelation_curves(
    weight_snapshots: list[np.ndarray],
    waiting_times: list[int],
    max_t: int | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Compute autocorrelation curves for multiple waiting times.

    Parameters
    ----------
    weight_snapshots : list of np.ndarray
        Weight vectors at each epoch.
    waiting_times : list of int
        List of t_w values to compute.
    max_t : int or None
        Maximum lag.

    Returns
    -------
    dict mapping t_w -> (t_values, c_values)
    """
    curves = {}
    for t_w in waiting_times:
        if t_w < len(weight_snapshots):
            t_vals, c_vals = compute_autocorrelation(weight_snapshots, t_w, max_t)
            if len(t_vals) > 0:
                curves[t_w] = (t_vals, c_vals)
    return curves
