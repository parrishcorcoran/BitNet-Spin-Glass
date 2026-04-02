"""Measurement 2: Edwards-Anderson order parameter q_EA.

q_EA = (1/N) * sum( <w_i>^2 )

where <w_i> is the time average of weight i over the last EA_AVERAGING_EPOCHS epochs.
One value per model. Measures how "frozen" each weight position is.
"""

import numpy as np

from src.config import EA_AVERAGING_EPOCHS


def compute_edwards_anderson(
    weight_snapshots: list[np.ndarray],
    n_averaging_epochs: int = EA_AVERAGING_EPOCHS,
) -> float:
    """Compute q_EA from a sequence of weight snapshots.

    Parameters
    ----------
    weight_snapshots : list of np.ndarray
        Weight matrices at consecutive epochs. Each has the same shape.
        Uses the last n_averaging_epochs snapshots.
    n_averaging_epochs : int
        How many of the latest snapshots to average over.

    Returns
    -------
    float
        q_EA value. Higher = more frozen weights.
    """
    if len(weight_snapshots) < n_averaging_epochs:
        n_averaging_epochs = len(weight_snapshots)

    # Stack last N snapshots: shape (N, *weight_shape)
    recent = np.stack(weight_snapshots[-n_averaging_epochs:], axis=0)

    # Time average: <w_i> for each weight position
    time_avg = recent.mean(axis=0)

    # q_EA = (1/N) * sum(<w_i>^2)
    q_ea = float(np.mean(time_avg ** 2))
    return q_ea


def compute_edwards_anderson_per_layer(
    weight_snapshots_per_layer: list[list[np.ndarray]],
    n_averaging_epochs: int = EA_AVERAGING_EPOCHS,
) -> list[float]:
    """Compute q_EA separately for each layer.

    Parameters
    ----------
    weight_snapshots_per_layer : list of list of np.ndarray
        weight_snapshots_per_layer[layer_idx][epoch_idx] = weight matrix.

    Returns
    -------
    list of float
        q_EA per layer.
    """
    return [
        compute_edwards_anderson(snapshots, n_averaging_epochs)
        for snapshots in weight_snapshots_per_layer
    ]
