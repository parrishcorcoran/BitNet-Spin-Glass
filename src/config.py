"""Central configuration for BitNet Spin Glass experiments.

All hyperparameters, model sizes, and experiment settings live here.
No magic numbers anywhere else in the codebase.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class WeightMode(Enum):
    FP = "fp"
    TERNARY = "ternary"


class ActivationMode(Enum):
    RELU = "relu"
    SIGN = "sign"


class DatasetName(Enum):
    MNIST = "mnist"
    CIFAR10 = "cifar10"
    RANDOM_LABELS_MNIST = "random_labels_mnist"


# ---------------------------------------------------------------------------
# Model size configs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSize:
    name: str
    hidden_layers: int
    width: int
    approx_params: int
    overparam_ratio: float


MODEL_SIZES: dict[str, ModelSize] = {
    "XS": ModelSize("XS", hidden_layers=2, width=128, approx_params=110_000, overparam_ratio=1.8),
    "S":  ModelSize("S",  hidden_layers=3, width=256, approx_params=260_000, overparam_ratio=4.3),
    "M":  ModelSize("M",  hidden_layers=3, width=512, approx_params=530_000, overparam_ratio=8.8),
    "L":  ModelSize("L",  hidden_layers=4, width=512, approx_params=790_000, overparam_ratio=13.0),
    "XL": ModelSize("XL", hidden_layers=6, width=1024, approx_params=7_000_000, overparam_ratio=117.0),
}


# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Optimizer
    optimizer: str = "adam"
    betas: tuple[float, float] = (0.9, 0.95)
    batch_size: int = 128
    gradient_clip_max_norm: float = 1.0

    # Weight-mode-dependent defaults (set by from_modes())
    lr: float = 3e-4
    weight_decay: float = 0.01
    weight_decay_off_epoch: Optional[int] = None  # None = never turn off

    # Schedule
    lr_schedule: str = "cosine"

    # Epochs (dataset-dependent)
    epochs: int = 200  # MNIST default; CIFAR-10 uses 500

    # Init
    init: str = "kaiming_normal"

    @classmethod
    def from_modes(
        cls,
        weight_mode: WeightMode,
        dataset: DatasetName,
    ) -> "TrainConfig":
        """Return the correct hyperparameters for a (weight, dataset) pair."""
        epochs = 500 if dataset == DatasetName.CIFAR10 else 200

        if weight_mode == WeightMode.TERNARY:
            return cls(
                lr=1e-3,
                weight_decay=0.01,
                weight_decay_off_epoch=50,
                epochs=epochs,
            )
        else:
            return cls(
                lr=3e-4,
                weight_decay=0.01,
                weight_decay_off_epoch=None,
                epochs=epochs,
            )


# ---------------------------------------------------------------------------
# Checkpoint schedule
# ---------------------------------------------------------------------------

def checkpoint_epochs(total_epochs: int) -> list[int]:
    """Return sorted list of epochs at which to save checkpoints."""
    # Epoch 0 = random init (ALWAYS saved)
    epochs = {0}

    # Epochs 1-20: every epoch
    for e in range(1, min(21, total_epochs + 1)):
        epochs.add(e)

    # Coarser schedule after that
    for e in [25, 30, 40, 50, 75, 100, 150, 200, 300, 400, 500]:
        if e <= total_epochs:
            epochs.add(e)

    return sorted(epochs)


# ---------------------------------------------------------------------------
# Replica counts
# ---------------------------------------------------------------------------

# Primary config (M size): 100 replicas per dataset
PRIMARY_REPLICAS: int = 100
# Other sizes: 30 replicas
SCALING_REPLICAS: int = 30
# Autocorrelation runs (save ALL epochs): 10 replicas
AUTOCORRELATION_REPLICAS: int = 10

PRIMARY_SIZE: str = "M"


# ---------------------------------------------------------------------------
# Four experiment configurations
# ---------------------------------------------------------------------------

EXPERIMENT_CONFIGS: list[tuple[WeightMode, ActivationMode]] = [
    (WeightMode.FP, ActivationMode.RELU),        # Practical baseline
    (WeightMode.TERNARY, ActivationMode.RELU),    # BitNet-like practical
    (WeightMode.FP, ActivationMode.SIGN),          # Ising-exact baseline
    (WeightMode.TERNARY, ActivationMode.SIGN),     # Full Ising analogy
]


# ---------------------------------------------------------------------------
# Learning rate sweep (Phase 6: temperature scanning)
# ---------------------------------------------------------------------------

LR_SWEEP_VALUES: list[float] = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
LR_SWEEP_REPLICAS: int = 100
LR_SWEEP_SIZE: str = "M"
LR_SWEEP_DATASET: DatasetName = DatasetName.MNIST


# ---------------------------------------------------------------------------
# Analysis parameters
# ---------------------------------------------------------------------------

# Frustration
FRUSTRATION_GAUGE_TRIALS: int = 5000  # 5000 for publication quality
FRUSTRATION_GAUGE_TRIALS_QUICK: int = 1000  # For quick checks

# Overlap histograms
OVERLAP_HISTOGRAM_BINS: int = 100

# Edwards-Anderson: average over last N epochs
EA_AVERAGING_EPOCHS: int = 20

# Hungarian alignment tie-breaking seeds
ALIGNMENT_TIE_BREAKING_SEEDS: int = 5

# Figure style
FIGURE_DPI: int = 300
FIGURE_WIDTH_INCHES: float = 3.5  # single-column


# ---------------------------------------------------------------------------
# Dataset dimensions
# ---------------------------------------------------------------------------

DATASET_INPUT_DIM: dict[str, int] = {
    "mnist": 784,           # 28x28 flattened
    "random_labels_mnist": 784,
    "cifar10": 3072,        # 32x32x3 flattened
}

DATASET_OUTPUT_DIM: int = 10  # All datasets have 10 classes


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_DIR: str = "results"
FIGURES_DIR: str = "figures"
CHECKPOINTS_DIR: str = "results/checkpoints"
