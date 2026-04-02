"""Plain MLP model for spin glass experiments.

Architecture:
  Input → Linear → [BN if sign] → Activation → ... → Linear → Output

No skip connections, no dropout, no bias terms.
Batch normalization ONLY with sign activations (required for convergence).
"""

import torch
import torch.nn as nn

from src.bitlinear import BitLinear
from src.sign_ste import sign_ste
from src.config import (
    WeightMode,
    ActivationMode,
    ModelSize,
    MODEL_SIZES,
    DATASET_INPUT_DIM,
    DATASET_OUTPUT_DIM,
)


class SpinGlassMLP(nn.Module):
    """Configurable MLP for spin glass replica experiments.

    Parameters
    ----------
    size_name : str
        Key into MODEL_SIZES (e.g. "M").
    weight_mode : WeightMode
        FP or TERNARY.
    activation_mode : ActivationMode
        RELU or SIGN.
    input_dim : int
        Flattened input size (784 for MNIST, 3072 for CIFAR-10).
    """

    def __init__(
        self,
        size_name: str,
        weight_mode: WeightMode,
        activation_mode: ActivationMode,
        input_dim: int = 784,
    ) -> None:
        super().__init__()
        self.size_name = size_name
        self.weight_mode = weight_mode
        self.activation_mode = activation_mode

        size: ModelSize = MODEL_SIZES[size_name]
        self.width = size.width
        self.num_hidden = size.hidden_layers

        LinearClass = BitLinear if weight_mode == WeightMode.TERNARY else _FPLinearNoBias
        use_bn = activation_mode == ActivationMode.SIGN

        layers: list[nn.Module] = []

        # Input layer
        in_dim = input_dim
        for i in range(size.hidden_layers):
            out_dim = size.width
            layers.append(LinearClass(in_dim, out_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(_ActivationModule(activation_mode))
            in_dim = out_dim

        # Output layer (always FP linear, no activation, no BN)
        # Use FP linear for output regardless of weight mode — standard practice
        layers.append(_FPLinearNoBias(in_dim, DATASET_OUTPUT_DIM))

        self.layers = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming normal initialization for all linear layers."""
        for m in self.modules():
            if isinstance(m, (nn.Linear, BitLinear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten input
        x = x.view(x.size(0), -1)
        return self.layers(x)

    def get_weight_matrices(self) -> list[torch.Tensor]:
        """Return list of weight matrices (detached) for all linear layers."""
        weights = []
        for m in self.modules():
            if isinstance(m, (nn.Linear, BitLinear)):
                weights.append(m.weight.detach().clone())
        return weights

    def get_ternary_weight_matrices(self) -> list[torch.Tensor]:
        """Return ternary-quantized weight matrices for BitLinear layers.

        For FP linear layers, returns the raw weights.
        """
        weights = []
        for m in self.modules():
            if isinstance(m, BitLinear):
                weights.append(m.get_ternary_weights())
            elif isinstance(m, nn.Linear):
                weights.append(m.weight.detach().clone())
        return weights

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class _FPLinearNoBias(nn.Linear):
    """Full-precision linear layer with no bias."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features, bias=False)


class _ActivationModule(nn.Module):
    """Wraps activation function as a module for nn.Sequential."""

    def __init__(self, mode: ActivationMode) -> None:
        super().__init__()
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == ActivationMode.RELU:
            return torch.relu(x)
        else:
            return sign_ste(x)
