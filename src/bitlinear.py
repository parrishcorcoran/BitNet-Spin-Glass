"""BitLinear: ternary {-1, 0, +1} weight layer with straight-through estimator."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BitLinear(nn.Linear):
    """Linear layer whose forward pass uses ternary weights via STE.

    During forward: weights are quantized to {-1, 0, +1} scaled by mean(|w|).
    During backward: gradients flow through as if quantization didn't happen (STE).
    No bias term (bias=False enforced).
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.weight.abs().mean().clamp(min=1e-5)
        w_ternary = (self.weight / scale).round().clamp(-1, 1)
        # STE: forward uses quantized weights, backward uses original
        w_ste = self.weight + (w_ternary * scale - self.weight).detach()
        return F.linear(x, w_ste)

    def get_ternary_weights(self) -> torch.Tensor:
        """Return the actual ternary weight values {-1, 0, +1} (no scale)."""
        with torch.no_grad():
            scale = self.weight.abs().mean().clamp(min=1e-5)
            return (self.weight / scale).round().clamp(-1, 1)
