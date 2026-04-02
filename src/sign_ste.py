"""Sign activation with clipped straight-through estimator."""

import torch


class SignSTE(torch.autograd.Function):
    """sign(x) in forward, clipped STE in backward.

    Forward: returns sign(x) in {-1, 0, +1}.
    Backward: passes gradient only where |x| <= 1 (clipped STE).
    """

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return torch.sign(x)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        grad = grad_output.clone()
        grad[x.abs() > 1] = 0
        return grad


def sign_ste(x: torch.Tensor) -> torch.Tensor:
    """Functional wrapper for SignSTE."""
    return SignSTE.apply(x)
