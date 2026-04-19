"""Custom Taylor Gate Layer for physics-informed transit classification.

The Taylor layer uses the 2nd-order Taylor approximation of cos(x) as a
physically motivated gate that passes only transit-dip-shaped signals and
zeros out everything else (stellar noise, instrumental trends, etc.).

Why cos(x) instead of sin(x):
    After phase-folding, the transit dip is centered at phase = 0.
    cos(x) has its maximum at x = 0, so -A·cos(x) has its minimum
    (deepest dip) at x = 0 — exactly where the transit is.
    sin(x) has its maximum at x = π/2, which would place the gate
    dip 90 degrees away from the actual transit. Using cos(x)
    eliminates this phase misalignment.

Math:
    cos(x) ≈ 1 - x²/2   (2nd-order Taylor, accurate for |x| < π)

    Forward:  y = min(0, -A · (1 - x²/2))
    Backward: ∂y/∂A = -(1 - x²/2) · mask
              ∂y/∂x = A · x · mask
              where mask = 1 where output < 0, else 0

    Dip shape:  centered at x=0, zero-crossings at x = ±√2 ≈ ±1.41 rad
    Dip width:  2√2 ≈ 2.83 rad out of 2π total (≈ 45% of phase)

The min(0, ...) acts as a hard gate: baseline regions produce zero output
(and zero gradient), so only the transit dip contributes to learning.
"""

import torch
import torch.nn as nn
from torch.autograd import Function


class TaylorGateFunction(Function):
    """Autograd function implementing y = min(0, -A · (1 - x²/2)).

    We write a custom Function (instead of relying on autograd) so that
    the gradient computation is explicit and inspectable. This matters
    for a learning project — you can print/plot every intermediate value
    and verify the math by hand.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        A: torch.Tensor,
    ) -> torch.Tensor:
        """Compute y = min(0, -A · (1 - x²/2)).

        Args:
            ctx: Context object for saving tensors needed in backward.
            x: Phase values, shape (batch, seq_len). Should be in [-π, π].
            A: Learnable amplitude scalar, shape (1,). Controls dip depth.

        Returns:
            Gate output, same shape as x. Zero at baseline, negative in dip.
        """
        # Taylor approximation of cos(x): 1 - x²/2
        # cos(0) = 1 (maximum), so -A·cos(0) = -A (deepest dip at phase 0)
        taylor_cos = 1.0 - (x ** 2) / 2.0

        # Flip and scale: negative means "dip direction"
        # At x=0: z = -A·1 = -A (most negative → deepest dip)
        # At x=±√2: z = -A·0 = 0 (zero crossing)
        # At x=±π: z = -A·(1-π²/2) = -A·(-3.93) = +3.93A (positive → clipped)
        z = -A * taylor_cos

        # Hard gate: clamp positive values to zero
        y = torch.clamp(z, max=0.0)

        # mask = 1 where z < 0 (the dip region), 0 elsewhere
        mask = (z < 0).float()

        ctx.save_for_backward(x, A, mask)
        return y

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute gradients for x and A.

        Args:
            ctx: Context with saved tensors from forward.
            grad_output: Upstream gradient ∂L/∂y, same shape as y.

        Returns:
            Tuple of (grad_x, grad_A):
                grad_x: ∂L/∂x = grad_output · A · x · mask
                grad_A: ∂L/∂A = grad_output · (-(1 - x²/2)) · mask
                        summed over all elements (A is a scalar).
        """
        x, A, mask = ctx.saved_tensors

        # ∂z/∂x = -A · d/dx(1 - x²/2) = -A · (-x) = A·x
        # ∂y/∂x = ∂y/∂z · ∂z/∂x = mask · A · x
        grad_x = grad_output * (A * x) * mask

        # ∂z/∂A = -(1 - x²/2)
        # ∂y/∂A = ∂y/∂z · ∂z/∂A = mask · (-(1 - x²/2))
        # Sum over all elements because A is a single scalar
        taylor_cos = 1.0 - (x ** 2) / 2.0
        grad_A = (grad_output * (-taylor_cos) * mask).sum().unsqueeze(0)

        return grad_x, grad_A


class TaylorGateLayer(nn.Module):
    """Physics-informed gate layer for transit dip detection.

    Uses the Taylor expansion of cos(x) to model a symmetric dip
    centered at phase = 0 (where fold() places the transit).

    Args:
        init_amplitude: Initial value for A. Start near expected transit
            depth (e.g., 0.01 for a 1% dip). The network will adjust.

    Example:
        >>> gate = TaylorGateLayer(init_amplitude=0.01)
        >>> phase = torch.linspace(-torch.pi, torch.pi, 200)
        >>> output = gate(phase.unsqueeze(0))  # add batch dim
        >>> # output is 0 at baseline, negative at phase=0 (dip center)
    """

    def __init__(self, init_amplitude: float = 0.01) -> None:
        super().__init__()
        self.A = nn.Parameter(torch.tensor([init_amplitude]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the Taylor gate.

        Args:
            x: Phase values in [-π, π], shape (batch, seq_len).

        Returns:
            Gated output, same shape. Zero outside dip, negative inside.
        """
        return TaylorGateFunction.apply(x, self.A)
