"""V8 Taylor Gate — learnable amplitude A, curvature B, and phase offset t0.

Why add B:
    V5/V6 used cos(x) ≈ 1 - x²/2, which produces a rounded dip with no
    flat bottom. Real planet transits have a flat bottom (light curve
    plateaus during full occultation). Grazing eclipsing binaries have
    V-shaped dips with no flat bottom. Adding the next even Taylor term
    gives a family of shapes controlled by a single learnable parameter:

        cos(x) ≈ 1 - x²/2 + B · x⁴/24

        B = 1 → matches 4th-order cos, flatter bottom (planet-like U-shape)
        B = 0 → V5 shape (intermediate)
        B < 0 → sharper V (grazing-EB-like)

Why learnable t0:
    Folding is done at the archive epoch, which may be imperfectly measured.
    A small t0 offset lets the gate align to where the dip actually is.
    Kept tiny in practice (< 0.1 rad); we clamp phase-t0 to [-π, π] so the
    gate always sees valid input.

Custom autograd Function:
    Implementing backward by hand keeps the math inspectable for gradcheck.
    Let f(x) = 1 - x²/2 + B · x⁴/24 and z = -A · f(x).
    y = min(0, z) = z · mask, where mask = 1 if z < 0 else 0.

        ∂z/∂x = -A · f'(x) = -A · (-x + B · x³/6) = A · (x - B · x³/6)
        ∂z/∂A = -f(x) = -(1 - x²/2 + B · x⁴/24)
        ∂z/∂B = -A · x⁴/24

    The Module computes x = clamp(phase - t0, -π, π) and passes x, A, B
    into the Function. Autograd handles t0's chain rule automatically
    (∂x/∂t0 = -1), which is checked by gradcheck against the analytic
    expression above.
"""

import math

import torch
import torch.nn as nn
from torch.autograd import Function


class TaylorGateV8Function(Function):
    """Autograd function for y = min(0, -A · (1 - x²/2 + B · x⁴/24))."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
    ) -> torch.Tensor:
        f = 1.0 - (x ** 2) / 2.0 + B * (x ** 4) / 24.0
        z = -A * f
        y = torch.clamp(z, max=0.0)
        mask = (z < 0).to(dtype=x.dtype)
        ctx.save_for_backward(x, A, B, mask)
        return y

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, A, B, mask = ctx.saved_tensors

        # ∂y/∂x = mask · A · (x - B · x³/6)
        grad_x = grad_output * A * (x - B * (x ** 3) / 6.0) * mask

        # ∂y/∂A = mask · -(1 - x²/2 + B · x⁴/24)  (summed → scalar grad)
        f = 1.0 - (x ** 2) / 2.0 + B * (x ** 4) / 24.0
        grad_A = (grad_output * (-f) * mask).sum().reshape(A.shape)

        # ∂y/∂B = mask · -A · (x⁴/24)  (summed → scalar grad)
        grad_B = (grad_output * (-A * (x ** 4) / 24.0) * mask).sum().reshape(B.shape)

        return grad_x, grad_A, grad_B


class TaylorGateLayerV8(nn.Module):
    """V8 gate: learnable A (amplitude), B (curvature), t0 (phase offset).

    Args:
        init_amplitude: Starting A (transit depth scale).
        init_B:         Starting B. 1.0 = flat-bottom (planet prior);
                        0.0 = quadratic (V5 shape); negative = V-shape.
        init_t0:        Starting phase offset. Keep near 0 (archive epoch
                        is already a reasonable fold center).

    Forward input:
        phase: shape (batch, seq_len), expected in [-π, π].
    """

    def __init__(
        self,
        init_amplitude: float = 0.01,
        init_B: float = 1.0,
        init_t0: float = 0.0,
    ) -> None:
        super().__init__()
        self.A = nn.Parameter(torch.tensor([init_amplitude]))
        self.B = nn.Parameter(torch.tensor([init_B]))
        self.t0 = nn.Parameter(torch.tensor([init_t0]))

    def forward(self, phase: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(phase - self.t0, min=-math.pi, max=math.pi)
        return TaylorGateV8Function.apply(x, self.A, self.B)
