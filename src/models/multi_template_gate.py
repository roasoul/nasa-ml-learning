"""V10 Multi-Template Taylor Gate Bank.

Five parallel gates, each with fixed morphology and a learnable amplitude.
The idea: a single global B (V8/V9) can only learn a population-averaged
shape. Replacing it with N parallel gates — each one a *specific* dip
morphology — lets the CNN pick per-sample which gate fired.

Gate menu:
    G1  U-shape planet          y = min(0, -A1·(1 - x²/2 + x⁴/24))
    G2  V-shape                  y = min(0, -A2·(1 - x²/2))
    G3  inverted secondary       y = max(0,  A3·(1 - x²/2))
    G4  asymmetric ingress       y = min(0, -A4·(x - x³/6))
    G5  narrow Gaussian          y = min(0, -A5·exp(-x² / 0.5))

Each A_i is clamped internally to min=0.001 (via `.clamp` on the forward
pass, so gradient still flows through for A_i > 0.001). No need for a
post-step clamp — the constraint lives inside the forward.

Output shape: (batch, 5, seq_len). The V10 CNN stacks this with
[primary, secondary, odd/even] to get a 8-channel input.
"""

import math

import torch
import torch.nn as nn


_GAUSS_SCALE = 0.5   # exp(-x² / _GAUSS_SCALE); 0.5 gives a narrow spike


class MultiTemplateGateBank(nn.Module):
    def __init__(self, init_amplitude: float = 0.01) -> None:
        super().__init__()
        self.A1 = nn.Parameter(torch.tensor([init_amplitude]))
        self.A2 = nn.Parameter(torch.tensor([init_amplitude]))
        self.A3 = nn.Parameter(torch.tensor([init_amplitude]))
        self.A4 = nn.Parameter(torch.tensor([init_amplitude]))
        self.A5 = nn.Parameter(torch.tensor([init_amplitude]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the 5-gate bank.

        Args:
            x: Phase, shape (batch, seq_len). Clamped to [-π, π] inside.

        Returns:
            Gate outputs stacked along a new channel dim — shape
            (batch, 5, seq_len). Ordering: [G1, G2, G3, G4, G5].
        """
        x = x.clamp(min=-math.pi, max=math.pi)

        A1 = self.A1.clamp(min=0.001)
        A2 = self.A2.clamp(min=0.001)
        A3 = self.A3.clamp(min=0.001)
        A4 = self.A4.clamp(min=0.001)
        A5 = self.A5.clamp(min=0.001)

        # G1 — planet U-shape. Fixed curvature B=1.
        g1 = torch.clamp(-A1 * (1.0 - x ** 2 / 2.0 + x ** 4 / 24.0), max=0.0)

        # G2 — V-shape (quadratic, no flat bottom). Fixed B=0.
        g2 = torch.clamp(-A2 * (1.0 - x ** 2 / 2.0), max=0.0)

        # G3 — inverted secondary eclipse bump. Fires positive.
        g3 = torch.clamp(A3 * (1.0 - x ** 2 / 2.0), min=0.0)

        # G4 — asymmetric ingress (odd function). Fires only for x > 0.
        g4 = torch.clamp(-A4 * (x - x ** 3 / 6.0), max=0.0)

        # G5 — narrow Gaussian (spot crossing).
        g5 = torch.clamp(-A5 * torch.exp(-x ** 2 / _GAUSS_SCALE), max=0.0)

        return torch.stack([g1, g2, g3, g4, g5], dim=1)

    def amplitudes(self) -> dict[str, float]:
        """Return current clamped A values as floats — for logging."""
        return {
            "A1": float(self.A1.clamp(min=0.001).item()),
            "A2": float(self.A2.clamp(min=0.001).item()),
            "A3": float(self.A3.clamp(min=0.001).item()),
            "A4": float(self.A4.clamp(min=0.001).item()),
            "A5": float(self.A5.clamp(min=0.001).item()),
        }
