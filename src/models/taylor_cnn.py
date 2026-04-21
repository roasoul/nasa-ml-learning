"""Four-channel Taylor-CNN classifier (V6) — primary + secondary + odd/even diff.

Architecture:
    The Taylor gate models the ideal transit dip shape from phase alone.
    The CNN receives FOUR stacked channels:
        Channel 0: gate output on primary phase — physics prior, a symmetric
                   dip at phase=0.
        Channel 1: primary flux   (LC folded at TCE epoch).
        Channel 2: secondary flux (LC folded at epoch + period/2 — secondary
                   eclipse shows up as dip at phase=0 for correct-period EBs).
        Channel 3: odd_fold - even_fold diff — alternating depth shows up as
                   a non-zero dip at phase=0 for period-doubled EBs; flat for
                   planets and correct-period EBs.

    Planet: channels 2 and 3 are both flat noise; channel 1 has a dip.
    EB correct period: channel 2 has a dip; channel 3 is flat.
    EB doubled period: channel 2 is flat; channel 3 has a dip.

Data flow with shapes (batch=B, seq_len=200):
    phase, primary, secondary, odd_even : each (B, 200)
        │       │        │         │
        ▼       │        │         │
    TaylorGate  │        │         │
    gate_out (B,200)     │         │
        │       ▼        ▼         ▼
    stack → (B, 4, 200)  [ch0=gate, ch1=primary, ch2=secondary, ch3=odd/even]
        │
    BatchNorm1d(4)
    Conv1d(4→8, k=7) + ReLU
    Conv1d(8→16, k=5) + ReLU
    AdaptiveAvgPool → (B, 16)
        │
    Linear(16, 1) → sigmoid → p(planet)

Taylor gate stays on primary only: the physics-informed layer encodes
"a real planet transit is a symmetric dip at phase=0" — a statement
about the primary view. The other channels exist to contradict the
primary; we want the CNN to learn its own filters there.
"""

import torch
import torch.nn as nn

from src.models.taylor_layer import TaylorGateLayer


class TaylorCNN(nn.Module):
    """Four-channel physics-informed transit classifier (V6).

    Args:
        init_amplitude: Starting value for the Taylor gate's learnable A.
        n_filters_1: Filters in the first Conv1d layer.
        n_filters_2: Filters in the second Conv1d layer.

    Example:
        >>> model = TaylorCNN(init_amplitude=0.01)
        >>> phase = torch.linspace(-3.14, 3.14, 200).unsqueeze(0)
        >>> p = torch.randn(1, 200) * 0.005
        >>> s = torch.randn(1, 200) * 0.005
        >>> oe = torch.randn(1, 200) * 0.005
        >>> prob = model(phase, p, s, oe)
    """

    def __init__(
        self,
        init_amplitude: float = 0.01,
        n_filters_1: int = 8,
        n_filters_2: int = 16,
    ) -> None:
        super().__init__()

        self.taylor_gate = TaylorGateLayer(init_amplitude=init_amplitude)

        # CNN over 4 channels: [gate, primary, secondary, odd/even_diff]
        self.cnn = nn.Sequential(
            nn.BatchNorm1d(4),
            nn.Conv1d(
                in_channels=4,
                out_channels=n_filters_1,
                kernel_size=7,
                padding=3,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=n_filters_1,
                out_channels=n_filters_2,
                kernel_size=5,
                padding=2,
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Linear(n_filters_2, 1)

    def forward(
        self,
        phase: torch.Tensor,
        primary_flux: torch.Tensor,
        secondary_flux: torch.Tensor,
        odd_even_diff: torch.Tensor,
    ) -> torch.Tensor:
        """Classify a light curve as planet vs. not-planet.

        Args:
            phase: Phase in [-π, π], shape (batch, seq_len).
            primary_flux:   Primary-view flux, shape (batch, seq_len).
            secondary_flux: Secondary-view flux (fold at epoch + period/2),
                            shape (batch, seq_len).
            odd_even_diff:  Odd-fold minus even-fold, shape (batch, seq_len).

        Returns:
            Planet probability, shape (batch, 1). Values in [0, 1].
        """
        gate_out = self.taylor_gate(phase)

        x = torch.stack(
            [gate_out, primary_flux, secondary_flux, odd_even_diff], dim=1
        )

        cnn_out = self.cnn(x)
        features = cnn_out.squeeze(2)

        logits = self.classifier(features)
        return torch.sigmoid(logits)
