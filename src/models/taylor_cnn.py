"""Three-channel Taylor-CNN classifier (V5) with primary + secondary views.

Architecture:
    The Taylor gate models the ideal transit dip shape from phase alone.
    The CNN receives THREE stacked channels:
        Channel 0: gate output on primary phase (physics: what a transit
                   should look like, symmetric dip at phase=0).
        Channel 1: primary flux   (LC folded at the TCE epoch).
        Channel 2: secondary flux (LC folded at epoch + period/2 — any
                   secondary eclipse shows up as a dip at phase=0 here).

    Planets: channel 2 is flat noise.
    Eclipsing binaries: channel 2 has a clear dip, matching channel 1.
    The CNN can learn to compare channels 1 and 2 — if they both look
    transit-like, suppress the output; if only channel 1 looks transit-like,
    trust the classification.

Data flow with shapes (batch=B, seq_len=200):
    phase: (B, 200)  primary_flux: (B, 200)  secondary_flux: (B, 200)
        │                   │                     │
        ▼                   │                     │
    TaylorGate              │                     │
    → gate_out (B, 200)     │                     │
        │                   ▼                     ▼
    stack → (B, 3, 200)     [ch0=gate, ch1=primary, ch2=secondary]
        │
    BatchNorm1d(3)
    Conv1d(3→8, k=7) + ReLU
    Conv1d(8→16, k=5) + ReLU
    AdaptiveAvgPool → (B, 16)
        │
    Linear(16, 1) → sigmoid → p(planet)

Why the Taylor gate stays on the primary only:
    The physics-informed layer encodes "a real planet transit is a
    symmetric dip at phase=0." That statement is about the primary
    view. The secondary channel exists to contradict the primary on
    EBs — we want the CNN to learn its own filter there, not to
    re-impose the same dip template.
"""

import torch
import torch.nn as nn

from src.models.taylor_layer import TaylorGateLayer


class TaylorCNN(nn.Module):
    """Three-channel physics-informed transit classifier (V5).

    Args:
        init_amplitude: Starting value for the Taylor gate's learnable
            amplitude A. Should be near expected transit depth.
        n_filters_1: Number of filters in the first CNN layer.
        n_filters_2: Number of filters in the second CNN layer.

    Example:
        >>> model = TaylorCNN(init_amplitude=0.01)
        >>> phase = torch.linspace(-3.14, 3.14, 200).unsqueeze(0)
        >>> primary = torch.randn(1, 200) * 0.005
        >>> secondary = torch.randn(1, 200) * 0.005
        >>> prob = model(phase, primary, secondary)
    """

    def __init__(
        self,
        init_amplitude: float = 0.01,
        n_filters_1: int = 8,
        n_filters_2: int = 16,
    ) -> None:
        super().__init__()

        # --- Physics model (primary view only) ---
        self.taylor_gate = TaylorGateLayer(init_amplitude=init_amplitude)

        # --- CNN over 3 channels: [gate, primary, secondary] ---
        # BatchNorm first: gate and flux values are tiny (~0.01) while conv
        # weights initialize at ~0.1. Without normalization, gradients are
        # too small for efficient learning. BatchNorm1d(3) independently
        # normalizes each channel to zero mean / unit variance.
        self.cnn = nn.Sequential(
            nn.BatchNorm1d(3),

            # Layer 1: (B, 3, 200) → (B, 8, 200)
            nn.Conv1d(
                in_channels=3,
                out_channels=n_filters_1,
                kernel_size=7,
                padding=3,
            ),
            nn.ReLU(),

            # Layer 2: (B, 8, 200) → (B, 16, 200)
            nn.Conv1d(
                in_channels=n_filters_1,
                out_channels=n_filters_2,
                kernel_size=5,
                padding=2,
            ),
            nn.ReLU(),

            # Pool: (B, 16, 200) → (B, 16, 1)
            nn.AdaptiveAvgPool1d(1),
        )

        # --- Classifier ---
        self.classifier = nn.Linear(n_filters_2, 1)

    def forward(
        self,
        phase: torch.Tensor,
        primary_flux: torch.Tensor,
        secondary_flux: torch.Tensor,
    ) -> torch.Tensor:
        """Classify a light curve as planet vs. not-planet.

        Args:
            phase: Phase values in [-π, π], shape (batch, seq_len).
            primary_flux: Primary-view flux (folded at TCE epoch),
                baseline-subtracted, shape (batch, seq_len).
            secondary_flux: Secondary-view flux (folded at epoch +
                period/2), baseline-subtracted, shape (batch, seq_len).

        Returns:
            Planet probability, shape (batch, 1). Values in [0, 1].
        """
        gate_out = self.taylor_gate(phase)  # (B, seq_len)

        # 3-channel stack
        x = torch.stack([gate_out, primary_flux, secondary_flux], dim=1)

        cnn_out = self.cnn(x)          # (B, 16, 1)
        features = cnn_out.squeeze(2)  # (B, 16)

        logits = self.classifier(features)
        prob = torch.sigmoid(logits)

        return prob
