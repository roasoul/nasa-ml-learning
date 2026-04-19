"""Two-channel Taylor-CNN classifier for transit/no-transit detection.

Architecture:
    The Taylor gate models the ideal transit dip shape from phase alone.
    The CNN receives TWO channels stacked together:
        Channel 0: gate output (physics model of what a transit looks like)
        Channel 1: raw flux (actual observed data)

    For a real transit, channels 0 and 1 look similar in the dip region.
    For a non-transit, channel 0 still shows a dip but channel 1 is noise.
    The CNN learns this comparison through its filters — early filters can
    learn "subtract channel 0 from channel 1" (the residual) or "multiply
    them" (correlation), or whatever comparison works best.

Data flow with shapes (batch=B, seq_len=200):
    phase: (B, 200)  flux: (B, 200)
        │                   │
        ▼                   │
    TaylorGate              │
    → gate_out (B, 200)     │
        │                   │
        ▼                   ▼
    stack → (B, 2, 200)     [ch0=gate, ch1=flux]
        │
    Conv1d(2→8, k=7) + ReLU
    Conv1d(8→16, k=5) + ReLU
    AdaptiveAvgPool → (B, 16)
        │
    Linear(16, 1) → sigmoid → p(transit)

Why this works:
    A Conv1d filter with 2 input channels computes:
        out[t] = w0 · gate[t-k:t+k] + w1 · flux[t-k:t+k] + bias
    If the network learns w0 ≈ -1 and w1 ≈ +1, this IS the residual.
    But it can also learn correlations, ratios, or other comparisons
    that might work better than a fixed subtraction.
"""

import torch
import torch.nn as nn

from src.models.taylor_layer import TaylorGateLayer


class TaylorCNN(nn.Module):
    """Two-channel physics-informed transit classifier.

    Args:
        init_amplitude: Starting value for the Taylor gate's learnable
            amplitude A. Should be near expected transit depth.
        n_filters_1: Number of filters in the first CNN layer.
        n_filters_2: Number of filters in the second CNN layer.

    Example:
        >>> model = TaylorCNN(init_amplitude=0.01)
        >>> phase = torch.linspace(-3.14, 3.14, 200).unsqueeze(0)
        >>> flux = torch.randn(1, 200) * 0.005
        >>> prob = model(phase, flux)  # → tensor([[0.4832]])
    """

    def __init__(
        self,
        init_amplitude: float = 0.01,
        n_filters_1: int = 8,
        n_filters_2: int = 16,
    ) -> None:
        super().__init__()

        # --- Physics model ---
        self.taylor_gate = TaylorGateLayer(init_amplitude=init_amplitude)

        # --- CNN ---
        # 2 input channels: [gate_output, flux]
        # BatchNorm first: gate and flux values are tiny (~0.01), but conv
        # weights initialize at ~0.1. Without normalization, gradients are
        # too small for efficient learning. BatchNorm1d(2) independently
        # normalizes each channel to zero mean / unit variance.
        self.cnn = nn.Sequential(
            # Normalize inputs: (B, 2, 200) → (B, 2, 200) with μ=0, σ=1
            nn.BatchNorm1d(2),

            # Layer 1: (B, 2, 200) → (B, 8, 200)
            nn.Conv1d(
                in_channels=2,           # gate + flux
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
        flux: torch.Tensor,
    ) -> torch.Tensor:
        """Classify a light curve as transit or non-transit.

        Args:
            phase: Phase values in [-π, π], shape (batch, seq_len).
            flux: Preprocessed flux (baseline-subtracted), shape (batch, seq_len).

        Returns:
            Transit probability, shape (batch, 1). Values in [0, 1].
        """
        # Physics model: compute ideal transit shape from phase
        gate_out = self.taylor_gate(phase)  # (B, seq_len)

        # Stack gate output and flux as 2-channel input for CNN
        # gate_out: what a transit SHOULD look like (physics)
        # flux: what we actually OBSERVED (data)
        # Shape: (B, seq_len) + (B, seq_len) → (B, 2, seq_len)
        x = torch.stack([gate_out, flux], dim=1)

        # CNN processes both channels together
        cnn_out = self.cnn(x)          # (B, 16, 1)
        features = cnn_out.squeeze(2)  # (B, 16)

        # Classify
        logits = self.classifier(features)
        prob = torch.sigmoid(logits)

        return prob
