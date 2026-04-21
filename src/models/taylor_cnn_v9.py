"""V9 Taylor-CNN: physics loss, not physics input.

V8.5 fused shape features into the classifier's input — the BCE found
pathways that bypassed them and A collapsed to zero. V9 moves the
shape features out of the forward pass and into the loss function,
so the CNN cannot game them.

Architecture:
    - 4-channel input identical to V6: [gate, primary, secondary,
      odd/even]. Classifier is Linear(16, 1) — same 914 params
      baseline.
    - Gate is the V8 Taylor gate (learnable A, B, t0) with A
      post-step-clamped to min=0.001 in the training loop so the
      gate can't die.
    - `compute_shape_features(primary_flux)` produces per-sample
      (t12_t14, auc_norm) for use in the DynamicGeometryLoss
      training penalty only. Never touches the classifier.

Training loss:
    L = BCE + 0.01 · |B|.mean() + L_geometry

    where L_geometry scales each sample's shape penalty by
    sigmoid(koi_model_snr − 12) so high-SNR samples carry the
    physics prior and low-SNR samples retain learner flexibility.
"""

import torch
import torch.nn as nn

from src.models.taylor_layer_v8 import TaylorGateLayerV8


_FIXED_GRID_LEN = 200
_SOFT_MASK_K = 50.0


class TaylorCNNv9(nn.Module):
    def __init__(
        self,
        init_amplitude: float = 0.01,
        init_B: float = 1.0,
        init_t0: float = 0.0,
        n_filters_1: int = 8,
        n_filters_2: int = 16,
    ) -> None:
        super().__init__()
        self.taylor_gate = TaylorGateLayerV8(
            init_amplitude=init_amplitude,
            init_B=init_B,
            init_t0=init_t0,
        )
        self.cnn = nn.Sequential(
            nn.BatchNorm1d(4),
            nn.Conv1d(4, n_filters_1, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(n_filters_1, n_filters_2, kernel_size=5, padding=2),
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
        gate_out = self.taylor_gate(phase)
        x = torch.stack(
            [gate_out, primary_flux, secondary_flux, odd_even_diff], dim=1
        )
        cnn_out = self.cnn(x).squeeze(2)
        return torch.sigmoid(self.classifier(cnn_out))

    def compute_shape_features(
        self,
        primary_flux: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (t12_t14, auc_norm) per sample. Used by the geometry
        loss; not part of the forward pass."""
        device = primary_flux.device
        dtype = primary_flux.dtype
        A = self.taylor_gate.A.abs().clamp(min=0.001) + 1e-8

        depth = (-primary_flux).clamp(min=0.0)
        depth_norm = depth / A

        k = _SOFT_MASK_K
        dip_all = torch.sigmoid(k * (depth_norm - 0.1))
        flat_bot = torch.sigmoid(k * (depth_norm - 0.8))
        ingress = dip_all - flat_bot
        total = (1.0 - dip_all) + ingress + flat_bot + 1e-8
        ingress = ingress / total
        flat_bot = flat_bot / total

        t14 = (ingress + flat_bot).mean(dim=1) + 1e-8
        t12_t14 = ingress.mean(dim=1) / t14

        phase_fixed = torch.linspace(
            -1.0, 1.0, _FIXED_GRID_LEN, device=device, dtype=dtype
        ).expand_as(depth)
        auc_raw = torch.trapz(depth, phase_fixed, dim=-1)
        auc_norm = auc_raw / A

        return t12_t14, auc_norm

    def sparsity_loss(self) -> torch.Tensor:
        return 0.01 * self.taylor_gate.B.abs().mean()

    def clamp_A(self, min_val: float = 0.001) -> None:
        """Post-step clamp on A to keep the gate alive."""
        with torch.no_grad():
            self.taylor_gate.A.data.abs_().clamp_(min=min_val)
