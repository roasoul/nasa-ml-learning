"""V10 Taylor-CNN — 5-gate bank + 3 observation channels (8-channel input).

Architecture:
    phase → MultiTemplateGateBank → (B, 5, 200)   # G1…G5
    primary, secondary, odd/even  → 3 channels    # (B, 3, 200)
    stack → (B, 8, 200)
    BatchNorm1d(8) → Conv1d(8→8, k=7) → ReLU
                   → Conv1d(8→16, k=5) → ReLU
                   → AdaptiveAvgPool1d(1) → (B, 16)
    Linear(16, 1) → sigmoid → p(planet)

`compute_shape_features(primary_flux)` is retained (same formulation
as V9) so InvertedGeometryLoss can be applied during training. Shape
features are loss-only; they do not enter the classifier input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.multi_template_gate import MultiTemplateGateBank


_FIXED_GRID_LEN = 200
_SOFT_MASK_K = 50.0


class TaylorCNNv10(nn.Module):
    def __init__(
        self,
        init_amplitude: float = 0.01,
        n_filters_1: int = 8,
        n_filters_2: int = 16,
    ) -> None:
        super().__init__()
        self.gate_bank = MultiTemplateGateBank(init_amplitude=init_amplitude)
        self.cnn = nn.Sequential(
            nn.BatchNorm1d(8),
            nn.Conv1d(8, n_filters_1, kernel_size=7, padding=3),
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
        gates = self.gate_bank(phase)                          # (B, 5, L)
        obs = torch.stack(
            [primary_flux, secondary_flux, odd_even_diff], dim=1
        )                                                       # (B, 3, L)
        x = torch.cat([gates, obs], dim=1)                     # (B, 8, L)
        cnn_out = self.cnn(x).squeeze(2)
        return torch.sigmoid(self.classifier(cnn_out))

    def compute_shape_features(
        self, primary_flux: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-sample (t12_t14, auc_norm) from primary_flux. Uses G1's
        amplitude (the canonical planet-U-shape gate) as the depth norm."""
        device = primary_flux.device
        dtype = primary_flux.dtype
        A = self.gate_bank.A1.clamp(min=0.001) + 1e-8

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

    def clamp_amplitudes(self, min_val: float = 0.001) -> None:
        """Post-step positivity clamp on all 5 amplitudes."""
        with torch.no_grad():
            for A in [self.gate_bank.A1, self.gate_bank.A2,
                      self.gate_bank.A3, self.gate_bank.A4,
                      self.gate_bank.A5]:
                A.data.clamp_(min=min_val)
