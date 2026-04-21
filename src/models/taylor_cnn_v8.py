"""V8 / V8.5 Taylor-CNN with learnable B and shape-feature fusion.

V8:   Same 4-channel architecture as V6 but with TaylorGateLayerV8
      (learnable A, B, t0). The CNN still consumes [gate, primary,
      secondary, odd/even]; `use_shape_features=False` selects this mode.

V8.5 (v2): Adds a 5-dim shape-feature vector fused into the classifier:

          shape = [B, AUC_raw, AUC_norm, T12_T14, flat_bottom]

      The four morphology features are computed per-sample from
      `primary_flux` — the actual observed dip — using the gate's
      learned A as the depth normalizer. B is the single global
      learned curvature and is broadcast as a shared prior (all
      samples in a batch see the same B).

      Earlier V8.5-v1 extracted features from gate_output; that
      collapsed to a batch-constant vector because the folded phase
      grid is identical for every TCE in this dataset. V8.5-v2
      moves the feature extractor onto primary_flux so each sample
      produces its own shape signature.

Shape-feature definitions (V8.5-v2, per sample unless noted):

    AUC_raw     — trapz( (-primary_flux).clamp(min=0) ) on a fixed
                  [-1, 1] phase grid. Instrument-agnostic dip-energy
                  integral. Positive; 0 for pure baseline.
    AUC_norm    — AUC_raw / A. Pulls A out of the integral so the
                  feature reads "how much of the available dip area
                  the observed signal fills". Fixes the small-deep
                  planet vs large-shallow EB degeneracy.
    T12_T14     — ingress fraction = ingress.mean / (ingress+flat).mean.
                  Mutually-exclusive soft masks (baseline + ingress +
                  flat_bot = 1 by construction) keep this from drifting
                  to 0.5 at low SNR. High for sharp-ingress V-shapes
                  (grazing EBs), low for U-shapes (real planets).
    flat_bottom — mean soft flat-bottom mask. Large for planet U-shape,
                  ≈0 for grazing-EB V-shape.
    B           — GLOBAL learned curvature parameter (same value for
                  every sample in the batch). Retained as a shared
                  prior so the CNN head sees the population template.

Soft-mask sharpness:
    k=50 applied to depth_norm = depth/A. That keeps the effective
    sharpness in normalized-depth space rather than raw-flux space,
    so the cutoffs at 0.1·A (dip threshold) and 0.8·A (flat-bottom
    threshold) are scale-invariant.

Physics rationale: see CLAUDE.md V8 section.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.taylor_layer_v8 import TaylorGateLayerV8


_SOFT_MASK_K = 50.0
_FIXED_GRID_LEN = 200


class TaylorCNNv8(nn.Module):
    """V8 / V8.5 physics-informed classifier.

    Args:
        init_amplitude:   Starting A.
        init_B:           Starting B (1.0 = planet U-shape prior).
        init_t0:          Starting phase offset.
        n_filters_1:      First Conv1d filters.
        n_filters_2:      Second Conv1d filters.
        use_shape_features:
            False → V8 (4-channel CNN only, classifier = Linear(16, 1)).
            True  → V8.5 (CNN 16-d features + 5 shape features fused,
                          classifier = Linear(21, 1)).

    Forward returns probabilities; `.last_shape_features` holds the
    most recent (batch, 5) shape-feature tensor for inspection.
    """

    def __init__(
        self,
        init_amplitude: float = 0.01,
        init_B: float = 1.0,
        init_t0: float = 0.0,
        n_filters_1: int = 8,
        n_filters_2: int = 16,
        use_shape_features: bool = True,
    ) -> None:
        super().__init__()
        self.use_shape_features = use_shape_features

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

        fused_dim = n_filters_2 + (5 if use_shape_features else 0)
        self.classifier = nn.Linear(fused_dim, 1)

        self.last_shape_features: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Shape-feature computation (V8.5)
    # ------------------------------------------------------------------

    def _compute_shape_features(
        self,
        primary_flux: torch.Tensor,
    ) -> torch.Tensor:
        """Build the 5-dim shape feature per sample from primary_flux.

        primary_flux: (batch, seq_len). Baseline ≈ 0, dip ≈ -A (negative).
        returns:      (batch, 5) float tensor.
        """
        batch_size = primary_flux.shape[0]
        device = primary_flux.device
        dtype = primary_flux.dtype
        A = self.taylor_gate.A.abs() + 1e-8
        B_val = self.taylor_gate.B

        # ---- Per-sample depth signal in normalized depth-space -------
        # Flip sign so dip becomes positive; clamp baseline noise to 0.
        depth = (-primary_flux).clamp(min=0.0)          # (batch, L)
        depth_norm = depth / A                          # ~1 at full dip

        # ---- Fixed-grid resample of depth (instrument-agnostic AUC) --
        depth_resampled = F.interpolate(
            depth.unsqueeze(1),
            size=_FIXED_GRID_LEN,
            mode="linear",
            align_corners=True,
        ).squeeze(1)  # (batch, 200)

        phase_fixed = torch.linspace(
            -1.0, 1.0, _FIXED_GRID_LEN, device=device, dtype=dtype
        )

        auc_raw = torch.trapz(depth_resampled, phase_fixed, dim=-1)  # (batch,)
        auc_norm = auc_raw / A

        # ---- Normalized soft masks (on raw grid, per sample) ----------
        # Thresholds in normalized depth-space:
        #   depth_norm < 0.1 → baseline   (no dip)
        #   0.1 ≤ depth_norm < 0.8 → ingress/egress walls
        #   depth_norm ≥ 0.8 → flat-bottom floor
        k = _SOFT_MASK_K
        dip_all = torch.sigmoid(k * (depth_norm - 0.1))   # 1 inside the dip
        flat_bot = torch.sigmoid(k * (depth_norm - 0.8))  # 1 near the floor
        ingress = dip_all - flat_bot

        # Normalize — baseline is the complement of dip_all, so
        # baseline + ingress + flat_bot = 1 before the safety epsilon.
        total = (1.0 - dip_all) + ingress + flat_bot + 1e-8
        ingress = ingress / total
        flat_bot = flat_bot / total

        t14 = (ingress + flat_bot).mean(dim=1) + 1e-8
        t12_t14 = ingress.mean(dim=1) / t14
        flat_bottom = flat_bot.mean(dim=1)

        B_broadcast = B_val.expand(batch_size).to(dtype=dtype)

        shape = torch.stack(
            [B_broadcast, auc_raw, auc_norm, t12_t14, flat_bottom],
            dim=1,
        )
        return shape

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

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
        cnn_out = self.cnn(x).squeeze(2)  # (batch, n_filters_2)

        if self.use_shape_features:
            shape = self._compute_shape_features(primary_flux)
            self.last_shape_features = shape.detach()
            features = torch.cat([cnn_out, shape], dim=1)
        else:
            self.last_shape_features = None
            features = cnn_out

        logits = self.classifier(features)
        return torch.sigmoid(logits)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def sparsity_loss(self) -> torch.Tensor:
        """0.01 · |B|.mean() — pushes B toward 0 unless data supports it."""
        return 0.01 * self.taylor_gate.B.abs().mean()
