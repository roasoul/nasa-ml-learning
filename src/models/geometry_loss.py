"""Dynamic geometry loss (for V9 experimentation — not used in V8/V8.5).

Combines two shape-consistency penalties (ingress-sharpness and AUC purity)
under a data-driven weight that scales with detection SNR:

    snr_weight = sigmoid(koi_model_snr - snr_pivot)
    lambda     = lambda_min + (lambda_max - lambda_min) * snr_weight

    L_shape = lambda * planet_prob * t12_t14
    L_auc   = lambda * planet_prob * (1 - auc_norm)

Why SNR-weighted: at low SNR the trapezoid-shape features are noisy
(T12/T14 drifts toward 0.5, AUC/A toward 0.5) so the penalty would
punish correct planet predictions. At high SNR the features are
trustworthy enough to enforce as a physics prior.

Reserved for V9 — not wired into the V8/V8.5 training loop.
"""

import torch
import torch.nn as nn


class DynamicGeometryLoss(nn.Module):
    def __init__(self, lambda_min=0.01, lambda_max=1.0, snr_pivot=12.0):
        super().__init__()
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.snr_pivot = snr_pivot

    def forward(self, planet_prob, t12_t14, auc_norm, koi_snr):
        snr_weight = torch.sigmoid(koi_snr - self.snr_pivot)
        lam = self.lambda_min + (self.lambda_max - self.lambda_min) * snr_weight
        L_shape = (lam * planet_prob * t12_t14).mean()
        L_auc = (lam * planet_prob * (1.0 - auc_norm)).mean()
        return L_shape + L_auc
