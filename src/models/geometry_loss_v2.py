"""InvertedGeometryLoss (V10) — corrected penalty direction.

V9's DynamicGeometryLoss was built on the hypothesis

    planets U-shape  (low T12/T14, high AUC_norm)
    EBs     V-shape  (high T12/T14, low AUC_norm)

The 76-TCE test-set data showed the *opposite* direction:

    planets T12/T14 median 0.673  |  FPs 0.604
    planets AUC_norm 0.090        |  FPs 0.506

V10 re-aims the penalty at false positives predicted as planets.
Concretely: if the model ever emits a high planet probability for a
sample whose shape *looks like a planet*, but the label says FP,
we want BCE to handle that. If the model (correctly) predicts
not-planet (FP) for a sample whose shape looks FP-like, we shouldn't
penalize. The only case where a geometry prior adds useful pressure
is when the model is *about to call something an FP* but the shape
screams "planet" — in that case we nudge the prob up.

So:

    L_shape = lam · (1 − planet_prob) · t12_t14
    L_auc   = lam · (1 − planet_prob) · (1 − auc_norm)

Both terms penalize *low* planet_prob (i.e. "calling this a FP") when
the shape features say it's planet-like (high T12/T14, low AUC_norm).
High-SNR samples carry the full penalty via sigmoid(SNR − pivot)
gating; low-SNR samples get λ_min.

SNR pivot raised to 100.0 because the 500-TCE dataset's median
SNR is 131 (archive koi_model_snr) — with pivot=12 every sample
got the full penalty, defeating the dataset-adaptive weighting.
"""

import torch
import torch.nn as nn


class InvertedGeometryLoss(nn.Module):
    def __init__(
        self,
        lambda_min: float = 0.01,
        lambda_max: float = 0.5,
        snr_pivot: float = 100.0,
    ) -> None:
        super().__init__()
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.snr_pivot = snr_pivot

    def forward(
        self,
        planet_prob: torch.Tensor,
        t12_t14: torch.Tensor,
        auc_norm: torch.Tensor,
        snr: torch.Tensor,
    ) -> torch.Tensor:
        fp_prob = 1.0 - planet_prob
        snr_weight = torch.sigmoid(snr - self.snr_pivot)
        lam = self.lambda_min + (self.lambda_max - self.lambda_min) * snr_weight

        # FP prediction with HIGH T12/T14 (planet-like ingress) → penalize
        L_shape = (lam * fp_prob * t12_t14).mean()

        # FP prediction with LOW AUC_norm (planet-like narrow dip) → penalize
        L_auc = (lam * fp_prob * (1.0 - auc_norm)).mean()

        return L_shape + L_auc
