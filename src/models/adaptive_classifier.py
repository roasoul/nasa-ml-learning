"""Adaptive PINN Classifier — SNR-routed ensemble of V6b, V10, V10 log-R*.

Three base models, four operating modes:

    discovery    p_v6b > 0.4  OR  p_v10 > 0.4  OR  p_log > 0.4
    balanced     p_v6b > 0.5  AND p_v10 > 0.5
    lightweight  p_v10 > 0.5
    auto         SNR < 20    -> discovery   (catch rare, weak signals)
                 20 <= SNR < 100 -> lightweight (bulk-TCE common case)
                 SNR >= 100  -> balanced    (strong signals, tight precision)

Notes on inputs:
    * Each of the three underlying models (TaylorCNN for V6b and
      TaylorCNNv10 for both V10 variants) already applies a sigmoid
      in its forward, so the returned values are probabilities. We do
      not re-apply sigmoid.
    * The V10 log-R* model was trained on flux channels pre-multiplied
      by log1p(R*/R_sun) for each sample (see scripts/exp5b_clipped_rstar.py).
      To keep inputs on the training distribution we apply the same
      scaling before calling that model; callers should pass the
      target's stellar_radius in solar units. If unknown, 1.0 solar is
      a neutral default.

Single-sample API by default (matches the project convention). Each
method also accepts batched tensors, and returns a tensor of per-sample
predictions when called that way.
"""

from __future__ import annotations

import os

import torch

from src.models.taylor_cnn import TaylorCNN
from src.models.taylor_cnn_v10 import TaylorCNNv10


class AdaptivePINNClassifier:
    VALID_MODES = ("discovery", "balanced", "lightweight", "auto")

    PRODUCTION_DIR = "src/models/production"

    def __init__(self, model_dir: str | None = None, device: str | torch.device | None = None):
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        model_dir = model_dir or self.PRODUCTION_DIR
        self.v6b = self._load(TaylorCNN, os.path.join(model_dir, "v6b_recall947.pt"))
        self.v10 = self._load(TaylorCNNv10, os.path.join(model_dir, "v10_f1861.pt"))
        self.v10_log = self._load(TaylorCNNv10, os.path.join(model_dir, "v10_log_mdwarf.pt"))

    def _load(self, cls, path: str):
        blob = torch.load(path, weights_only=False, map_location=self.device)
        state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
        model = cls(init_amplitude=0.01).to(self.device)
        model.load_state_dict(state)
        model.eval()
        return model

    @staticmethod
    def _as_batch(t: torch.Tensor) -> torch.Tensor:
        return t.unsqueeze(0) if t.dim() == 1 else t

    def _log_scale_vector(self, stellar_radius, batch_size: int) -> torch.Tensor:
        """Per-sample log-R* scaling that matches the training recipe."""
        if isinstance(stellar_radius, torch.Tensor):
            r = stellar_radius.clamp(min=0.01).view(-1).to(self.device)
            if r.numel() == 1 and batch_size > 1:
                r = r.expand(batch_size)
        else:
            r = torch.full((batch_size,), float(stellar_radius), device=self.device).clamp(min=0.01)
        return torch.log1p(r).unsqueeze(1)

    def _probs(self, phase, flux, secondary, odd_even, stellar_radius=1.0):
        """Forward pass through all three models. Returns (p_v6b, p_v10, p_log)
        as 1-D tensors on self.device, one entry per sample."""
        phase = self._as_batch(phase).to(self.device)
        flux = self._as_batch(flux).to(self.device)
        secondary = self._as_batch(secondary).to(self.device)
        odd_even = self._as_batch(odd_even).to(self.device)

        scale = self._log_scale_vector(stellar_radius, batch_size=flux.size(0))
        flux_log = flux * scale
        secondary_log = secondary * scale
        odd_even_log = odd_even * scale

        with torch.no_grad():
            p_v6b = self.v6b(phase, flux, secondary, odd_even).squeeze(-1)
            p_v10 = self.v10(phase, flux, secondary, odd_even).squeeze(-1)
            p_log = self.v10_log(phase, flux_log, secondary_log, odd_even_log).squeeze(-1)
        return p_v6b, p_v10, p_log

    @staticmethod
    def _route_mode(snr) -> str:
        if snr is None:
            return "balanced"
        if snr < 20:
            return "discovery"
        if snr < 100:
            return "lightweight"
        return "balanced"

    @staticmethod
    def _combine(p_v6b: torch.Tensor, p_v10: torch.Tensor, p_log: torch.Tensor,
                 mode: str) -> torch.Tensor:
        if mode == "discovery":
            stacked = torch.stack([p_v6b, p_v10, p_log], dim=0)
            return (stacked.max(dim=0).values > 0.4).long()
        if mode == "balanced":
            return ((p_v6b > 0.5) & (p_v10 > 0.5)).long()
        if mode == "lightweight":
            return (p_v10 > 0.5).long()
        raise ValueError(f"Unknown mode: {mode!r}")

    def predict(self, phase, flux, secondary, odd_even,
                snr=None, mode: str = "auto", stellar_radius=1.0):
        """Return a planet/FP decision as an int for single-sample input, or
        a long tensor of decisions for batched input."""
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got {mode!r}")
        concrete = self._route_mode(snr) if mode == "auto" else mode
        p_v6b, p_v10, p_log = self._probs(phase, flux, secondary, odd_even, stellar_radius)
        preds = self._combine(p_v6b, p_v10, p_log, concrete)
        return int(preds.item()) if preds.numel() == 1 else preds.cpu()

    def predict_with_report(self, phase, flux, secondary, odd_even,
                            snr=None, stellar_radius=1.0) -> dict:
        """Return a dict with prediction, probabilities, and routing info.

        Accepts a single sample (1-D tensors). For batched evaluation use
        predict_batch() below.
        """
        p_v6b, p_v10, p_log = self._probs(phase, flux, secondary, odd_even, stellar_radius)
        p_v6b_s = float(p_v6b.view(-1)[0].item())
        p_v10_s = float(p_v10.view(-1)[0].item())
        p_log_s = float(p_log.view(-1)[0].item())
        mode_used = self._route_mode(snr)
        decision = int(self._combine(
            torch.tensor([p_v6b_s]), torch.tensor([p_v10_s]), torch.tensor([p_log_s]), mode_used,
        ).item())
        return {
            "prediction": "PLANET" if decision else "FP",
            "mode_used": mode_used,
            "snr": snr,
            "stellar_radius": stellar_radius,
            "p_v6b": round(p_v6b_s, 3),
            "p_v10": round(p_v10_s, 3),
            "p_log": round(p_log_s, 3),
            "confidence": round(max(p_v6b_s, p_v10_s, p_log_s), 3),
        }

    def predict_batch(self, phase, flux, secondary, odd_even,
                      snr=None, mode: str = "auto", stellar_radius=1.0) -> dict:
        """Batched convenience: return both the per-sample decisions and the
        three per-sample probabilities, routing per-sample via SNR when mode='auto'.

        snr and stellar_radius may be scalars (applied to all samples) or 1-D
        tensors/arrays of length batch_size.
        """
        p_v6b, p_v10, p_log = self._probs(phase, flux, secondary, odd_even, stellar_radius)
        bs = p_v6b.numel()

        if mode != "auto":
            preds = self._combine(p_v6b, p_v10, p_log, mode)
            modes_used = [mode] * bs
        else:
            if snr is None:
                modes_used = ["balanced"] * bs
            else:
                snr_t = torch.as_tensor(snr).view(-1)
                if snr_t.numel() == 1 and bs > 1:
                    snr_t = snr_t.expand(bs)
                modes_used = [self._route_mode(float(s)) for s in snr_t.tolist()]

            preds = torch.zeros(bs, dtype=torch.long)
            for i, m in enumerate(modes_used):
                preds[i] = self._combine(
                    p_v6b[i:i + 1], p_v10[i:i + 1], p_log[i:i + 1], m,
                ).item()

        return {
            "preds": preds.cpu(),
            "p_v6b": p_v6b.cpu(),
            "p_v10": p_v10.cpu(),
            "p_log": p_log.cpu(),
            "modes_used": modes_used,
        }
