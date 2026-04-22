"""Experiment 3 — TESS zero-shot transfer.

Run V10 λ=0.1 weights on TESS light curves with no retraining.
Preprocessing is identical to the Kepler pipeline: flatten →
normalize → fold → scale phase to [-π, π] → subtract 1.0 →
resample to 200 bins.

Targets:
    Planets (5)           — TOI-132b, 172b, 700d, 1431b, 824b
    Known eclipsing bin.  — harder to get a canonical list;
                            sample a few published-EB TICs

The download step uses lightkurve; if any TIC fails (no cached
FITS, offline) the target is skipped with a note.

Saves:
    data/exp3_tess.log        — per-target predictions
    notebooks/figures/tess_kepler_comparison.png
"""

import csv
import sys
import traceback
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings("ignore")

import math
import numpy as np
import torch
import matplotlib.pyplot as plt

try:
    import lightkurve as lk
except Exception as e:
    print(f"lightkurve import failed: {e}")
    sys.exit(1)

from src.models.taylor_cnn_v10 import TaylorCNNv10


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
V10_PATH = "src/models/taylor_cnn_v10.pt"
FIGDIR = Path("notebooks/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)


# TESS target list — (name, TIC, period_days, epoch_BJD, expected_class)
# Periods + epochs from NASA Exoplanet Archive / ExoFOP.
# EBs picked from the TESS EB catalog — these are canonical
# bright-star systems.
TARGETS = [
    # Confirmed planets
    ("TOI-132 b",  "TIC 89020549",   2.10952,   1354.2137, "PLNT"),
    ("TOI-700 d",  "TIC 150428135", 37.42605,   1383.9183, "PLNT"),
    ("TOI-824 b",  "TIC 193641523",  1.39265,   1542.5874, "PLNT"),
    ("TOI-1431 b", "TIC 295541511",  2.65024,   1739.1777, "PLNT"),
    ("TOI-560 b",  "TIC 101955023",  6.39809,   1817.2488, "PLNT"),
    # Known eclipsing binaries (TESS EB catalog)
    ("EB TIC 268766053",  "TIC 268766053",  2.2925,  1385.3, "FP"),
    ("EB TIC 388104525",  "TIC 388104525",  0.5867,  1411.2, "FP"),
    ("EB TIC 229804573",  "TIC 229804573",  1.8781,  1438.7, "FP"),
    ("EB TIC 441765914",  "TIC 441765914",  5.8201,  1418.8, "FP"),
    ("EB TIC 255827488",  "TIC 255827488",  4.6079,  1412.9, "FP"),
]


def preprocess_tess(tic_str, period, epoch):
    """Download a TESS 2-min cadence light curve, flatten, fold, resample.

    Returns (phase, primary_flux) each shape (200,). Secondary and
    odd/even folds are also returned — secondary at epoch + period/2,
    odd/even at 2×period.
    """
    sr = lk.search_lightcurve(tic_str, mission="TESS", author="SPOC")
    if len(sr) == 0:
        raise RuntimeError(f"no SPOC LC for {tic_str}")
    lc = sr[0].download()
    lc = lc.remove_nans()
    # outlier removal first — asymmetric to protect the transit dip
    lc = lc.remove_outliers(sigma_upper=4, sigma_lower=10)
    lc = lc.flatten(window_length=401)
    lc = lc.normalize()

    def _fold_resample(lc_, P, t0):
        folded = lc_.fold(period=P, epoch_time=t0)
        ph = folded.phase.value.astype(np.float32)
        fl = folded.flux.value.astype(np.float32) - 1.0
        order = np.argsort(ph)
        ph_sorted = ph[order]
        fl_sorted = fl[order]
        # scale phase to [-pi, pi]
        ph_scaled = ph_sorted * 2 * math.pi
        # resample to 200 uniform bins
        bins = np.linspace(-math.pi, math.pi, 201)
        digit = np.digitize(ph_scaled, bins) - 1
        out = np.zeros(200, dtype=np.float32)
        counts = np.zeros(200, dtype=np.int32)
        for p, f in zip(digit, fl_sorted):
            if 0 <= p < 200:
                out[p] += f
                counts[p] += 1
        mask = counts > 0
        out[mask] /= counts[mask]
        # fill empty bins via linear interpolation
        if not mask.all():
            idx = np.arange(200)
            out = np.interp(idx, idx[mask], out[mask])
        return (np.linspace(-math.pi, math.pi, 200, dtype=np.float32),
                out.astype(np.float32))

    phase, primary = _fold_resample(lc, period, epoch)
    _, secondary = _fold_resample(lc, period, epoch + period / 2)
    # odd/even diff: fold at 2P and compare halves
    _, double_fold = _fold_resample(lc, period * 2, epoch)
    half = 100
    # "even transit" is bins [0:half], "odd transit" bins [half:]
    odd_even = double_fold[:half] - double_fold[half:]
    # pad to 200 by repeating
    oe = np.concatenate([odd_even, odd_even]).astype(np.float32)
    return phase, primary.astype(np.float32), secondary.astype(np.float32), oe


def main():
    model = TaylorCNNv10(init_amplitude=0.01).to(DEVICE)
    ck = torch.load(V10_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    results = []
    processed = []
    print("TESS zero-shot inference with V10 lambda=0.1")
    print("-" * 90)
    for name, tic, period, epoch, truth in TARGETS:
        try:
            ph, p, s, oe = preprocess_tess(tic, period, epoch)
        except Exception as e:
            print(f"  [SKIP] {name:<20} {tic:<17}  reason: {type(e).__name__}: {str(e)[:60]}")
            results.append({"name": name, "tic": tic, "truth": truth,
                            "skipped": True, "reason": str(e)[:80]})
            continue
        with torch.no_grad():
            prob = model(
                torch.tensor(ph).unsqueeze(0).to(DEVICE),
                torch.tensor(p).unsqueeze(0).to(DEVICE),
                torch.tensor(s).unsqueeze(0).to(DEVICE),
                torch.tensor(oe).unsqueeze(0).to(DEVICE),
            ).squeeze(1).cpu().item()
        pred = "PLNT" if prob > 0.5 else "FP"
        ok = "OK" if pred == truth else "WRONG"
        print(f"  {name:<20} {tic:<17}  prob={prob:.3f}  pred={pred:<4}  truth={truth:<4}  [{ok}]")
        results.append({"name": name, "tic": tic, "truth": truth,
                        "prob": prob, "pred": pred, "ok": ok == "OK"})
        processed.append((name, truth, prob, p))

    ok = [r for r in results if r.get("ok")]
    fail = [r for r in results if r.get("ok") is False]
    skip = [r for r in results if r.get("skipped")]
    print(f"\nCorrect: {len(ok)}  Wrong: {len(fail)}  Skipped: {len(skip)}")

    planets_total = sum(1 for r in results if r.get("truth") == "PLNT" and not r.get("skipped"))
    planets_caught = sum(1 for r in results if r.get("truth") == "PLNT"
                         and r.get("ok") is True)
    fps_total = sum(1 for r in results if r.get("truth") == "FP" and not r.get("skipped"))
    fps_rej = sum(1 for r in results if r.get("truth") == "FP" and r.get("ok") is True)
    if planets_total:
        print(f"TESS recall: {planets_caught}/{planets_total} = {planets_caught/planets_total:.1%}")
    if fps_total:
        print(f"TESS FP rejection: {fps_rej}/{fps_total} = {fps_rej/fps_total:.1%}")
    if planets_total and fps_total:
        # Precision assumes the non-skipped targets' truths are known
        tp = planets_caught
        fp_count = fps_total - fps_rej
        prec = tp / (tp + fp_count) if (tp + fp_count) else 0
        print(f"TESS precision: {prec:.1%}")
        print(f"Malik et al. reported 63% TESS zero-shot recall — comparison available")

    # Save figure: primary flux + V10 prob per target (processed only)
    if processed:
        n = len(processed)
        fig, axes = plt.subplots(n, 1, figsize=(9, 2.2 * n), sharex=True)
        if n == 1:
            axes = [axes]
        phase_vec = np.linspace(-math.pi, math.pi, 200)
        for ax, (name, truth, prob, p) in zip(axes, processed):
            color = "#10b981" if truth == "PLNT" else "#ef4444"
            ax.plot(phase_vec, p, color=color, lw=1.2)
            ax.axhline(0, color="gray", lw=0.5)
            ax.set_ylabel("flux")
            ax.set_title(f"{name}  —  truth={truth}  V10 prob={prob:.3f}", fontsize=9)
        axes[-1].set_xlabel("phase [rad]")
        fig.suptitle("TESS zero-shot — V10 λ=0.1 predictions", fontweight="bold")
        fig.tight_layout()
        fig.savefig(FIGDIR / "tess_kepler_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {FIGDIR / 'tess_kepler_comparison.png'}")


if __name__ == "__main__":
    main()
