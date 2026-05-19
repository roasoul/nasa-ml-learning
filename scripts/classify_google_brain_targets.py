# ═══════════════════════════════════════════
# PRODUCTION MODEL PROTECTION
# NEVER save to src/models/production/
# Save experiments to src/models/ only
# with descriptive version suffix
# ═══════════════════════════════════════════

"""Classify Kepler-90i and Kepler-80g with the V10+V6b AND ensemble.

Both were originally discovered by Google Brain / AstroNet (Shallue &
Vanderburg 2017, AJ 155 94). These are extremely shallow transits
(~172-200 ppm) — well below the 1000-50000 ppm training range.
The test probes whether the physics gate bank generalises to sub-ppm
signals that the CNN has never seen during training.

Epochs are in BKJD = BJD - 2454833, matching lightkurve Kepler time.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")   # headless — save to file, no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.kepler import download_lightcurve, preprocess_lightcurve
from src.models.adaptive_classifier import AdaptivePINNClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIGURE_PATH = "notebooks/figures/google_brain_targets.png"

TARGETS = [
    {
        "name":    "Kepler-90i",
        "koi":     "K00351.08",
        "kic":     11442793,
        "period":  14.44912,
        "epoch":   2455012.48 - 2454833,   # 179.48 BKJD
        "depth":   200,
        "r_earth": 1.32,
        "truth":   "CONFIRMED PLANET",
    },
    {
        "name":    "Kepler-80g",
        "koi":     "K00500.06",
        "kic":     4852528,
        "period":  0.98678,
        "epoch":   2454968.97 - 2454833,   # 135.97 BKJD
        "depth":   172,
        "r_earth": 1.53,
        "truth":   "CONFIRMED PLANET",
    },
]

# ── Load ensemble ─────────────────────────────────────────────────────────────
print(f"Device: {DEVICE}")
print("Loading V10+V6b production ensemble...")
# AdaptivePINNClassifier is NOT an nn.Module — pass device to constructor
model = AdaptivePINNClassifier(model_dir="src/models/production", device=DEVICE)

# ── Download, preprocess, score ───────────────────────────────────────────────
results = []

for target in TARGETS:
    print(f"\n{'='*50}")
    print(f"  {target['name']}  ({target['koi']})")
    print(f"  KIC {target['kic']}  P={target['period']}d  depth~{target['depth']} ppm")
    print(f"{'='*50}")

    try:
        print("  Downloading all Kepler quarters from MAST...")
        lc = download_lightcurve(target["kic"])
        print(f"  Downloaded {len(lc)} cadences")

        print("  Preprocessing (primary + secondary + odd/even)...")
        phase, primary, secondary, odd_even = preprocess_lightcurve(
            lc, target["period"], target["epoch"], n_points=200
        )

        ph = phase.unsqueeze(0).to(DEVICE)
        pr = primary.unsqueeze(0).to(DEVICE)
        sc = secondary.unsqueeze(0).to(DEVICE)
        oe = odd_even.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            p_v10 = float(model.v10(ph, pr, sc, oe).item())
            p_v6b = float(model.v6b(ph, pr, sc, oe).item())

        ensemble_pred = int(p_v10 > 0.5 and p_v6b > 0.5)
        verdict = "PLANET [CORRECT]" if ensemble_pred == 1 else "FP     [WRONG]"

        print(f"  p_v10:    {p_v10:.4f}")
        print(f"  p_v6b:    {p_v6b:.4f}")
        print(f"  Ensemble: {verdict}")
        print(f"  Truth:    {target['truth']}")
        print(f"  Primary dip: {float(primary.min()):.5f}  "
              f"({float(primary.min())*1e6:.0f} ppm)")

        results.append({
            **target,
            "p_v10":    p_v10,
            "p_v6b":    p_v6b,
            "ensemble": ensemble_pred,
            "verdict":  verdict,
            "phase":    phase,
            "primary":  primary,
        })

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("GOOGLE BRAIN DISCOVERY TARGETS -- TAYLOR-CNN VERDICT")
print("=" * 65)
print(f"{'Name':<14} {'Depth':>7} {'p_v10':>7} {'p_v6b':>7} {'Verdict':<22} Truth")
print("-" * 65)
for r in results:
    print(f"{r['name']:<14} {r['depth']:>6}ppm "
          f"{r['p_v10']:>7.4f} {r['p_v6b']:>7.4f} "
          f"{r['verdict']:<22} {r['truth']}")

n_correct = sum(1 for r in results if r["ensemble"] == 1)
print(f"\nRecovery: {n_correct}/{len(results)} confirmed planets correctly classified")
if n_correct == len(results):
    print("Both Google Brain targets recovered by physics gate bank.")

# ── Plot ──────────────────────────────────────────────────────────────────────
if results:
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        phase_np = r["phase"].numpy()
        flux_ppm = r["primary"].numpy() * 1e6

        ax.plot(phase_np, flux_ppm, "b.", markersize=2, alpha=0.6, label="flux")
        ax.axhline(0,  color="gray",  linestyle="--", alpha=0.4)
        ax.axvline(0,  color="green", linestyle=":",  alpha=0.7, label="transit center")

        verdict_short = "PLANET" if r["ensemble"] == 1 else "FP"
        color = "green" if r["ensemble"] == 1 else "red"
        ax.set_title(
            f"{r['name']}  ({r['depth']} ppm,  R={r['r_earth']} R_E)\n"
            f"p_v10={r['p_v10']:.4f}  p_v6b={r['p_v6b']:.4f}  "
            f"-> {verdict_short}",
            color=color, fontsize=10,
        )
        ax.set_xlabel("Phase (radians)")
        ax.set_ylabel("Flux - 1 (ppm)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle(
        "Taylor-CNN vs Google Brain Discovery Targets\n"
        "(Shallue & Vanderburg 2017 -- AstroNet originals)",
        fontweight="bold", fontsize=12,
    )
    plt.tight_layout()
    Path(FIGURE_PATH).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {FIGURE_PATH}")
