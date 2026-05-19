# ═══════════════════════════════════════════
# PRODUCTION MODEL PROTECTION
# NEVER save to src/models/production/
# Save experiments to src/models/ only
# with descriptive version suffix
# ═══════════════════════════════════════════

"""Phase 1 scoring: V10+V6b AND ensemble on the 500-TCE Kepler dataset.

Validates the scoring pipeline before the full DR25 batch run.
Writes data/kepler_500_scores.csv with per-TCE probabilities and
flags four case types for manual review:
  MISSED_PLANET_CANDIDATE   archive=FP,        model=PLANET
  SUSPICIOUS_CONFIRMATION   archive=CONFIRMED,  model=FP
  high_confidence_planet    both p>0.85
  high_confidence_fp        both p<0.15
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from src.models.adaptive_classifier import AdaptivePINNClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.5

# ── Load production ensemble ──────────────────────────────────────────────────
model = AdaptivePINNClassifier(
    model_dir="src/models/production",
    device=DEVICE,
)

# ── Load 500-TCE dataset ──────────────────────────────────────────────────────
d = torch.load("data/kepler_tce_v6.pt", weights_only=False)
phases    = d["phases"]
primary   = d["fluxes"]
secondary = d["fluxes_secondary"]
odd_even  = d["fluxes_odd_even"]
labels    = d["labels"].numpy().astype(int)
names     = d["names"]
depths    = np.array(d["depths_ppm"], dtype=float)
stellar_r = d["stellar_radius"].numpy()

# dataset has no SNR cache — use 50.0 (routes to "lightweight" = V10 only for
# auto-mode, but we score individually below to report all three probabilities)
SNR_DEFAULT = 50.0

# ── Score every TCE ───────────────────────────────────────────────────────────
results = []
for i in range(len(names)):
    ph  = phases[i].unsqueeze(0).to(DEVICE)
    pr  = primary[i].unsqueeze(0).to(DEVICE)
    sc  = secondary[i].unsqueeze(0).to(DEVICE)
    oe  = odd_even[i].unsqueeze(0).to(DEVICE)
    r_s = float(stellar_r[i]) if not np.isnan(stellar_r[i]) else 1.0

    with torch.no_grad():
        p_v10 = float(model.v10(ph, pr, sc, oe).item())
        p_v6b = float(model.v6b(ph, pr, sc, oe).item())

    ensemble_pred = int(p_v10 > THRESHOLD and p_v6b > THRESHOLD)
    truth = int(labels[i])

    if truth == 0 and ensemble_pred == 1:
        case = "MISSED_PLANET_CANDIDATE"        # archive says FP, model says planet
    elif truth == 1 and ensemble_pred == 0:
        case = "SUSPICIOUS_CONFIRMATION"        # archive confirmed, model disagrees
    elif truth == 1 and p_v10 > 0.85 and p_v6b > 0.85:
        case = "high_confidence_planet"
    elif truth == 0 and p_v10 < 0.15 and p_v6b < 0.15:
        case = "high_confidence_fp"
    else:
        case = "agreement"

    results.append({
        "koi_id":        names[i],
        "depth_ppm":     round(float(depths[i]), 1),
        "stellar_r":     round(r_s, 3),
        "truth":         truth,
        "p_v10":         round(p_v10, 4),
        "p_v6b":         round(p_v6b, 4),
        "ensemble_pred": ensemble_pred,
        "case":          case,
    })

df = pd.DataFrame(results)

# ── Print summary ─────────────────────────────────────────────────────────────
print("=" * 60)
print("V10+V6b AND ensemble on 500-TCE Kepler dataset")
print("=" * 60)
total = len(df)
agree = (df["truth"] == df["ensemble_pred"]).sum()
tp = ((df["truth"] == 1) & (df["ensemble_pred"] == 1)).sum()
tn = ((df["truth"] == 0) & (df["ensemble_pred"] == 0)).sum()
fp = ((df["truth"] == 0) & (df["ensemble_pred"] == 1)).sum()
fn = ((df["truth"] == 1) & (df["ensemble_pred"] == 0)).sum()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

print(f"Device: {DEVICE}")
print(f"Total TCEs scored:        {total}")
print(f"Accuracy:                 {agree}/{total} ({agree/total:.1%})")
print(f"TP={tp}  TN={tn}  FP={fp}  FN={fn}")
print(f"Precision:                {precision:.3f}  ({precision:.1%})")
print(f"Recall:                   {recall:.3f}  ({recall:.1%})")
print(f"F1:                       {f1:.3f}")

missed    = df[df["case"] == "MISSED_PLANET_CANDIDATE"]
suspicious = df[df["case"] == "SUSPICIOUS_CONFIRMATION"]
hi_planet = df[df["case"] == "high_confidence_planet"]
hi_fp     = df[df["case"] == "high_confidence_fp"]

print(f"\nMISSED PLANET CANDIDATES: {len(missed)}")
print("  (archive=FP, model=PLANET — worth manual review)")
if len(missed) > 0:
    print(missed[["koi_id", "depth_ppm", "stellar_r",
                   "p_v10", "p_v6b"]].to_string(index=False))

print(f"\nSUSPICIOUS CONFIRMATIONS: {len(suspicious)}")
print("  (archive=CONFIRMED, model=FP — possible mis-classification)")
if len(suspicious) > 0:
    print(suspicious[["koi_id", "depth_ppm", "stellar_r",
                       "p_v10", "p_v6b"]].to_string(index=False))

print(f"\nHigh-confidence planets:  {len(hi_planet)}")
print(f"High-confidence FPs:      {len(hi_fp)}")

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = "data/kepler_500_scores.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
