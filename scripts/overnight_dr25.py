# ═══════════════════════════════════════════
# PRODUCTION MODEL PROTECTION
# NEVER save to src/models/production/
# Save experiments to src/models/ only
# with descriptive version suffix
# ═══════════════════════════════════════════

"""Overnight batch: download 1000 Kepler DR25 TCEs and score with V10+V6b ensemble.

Downloads 500 confirmed + 500 FP (seed=99, different sample from the existing
500-TCE training set), scores with the AND ensemble, and writes
data/kepler_dr25_1k_scores.csv. The .pt file is saved to
data/kepler_dr25_1k.pt so it can be used for future experiments.

Resume: if kepler_dr25_1k.pt already exists it is used as the resume base —
already-downloaded TCEs are skipped and new ones appended.

Usage:
    .\\nasaml\\Scripts\\python.exe scripts\\overnight_dr25.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from src.data.build_dataset import build_dataset
from src.models.adaptive_classifier import AdaptivePINNClassifier

# ── Config ────────────────────────────────────────────────────────────────────
N_PER_CLASS  = 500          # 500 confirmed + 500 FP = 1000 total
SEED         = 99           # different from training-set seed=42
OUTPUT_PT    = "data/kepler_dr25_1k.pt"
OUTPUT_CSV   = "data/kepler_dr25_1k_scores.csv"
THRESHOLD    = 0.5
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Step 1: Download ──────────────────────────────────────────────────────────
print("=" * 60)
print("PHASE 2 — overnight DR25 batch (1000 TCEs)")
print("=" * 60)

resume_path = OUTPUT_PT if Path(OUTPUT_PT).exists() else None
if resume_path:
    print(f"Resuming from {resume_path}")
else:
    print(f"Fresh download -> {OUTPUT_PT}")

t_download_start = time.time()
build_dataset(
    n_per_class=N_PER_CLASS,
    n_points=200,
    output_path=OUTPUT_PT,
    mission="Kepler",
    seed=SEED,
    resume=resume_path,
)
download_elapsed = time.time() - t_download_start
print(f"\nDownload finished in {download_elapsed/60:.1f} min")

# ── Step 2: Score ─────────────────────────────────────────────────────────────
print("\nLoading production ensemble...")
model = AdaptivePINNClassifier(model_dir="src/models/production", device=DEVICE)

print(f"Loading dataset {OUTPUT_PT}...")
d = torch.load(OUTPUT_PT, weights_only=False)
phases    = d["phases"]
primary   = d["fluxes"]
secondary = d["fluxes_secondary"]
odd_even  = d["fluxes_odd_even"]
labels    = d["labels"].numpy().astype(int)
names     = d["names"]
depths    = np.array(d["depths_ppm"], dtype=float)
stellar_r = d["stellar_radius"].numpy()

print(f"Scoring {len(names)} TCEs on {DEVICE}...")
t_score_start = time.time()
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
        case = "MISSED_PLANET_CANDIDATE"
    elif truth == 1 and ensemble_pred == 0:
        case = "SUSPICIOUS_CONFIRMATION"
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

score_elapsed = time.time() - t_score_start
df = pd.DataFrame(results)

# ── Step 3: Report ────────────────────────────────────────────────────────────
total  = len(df)
agree  = (df["truth"] == df["ensemble_pred"]).sum()
tp     = ((df["truth"] == 1) & (df["ensemble_pred"] == 1)).sum()
tn     = ((df["truth"] == 0) & (df["ensemble_pred"] == 0)).sum()
fp_cnt = ((df["truth"] == 0) & (df["ensemble_pred"] == 1)).sum()
fn     = ((df["truth"] == 1) & (df["ensemble_pred"] == 0)).sum()
prec   = tp / (tp + fp_cnt) if (tp + fp_cnt) > 0 else 0.0
rec    = tp / (tp + fn)     if (tp + fn)     > 0 else 0.0
f1     = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

print("\n" + "=" * 60)
print("V10+V6b AND ensemble on 1000-TCE DR25 batch")
print("=" * 60)
print(f"Device:        {DEVICE}")
print(f"Total scored:  {total}")
print(f"Accuracy:      {agree}/{total}  ({agree/total:.1%})")
print(f"TP={tp}  TN={tn}  FP={fp_cnt}  FN={fn}")
print(f"Precision:     {prec:.3f}  ({prec:.1%})")
print(f"Recall:        {rec:.3f}  ({rec:.1%})")
print(f"F1:            {f1:.3f}")
print(f"Scoring time:  {score_elapsed:.1f}s")

for label, case_key, note in [
    ("MISSED PLANET CANDIDATES", "MISSED_PLANET_CANDIDATE",
     "(archive=FP, model=PLANET)"),
    ("SUSPICIOUS CONFIRMATIONS", "SUSPICIOUS_CONFIRMATION",
     "(archive=CONFIRMED, model=FP)"),
]:
    sub = df[df["case"] == case_key]
    print(f"\n{label}: {len(sub)}  {note}")
    if len(sub) > 0:
        print(sub[["koi_id", "depth_ppm", "stellar_r",
                    "p_v10", "p_v6b"]].to_string(index=False))

hi_p = (df["case"] == "high_confidence_planet").sum()
hi_f = (df["case"] == "high_confidence_fp").sum()
print(f"\nHigh-confidence planets (both >0.85): {hi_p}")
print(f"High-confidence FPs   (both <0.15): {hi_f}")

# ── Step 4: Save ──────────────────────────────────────────────────────────────
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved: {OUTPUT_CSV}")
print(f"Total wall time: {(time.time() - t_download_start)/60:.1f} min")
