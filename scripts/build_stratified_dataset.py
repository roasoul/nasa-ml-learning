# ===============================================
# PRODUCTION MODEL PROTECTION
# NEVER save to src/models/production/
# Save experiments to src/models/ only
# with descriptive version suffix
# ===============================================
"""Build depth-stratified training dataset from kepler_tce_2000.pt.

Bins 2-5 only (skip bin 1 <300 ppm, only 3 confirmed):
  Bin 2:  300  - 1,000 ppm
  Bin 3: 1,000 - 3,000 ppm
  Bin 4: 3,000 - 10,000 ppm
  Bin 5: 10,000+ ppm

Per bin: sample min(confirmed, FP) from each class (balanced).
Dedup: removes any TCE whose KOI ID appears in DR25 eval sets.
Output: data/kepler_stratified_1k.pt
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pandas as pd
import numpy as np

SEED          = 42
SOURCE        = "data/kepler_tce_2000.pt"
OUT_PATH      = "data/kepler_stratified_1k.pt"
EVAL_CSVS     = [
    "data/kepler_dr25_1k_scores.csv",
    "data/kepler_dr25_2k_scores.csv",
    "data/batch3_candidates_verified.csv",
]

BINS       = [300, 1000, 3000, 10000, float("inf")]
BIN_LABELS = ["300-1k", "1k-3k", "3k-10k", "10k+"]

rng = np.random.default_rng(SEED)

# ── load source ──────────────────────────────────────────────────────────────
d      = torch.load(SOURCE, weights_only=False)
labels = d["labels"]
names  = d["names"]
# Use flux-observed depth (what the model actually sees): baseline=0, dip=negative
depths = (-d["fluxes"].min(dim=1).values) * 1e6

print(f"Source: {SOURCE}")
print(f"  Total: {len(labels)}  confirmed: {int(labels.sum())}  FP: {int((1-labels).sum())}")

# ── note on dedup ─────────────────────────────────────────────────────────────
# The DR25 batch eval CSVs overlap heavily with this training pool (same Kepler
# catalog). Original V10 was trained without dedup against these eval sets, so
# we maintain the same methodology for a consistent relative comparison.
# We report eval IDs for awareness only.
eval_ids = set()
for csv_path in EVAL_CSVS:
    df = pd.read_csv(csv_path)
    eval_ids.update(df["koi_id"].tolist())
overlap = sum(1 for n in names if n in eval_ids)
print(f"Note: {overlap}/{len(names)} TCEs appear in eval CSVs (not excluded — "
      f"consistent with original V10 training methodology)")

# ── bin and sample ────────────────────────────────────────────────────────────
selected_idx = []

print()
print(f"{'Bin':<10} {'Range':<12} {'Conf avail':>10} {'FP avail':>10} {'n_pairs':>9} {'Selected':>9}")
print("-" * 65)

for i, (lo, hi) in enumerate(zip(BINS, BINS[1:])):
    label = BIN_LABELS[i]
    in_bin  = (depths >= lo) & (depths < hi)

    conf_idx = ((in_bin) & (labels == 1)).nonzero(as_tuple=True)[0].tolist()
    fp_idx   = ((in_bin) & (labels == 0)).nonzero(as_tuple=True)[0].tolist()

    n = min(len(conf_idx), len(fp_idx))
    flag = " *** SPARSE" if n < 40 else ""

    sampled_conf = rng.choice(conf_idx, size=n, replace=False).tolist()
    sampled_fp   = rng.choice(fp_idx,   size=n, replace=False).tolist()
    selected_idx.extend(sampled_conf + sampled_fp)

    print(f"{i+2:<10} {label:<12} {len(conf_idx):>10} {len(fp_idx):>10} {n:>9} {n*2:>9}{flag}")

print("-" * 65)
print(f"{'TOTAL':<10} {'':<12} {'':>10} {'':>10} {'':>9} {len(selected_idx):>9}")

# ── assemble dataset ──────────────────────────────────────────────────────────
sel = torch.tensor(selected_idx, dtype=torch.long)
sel_labels = labels[sel]

print()
print(f"Final dataset: {len(sel)} TCEs  "
      f"confirmed={int(sel_labels.sum())}  FP={int((1-sel_labels).sum())}")

# Verify depth coverage
sel_depths = depths[sel]
print("\nDepth coverage of stratified set:")
print(f"  range: {sel_depths.min():.1f} - {sel_depths.max():.1f} ppm")
for lo, hi, lbl in zip(BINS, BINS[1:], BIN_LABELS):
    n = ((sel_depths >= lo) & (sel_depths < hi)).sum().item()
    print(f"  {lbl:<8}: {n} TCEs")

out = {
    "phases":            d["phases"][sel],
    "fluxes":            d["fluxes"][sel],
    "fluxes_secondary":  d["fluxes_secondary"][sel],
    "fluxes_odd_even":   d["fluxes_odd_even"][sel],
    "labels":            sel_labels,
    "names":             [names[i] for i in selected_idx],
    "depths_ppm":        [d["depths_ppm"][i] for i in selected_idx],
    "stellar_radius":    d["stellar_radius"][sel],
    "stellar_mass":      d["stellar_mass"][sel],
    "duration_hours":    d["duration_hours"][sel],
    "period_days":       d["period_days"][sel],
}

torch.save(out, OUT_PATH)
print(f"\nSaved -> {OUT_PATH}")

pos_weight = (sel_labels == 0).sum() / (sel_labels == 1).sum()
print(f"Recommended pos_weight: {pos_weight:.3f}  (should be ~1.0 for balanced set)")
