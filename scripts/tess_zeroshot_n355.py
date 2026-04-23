"""N=355 TESS zero-shot validation of V10 in discovery mode.

MAST CAOMv240 was offline, blocking the curated N=50 fetch.
This offline variant uses data/tess_tce_400.pt (built by
build_dataset.py --mission TESS for the Exp 4b TESS-native
training experiment), and runs the production V10 weights
(src/models/production/v10_f1861.pt) over all 355 TCEs.

The preprocessed channels in that file use the identical
pipeline the paper's Exp 3 uses (flatten sigma_upper=4
sigma_lower=10, normalize, fold, scale to [-pi, pi], subtract 1).
V10 production weights were trained on 500 Kepler TCEs only —
they have never seen any TESS data — so this remains zero-shot.

Mode: discovery — predict PLANET if p_v10 > 0.4
Also reports the strict-threshold result (p > 0.5) as a reference.

Outputs:
    data/tess_zeroshot_n355.csv
    notebooks/figures/tess_zeroshot_n355.png
"""

# ════════════════════════════════════════════════════════
# PRODUCTION MODEL PROTECTION
# Loads READ-ONLY from src/models/production/v10_f1861.pt
# No model weights are saved by this script.
# ════════════════════════════════════════════════════════

import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.models.taylor_cnn_v10 import TaylorCNNv10


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "src/models/production/v10_f1861.pt"
DATA_PATH = "data/tess_tce_400.pt"
THRESHOLD_DISCOVERY = 0.40
THRESHOLD_STRICT = 0.50

FIGDIR = Path("notebooks/figures")
DATADIR = Path("data")
FIGDIR.mkdir(parents=True, exist_ok=True)


def compute_snr(primary_flux_row: np.ndarray) -> float:
    depth = float(np.abs(np.min(primary_flux_row)))
    phase_vec = np.linspace(-math.pi, math.pi, 200)
    oot = np.abs(phase_vec) > 0.6 * math.pi
    noise = float(np.std(primary_flux_row[oot])) if oot.any() else float(np.std(primary_flux_row))
    return 0.0 if noise < 1e-6 else depth / noise


def metrics(y_true: np.ndarray, y_pred: np.ndarray):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn)
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "recall": recall, "precision": prec, "f1": f1, "accuracy": acc,
    }


def main():
    print("="*90)
    print("TESS zero-shot N=355 — V10 discovery mode")
    print(f"Model   : {MODEL_PATH}")
    print(f"Data    : {DATA_PATH}")
    print("="*90)

    d = torch.load(DATA_PATH, weights_only=False)
    phases = d["phases"].to(DEVICE)
    fluxes = d["fluxes"].to(DEVICE)
    fluxes_secondary = d["fluxes_secondary"].to(DEVICE)
    fluxes_odd_even = d["fluxes_odd_even"].to(DEVICE)
    labels = d["labels"].cpu().numpy().astype(int)
    names = d["names"]
    depths_ppm = d["depths_ppm"]
    periods = d["period_days"].cpu().numpy()
    N = len(names)
    print(f"Loaded {N} TCEs  "
          f"(planets {int(labels.sum())}, FPs {int(N - labels.sum())})")

    model = TaylorCNNv10(init_amplitude=0.01).to(DEVICE)
    ck = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    # Batch inference — dataset is small
    with torch.no_grad():
        probs = model(phases, fluxes, fluxes_secondary, fluxes_odd_even)
        probs = probs.squeeze(1).cpu().numpy()

    # Per-TCE SNR from the primary-flux channel
    primary_np = fluxes.cpu().numpy()
    snrs = np.array([compute_snr(primary_np[i]) for i in range(N)])

    # Metrics
    pred_disc = (probs > THRESHOLD_DISCOVERY).astype(int)
    pred_strict = (probs > THRESHOLD_STRICT).astype(int)
    m_disc = metrics(labels, pred_disc)
    m_strict = metrics(labels, pred_strict)

    print("\n" + "="*90)
    print(f"Discovery  (p > {THRESHOLD_DISCOVERY})")
    print("="*90)
    print(f"  Confusion : TP={m_disc['tp']:3d}  TN={m_disc['tn']:3d}  "
          f"FP={m_disc['fp']:3d}  FN={m_disc['fn']:3d}")
    print(f"  Recall    : {m_disc['recall']:.1%}")
    print(f"  Precision : {m_disc['precision']:.1%}")
    print(f"  F1        : {m_disc['f1']:.3f}")
    print(f"  Accuracy  : {m_disc['accuracy']:.1%}")

    print("\n" + "="*90)
    print(f"Strict     (p > {THRESHOLD_STRICT})")
    print("="*90)
    print(f"  Confusion : TP={m_strict['tp']:3d}  TN={m_strict['tn']:3d}  "
          f"FP={m_strict['fp']:3d}  FN={m_strict['fn']:3d}")
    print(f"  Recall    : {m_strict['recall']:.1%}")
    print(f"  Precision : {m_strict['precision']:.1%}")
    print(f"  F1        : {m_strict['f1']:.3f}")
    print(f"  Accuracy  : {m_strict['accuracy']:.1%}")

    print("\nPaper reference (N=8 curated, Exp 3): recall 100%, prec 80%, F1 0.889")
    print("Paper reference (V10 Kepler 76-TCE test): recall 89.5%, prec 82.9%, F1 0.861")

    # Missed planets at discovery threshold
    missed_idx = np.where((labels == 1) & (pred_disc == 0))[0]
    if len(missed_idx):
        print(f"\nMissed planets at p > {THRESHOLD_DISCOVERY}  (N={len(missed_idx)}):")
        order = np.argsort(probs[missed_idx])
        for i in missed_idx[order][:20]:
            print(f"  {names[i]:<16}  P={periods[i]:6.3f}d  "
                  f"depth={depths_ppm[i]:>7.0f}ppm  "
                  f"SNR={snrs[i]:5.1f}  p_v10={probs[i]:.3f}")
        if len(missed_idx) > 20:
            print(f"  ... +{len(missed_idx) - 20} more")

    # SNR of caught vs missed planets
    planet_mask = labels == 1
    caught_mask = planet_mask & (pred_disc == 1)
    missed_mask = planet_mask & (pred_disc == 0)
    if caught_mask.any():
        print(f"\nSNR caught planets  : median={np.median(snrs[caught_mask]):.1f}  "
              f"IQR=[{np.percentile(snrs[caught_mask], 25):.1f}, "
              f"{np.percentile(snrs[caught_mask], 75):.1f}]")
    if missed_mask.any():
        print(f"SNR missed planets  : median={np.median(snrs[missed_mask]):.1f}  "
              f"IQR=[{np.percentile(snrs[missed_mask], 25):.1f}, "
              f"{np.percentile(snrs[missed_mask], 75):.1f}]")

    # FPs that leaked through
    fp_leaked = np.where((labels == 0) & (pred_disc == 1))[0]
    if len(fp_leaked):
        print(f"\nFPs leaked through  : {len(fp_leaked)}")
        order = np.argsort(-probs[fp_leaked])
        for i in fp_leaked[order][:10]:
            print(f"  {names[i]:<16}  P={periods[i]:6.3f}d  "
                  f"depth={depths_ppm[i]:>7.0f}ppm  "
                  f"SNR={snrs[i]:5.1f}  p_v10={probs[i]:.3f}")

    # Save CSV
    csv_path = DATADIR / "tess_zeroshot_n355.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "period_d", "depth_ppm", "snr",
                    "prob", "pred_discovery", "pred_strict", "truth"])
        for i in range(N):
            w.writerow([names[i], float(periods[i]), float(depths_ppm[i]),
                        float(snrs[i]), float(probs[i]),
                        int(pred_disc[i]), int(pred_strict[i]),
                        int(labels[i])])
    print(f"\nSaved {csv_path}")

    # Figure: prob distribution + SNR scatter
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    bins = np.linspace(0, 1, 26)
    planet_probs = probs[labels == 1]
    fp_probs = probs[labels == 0]
    ax1.hist(planet_probs, bins=bins, color="#10b981", alpha=0.75,
             label=f"Planet (N={len(planet_probs)})")
    ax1.hist(fp_probs, bins=bins, color="#ef4444", alpha=0.75,
             label=f"FP (N={len(fp_probs)})")
    ax1.axvline(THRESHOLD_DISCOVERY, color="black", linestyle="--",
                label=f"discovery thr={THRESHOLD_DISCOVERY}")
    ax1.axvline(THRESHOLD_STRICT, color="gray", linestyle=":",
                label=f"strict thr={THRESHOLD_STRICT}")
    ax1.set_xlabel("V10 probability")
    ax1.set_ylabel("count")
    ax1.set_title(f"p_v10 distribution  (N={N})")
    ax1.legend()

    for i in range(N):
        truth_color = "#10b981" if labels[i] == 1 else "#ef4444"
        ok = labels[i] == pred_disc[i]
        marker = "o" if ok else "x"
        ax2.scatter(max(snrs[i], 0.1), probs[i],
                    c=truth_color, marker=marker, s=22,
                    alpha=0.6, edgecolors="black", linewidths=0.3)
    ax2.axhline(THRESHOLD_DISCOVERY, color="black", linestyle="--", alpha=0.5)
    ax2.set_xscale("log")
    ax2.set_xlabel("SNR (log)")
    ax2.set_ylabel("V10 probability")
    ax2.set_title("SNR vs p_v10  (x = misclassified @ disc.)")

    fig.suptitle(
        f"TESS zero-shot N={N} — V10 @ disc. thr {THRESHOLD_DISCOVERY}   "
        f"recall={m_disc['recall']:.1%}  prec={m_disc['precision']:.1%}  "
        f"F1={m_disc['f1']:.3f}",
        fontweight="bold",
    )
    fig.tight_layout()
    fig_path = FIGDIR / "tess_zeroshot_n355.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
