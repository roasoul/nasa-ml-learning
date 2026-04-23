"""N=355 TESS zero-shot — depth-normalized inference test.

Hypothesis: V10 misses deep TESS transits (>10k ppm) because its
gate amplitudes peak near A5 ≈ 0.025 (tuned to Kepler 1-5k ppm).
At 50k ppm the primary channel is 40× larger than any gate output,
so the Conv1d kernel can no longer read the gate-vs-primary
correlation structure it learned.

Fix test: for each TCE, rescale primary / secondary / odd_even by
a single scalar so that min(primary) = -TARGET_DEPTH. The three
channels use the SAME scale so their relative magnitudes (which
encode the EB signature via secondary-to-primary depth ratio)
are preserved.

Sweep TARGET_DEPTH ∈ {0.001, 0.0015, 0.002, 0.003, 0.005}
(≈ 1000-5000 ppm, centred on V10's training distribution).

Outputs:
    data/tess_zeroshot_n355_depthnorm.csv
    notebooks/figures/tess_zeroshot_n355_depthnorm.png
"""

# ════════════════════════════════════════════════════════
# PRODUCTION MODEL PROTECTION
# Loads READ-ONLY from src/models/production/v10_f1861.pt
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
THRESHOLD = 0.40

TARGET_DEPTHS = [0.0010, 0.0015, 0.0020, 0.0030, 0.0050]

DEPTH_BINS = [
    (0, 2000, "0-2k"),
    (2000, 5000, "2-5k"),
    (5000, 10000, "5-10k"),
    (10000, 20000, "10-20k"),
    (20000, 50000, "20-50k"),
]


def metrics(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) else 0.0
    return recall, prec, f1, tp, tn, fp, fn


def infer(model, phases, primary, secondary, oe):
    with torch.no_grad():
        return model(phases, primary, secondary, oe).squeeze(1).cpu().numpy()


def main():
    print("="*90)
    print("TESS N=355 zero-shot — depth-normalized inference sweep")
    print("="*90)

    d = torch.load(DATA_PATH, weights_only=False)
    phases = d["phases"].to(DEVICE)
    primary = d["fluxes"].to(DEVICE)
    secondary = d["fluxes_secondary"].to(DEVICE)
    oe = d["fluxes_odd_even"].to(DEVICE)
    labels = d["labels"].cpu().numpy().astype(int)
    names = d["names"]
    depths_ppm = np.array(d["depths_ppm"], dtype=float)
    N = len(names)

    model = TaylorCNNv10(init_amplitude=0.01).to(DEVICE)
    ck = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    # Baseline (no normalization)
    probs_base = infer(model, phases, primary, secondary, oe)
    pred_base = (probs_base > THRESHOLD).astype(int)
    r, p, f, *_ = metrics(labels, pred_base)
    print(f"\nBaseline (no norm)     recall={r:.1%}  prec={p:.1%}  F1={f:.3f}")

    # Each TCE's actual primary depth, as a tensor of shape (N,)
    actual_depth = (-primary.min(dim=1).values).clamp(min=1e-6)  # (N,)

    all_results = {}

    for td in TARGET_DEPTHS:
        # scale = target / actual; applied to all three obs channels
        scale = (td / actual_depth).unsqueeze(1)  # (N, 1)
        p_norm = primary * scale
        s_norm = secondary * scale
        oe_norm = oe * scale
        probs = infer(model, phases, p_norm, s_norm, oe_norm)
        pred = (probs > THRESHOLD).astype(int)
        r, p, f, tp, tn, fp, fn = metrics(labels, pred)
        all_results[td] = {
            "probs": probs, "pred": pred,
            "recall": r, "precision": p, "f1": f,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        }
        print(f"Target depth = {td:.4f}  "
              f"recall={r:.1%}  prec={p:.1%}  F1={f:.3f}  "
              f"(TP={tp} FP={fp} FN={fn} TN={tn})")

    # Best target depth by F1
    best_td = max(all_results, key=lambda t: all_results[t]["f1"])
    best = all_results[best_td]
    print(f"\nBest by F1: target_depth={best_td:.4f}  F1={best['f1']:.3f}")

    # Depth-stratified comparison: baseline vs best
    print("\n" + "="*90)
    print(f"Depth-stratified recall: baseline vs best (target_depth={best_td:.4f})")
    print("="*90)
    print(f"{'Depth bin':<10} {'Nplanet':>8} {'base rec':>10} {'norm rec':>10} "
          f"{'delta':>7} {'NFP':>5} {'base prec':>10} {'norm prec':>10}")
    print("-"*90)
    for lo, hi, label in DEPTH_BINS:
        m = (depths_ppm >= lo) & (depths_ppm < hi)
        pm = m & (labels == 1)
        fm = m & (labels == 0)
        if not pm.any():
            continue
        br = pred_base[pm].mean()
        nr = best["pred"][pm].mean()
        # per-bin precision (of discoveries that fell in this depth bin)
        bp_tp = ((pred_base == 1) & pm).sum()
        bp_fp = ((pred_base == 1) & fm).sum()
        bp = bp_tp / (bp_tp + bp_fp) if (bp_tp + bp_fp) else float("nan")
        np_tp = ((best["pred"] == 1) & pm).sum()
        np_fp = ((best["pred"] == 1) & fm).sum()
        npr = np_tp / (np_tp + np_fp) if (np_tp + np_fp) else float("nan")
        delta = nr - br
        print(f"{label:<10} {pm.sum():>8d} {br:>10.1%} {nr:>10.1%} "
              f"{delta:>+6.1%}  {fm.sum():>5d} {bp:>10.1%} {npr:>10.1%}")

    # Save CSV — baseline + best norm per TCE
    out_csv = Path("data/tess_zeroshot_n355_depthnorm.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "name", "depth_ppm", "truth",
            "prob_baseline", "pred_baseline",
            f"prob_norm_{best_td:.4f}", f"pred_norm_{best_td:.4f}",
            "target_depth",
        ])
        for i in range(N):
            w.writerow([
                names[i], float(depths_ppm[i]), int(labels[i]),
                float(probs_base[i]), int(pred_base[i]),
                float(best["probs"][i]), int(best["pred"][i]),
                best_td,
            ])
    print(f"\nSaved {out_csv}")

    # Figure: sweep curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    tds = list(all_results.keys())
    recalls = [all_results[t]["recall"] for t in tds]
    precs = [all_results[t]["precision"] for t in tds]
    f1s = [all_results[t]["f1"] for t in tds]
    ax1.plot(tds, recalls, "o-", color="#10b981", label="recall")
    ax1.plot(tds, precs, "s-", color="#ef4444", label="precision")
    ax1.plot(tds, f1s, "^-", color="#2563eb", label="F1")
    br, bp, bf, *_ = metrics(labels, pred_base)
    ax1.axhline(br, color="#10b981", linestyle=":", alpha=0.4, label=f"base recall={br:.2f}")
    ax1.axhline(bp, color="#ef4444", linestyle=":", alpha=0.4, label=f"base prec={bp:.2f}")
    ax1.axhline(bf, color="#2563eb", linestyle=":", alpha=0.4, label=f"base F1={bf:.2f}")
    ax1.set_xlabel("target depth")
    ax1.set_ylabel("score")
    ax1.set_title(f"Depth-norm sweep @ p > {THRESHOLD}")
    ax1.set_xscale("log")
    ax1.legend(fontsize=8, loc="center right")
    ax1.grid(alpha=0.3)

    # Per-bin recall comparison
    labels_plot = []
    base_recall = []
    norm_recall = []
    for lo, hi, label in DEPTH_BINS:
        m = (depths_ppm >= lo) & (depths_ppm < hi)
        pm = m & (labels == 1)
        if not pm.any():
            continue
        labels_plot.append(label)
        base_recall.append(pred_base[pm].mean())
        norm_recall.append(best["pred"][pm].mean())
    x = np.arange(len(labels_plot))
    w = 0.38
    ax2.bar(x - w/2, base_recall, w, color="#94a3b8", label="baseline")
    ax2.bar(x + w/2, norm_recall, w, color="#10b981",
            label=f"depth-norm  td={best_td:.4f}")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_plot)
    ax2.set_ylabel("recall")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Recall by depth bin")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"V10 TESS N=355 — depth-normalized inference  "
        f"(best F1 {best['f1']:.3f} @ td={best_td:.4f})",
        fontweight="bold",
    )
    fig.tight_layout()
    out_png = Path("notebooks/figures/tess_zeroshot_n355_depthnorm.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
