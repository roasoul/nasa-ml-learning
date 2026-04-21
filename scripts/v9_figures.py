"""V9 lambda-sweep diagnostic plots.

Produces three figures:

    (a) notebooks/figures/v9_metrics_vs_lambda.png
        Precision / recall / F1 vs lambda_max, with V6 baseline line.

    (b) notebooks/figures/v9_A_trace.png
        Learned A value over training epochs for each lambda. Confirms
        the A.clamp(min=0.001) constraint held but shows A hit floor
        quickly (gate effectively dead, carried by the clamp).

    (c) notebooks/figures/v9_T12T14_distribution.png
        Per-sample T12/T14 on the test set, split by class, for the
        winning lambda. Shows the DIRECTION of the shape signal:
        planet T12/T14 median vs FP median on THIS dataset.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch


FIGDIR = Path("notebooks/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)
V9_RESULTS = "data/v9_results.pt"
V6_PRECISION = 0.767
V6_F1 = 0.815
V6_RECALL = 0.868


def main():
    r = torch.load(V9_RESULTS, weights_only=False)
    sweep = r["sweep"]
    per_sample = r["per_sample"]
    A_traces = r["A_traces"]
    best_lam = r["best_lambda"]
    test_labels = r["test_labels"].cpu().numpy()

    lambdas = sorted(sweep.keys())
    accs = [sweep[l]["accuracy"] for l in lambdas]
    precs = [sweep[l]["precision"] for l in lambdas]
    recs = [sweep[l]["recall"] for l in lambdas]
    f1s = [sweep[l]["f1"] for l in lambdas]

    # ---- (a) Metrics vs lambda ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lambdas, precs, "o-", label="Precision", color="#1d4ed8", lw=2)
    ax.plot(lambdas, recs, "s-", label="Recall", color="#059669", lw=2)
    ax.plot(lambdas, f1s, "^-", label="F1", color="#dc2626", lw=2)
    ax.axhline(V6_PRECISION, ls=":", color="#1d4ed8", alpha=0.6, label=f"V6 prec {V6_PRECISION:.1%}")
    ax.axhline(V6_F1, ls=":", color="#dc2626", alpha=0.6, label=f"V6 F1 {V6_F1:.3f}")
    ax.set_xlabel("lambda_max (DynamicGeometryLoss)")
    ax.set_ylabel("metric")
    ax.set_title("V9 metrics vs lambda_max — none beats V6 baseline")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(FIGDIR / "v9_metrics_vs_lambda.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIGDIR / 'v9_metrics_vs_lambda.png'}")

    # ---- (b) A trajectory per lambda ----
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"0.1": "#1d4ed8", "0.5": "#059669", "1.0": "#dc2626"}
    for lam in lambdas:
        trace = A_traces[lam]
        ax.plot(range(1, len(trace) + 1), trace, lw=1.6,
                label=f"lambda_max = {lam}  (A_final={trace[-1]:.4f})",
                color=colors.get(f"{lam:.1f}", None))
    ax.axhline(0.001, ls="--", color="gray", alpha=0.7, label="clamp floor (0.001)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Taylor gate A")
    ax.set_yscale("log")
    ax.set_title("V9 — A drifts to the clamp floor for lambda in {0.1, 0.5}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIGDIR / "v9_A_trace.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIGDIR / 'v9_A_trace.png'}")

    # ---- (c) T12/T14 distribution for winning lambda ----
    win = per_sample[best_lam]
    t12_t14 = win["t12_t14"].cpu().numpy()
    auc_norm = win["auc_norm"].cpu().numpy()
    p_t = t12_t14[test_labels == 1]; fp_t = t12_t14[test_labels == 0]
    p_a = auc_norm[test_labels == 1]; fp_a = auc_norm[test_labels == 0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(p_t, bins=20, alpha=0.7, color="#10b981",
                 label=f"Confirmed planets (n={len(p_t)}, med={np.median(p_t):.3f})")
    axes[0].hist(fp_t, bins=20, alpha=0.7, color="#ef4444",
                 label=f"False positives (n={len(fp_t)}, med={np.median(fp_t):.3f})")
    axes[0].axvline(np.median(p_t), color="#10b981", ls="--")
    axes[0].axvline(np.median(fp_t), color="#ef4444", ls="--")
    axes[0].set_xlabel("T12/T14  (normalized ingress fraction)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"T12/T14 by class — lambda_max={best_lam}\n"
                      f"Sign is opposite to naive expectation: planets HIGHER, FPs LOWER")
    axes[0].legend(fontsize=9)
    axes[0].set_xlim(0, 1)

    # Clip AUC_norm for plotting; extreme-FP tails can reach 8+
    axes[1].hist(np.clip(p_a, 0, 3), bins=20, alpha=0.7, color="#10b981",
                 label=f"Confirmed planets (n={len(p_a)}, med={np.median(p_a):.3f})")
    axes[1].hist(np.clip(fp_a, 0, 3), bins=20, alpha=0.7, color="#ef4444",
                 label=f"False positives (n={len(fp_a)}, med={np.median(fp_a):.3f})")
    axes[1].axvline(np.median(p_a), color="#10b981", ls="--")
    axes[1].axvline(np.median(fp_a), color="#ef4444", ls="--")
    axes[1].set_xlabel("AUC / A")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"AUC_norm by class — lambda_max={best_lam}\n"
                      f"~5x separation; penalty direction in loss inverted for this data")
    axes[1].legend(fontsize=9)

    fig.suptitle(f"V9 shape-feature distributions (winning lambda_max={best_lam})",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGDIR / "v9_T12T14_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIGDIR / 'v9_T12T14_distribution.png'}")


if __name__ == "__main__":
    main()
