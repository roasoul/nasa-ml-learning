"""V10 diagnostic figures.

    (a) v10_metrics_vs_lambda.png    Precision / recall / F1 sweep, V6 line
    (b) v10_amplitude_traces.png     A1..A5 over training epochs, per lambda
    (c) v10_learned_templates.png    Each gate's learned shape at best lambda
    (d) v10_gate_primary_corr.png    Per-sample correlation between each gate
                                     output and primary_flux, split by class.
                                     Replaces the naive "mean |gate|" heatmap
                                     which is batch-constant (gates are
                                     phase-only, phase is identical per TCE).
    (e) v10_T12T14_distribution.png  Shape-feature distributions at best λ.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.models.taylor_cnn_v10 import TaylorCNNv10


FIGDIR = Path("notebooks/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = "data/kepler_tce_v6.pt"
V10_RESULTS = "data/v10_results.pt"
V10_MODEL = "src/models/taylor_cnn_v10.pt"
V6_PREC = 0.767
V6_F1 = 0.815
V6_REC = 0.868


def main():
    r = torch.load(V10_RESULTS, weights_only=False)
    d = torch.load(DATA_PATH, weights_only=False)
    sweep = r["sweep"]
    per_sample = r["per_sample"]
    amp_traces = r["amp_traces"]
    best_lam = r["best_lambda"]
    test_labels = r["test_labels"].cpu().numpy()
    test_idx = r["test_idx"]

    lambdas = sorted(sweep.keys())
    precs = [sweep[l]["precision"] for l in lambdas]
    recs = [sweep[l]["recall"] for l in lambdas]
    f1s = [sweep[l]["f1"] for l in lambdas]

    # ---- (a) Metrics vs lambda ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lambdas, precs, "o-", label="Precision", color="#1d4ed8", lw=2)
    ax.plot(lambdas, recs, "s-", label="Recall", color="#059669", lw=2)
    ax.plot(lambdas, f1s, "^-", label="F1", color="#dc2626", lw=2)
    ax.axhline(V6_PREC, ls=":", color="#1d4ed8", alpha=0.6, label=f"V6 prec {V6_PREC:.1%}")
    ax.axhline(V6_F1, ls=":", color="#dc2626", alpha=0.6, label=f"V6 F1 {V6_F1:.3f}")
    ax.axhline(0.80, ls="--", color="gray", alpha=0.5, label="target prec 80%")
    ax.set_xlabel("lambda_max (InvertedGeometryLoss)")
    ax.set_ylabel("metric")
    ax.set_title("V10 metrics vs lambda_max — lambda=0.1 beats V6 baseline")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.3)
    ax.set_ylim(0.5, 1.0)
    fig.tight_layout()
    fig.savefig(FIGDIR / "v10_metrics_vs_lambda.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIGDIR / 'v10_metrics_vs_lambda.png'}")

    # ---- (b) Amplitude traces ----
    fig, axes = plt.subplots(1, len(lambdas), figsize=(15, 4.5), sharey=True)
    gate_colors = {"A1": "#2563eb", "A2": "#059669", "A3": "#7c3aed",
                   "A4": "#ea580c", "A5": "#dc2626"}
    for ax, lam in zip(axes, lambdas):
        traces = amp_traces[lam]
        for k, color in gate_colors.items():
            ax.plot(range(1, len(traces[k]) + 1), traces[k],
                    lw=1.5, label=k, color=color)
        ax.axhline(0.001, ls="--", color="gray", alpha=0.5, label="clamp floor")
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_title(f"lambda_max = {lam}")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("amplitude (log)")
    axes[-1].legend(fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle("V10 — amplitude trajectories per lambda (A1 hits floor in every run)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGDIR / "v10_amplitude_traces.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIGDIR / 'v10_amplitude_traces.png'}")

    # ---- (c) Learned templates ----
    model = TaylorCNNv10()
    ckpt = torch.load(V10_MODEL, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    phase = torch.linspace(-math.pi, math.pi, 200).unsqueeze(0)
    with torch.no_grad():
        gates = model.gate_bank(phase).squeeze(0).numpy()  # (5, 200)
    amps = model.gate_bank.amplitudes()
    names_g = [
        f"G1 planet U  (A={amps['A1']:.4f})",
        f"G2 V-shape    (A={amps['A2']:.4f})",
        f"G3 inv. sec.  (A={amps['A3']:.4f})",
        f"G4 asymmetric (A={amps['A4']:.4f})",
        f"G5 Gaussian   (A={amps['A5']:.4f})",
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2563eb", "#059669", "#7c3aed", "#ea580c", "#dc2626"]
    for i, n in enumerate(names_g):
        ax.plot(phase.squeeze(0).numpy(), gates[i], lw=2, label=n, color=colors[i])
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel("phase [rad]")
    ax.set_ylabel("gate output")
    ax.set_title(f"V10 learned gate templates at winning lambda_max = {best_lam}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "v10_learned_templates.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIGDIR / 'v10_learned_templates.png'}")

    # ---- (d) Per-sample gate × primary correlation by class ----
    # Gates are phase-only so mean-|gate| is batch-constant. The per-sample
    # signal is how each gate matches each TCE's primary_flux.
    phases_test = d["phases"][torch.tensor(test_idx)].numpy()
    fluxes_test = d["fluxes"][torch.tensor(test_idx)].numpy()
    gates_np = gates  # (5, 200), fixed template per gate
    n_test = len(test_idx)
    corr = np.zeros((n_test, 5))
    for i in range(n_test):
        f_i = fluxes_test[i]
        for g in range(5):
            v = gates_np[g]
            std_f = f_i.std() + 1e-12
            std_v = v.std() + 1e-12
            corr[i, g] = float(((f_i - f_i.mean()) * (v - v.mean())).mean() / (std_f * std_v))

    # Heatmap ordered by class
    order = np.argsort(-test_labels)  # planets (1) first
    corr_sorted = corr[order]
    names_short = ["G1", "G2", "G3", "G4", "G5"]
    fig, ax = plt.subplots(figsize=(10, 12))
    im = ax.imshow(corr_sorted, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    n_planet = int(test_labels.sum())
    ax.axhline(n_planet - 0.5, color="black", lw=2)
    ax.set_xticks(range(5), labels=names_short)
    ax.set_ylabel(f"test TCE (planets rows 0..{n_planet-1}, FPs rows {n_planet}..{n_test-1})")
    ax.set_xlabel("gate")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Pearson corr(gate template, primary_flux)")
    ax.set_title("V10 gate-vs-primary correlation heatmap — planets above line, FPs below")
    fig.tight_layout()
    fig.savefig(FIGDIR / "v10_gate_primary_corr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIGDIR / 'v10_gate_primary_corr.png'}")

    # Summary numbers
    p_mean = corr[test_labels == 1].mean(axis=0)
    fp_mean = corr[test_labels == 0].mean(axis=0)
    print(f"\nPer-gate mean correlation with primary_flux on test set:")
    print(f"  {'gate':<4} {'planet mean':>13} {'FP mean':>13} {'diff':>8}")
    for i, n in enumerate(names_short):
        print(f"  {n:<4} {p_mean[i]:>+13.4f} {fp_mean[i]:>+13.4f} {p_mean[i]-fp_mean[i]:>+8.4f}")

    # ---- (e) Shape-feature distribution at winning lambda ----
    win = per_sample[best_lam]
    t12_t14 = win["t12_t14"].cpu().numpy()
    auc_norm = win["auc_norm"].cpu().numpy()
    p_t = t12_t14[test_labels == 1]; fp_t = t12_t14[test_labels == 0]
    p_a = auc_norm[test_labels == 1]; fp_a = auc_norm[test_labels == 0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(p_t, bins=20, alpha=0.7, color="#10b981",
                 label=f"Planets (n={len(p_t)}, med={np.median(p_t):.3f})")
    axes[0].hist(fp_t, bins=20, alpha=0.7, color="#ef4444",
                 label=f"FPs (n={len(fp_t)}, med={np.median(fp_t):.3f})")
    axes[0].axvline(np.median(p_t), color="#10b981", ls="--")
    axes[0].axvline(np.median(fp_t), color="#ef4444", ls="--")
    axes[0].set_xlabel("T12/T14"); axes[0].set_ylabel("count")
    axes[0].set_title(f"T12/T14 by class — V10 lambda={best_lam}")
    axes[0].legend(fontsize=9)
    axes[0].set_xlim(0, 1)

    axes[1].hist(np.clip(p_a, 0, 3), bins=20, alpha=0.7, color="#10b981",
                 label=f"Planets (n={len(p_a)}, med={np.median(p_a):.3f})")
    axes[1].hist(np.clip(fp_a, 0, 3), bins=20, alpha=0.7, color="#ef4444",
                 label=f"FPs (n={len(fp_a)}, med={np.median(fp_a):.3f})")
    axes[1].axvline(np.median(p_a), color="#10b981", ls="--")
    axes[1].axvline(np.median(fp_a), color="#ef4444", ls="--")
    axes[1].set_xlabel("AUC / A1"); axes[1].set_ylabel("count")
    axes[1].set_title(f"AUC_norm by class — V10 lambda={best_lam}")
    axes[1].legend(fontsize=9)

    fig.suptitle(f"V10 shape-feature distributions (lambda_max={best_lam})",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGDIR / "v10_T12T14_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIGDIR / 'v10_T12T14_distribution.png'}")


if __name__ == "__main__":
    main()
