"""V8 / V8.5 figure generation.

Produces four figures:

    (a) B histogram — confirmed planets vs FPs.
        Since B is a single global parameter in V8.5, this plot shows
        the *closed-form per-sample B* fit on each TCE's primary_flux,
        which is the right per-sample quantity (same estimator used
        by the SS-flag validation, Figure 7).
    (b) AUC_norm histogram — confirmed planets vs FPs.
    (c) Scatter of per-sample B vs AUC_norm, colored by class.
    (d) K03745.01 gate output and primary-flux dip — does the
        learned gate capture the V-shape?

Files → notebooks/figures/{b_histogram,auc_norm_histogram,
                            b_vs_aucnorm_scatter,K03745_gate}.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.models.taylor_cnn_v8 import TaylorCNNv8


FIGDIR = Path("notebooks/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = "data/kepler_tce_v6.pt"
RESULTS_PATH = "data/v8_results.pt"
V85_PATH = "src/models/taylor_cnn_v85.pt"


def fit_B_closed_form(phase: np.ndarray, flux: np.ndarray) -> tuple[float, float]:
    """Closed-form least-squares fit of y ≈ -A · (1 - x²/2 + B · x⁴/24).

    Only fits where the observed flux is negative (inside the dip) — baseline
    regions have no signal about A or B and would only add noise.

    Model rewritten for linearity in (A, AB):
        y = -A + A·x²/2 - (A·B)·x⁴/24
        y = -A + A·x²/2 + C·x⁴/24         where C = -A·B
    Let θ = (A, C), design matrix:
        X = [ -1 + x²/2,   x⁴/24 ]
    Solve X·θ = y via lstsq. Then B = -C / A.

    Returns (A, B). If A comes out ≤ 0 (no dip), returns (0, 0).
    """
    mask = flux < 0
    if mask.sum() < 5:
        return 0.0, 0.0
    x = phase[mask]
    y = flux[mask]
    X = np.stack([-1.0 + x ** 2 / 2.0, x ** 4 / 24.0], axis=1)
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    A_fit, C_fit = theta
    if A_fit <= 1e-6:
        return 0.0, 0.0
    B_fit = -C_fit / A_fit
    return float(A_fit), float(B_fit)


def compute_per_sample_B(phases: torch.Tensor, fluxes: torch.Tensor) -> np.ndarray:
    ph = phases.cpu().numpy()
    fl = fluxes.cpu().numpy()
    n = fl.shape[0]
    Bs = np.zeros(n, dtype=np.float64)
    for i in range(n):
        _, Bs[i] = fit_B_closed_form(ph[i], fl[i])
    return Bs


def main():
    d = torch.load(DATA_PATH, weights_only=False)
    r = torch.load(RESULTS_PATH, weights_only=False)

    labels = r["test_labels"].cpu().numpy()
    test_idx = r["test_idx"]
    shape_feats = r["v85"]["shape_features"].cpu().numpy()
    probs = r["v85"]["probs"].cpu().numpy()

    # Per-sample B via closed-form fit on primary flux
    phases_all = d["phases"]
    fluxes_all = d["fluxes"]
    names = d["names"]

    phases_test = phases_all[torch.tensor(test_idx)]
    fluxes_test = fluxes_all[torch.tensor(test_idx)]
    B_per_sample = compute_per_sample_B(phases_test, fluxes_test)

    B_p = B_per_sample[labels == 1]
    B_fp = B_per_sample[labels == 0]
    auc_p = shape_feats[labels == 1, 2]
    auc_fp = shape_feats[labels == 0, 2]

    print(f"B per-sample (closed-form): planets n={len(B_p)}, FPs n={len(B_fp)}")
    print(f"  Planets B: median={np.median(B_p):+.3f}  mean={np.mean(B_p):+.3f}  "
          f"min={B_p.min():+.3f}  max={B_p.max():+.3f}")
    print(f"  FPs     B: median={np.median(B_fp):+.3f}  mean={np.mean(B_fp):+.3f}  "
          f"min={B_fp.min():+.3f}  max={B_fp.max():+.3f}")

    # -------- (a) B histogram --------
    fig, ax = plt.subplots(figsize=(8, 5))
    # Clip to [-5, 5] for plotting — fits on pure noise can go to extremes
    B_p_clip = np.clip(B_p, -5, 5)
    B_fp_clip = np.clip(B_fp, -5, 5)
    ax.hist(B_p_clip, bins=25, alpha=0.65, color="#10b981",
            label=f"Confirmed planets (n={len(B_p)})")
    ax.hist(B_fp_clip, bins=25, alpha=0.65, color="#ef4444",
            label=f"False positives (n={len(B_fp)})")
    ax.axvline(np.median(B_p), color="#10b981", ls="--",
               label=f"planet median={np.median(B_p):+.2f}")
    ax.axvline(np.median(B_fp), color="#ef4444", ls="--",
               label=f"FP median={np.median(B_fp):+.2f}")
    ax.set_xlabel("Per-sample B (closed-form fit on primary_flux)")
    ax.set_ylabel("Count")
    ax.set_title("(a) Learned shape parameter B by class\n"
                 "U-shape planets expected near B=+1; V-shape EBs expected near B=0 or negative")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "v8_B_histogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIGDIR / 'v8_B_histogram.png'}")

    # -------- (b) AUC_norm histogram --------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(np.clip(auc_p, 0, 5), bins=25, alpha=0.65, color="#10b981",
            label=f"Confirmed planets (n={len(auc_p)})")
    ax.hist(np.clip(auc_fp, 0, 5), bins=25, alpha=0.65, color="#ef4444",
            label=f"False positives (n={len(auc_fp)})")
    ax.axvline(np.median(auc_p), color="#10b981", ls="--",
               label=f"planet median={np.median(auc_p):.2f}")
    ax.axvline(np.median(auc_fp), color="#ef4444", ls="--",
               label=f"FP median={np.median(auc_fp):.2f}")
    ax.set_xlabel("AUC / A  (dip-energy normalized by learned amplitude)")
    ax.set_ylabel("Count")
    ax.set_title("(b) AUC_norm by class\n"
                 "Deep-dip EBs give AUC_norm >> 1; planets cluster near 0.1–0.3")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "v8_AUCnorm_histogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIGDIR / 'v8_AUCnorm_histogram.png'}")

    # -------- (c) Scatter B vs AUC/A --------
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(np.clip(B_fp, -5, 5), np.clip(auc_fp, 0, 5),
               c="#ef4444", s=50, alpha=0.7, label=f"FP (n={len(B_fp)})")
    ax.scatter(np.clip(B_p, -5, 5), np.clip(auc_p, 0, 5),
               c="#10b981", s=50, alpha=0.7, label=f"Planet (n={len(B_p)})")
    ax.set_xlabel("Per-sample B (closed-form fit)")
    ax.set_ylabel("AUC / A")
    ax.set_title("(c) Shape scatter — B vs AUC_norm, colored by class")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "v8_B_vs_AUCnorm.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIGDIR / 'v8_B_vs_AUCnorm.png'}")

    # -------- (d) K03745.01 gate output --------
    # Find K03745.01 in the full dataset and plot its gate output
    koi = "K03745.01"
    if koi in names:
        idx = names.index(koi)
    else:
        # fallback — any name containing 03745
        candidates = [i for i, n in enumerate(names) if "03745" in n]
        idx = candidates[0] if candidates else 0
        print(f"K03745.01 not exact; using index {idx} ({names[idx]})")

    model = TaylorCNNv8(use_shape_features=True)
    ckpt = torch.load(V85_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ph = phases_all[idx:idx+1]
    pf = fluxes_all[idx:idx+1]
    sf = d["fluxes_secondary"][idx:idx+1]
    oe = d["fluxes_odd_even"][idx:idx+1]
    with torch.no_grad():
        _ = model(ph, pf, sf, oe)
        gate_out = model.taylor_gate(ph).squeeze(0).cpu().numpy()

    # Per-sample B fit for the title
    A_k, B_k = fit_B_closed_form(ph.squeeze(0).cpu().numpy(), pf.squeeze(0).cpu().numpy())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    phase_vec = ph.squeeze(0).cpu().numpy()
    axes[0].plot(phase_vec, pf.squeeze(0).cpu().numpy(),
                 color="#1f77b4", lw=1.2)
    axes[0].axhline(0, color="gray", lw=0.5)
    axes[0].set_xlabel("phase [rad]")
    axes[0].set_ylabel("primary_flux")
    axes[0].set_title(f"{names[idx]} primary flux\n"
                      f"closed-form fit: A={A_k:+.4f}  B={B_k:+.3f}")
    axes[0].grid(alpha=0.3)

    axes[1].plot(phase_vec, gate_out, color="#ef4444", lw=1.5)
    axes[1].axhline(0, color="gray", lw=0.5)
    axes[1].set_xlabel("phase [rad]")
    axes[1].set_ylabel("gate_output (learned template)")
    axes[1].set_title(f"V8.5 learned gate at {names[idx]}\n"
                      f"global A={model.taylor_gate.A.item():+.4f}  "
                      f"global B={model.taylor_gate.B.item():+.3f}")
    axes[1].grid(alpha=0.3)

    fig.suptitle("(d) K03745.01 — V-shape FP case", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGDIR / "v8_K03745_gate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIGDIR / 'v8_K03745_gate.png'}")


if __name__ == "__main__":
    main()
