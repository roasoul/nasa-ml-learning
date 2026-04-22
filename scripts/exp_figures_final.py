"""Final-session figures for the experiment round.

Figure 8 (recolored):  notebooks/figures/gate_activation_heatmap.png
    Per-sample gate-vs-primary Pearson correlation on the 76-TCE test
    set, rows colored by class (green=planet, red=FP) via an overlay
    color column.

Figure 10: notebooks/figures/ensemble_scatter.png
    V6 Config B prob vs V10 lambda=0.1 prob, colored by class. Shows
    where the AND ensemble lives.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm

from src.models.taylor_cnn import TaylorCNN
from src.models.taylor_cnn_v10 import TaylorCNNv10


FIGDIR = Path("notebooks/figures")
DATA_PATH = "data/kepler_tce_v6.pt"
V6_PATH = "src/models/taylor_cnn_v6.pt"
V10_PATH = "src/models/taylor_cnn_v10.pt"
SEED_SPLIT = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_split():
    d = torch.load(DATA_PATH, weights_only=False)
    y = d["labels"]
    torch.manual_seed(SEED_SPLIT)
    conf = (y == 1).nonzero(as_tuple=True)[0]
    fp = (y == 0).nonzero(as_tuple=True)[0]
    conf = conf[torch.randperm(len(conf))]
    fp = fp[torch.randperm(len(fp))]
    def split(idx, tf=0.7, vf=0.15):
        n = len(idx); nt = int(n * tf); nv = int(n * vf)
        return idx[:nt], idx[nt:nt + nv], idx[nt + nv:]
    _, _, cte = split(conf); _, _, fte = split(fp)
    return d, torch.cat([cte, fte])


def infer(model, d, idx):
    model.eval()
    with torch.no_grad():
        phase = d["phases"][idx].to(DEVICE)
        prim = d["fluxes"][idx].to(DEVICE)
        sec = d["fluxes_secondary"][idx].to(DEVICE)
        oe = d["fluxes_odd_even"][idx].to(DEVICE)
        if isinstance(model, TaylorCNN):
            return model(phase, prim, sec, oe).squeeze(1).cpu().numpy()
        return model(phase, prim, sec, oe).squeeze(1).cpu().numpy()


def main():
    d, test_idx = test_split()
    labels = d["labels"][test_idx].cpu().numpy()
    names = [d["names"][i] for i in test_idx.tolist()]

    v6 = TaylorCNN(init_amplitude=0.01).to(DEVICE)
    v6.load_state_dict(torch.load(V6_PATH, map_location=DEVICE, weights_only=False)["state_dict"])
    v10 = TaylorCNNv10(init_amplitude=0.01).to(DEVICE)
    v10.load_state_dict(torch.load(V10_PATH, map_location=DEVICE, weights_only=False)["state_dict"])

    probs_v6 = infer(v6, d, test_idx)
    probs_v10 = infer(v10, d, test_idx)

    # Gate templates from V10
    phase_t = torch.linspace(-math.pi, math.pi, 200, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        gates = v10.gate_bank(phase_t).squeeze(0).cpu().numpy()

    fluxes = d["fluxes"][test_idx].cpu().numpy()
    n = len(test_idx)
    corr = np.zeros((n, 5))
    for i in range(n):
        f = fluxes[i]
        df = f - f.mean()
        std_f = f.std() + 1e-12
        for g in range(5):
            v = gates[g]
            dv = v - v.mean()
            std_v = v.std() + 1e-12
            corr[i, g] = ((df * dv).mean() / (std_f * std_v))

    # Figure 8 recolor — sort by class, then by V10 prob
    p_idx = np.where(labels == 1)[0]
    fp_idx = np.where(labels == 0)[0]
    p_idx = p_idx[np.argsort(-probs_v10[p_idx])]
    fp_idx = fp_idx[np.argsort(-probs_v10[fp_idx])]
    order = np.concatenate([p_idx, fp_idx])

    class_col = np.array([[1.0] if labels[i] == 1 else [0.0] for i in order])
    heat = corr[order]

    fig, axes = plt.subplots(1, 3, figsize=(12, 13),
                             gridspec_kw={"width_ratios": [0.8, 10, 0.8]})
    # Class stripe (green=planet, red=FP)
    axes[0].imshow(class_col, aspect="auto",
                   cmap=plt.cm.colors.ListedColormap(["#ef4444", "#10b981"]),
                   vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_xticks([0], labels=["class"])
    axes[0].set_yticks(range(len(order)),
                       labels=[names[i] for i in order], fontsize=6)
    axes[0].axhline(len(p_idx) - 0.5, color="black", lw=2)

    # Heatmap
    norm = TwoSlopeNorm(vmin=-0.8, vcenter=0.0, vmax=0.8)
    im = axes[1].imshow(heat, aspect="auto", cmap="RdBu_r", norm=norm,
                        interpolation="nearest")
    axes[1].set_xticks(range(5), labels=["G1 planet U", "G2 V-shape",
                                          "G3 inv. sec.", "G4 asymmetric",
                                          "G5 Gaussian"])
    axes[1].set_yticks([])
    axes[1].axhline(len(p_idx) - 0.5, color="black", lw=2)
    axes[1].set_title("Figure 8 — V10 gate-vs-primary correlation heatmap\n"
                      "(class stripe: green=planet, red=FP)")
    plt.colorbar(im, ax=axes[1])

    # V10 prob side bar
    prob_col = np.array([[probs_v10[i]] for i in order])
    im2 = axes[2].imshow(prob_col, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    axes[2].set_xticks([0], labels=["V10\nP(planet)"])
    axes[2].set_yticks([])
    axes[2].axhline(len(p_idx) - 0.5, color="black", lw=2)
    plt.colorbar(im2, ax=axes[2])

    fig.tight_layout()
    fig.savefig(FIGDIR / "gate_activation_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIGDIR / 'gate_activation_heatmap.png'}")

    # Figure 10 — ensemble scatter V6 vs V10
    fig, ax = plt.subplots(figsize=(7, 7))
    planet_mask = labels == 1
    fp_mask = labels == 0
    ax.scatter(probs_v6[planet_mask], probs_v10[planet_mask],
               c="#10b981", s=60, alpha=0.8, label=f"Planet (n={planet_mask.sum()})",
               edgecolors="black", linewidths=0.3)
    ax.scatter(probs_v6[fp_mask], probs_v10[fp_mask],
               c="#ef4444", s=60, alpha=0.8, label=f"FP (n={fp_mask.sum()})",
               edgecolors="black", linewidths=0.3)
    ax.axhline(0.5, color="gray", ls="--", alpha=0.6)
    ax.axvline(0.5, color="gray", ls="--", alpha=0.6)
    ax.plot([0, 1], [0, 1], color="gray", ls=":", alpha=0.5)
    ax.set_xlabel("V6 Config B P(planet)")
    ax.set_ylabel("V10 λ=0.1 P(planet)")
    ax.set_title("Figure 10 — V6 vs V10 predictions on 76-TCE test set\n"
                 "AND ensemble hits F1=0.872 (top-right quadrant = both >0.5)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(FIGDIR / "ensemble_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIGDIR / 'ensemble_scatter.png'}")


if __name__ == "__main__":
    main()
