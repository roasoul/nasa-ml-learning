"""V6 Config C vs V10 per-TCE diagnostic on the 76-TCE test set.

Loads both models, runs them on the same seed=42 test split, and
reports:

    - FPs V10 newly catches (V6: planet wrong, V10: FP correct)
    - Planets V10 newly catches (V6: FN, V10: TP)
    - Planets V10 misses (V10: FN, V6: may or may not)
    - For each of those: V10 gate-vs-primary correlations to show
      which gate pattern triggered the catch.

Saves:
    data/v10_vs_v6_diagnostic.csv   — per-TCE comparison table
    notebooks/figures/v10_figure_8_gate_heatmap.png — Figure 8
"""

import csv
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


DATA_PATH = "data/kepler_tce_v6.pt"
V6_MODEL = "src/models/taylor_cnn_v6.pt"
V10_MODEL = "src/models/taylor_cnn_v10.pt"
SEED_SPLIT = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIGDIR = Path("notebooks/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)


def same_split_as_training():
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

    _, _, cte = split(conf)
    _, _, fte = split(fp)
    test_idx = torch.cat([cte, fte])
    return d, test_idx


def infer(model, d, idx):
    ph = d["phases"][idx].to(DEVICE)
    p = d["fluxes"][idx].to(DEVICE)
    s = d["fluxes_secondary"][idx].to(DEVICE)
    oe = d["fluxes_odd_even"][idx].to(DEVICE)
    model.eval()
    with torch.no_grad():
        probs = model(ph, p, s, oe).squeeze(1).cpu()
    return probs


def gate_templates(v10_model):
    """Return the 5 gate outputs on the fixed [-pi, pi] phase grid."""
    device = next(v10_model.parameters()).device
    phase = torch.linspace(-math.pi, math.pi, 200, device=device).unsqueeze(0)
    with torch.no_grad():
        gates = v10_model.gate_bank(phase).squeeze(0).cpu().numpy()  # (5, 200)
    return gates


def gate_primary_correlation(d, idx, gates_np):
    """Pearson correlation between each gate template and each TCE's primary_flux."""
    fluxes = d["fluxes"][idx].numpy()  # (N, 200)
    N = fluxes.shape[0]
    corr = np.zeros((N, 5))
    for i in range(N):
        f = fluxes[i]
        df = f - f.mean()
        std_f = f.std() + 1e-12
        for g in range(5):
            v = gates_np[g]
            dv = v - v.mean()
            std_v = v.std() + 1e-12
            corr[i, g] = float((df * dv).mean() / (std_f * std_v))
    return corr


def main():
    d, test_idx = same_split_as_training()
    names = d["names"]
    labels = d["labels"][test_idx].numpy()
    test_names = [names[i] for i in test_idx.tolist()]

    # Load V6 Config C
    v6 = TaylorCNN(init_amplitude=0.01).to(DEVICE)
    ck6 = torch.load(V6_MODEL, map_location=DEVICE, weights_only=False)
    v6.load_state_dict(ck6["state_dict"])

    # Load V10 winner
    v10 = TaylorCNNv10(init_amplitude=0.01).to(DEVICE)
    ck10 = torch.load(V10_MODEL, map_location=DEVICE, weights_only=False)
    v10.load_state_dict(ck10["state_dict"])

    probs_v6 = infer(v6, d, test_idx)
    probs_v10 = infer(v10, d, test_idx)
    preds_v6 = (probs_v6 > 0.5).int().numpy()
    preds_v10 = (probs_v10 > 0.5).int().numpy()
    probs_v6 = probs_v6.numpy()
    probs_v10 = probs_v10.numpy()

    # Sanity check metrics
    def confusion(preds, y):
        TP = int(((preds == 1) & (y == 1)).sum())
        TN = int(((preds == 0) & (y == 0)).sum())
        FP = int(((preds == 1) & (y == 0)).sum())
        FN = int(((preds == 0) & (y == 1)).sum())
        acc = (TP + TN) / len(y)
        prec = TP / (TP + FP) if TP + FP else 0
        rec = TP / (TP + FN) if TP + FN else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        return TP, TN, FP, FN, acc, prec, rec, f1

    v6_conf = confusion(preds_v6, labels)
    v10_conf = confusion(preds_v10, labels)
    print(f"V6  : TP={v6_conf[0]} TN={v6_conf[1]} FP={v6_conf[2]} FN={v6_conf[3]}"
          f"  acc={v6_conf[4]:.1%} prec={v6_conf[5]:.1%} rec={v6_conf[6]:.1%} F1={v6_conf[7]:.3f}")
    print(f"V10 : TP={v10_conf[0]} TN={v10_conf[1]} FP={v10_conf[2]} FN={v10_conf[3]}"
          f"  acc={v10_conf[4]:.1%} prec={v10_conf[5]:.1%} rec={v10_conf[6]:.1%} F1={v10_conf[7]:.3f}")

    # Gate correlations on test set
    gates_np = gate_templates(v10)
    corr = gate_primary_correlation(d, test_idx, gates_np)

    # Categorize each TCE
    # V6 wrong but V10 right
    # V10 wrong but V6 right
    # Both wrong (shared errors)
    rows = []
    for i in range(len(test_idx)):
        row = {
            "name": test_names[i],
            "label": int(labels[i]),
            "v6_prob": float(probs_v6[i]),
            "v6_pred": int(preds_v6[i]),
            "v10_prob": float(probs_v10[i]),
            "v10_pred": int(preds_v10[i]),
            "G1_corr": float(corr[i, 0]),
            "G2_corr": float(corr[i, 1]),
            "G3_corr": float(corr[i, 2]),
            "G4_corr": float(corr[i, 3]),
            "G5_corr": float(corr[i, 4]),
        }
        v6_ok = row["v6_pred"] == row["label"]
        v10_ok = row["v10_pred"] == row["label"]
        if v6_ok and not v10_ok:
            row["delta"] = "V6_only_right"
        elif v10_ok and not v6_ok:
            row["delta"] = "V10_only_right"
        elif v6_ok and v10_ok:
            row["delta"] = "both_right"
        else:
            row["delta"] = "both_wrong"
        rows.append(row)

    # Summary splits
    v10_catches_fp = [r for r in rows if r["label"] == 0
                      and r["v6_pred"] == 1 and r["v10_pred"] == 0]  # V6 FP → V10 TN
    v10_catches_planet = [r for r in rows if r["label"] == 1
                          and r["v6_pred"] == 0 and r["v10_pred"] == 1]  # V6 FN → V10 TP
    v10_misses_planet = [r for r in rows if r["label"] == 1 and r["v10_pred"] == 0]  # V10 FN
    v10_creates_fp = [r for r in rows if r["label"] == 0
                      and r["v6_pred"] == 0 and r["v10_pred"] == 1]  # V6 TN → V10 FP

    def pretty(r):
        return (f"  {r['name']:<12} label={'PLNT' if r['label']==1 else 'FP  '}"
                f"  V6 p={r['v6_prob']:.2f}->{'PL' if r['v6_pred']==1 else 'FP'}"
                f"  V10 p={r['v10_prob']:.2f}->{'PL' if r['v10_pred']==1 else 'FP'}"
                f"  G_corr=[{r['G1_corr']:+.2f},{r['G2_corr']:+.2f},"
                f"{r['G3_corr']:+.2f},{r['G4_corr']:+.2f},{r['G5_corr']:+.2f}]")

    print(f"\n=== FPs V10 catches that V6 missed  (n={len(v10_catches_fp)}) ===")
    for r in v10_catches_fp:
        print(pretty(r))
    print(f"\n=== Planets V10 catches that V6 missed  (n={len(v10_catches_planet)}) ===")
    for r in v10_catches_planet:
        print(pretty(r))
    print(f"\n=== Planets V10 misses (total FN = {len(v10_misses_planet)}) ===")
    for r in v10_misses_planet:
        print(pretty(r))
    print(f"\n=== New FPs V10 introduces (V6 got right, V10 wrong) n={len(v10_creates_fp)} ===")
    for r in v10_creates_fp:
        print(pretty(r))

    # Net V10 gain
    print(f"\nNet: V10 gains {len(v10_catches_fp)} FP catches and {len(v10_catches_planet)} planet catches"
          f"  |  loses {len(v10_creates_fp)} new FPs and has "
          f"{len([r for r in v10_misses_planet if r['v6_pred']==1])} planets V6 had right but V10 lost")

    # Gate-pattern summary for the newly-caught FPs
    if v10_catches_fp:
        newly = np.array([[r["G1_corr"], r["G2_corr"], r["G3_corr"], r["G4_corr"], r["G5_corr"]]
                          for r in v10_catches_fp])
        print(f"\nNewly-caught FPs — mean gate correlation:")
        for i, n in enumerate(["G1", "G2", "G3", "G4", "G5"]):
            print(f"  {n}  mean corr = {newly[:, i].mean():+.3f}  "
                  f"(min {newly[:, i].min():+.3f}, max {newly[:, i].max():+.3f})")

    # Save CSV
    csv_path = Path("data/v10_vs_v6_diagnostic.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nSaved per-TCE table -> {csv_path}")

    # -------- Figure 8: gate heatmap (5 gates × 76 TCEs), colored by class --------
    # Rows ordered as: planets first, FPs second; within each class, sort by V10 prob.
    p_idx = np.where(labels == 1)[0]
    fp_idx = np.where(labels == 0)[0]
    p_idx = p_idx[np.argsort(-probs_v10[p_idx])]
    fp_idx = fp_idx[np.argsort(-probs_v10[fp_idx])]
    order = np.concatenate([p_idx, fp_idx])

    heat = corr[order]
    names_ordered = [test_names[i] for i in order]

    fig, axes = plt.subplots(1, 2, figsize=(14, 13),
                             gridspec_kw={"width_ratios": [10, 1]})
    norm = TwoSlopeNorm(vmin=-0.8, vcenter=0.0, vmax=0.8)
    im = axes[0].imshow(heat, aspect="auto", cmap="RdBu_r", norm=norm,
                        interpolation="nearest")
    axes[0].set_xticks(range(5),
                       labels=["G1\nplanet U", "G2\nV-shape", "G3\ninv. sec.",
                               "G4\nasymmetric", "G5\nGaussian"])
    axes[0].set_yticks(range(len(names_ordered)), labels=names_ordered, fontsize=7)
    axes[0].axhline(len(p_idx) - 0.5, color="black", lw=2)
    axes[0].set_title("Figure 8 — Per-TCE gate-template correlation\n"
                      "(rows: planets above line, FPs below; sorted by V10 planet probability)")
    cbar = plt.colorbar(im, ax=axes[0])
    cbar.set_label("Pearson corr(gate template, primary_flux)")

    # Side panel: predicted class & probability
    pred_col = np.array([[probs_v10[i]] for i in order])
    im2 = axes[1].imshow(pred_col, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    axes[1].set_xticks([0], labels=["V10\nP(planet)"])
    axes[1].set_yticks([])
    axes[1].axhline(len(p_idx) - 0.5, color="black", lw=2)
    plt.colorbar(im2, ax=axes[1])

    fig.tight_layout()
    fig.savefig(FIGDIR / "v10_figure_8_gate_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 8 -> {FIGDIR / 'v10_figure_8_gate_heatmap.png'}")


if __name__ == "__main__":
    main()
