"""Generate the 4 paper figures that weren't on disk yet.

    Figure 1  — diagnostic_phase_aligned.png
        Phase-aligned 4-channel diagnostic for a canonical planet and an
        archived false positive. Demonstrates what enters the CNN.

    Figure 2  — layer_activations.png
        Multi-template gate bank output (5 fixed-shape gates) for the same
        planet + FP. Shows which physical morphology each sample activates.

    Figure 12 — curriculum_pr_curve.png
        Precision-recall curves for V6b, V10 production, V10 log-R*, and
        V10 curriculum on the 76-TCE paper test set.

    Figure 13 — pr_operating_points.png
        Scatter of every ensemble operating point tried in this project,
        annotated, against the PR frontier.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.models.taylor_cnn import TaylorCNN
from src.models.taylor_cnn_v10 import TaylorCNNv10


KEPLER_V6_PATH = "data/kepler_tce_v6.pt"
PROD_V6B = "src/models/production/v6b_recall947.pt"
PROD_V10 = "src/models/production/v10_f1861.pt"
PROD_V10_LOG = "src/models/production/v10_log_mdwarf.pt"
CURRICULUM_MODEL = "src/models/taylor_cnn_v10_curriculum.pt"

FIG_DIR = Path("notebooks/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEED_SPLIT = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def stratified_split(labels, seed, tf=0.7, vf=0.15):
    torch.manual_seed(seed)
    conf = (labels == 1).nonzero(as_tuple=True)[0]
    fp = (labels == 0).nonzero(as_tuple=True)[0]
    conf = conf[torch.randperm(len(conf))]
    fp = fp[torch.randperm(len(fp))]
    def part(idx):
        n = len(idx); nt = int(n * tf); nv = int(n * vf)
        return idx[:nt], idx[nt:nt + nv], idx[nt + nv:]
    ct, cv, cte = part(conf); ft, fv, fte = part(fp)
    return torch.cat([cte, fte])


def load_model(cls, path):
    blob = torch.load(path, weights_only=False, map_location=DEVICE)
    state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    m = cls(init_amplitude=0.01).to(DEVICE)
    m.load_state_dict(state)
    m.eval()
    return m


def forward(m, phase, primary, secondary, oe):
    with torch.no_grad():
        return m(phase, primary, secondary, oe).squeeze(1).cpu()


def pr_curve(probs, labels, n=101):
    probs = probs.cpu().numpy() if isinstance(probs, torch.Tensor) else np.asarray(probs)
    y = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
    thresholds = np.linspace(0.0, 1.0, n)
    prec = np.zeros(n)
    rec = np.zeros(n)
    for i, t in enumerate(thresholds):
        preds = (probs > t).astype(int)
        TP = int(((preds == 1) & (y == 1)).sum())
        FP = int(((preds == 1) & (y == 0)).sum())
        FN = int(((preds == 0) & (y == 1)).sum())
        prec[i] = TP / (TP + FP) if TP + FP else 1.0
        rec[i] = TP / (TP + FN) if TP + FN else 0.0
    return thresholds, prec, rec


def metrics_bool(pred_bool, labels):
    preds = pred_bool.long().cpu().numpy()
    y = labels.long().cpu().numpy()
    TP = int(((preds == 1) & (y == 1)).sum())
    FP = int(((preds == 1) & (y == 0)).sum())
    FN = int(((preds == 0) & (y == 1)).sum())
    TN = int(((preds == 0) & (y == 0)).sum())
    prec = TP / (TP + FP) if TP + FP else 0.0
    rec = TP / (TP + FN) if TP + FN else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1, (TP, FP, TN, FN)


def pick_showcase_indices(names: list[str], labels: torch.Tensor):
    """Prefer Kepler-5b for the planet (a clean hot-Jupiter with a textbook dip)
    and K06235.01 for the FP. Fall back to any conf/FP if they are missing."""
    planet = None
    for cand in ("K00018.01", "K00017.01", "K00020.01"):
        if cand in names:
            planet = names.index(cand); break
    if planet is None:
        planet = int(((labels == 1).nonzero(as_tuple=True)[0])[0])

    fp = None
    for cand in ("K06235.01", "K00772.01"):
        if cand in names:
            fp = names.index(cand); break
    if fp is None:
        fp = int(((labels == 0).nonzero(as_tuple=True)[0])[0])
    return planet, fp


# --------------------------------------------------------------------
# Figure 1 — Phase-aligned 4-channel diagnostic
# --------------------------------------------------------------------
def figure_1(d, planet_i, fp_i):
    phase = d["phases"][0].cpu().numpy()  # shared grid
    planet_name = d["names"][planet_i]
    fp_name = d["names"][fp_i]

    panels = [
        ("Primary flux",    "fluxes"),
        ("Secondary flux",  "fluxes_secondary"),
        ("Odd/even diff",   "fluxes_odd_even"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 5.5), sharex=True)
    for col, (title, key) in enumerate(panels):
        y_p = d[key][planet_i].cpu().numpy()
        y_f = d[key][fp_i].cpu().numpy()
        axes[0, col].plot(phase, y_p, lw=1.2, color="#1f77b4")
        axes[0, col].axhline(0, color="grey", lw=0.5, alpha=0.5)
        axes[0, col].axvline(0, color="grey", lw=0.5, ls=":", alpha=0.5)
        axes[0, col].set_title(f"Planet · {title}", fontsize=10)

        axes[1, col].plot(phase, y_f, lw=1.2, color="#d62728")
        axes[1, col].axhline(0, color="grey", lw=0.5, alpha=0.5)
        axes[1, col].axvline(0, color="grey", lw=0.5, ls=":", alpha=0.5)
        axes[1, col].set_title(f"False positive · {title}", fontsize=10)
        axes[1, col].set_xlabel("Phase (rad)", fontsize=9)

    axes[0, 0].set_ylabel(f"{planet_name}\n(confirmed planet)", fontsize=9)
    axes[1, 0].set_ylabel(f"{fp_name}\n(false positive)", fontsize=9)
    fig.suptitle("Figure 1 — Phase-aligned 4-channel input after preprocessing",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIG_DIR / "diagnostic_phase_aligned.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# --------------------------------------------------------------------
# Figure 2 — Gate-bank activations (the "layer" diagnostic)
# --------------------------------------------------------------------
def figure_2(d, planet_i, fp_i, v10_model):
    """Gate templates are phase-only (identical across samples) — the per-sample
    signal is the correlation between each template and that sample's primary_flux.
    Show both: template as a grey curve, primary flux overlaid in colour, with
    Pearson correlation annotated."""
    phase_t = d["phases"][[planet_i]].to(DEVICE)
    with torch.no_grad():
        gates = v10_model.gate_bank(phase_t).cpu().numpy()[0]  # (5, 200)

    phase = d["phases"][0].cpu().numpy()
    gate_names = ["G1 planet U", "G2 V-shape", "G3 inv. secondary",
                  "G4 asymmetric", "G5 Gaussian"]

    fig, axes = plt.subplots(2, 5, figsize=(13, 5.5), sharex=True)
    for row, (idx, row_label, color) in enumerate([
        (planet_i, d["names"][planet_i] + "\n(planet)", "#1f77b4"),
        (fp_i,     d["names"][fp_i]     + "\n(FP)",     "#d62728"),
    ]):
        primary = d["fluxes"][idx].cpu().numpy()
        for j in range(5):
            ax = axes[row, j]
            # gate template as grey background, primary flux in colour
            ax.plot(phase, gates[j], lw=1.0, color="grey", alpha=0.6,
                    label="gate template" if (row == 0 and j == 0) else None)
            ax.plot(phase, primary, lw=1.2, color=color,
                    label="primary flux" if (row == 0 and j == 0) else None)
            ax.axhline(0, color="grey", lw=0.5, alpha=0.3)
            ax.axvline(0, color="grey", lw=0.5, ls=":", alpha=0.3)
            # Pearson correlation between gate and primary
            pc = np.corrcoef(gates[j], primary)[0, 1]
            ax.text(0.03, 0.04, f"r = {pc:+.2f}", transform=ax.transAxes,
                    fontsize=8, color="black",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))
            if row == 0:
                ax.set_title(gate_names[j], fontsize=9)
            if row == 1:
                ax.set_xlabel("Phase (rad)", fontsize=9)
        axes[row, 0].set_ylabel(row_label, fontsize=9)

    axes[0, 0].legend(loc="upper right", fontsize=7)
    fig.suptitle("Figure 2 — Fixed gate templates vs primary flux (Pearson r is "
                 "the per-sample signal V10's CNN uses)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = FIG_DIR / "layer_activations.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# --------------------------------------------------------------------
# Figure 12 — Curriculum PR curves
# --------------------------------------------------------------------
def figure_12(probs_dict, labels):
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    colors = {
        "V6b (prod)":          "#2ca02c",
        "V10 production":      "#1f77b4",
        "V10 log-R*":          "#9467bd",
        "V10 curriculum":      "#d62728",
    }
    markers = {
        "V6b (prod)":          "o",
        "V10 production":      "s",
        "V10 log-R*":          "^",
        "V10 curriculum":      "D",
    }
    for label, probs in probs_dict.items():
        _, prec, rec = pr_curve(probs, labels, n=101)
        ax.plot(rec, prec, lw=2, color=colors[label], label=label)
        # annotate threshold 0.5 marker
        p, r, f1, _ = metrics_bool((probs > 0.5), labels)
        ax.scatter([r], [p], s=60, marker=markers[label], color=colors[label],
                   edgecolor="black", linewidth=0.8, zorder=5)

    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Figure 12 — Precision-recall curves on the 76-TCE test set\n"
                 "(markers = threshold 0.5 operating point)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    out = FIG_DIR / "curriculum_pr_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# --------------------------------------------------------------------
# Figure 13 — Ensemble operating points scatter
# --------------------------------------------------------------------
def figure_13(probs_dict, labels, p_v6b, p_v10, p_log, p_curr):
    """Scatter of every ensemble / threshold combo we evaluated."""
    labels_t = labels

    points = []  # (name, precision, recall, style)

    # Single models @ 0.5
    for name, probs in probs_dict.items():
        p, r, f1, _ = metrics_bool((probs > 0.5), labels_t)
        points.append((f"{name} @0.5", p, r, "single"))

    # V6b + V10 AND @ 0.5 (session best)
    p, r, f1, _ = metrics_bool((p_v6b > 0.5) & (p_v10 > 0.5), labels_t)
    points.append(("V6b+V10 AND (F1 0.872)", p, r, "best"))

    # V6b + V10_curr AND sweep
    for thr in (0.40, 0.45, 0.50, 0.55, 0.60):
        pr_, rc_, f1_, _ = metrics_bool((p_v6b > 0.5) & (p_curr > thr), labels_t)
        points.append((f"V6b+V10_curr AND thr={thr:.2f}", pr_, rc_, "and_curr"))

    # 3-way AND at thresholds
    for thr in (0.40, 0.45, 0.50):
        pr_, rc_, f1_, _ = metrics_bool(
            (p_v6b > thr) & (p_v10 > thr) & (p_curr > thr), labels_t)
        points.append((f"3-way AND thr={thr:.2f}", pr_, rc_, "and3"))

    # Triple OR @ 0.4
    pr_, rc_, f1_, _ = metrics_bool(
        (p_v6b > 0.4) | (p_v10 > 0.4) | (p_log > 0.4), labels_t)
    points.append(("Triple OR @0.4", pr_, rc_, "or"))

    # 2-of-3 voting @ 0.5
    vote = (p_v6b > 0.5).int() + (p_v10 > 0.5).int() + (p_curr > 0.5).int()
    pr_, rc_, f1_, _ = metrics_bool(vote >= 2, labels_t)
    points.append(("2-of-3 vote @0.5", pr_, rc_, "vote"))

    style_map = {
        "single":  ("o", "#1f77b4", 70),
        "best":    ("*", "#d62728", 260),
        "and_curr":("D", "#9467bd", 70),
        "and3":    ("s", "#2ca02c", 70),
        "or":      ("^", "#ff7f0e", 110),
        "vote":    ("P", "#8c564b", 110),
    }

    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    # F1 isocontours
    R = np.linspace(0.02, 1.0, 200)
    P = np.linspace(0.02, 1.0, 200)
    RR, PP = np.meshgrid(R, P)
    F1 = np.where(RR + PP > 0, 2 * RR * PP / (RR + PP + 1e-12), 0)
    cs = ax.contour(RR, PP, F1, levels=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9],
                    colors="lightgrey", linewidths=0.8, alpha=0.7)
    ax.clabel(cs, inline=True, fontsize=7, fmt="F1=%.2f")

    seen_legend = set()
    for name, p, r, style in points:
        m, c, s = style_map[style]
        lbl = style if style not in seen_legend else None
        seen_legend.add(style)
        ax.scatter(r, p, s=s, marker=m, color=c, edgecolor="black",
                   linewidth=0.8, alpha=0.9, zorder=3,
                   label=None if lbl is None else {
                       "single":  "Single model @0.5",
                       "best":    "Session best (V6b+V10 AND)",
                       "and_curr":"V6b+V10_curr AND (thr sweep)",
                       "and3":    "3-way AND (thr sweep)",
                       "or":      "Triple OR @0.4",
                       "vote":    "2-of-3 vote",
                   }[lbl])

    # Annotate a few key points
    for name, p, r, style in points:
        if style in ("best", "or") or "thr=0.40" in name:
            ax.annotate(name, (r, p), xytext=(5, 5), textcoords="offset points",
                        fontsize=7.5, alpha=0.9)

    ax.set_xlim(0, 1.03); ax.set_ylim(0, 1.03)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Figure 13 — Ensemble operating points on the 76-TCE test set")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "pr_operating_points.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    print(f"Device: {DEVICE}")
    d = torch.load(KEPLER_V6_PATH, weights_only=False)
    test_idx = stratified_split(d["labels"], SEED_SPLIT)
    labels = d["labels"][test_idx]
    names = [d["names"][i] for i in test_idx.tolist()]

    phase = d["phases"][test_idx].to(DEVICE)
    primary = d["fluxes"][test_idx].to(DEVICE)
    secondary = d["fluxes_secondary"][test_idx].to(DEVICE)
    odd_even = d["fluxes_odd_even"][test_idx].to(DEVICE)
    r = d["stellar_radius"][test_idx].clamp(min=0.01).unsqueeze(1)
    scale = torch.log1p(r).to(DEVICE)

    v6b = load_model(TaylorCNN, PROD_V6B)
    v10 = load_model(TaylorCNNv10, PROD_V10)
    v10_log = load_model(TaylorCNNv10, PROD_V10_LOG)
    v10_curr = load_model(TaylorCNNv10, CURRICULUM_MODEL)

    p_v6b = forward(v6b, phase, primary, secondary, odd_even)
    p_v10 = forward(v10, phase, primary, secondary, odd_even)
    p_log = forward(v10_log, phase, primary * scale, secondary * scale, odd_even * scale)
    p_curr = forward(v10_curr, phase, primary, secondary, odd_even)

    probs_dict = {
        "V6b (prod)":      p_v6b,
        "V10 production":  p_v10,
        "V10 log-R*":      p_log,
        "V10 curriculum":  p_curr,
    }

    # Use the FULL dataset to pick showcase TCEs so we can reach Kepler-5b etc.
    planet_i, fp_i = pick_showcase_indices(d["names"], d["labels"])
    print(f"Figure 1/2 showcase: planet={d['names'][planet_i]}, FP={d['names'][fp_i]}")

    print("Generating figures...")
    figure_1(d, planet_i, fp_i)
    figure_2(d, planet_i, fp_i, v10)
    figure_12(probs_dict, labels)
    figure_13(probs_dict, labels, p_v6b, p_v10, p_log, p_curr)

    print("\nDone. Figures written to notebooks/figures/:")
    for f in ("diagnostic_phase_aligned", "layer_activations",
              "curriculum_pr_curve", "pr_operating_points"):
        path = FIG_DIR / f"{f}.png"
        kb = path.stat().st_size / 1024 if path.exists() else 0
        print(f"  {f}.png  ({kb:.1f} KB)")


if __name__ == "__main__":
    main()
