"""Figure 10 — F1 ablation bar chart, all configs on 76-TCE test set.
Figure 11 — precision/recall scatter for every ensemble strategy.

V4 and V5 numbers are reproduced from their original evaluation on a
16-TCE "easy" hot-Jupiter subset — NOT on the 76-TCE stratified test
set used from V6 onwards. We annotate that caveat directly on the
figure so the bar heights are comparable only within-test-set.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np


FIGDIR = Path("notebooks/figures")


# (label, prec, rec, f1, test_set, color)
# prec/rec/f1 are the reported numbers. test_set = "16" or "76".
ROWS = [
    ("V4",             0.667, 1.000, 0.800, "16", "#9ca3af"),
    ("V5",             0.727, 1.000, 0.842, "16", "#9ca3af"),
    ("V6 Config C",    0.767, 0.868, 0.815, "76", "#60a5fa"),
    ("V6 Config B",    0.720, 0.947, 0.818, "76", "#60a5fa"),
    ("V7.5",           0.767, 0.868, 0.815, "76", "#60a5fa"),
    ("V8",             0.733, 0.868, 0.795, "76", "#c084fc"),
    ("V8.5",           0.717, 0.868, 0.786, "76", "#c084fc"),
    ("V9 best",        0.721, 0.816, 0.765, "76", "#c084fc"),
    ("V10 lambda=0.1", 0.829, 0.895, 0.861, "76", "#10b981"),
    ("V10_5b log R*",  0.795, 0.921, 0.854, "76", "#10b981"),
    ("V6b+V10 AND",    0.850, 0.895, 0.872, "76", "#059669"),
    ("V6b+V10_5b AND", 0.850, 0.895, 0.872, "76", "#059669"),
    ("V10+V10_5b AND", 0.868, 0.868, 0.868, "76", "#059669"),
    ("Triple AND",     0.868, 0.868, 0.868, "76", "#059669"),
    ("V6b+V10 OR0.4",  0.685, 0.974, 0.804, "76", "#f59e0b"),
    ("Triple OR0.4",   0.667, 1.000, 0.800, "76", "#f59e0b"),
    ("TESS zero-shot", 0.800, 1.000, 0.889, "T",  "#be123c"),
]


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)

    # ---------- Figure 10 — F1 ablation bars ----------
    fig, ax = plt.subplots(figsize=(12, 5.5))
    labels = [r[0] for r in ROWS]
    f1s = [r[3] for r in ROWS]
    colors = [r[5] for r in ROWS]
    bars = ax.bar(labels, f1s, color=colors, edgecolor="#1f2937", linewidth=0.6)

    # Annotate bars
    for bar, r in zip(bars, ROWS):
        ts = r[4]
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.006,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=8)
        # Test-set tag
        tag = {"16": "16", "76": "76", "T": "TESS"}[ts]
        ax.text(bar.get_x() + bar.get_width() / 2, 0.01, tag,
                ha="center", va="bottom", fontsize=7, color="white",
                weight="bold")

    ax.axhline(0.815, ls=":", color="#2563eb", alpha=0.7, label="V6 C baseline F1=0.815")
    ax.axhline(0.861, ls=":", color="#10b981", alpha=0.7, label="V10 F1=0.861")
    ax.axhline(0.872, ls="--", color="#059669", alpha=0.9, label="best ensemble F1=0.872")

    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1.0)
    ax.set_title("Figure 10 — F1 ablation across all versions and ensembles\n"
                 "Test-set tag: 16 = 16-TCE hot-Jupiter subset (V4/V5); "
                 "76 = 76-TCE stratified DR25 test; TESS = 8-target zero-shot")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "ablation_f1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIGDIR / 'ablation_f1.png'}")

    # ---------- Figure 11 — precision/recall scatter ----------
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    for label, prec, rec, f1, ts, color in ROWS:
        marker = "o" if ts == "76" else ("s" if ts == "T" else "^")
        ax.scatter(rec, prec, s=120, c=color, marker=marker,
                   edgecolors="black", linewidths=0.6, alpha=0.9)
        ax.annotate(label, (rec, prec), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)
    ax.axvline(0.90, ls="--", color="#059669", alpha=0.5, label="recall >= 90% target")
    ax.axhline(0.80, ls="--", color="#2563eb", alpha=0.5, label="precision > 80% target")
    # Target quadrant
    ax.axvspan(0.90, 1.01, ymin=(0.80 - 0.45) / (1.0 - 0.45), color="#10b981", alpha=0.08)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Figure 11 — precision vs recall tradeoff\n"
                 "Circle = 76-TCE test set, Square = TESS zero-shot, Triangle = 16-TCE V4/V5")
    ax.set_xlim(0.45, 1.02); ax.set_ylim(0.45, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "pr_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIGDIR / 'pr_curve.png'}")


if __name__ == "__main__":
    main()
