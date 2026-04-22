"""Experiment 7 — V5 + V10 ensemble.

V5 is a 3-channel model (phase gate + primary + secondary — no odd/even
channel). It was trained on the *older 100-TCE* kepler_tce.pt dataset.
Our V10 test set is the 76-TCE slice of the 500-TCE kepler_tce_v6.pt
— so there may be train/test overlap for V5. We flag this caveat in
the report.

Ensemble strategies:
    A. 0.3 * v5 + 0.7 * v10, thr=0.5
    B. 0.5 * v5 + 0.5 * v10, thr=0.5
    C. v5 > 0.3 OR v10 > 0.5 (recall-first)
    D. v5 > 0.5 AND v10 > 0.5 (precision-first)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from src.models.taylor_cnn_v10 import TaylorCNNv10
from src.models.taylor_layer import TaylorGateLayer


DATA_PATH = "data/kepler_tce_v6.pt"
V5_PATH = "src/models/taylor_cnn_v5.pt"
V10_PATH = "src/models/taylor_cnn_v10.pt"
SEED_SPLIT = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TaylorCNNV5(nn.Module):
    """3-channel V5 architecture — phase gate + primary + secondary."""

    def __init__(self, init_amplitude=0.01):
        super().__init__()
        self.taylor_gate = TaylorGateLayer(init_amplitude=init_amplitude)
        self.cnn = nn.Sequential(
            nn.BatchNorm1d(3),
            nn.Conv1d(3, 8, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(16, 1)

    def forward(self, phase, primary_flux, secondary_flux):
        gate_out = self.taylor_gate(phase)
        x = torch.stack([gate_out, primary_flux, secondary_flux], dim=1)
        cnn_out = self.cnn(x).squeeze(2)
        return torch.sigmoid(self.classifier(cnn_out))


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


def metrics(preds, y):
    TP = int(((preds == 1) & (y == 1)).sum())
    TN = int(((preds == 0) & (y == 0)).sum())
    FP = int(((preds == 1) & (y == 0)).sum())
    FN = int(((preds == 0) & (y == 1)).sum())
    acc = (TP + TN) / len(y)
    prec = TP / (TP + FP) if TP + FP else 0.0
    rec = TP / (TP + FN) if TP + FN else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return TP, TN, FP, FN, acc, prec, rec, f1


def report(tag, preds, lab):
    TP, TN, FP, FN, acc, prec, rec, f1 = metrics(preds, lab)
    hit = "HIT" if (prec > 0.80 and rec >= 0.90) else ""
    print(f"  {tag:<40} acc={acc:>5.1%}  prec={prec:>5.1%}  "
          f"rec={rec:>5.1%}  F1={f1:>6.3f}  TP={TP} FP={FP} TN={TN} FN={FN}  {hit}")


def main():
    d, test_idx = test_split()
    lab = d["labels"][test_idx].cpu().int()
    names = d["names"]

    # V5 inference
    v5 = TaylorCNNV5(init_amplitude=0.01).to(DEVICE)
    ck5 = torch.load(V5_PATH, map_location=DEVICE, weights_only=False)
    v5.load_state_dict(ck5["state_dict"])
    v5.eval()
    with torch.no_grad():
        p_v5 = v5(
            d["phases"][test_idx].to(DEVICE),
            d["fluxes"][test_idx].to(DEVICE),
            d["fluxes_secondary"][test_idx].to(DEVICE),
        ).squeeze(1).cpu()

    # V10 inference
    v10 = TaylorCNNv10(init_amplitude=0.01).to(DEVICE)
    ck10 = torch.load(V10_PATH, map_location=DEVICE, weights_only=False)
    v10.load_state_dict(ck10["state_dict"])
    v10.eval()
    with torch.no_grad():
        p_v10 = v10(
            d["phases"][test_idx].to(DEVICE),
            d["fluxes"][test_idx].to(DEVICE),
            d["fluxes_secondary"][test_idx].to(DEVICE),
            d["fluxes_odd_even"][test_idx].to(DEVICE),
        ).squeeze(1).cpu()

    # Caveat: V5 was trained on the old 100-TCE dataset. Some of the
    # 76-TCE V10 test set may have been in V5's training set. We report
    # numbers as-is but note the leakage risk.
    print("CAVEAT: V5 was trained on the old 100-TCE kepler_tce.pt")
    print("        dataset. The V6/V10 test set may overlap V5's train.")
    print()
    print("V5 + V10 ensemble on the 76-TCE seed=42 test set")
    print("-" * 90)
    report("V5 alone (thr=0.5)",  (p_v5 > 0.5).int(), lab)
    report("V10 alone (thr=0.5)", (p_v10 > 0.5).int(), lab)
    print()

    # A: 0.3*v5 + 0.7*v10
    report("A. 0.3*v5 + 0.7*v10 (thr=0.5)",
           ((0.3 * p_v5 + 0.7 * p_v10) > 0.5).int(), lab)
    # B: 0.5*v5 + 0.5*v10
    report("B. 0.5*v5 + 0.5*v10 (thr=0.5)",
           ((0.5 * p_v5 + 0.5 * p_v10) > 0.5).int(), lab)
    # C: v5 > 0.3 OR v10 > 0.5
    report("C. v5>0.3 OR v10>0.5",
           ((p_v5 > 0.3) | (p_v10 > 0.5)).int(), lab)
    # D: v5 > 0.5 AND v10 > 0.5
    report("D. v5>0.5 AND v10>0.5",
           ((p_v5 > 0.5) & (p_v10 > 0.5)).int(), lab)


if __name__ == "__main__":
    main()
