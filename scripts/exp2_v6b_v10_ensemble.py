"""Experiment 2 — V6 Config B + V10 lambda=0.1 ensemble.

Stored taylor_cnn_v6.pt IS Config B already (we verified this in
the V10 diagnostic). Four ensemble strategies on the 76-TCE test
set, no retraining:

    1. simple average:   p = (p_v6b + p_v10) / 2, thr=0.5
    2. weighted:         p = 0.4*p_v6b + 0.6*p_v10, thr=0.5
    3. OR recall-first:  planet if p_v6b > 0.4 OR p_v10 > 0.4
    4. AND precision-first: planet if p_v6b > 0.5 AND p_v10 > 0.5

Target: precision > 80% AND recall > 90%.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.models.taylor_cnn import TaylorCNN
from src.models.taylor_cnn_v10 import TaylorCNNv10


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


def infer_v6(d, idx):
    m = TaylorCNN(init_amplitude=0.01).to(DEVICE)
    ck = torch.load(V6_PATH, map_location=DEVICE, weights_only=False)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    with torch.no_grad():
        return m(d["phases"][idx].to(DEVICE), d["fluxes"][idx].to(DEVICE),
                 d["fluxes_secondary"][idx].to(DEVICE),
                 d["fluxes_odd_even"][idx].to(DEVICE)).squeeze(1).cpu()


def infer_v10(d, idx):
    m = TaylorCNNv10(init_amplitude=0.01).to(DEVICE)
    ck = torch.load(V10_PATH, map_location=DEVICE, weights_only=False)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    with torch.no_grad():
        return m(d["phases"][idx].to(DEVICE), d["fluxes"][idx].to(DEVICE),
                 d["fluxes_secondary"][idx].to(DEVICE),
                 d["fluxes_odd_even"][idx].to(DEVICE)).squeeze(1).cpu()


def report(tag, preds, lab):
    TP, TN, FP, FN, acc, prec, rec, f1 = metrics(preds, lab)
    hit = "HIT" if (prec > 0.80 and rec >= 0.90) else ""
    print(f"  {tag:<40} acc={acc:>5.1%}  prec={prec:>5.1%}  "
          f"rec={rec:>5.1%}  F1={f1:>6.3f}  TP={TP} FP={FP} TN={TN} FN={FN}  {hit}")


def main():
    d, test_idx = test_split()
    lab = d["labels"][test_idx].cpu().int()

    p_v6 = infer_v6(d, test_idx)
    p_v10 = infer_v10(d, test_idx)

    print("V6 Config B vs V10 lambda=0.1 — ensemble strategies")
    print("-" * 90)
    report("V6 Config B alone (thr=0.5)", (p_v6 > 0.5).int(), lab)
    report("V10 alone (thr=0.5)",         (p_v10 > 0.5).int(), lab)
    print()

    # 1. Simple average
    avg = 0.5 * p_v6 + 0.5 * p_v10
    report("1. avg (thr=0.5)", (avg > 0.5).int(), lab)

    # 2. Weighted 40/60
    w = 0.4 * p_v6 + 0.6 * p_v10
    report("2. weighted 0.4*v6 + 0.6*v10 (thr=0.5)", (w > 0.5).int(), lab)

    # 3. OR recall-first
    or_pred = ((p_v6 > 0.4) | (p_v10 > 0.4)).int()
    report("3. OR (v6>0.4 or v10>0.4)", or_pred, lab)

    # 4. AND precision-first
    and_pred = ((p_v6 > 0.5) & (p_v10 > 0.5)).int()
    report("4. AND (v6>0.5 and v10>0.5)", and_pred, lab)


if __name__ == "__main__":
    main()
