"""V6 Config B + V10_5b log ensemble on same 76-TCE test set.

Since V10_5b expects R*-normalised fluxes and V6 expects raw fluxes,
we produce each model's predictions on its own input. The test-set
indices are identical (seed=42 split), so we can combine prob
vectors directly.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.models.taylor_cnn import TaylorCNN
from src.models.taylor_cnn_v10 import TaylorCNNv10


DATA_PATH = "data/kepler_tce_v6.pt"
V6_PATH = "src/models/taylor_cnn_v6.pt"
V10_PATH = "src/models/taylor_cnn_v10.pt"
V10_5B_PATH = "src/models/taylor_cnn_v10_5b.pt"
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
    prec = TP / (TP + FP) if TP + FP else 0
    rec = TP / (TP + FN) if TP + FN else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    return TP, TN, FP, FN, acc, prec, rec, f1


def infer_v6(d, idx):
    m = TaylorCNN(init_amplitude=0.01).to(DEVICE)
    m.load_state_dict(torch.load(V6_PATH, map_location=DEVICE, weights_only=False)["state_dict"])
    m.eval()
    with torch.no_grad():
        return m(d["phases"][idx].to(DEVICE),
                 d["fluxes"][idx].to(DEVICE),
                 d["fluxes_secondary"][idx].to(DEVICE),
                 d["fluxes_odd_even"][idx].to(DEVICE)).squeeze(1).cpu()


def infer_v10(d, idx, path):
    m = TaylorCNNv10(init_amplitude=0.01).to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=False)["state_dict"])
    m.eval()
    with torch.no_grad():
        return m(d["phases"][idx].to(DEVICE),
                 d["fluxes"][idx].to(DEVICE),
                 d["fluxes_secondary"][idx].to(DEVICE),
                 d["fluxes_odd_even"][idx].to(DEVICE)).squeeze(1).cpu()


def infer_v10_5b(d, idx):
    """V10_5b was trained with flux * log1p(R*/Rsun) scale. Apply same
    scale to the test inputs before inference."""
    scale = torch.log1p(d["stellar_radius"].clamp(min=0.01) / 1.0).unsqueeze(1)
    m = TaylorCNNv10(init_amplitude=0.01).to(DEVICE)
    m.load_state_dict(torch.load(V10_5B_PATH, map_location=DEVICE, weights_only=False)["state_dict"])
    m.eval()
    with torch.no_grad():
        p = d["fluxes"] * scale
        s = d["fluxes_secondary"] * scale
        oe = d["fluxes_odd_even"] * scale
        return m(d["phases"][idx].to(DEVICE),
                 p[idx].to(DEVICE),
                 s[idx].to(DEVICE),
                 oe[idx].to(DEVICE)).squeeze(1).cpu()


def report(tag, preds, lab):
    TP, TN, FP, FN, acc, prec, rec, f1 = metrics(preds, lab)
    hit = "HIT" if (prec > 0.80 and rec >= 0.90) else ""
    print(f"  {tag:<45} acc={acc:>5.1%}  prec={prec:>5.1%}  "
          f"rec={rec:>5.1%}  F1={f1:>6.3f}  TP={TP} FP={FP} TN={TN} FN={FN}  {hit}")


def main():
    d, test_idx = test_split()
    lab = d["labels"][test_idx].cpu().int()

    p_v6 = infer_v6(d, test_idx)
    p_v10 = infer_v10(d, test_idx, V10_PATH)
    p_v10_5b = infer_v10_5b(d, test_idx)

    print("Ensemble variants — V6b, V10, V10_5b (log R*)")
    print("-" * 95)
    report("V6 Config B",  (p_v6 > 0.5).int(), lab)
    report("V10 lambda=0.1", (p_v10 > 0.5).int(), lab)
    report("V10_5b log R*",  (p_v10_5b > 0.5).int(), lab)
    print()
    report("V6b + V10 AND",        ((p_v6 > 0.5) & (p_v10 > 0.5)).int(), lab)
    report("V6b + V10_5b AND",     ((p_v6 > 0.5) & (p_v10_5b > 0.5)).int(), lab)
    report("V10 + V10_5b AND",     ((p_v10 > 0.5) & (p_v10_5b > 0.5)).int(), lab)
    report("Triple AND",           ((p_v6 > 0.5) & (p_v10 > 0.5) & (p_v10_5b > 0.5)).int(), lab)
    report("V6b + V10 + V10_5b OR (0.4)",
           ((p_v6 > 0.4) | (p_v10 > 0.4) | (p_v10_5b > 0.4)).int(), lab)
    report("avg V6b+V10+V10_5b (thr=0.5)",
           (((p_v6 + p_v10 + p_v10_5b) / 3) > 0.5).int(), lab)


if __name__ == "__main__":
    main()
