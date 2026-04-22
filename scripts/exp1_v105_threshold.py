"""Experiment 1 — V10.5 threshold sweep.

Loads V10 lambda=0.1 weights. Sweeps decision threshold {0.30, 0.35,
0.40, 0.45, 0.50}. Zero retraining.

Also reports the four planets V10 misclassifies at threshold 0.50
with KOI, period, depth, SNR so we can diagnose whether the missed
one is K00254.01 again (dominant M-dwarf contamination candidate).
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.models.taylor_cnn_v10 import TaylorCNNv10


DATA_PATH = "data/kepler_tce_v6.pt"
SS_CACHE = "data/ss_flag_cache.csv"
V10_MODEL = "src/models/taylor_cnn_v10.pt"
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


def snr_lookup(names):
    out = {}
    with open(SS_CACHE, newline="") as f:
        for r in csv.DictReader(f):
            out[r["kepoi_name"]] = r
    return out


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


def main():
    d, test_idx = test_split()
    names = d["names"]

    model = TaylorCNNv10(init_amplitude=0.01).to(DEVICE)
    ck = torch.load(V10_MODEL, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    ph = d["phases"][test_idx].to(DEVICE)
    p = d["fluxes"][test_idx].to(DEVICE)
    s = d["fluxes_secondary"][test_idx].to(DEVICE)
    oe = d["fluxes_odd_even"][test_idx].to(DEVICE)
    lab = d["labels"][test_idx].cpu()

    with torch.no_grad():
        probs = model(ph, p, s, oe).squeeze(1).cpu()

    print("V10.5 threshold sweep (V10 lambda=0.1, no retraining)")
    print("-" * 70)
    print(f"{'thr':>5}  {'acc':>6} {'prec':>6} {'rec':>6} {'F1':>6}  "
          f"{'TP':>3} {'FP':>3} {'TN':>3} {'FN':>3}")
    for thr in [0.30, 0.35, 0.40, 0.45, 0.50]:
        preds = (probs > thr).int()
        TP, TN, FP, FN, acc, prec, rec, f1 = metrics(preds, lab)
        target_hit = "+" if (prec > 0.80 and rec >= 0.90) else " "
        print(f"{thr:>5.2f}  {acc:>5.1%} {prec:>5.1%} {rec:>5.1%} {f1:>6.3f}  "
              f"{TP:>3} {FP:>3} {TN:>3} {FN:>3} {target_hit}")

    # FN analysis at threshold 0.50
    print("\nV10 @ thr=0.50 — planets missed (FN):")
    snr_map = snr_lookup(names)
    pred_50 = (probs > 0.50).int()
    for i, orig in enumerate(test_idx.tolist()):
        if lab[i] == 1 and pred_50[i] == 0:
            nm = names[orig]
            period = float(d["period_days"][orig])
            depth_ppm = float(d["depths_ppm"][orig])
            snr = float(snr_map[nm]["snr"]) if nm in snr_map and snr_map[nm]["snr"] not in ("", "None") else float("nan")
            rstar = float(d["stellar_radius"][orig])
            print(f"  {nm:<12}  P={period:>8.3f}d  depth={depth_ppm:>7.0f}ppm  "
                  f"SNR={snr:>6.1f}  R*={rstar:>5.3f}Rsun  V10_prob={probs[i]:.3f}")

    # Lowest threshold that achieves recall >= 90%
    for thr in [0.50, 0.45, 0.40, 0.35, 0.30]:
        preds = (probs > thr).int()
        TP, TN, FP, FN, acc, prec, rec, f1 = metrics(preds, lab)
        if rec >= 0.90 and prec > 0.80:
            print(f"\nTarget hit: thr={thr}  prec={prec:.1%}  rec={rec:.1%}  F1={f1:.3f}")
            break
    else:
        print("\nNo threshold in {0.30..0.50} hits recall>=90% AND prec>80%")


if __name__ == "__main__":
    main()
