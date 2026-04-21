"""Threshold sweep on V6 Config C (from-scratch).

User requirement 4: "lower classification threshold from 0.5 to 0.4 and
measure impact on V6 Config C first" — checks whether a threshold move
restores 100% recall without crushing precision.

Retrains Config C from scratch (same seeds as run_v6.py), then sweeps
thresholds over [0.2, 0.7] and prints a precision/recall/F1 table.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn

from src.models.taylor_cnn import TaylorCNN


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "data/kepler_tce_v6.pt"
SEED_SPLIT = 42
SEED_INIT = 7


def load_splits():
    d = torch.load(DATA_PATH, weights_only=False)
    phases = d["phases"]
    p = d["fluxes"]
    s = d["fluxes_secondary"]
    oe = d["fluxes_odd_even"]
    y = d["labels"]

    torch.manual_seed(SEED_SPLIT)
    conf = (y == 1).nonzero(as_tuple=True)[0]
    fp = (y == 0).nonzero(as_tuple=True)[0]
    conf = conf[torch.randperm(len(conf))]
    fp = fp[torch.randperm(len(fp))]

    def split(idx, tf=0.7, vf=0.15):
        n = len(idx); nt = int(n * tf); nv = int(n * vf)
        return idx[:nt], idx[nt:nt + nv], idx[nt + nv:]

    ct, cv, cte = split(conf)
    ft, fv, fte = split(fp)

    def to_dev(idx):
        return (
            phases[idx].to(DEVICE), p[idx].to(DEVICE),
            s[idx].to(DEVICE), oe[idx].to(DEVICE),
            y[idx].to(DEVICE),
        )

    return to_dev(torch.cat([ct, ft])), to_dev(torch.cat([cv, fv])), to_dev(torch.cat([cte, fte]))


def train_config_c(train, val, n_epochs=200, patience=25, batch_size=16):
    torch.manual_seed(SEED_INIT)
    model = TaylorCNN(init_amplitude=0.01).to(DEVICE)
    crit = nn.BCELoss()
    opt = torch.optim.Adam([
        {"params": model.taylor_gate.parameters(), "lr": 1e-4},
        {"params": model.cnn.parameters(), "lr": 1e-3},
        {"params": model.classifier.parameters(), "lr": 1e-3},
    ])
    tp, tpp, tps, tpoe, tpl = train
    vp, vpp, vps, vpoe, vpl = val
    n_tr = len(tpl)
    best_val = float("inf"); best_state = None; wait = 0
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n_tr, device=DEVICE)
        for start in range(0, n_tr, batch_size):
            idx = perm[start:start + batch_size]
            pred = model(tp[idx], tpp[idx], tps[idx], tpoe[idx]).squeeze(1)
            loss = crit(pred, tpl[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = model(vp, vpp, vps, vpoe).squeeze(1)
            vl = crit(v, vpl).item()
        if vl < best_val:
            best_val = vl
            best_state = {k: vv.clone() for k, vv in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break
    model.load_state_dict(best_state)
    return model, best_val


def sweep_thresholds(model, test, thresholds):
    tp, tpp, tps, tpoe, tpl = test
    model.eval()
    with torch.no_grad():
        probs = model(tp, tpp, tps, tpoe).squeeze(1).cpu()
    tl = tpl.cpu()

    rows = []
    for thr in thresholds:
        preds = (probs > thr).float()
        TP = int(((preds == 1) & (tl == 1)).sum())
        TN = int(((preds == 0) & (tl == 0)).sum())
        FP = int(((preds == 1) & (tl == 0)).sum())
        FN = int(((preds == 0) & (tl == 1)).sum())
        acc = (TP + TN) / len(tl)
        prec = TP / (TP + FP) if TP + FP else 0.0
        rec = TP / (TP + FN) if TP + FN else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append({
            "thr": thr, "acc": acc, "prec": prec, "rec": rec, "f1": f1,
            "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        })
    return probs, rows


def main():
    print(f"Device: {DEVICE}")
    print("Loading V6 dataset and splits...")
    train, val, test = load_splits()
    print(f"  Train: {len(train[-1])}  Val: {len(val[-1])}  Test: {len(test[-1])}")

    print("\nRetraining V6 Config C (from scratch, seed-7 init)...")
    model, best_val = train_config_c(train, val)
    print(f"  Done. Best val loss = {best_val:.4f}")

    thresholds = np.linspace(0.20, 0.70, 11)
    probs, rows = sweep_thresholds(model, test, thresholds)

    print(f"\nThreshold sweep on test set (n={len(test[-1])}):")
    print(f"{'thr':>5} {'acc':>7} {'prec':>7} {'rec':>7} {'F1':>7} "
          f"{'TP':>3} {'FP':>3} {'TN':>3} {'FN':>3}")
    print("-" * 58)
    for r in rows:
        mark = "  *" if (r["rec"] >= 1.0 and r["prec"] > 0.80) else ""
        print(
            f"{r['thr']:.2f}  {r['acc']:>6.1%}  {r['prec']:>6.1%}  "
            f"{r['rec']:>6.1%}  {r['f1']:>6.3f}  "
            f"{r['TP']:>3} {r['FP']:>3} {r['TN']:>3} {r['FN']:>3}{mark}"
        )

    # Highlight threshold 0.40 (user's target) and threshold 0.50 (V6 default)
    def nearest(row_list, target):
        return min(row_list, key=lambda r: abs(r["thr"] - target))

    r050 = nearest(rows, 0.50)
    r040 = nearest(rows, 0.40)

    print(
        f"\nV6 default threshold 0.50: "
        f"acc={r050['acc']:.1%}  prec={r050['prec']:.1%}  rec={r050['rec']:.1%}  F1={r050['f1']:.3f}"
    )
    print(
        f"User-proposed     0.40: "
        f"acc={r040['acc']:.1%}  prec={r040['prec']:.1%}  rec={r040['rec']:.1%}  F1={r040['f1']:.3f}"
    )
    d_prec = r040["prec"] - r050["prec"]
    d_rec = r040["rec"] - r050["rec"]
    print(f"Delta (0.40 - 0.50): precision {d_prec:+.1%}  recall {d_rec:+.1%}")

    # Find threshold that first hits recall = 100%
    rec100 = [r for r in rows if r["rec"] >= 1.0]
    if rec100:
        r100 = max(rec100, key=lambda r: r["thr"])  # highest thr with recall=100
        print(
            f"\nHighest threshold keeping recall=100%: thr={r100['thr']:.2f}  "
            f"prec={r100['prec']:.1%}  F1={r100['f1']:.3f}"
        )
    else:
        print("\nNo threshold in sweep achieves recall=100%.")

    # Save sweep rows for the notebook
    out = {
        "probs": probs,
        "rows": rows,
    }
    Path("data").mkdir(parents=True, exist_ok=True)
    torch.save(out, "data/v6_config_c_threshold_sweep.pt")
    print("\nSaved sweep results to data/v6_config_c_threshold_sweep.pt")


if __name__ == "__main__":
    main()
