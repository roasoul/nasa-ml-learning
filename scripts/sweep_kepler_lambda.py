"""Sweep lambda_kepler to find the right Kepler-loss weight.

Default V7 lambda=0.1 made the Kepler term only 2.6% of BCE magnitude
(tiny gradient signal). This sweep trains at several lambda values and
reports precision/recall/F1 at threshold 0.50 for each.

Success target: precision > 80%, recall = 100%.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from src.models.taylor_cnn import TaylorCNN
from src.models.kepler_loss import (
    calculate_kepler_violation,
    kepler_loss,
    sparsity_loss,
)


DATA_PATH = "data/kepler_tce_v6.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED_SPLIT = 42
SEED_INIT = 7
LAMBDA_SPARSITY = 0.01
TOLERANCE = 0.2


def load_splits(asymmetric: bool = False):
    d = torch.load(DATA_PATH, weights_only=False)
    phases = d["phases"]; p = d["fluxes"]; s = d["fluxes_secondary"]
    oe = d["fluxes_odd_even"]; y = d["labels"]
    violations = calculate_kepler_violation(
        d["period_days"], d["duration_hours"] / 24.0,
        d["stellar_mass"], d["stellar_radius"],
        asymmetric=asymmetric,
    )
    torch.manual_seed(SEED_SPLIT)
    conf = (y == 1).nonzero(as_tuple=True)[0]; fp = (y == 0).nonzero(as_tuple=True)[0]
    conf = conf[torch.randperm(len(conf))]; fp = fp[torch.randperm(len(fp))]
    def split(idx, tf=0.7, vf=0.15):
        n = len(idx); nt = int(n * tf); nv = int(n * vf)
        return idx[:nt], idx[nt:nt + nv], idx[nt + nv:]
    ct, cv, cte = split(conf); ft, fv, fte = split(fp)
    def to_dev(idx):
        return (phases[idx].to(DEVICE), p[idx].to(DEVICE),
                s[idx].to(DEVICE), oe[idx].to(DEVICE),
                y[idx].to(DEVICE), violations[idx].to(DEVICE))
    return to_dev(torch.cat([ct, ft])), to_dev(torch.cat([cv, fv])), to_dev(torch.cat([cte, fte]))


def train_once(train, val, lambda_kep, n_epochs=200, patience=25, bs=16):
    torch.manual_seed(SEED_INIT)
    model = TaylorCNN(init_amplitude=0.01).to(DEVICE)
    bce = nn.BCELoss()
    opt = torch.optim.Adam([
        {"params": model.taylor_gate.parameters(), "lr": 1e-4},
        {"params": model.cnn.parameters(), "lr": 1e-3},
        {"params": model.classifier.parameters(), "lr": 1e-3},
    ])
    tp, tpp, tps, tpoe, tpl, tpv = train
    vp, vpp, vps, vpoe, vpl, _ = val
    n_tr = len(tpl); best_val = float("inf"); best_state = None; wait = 0
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n_tr, device=DEVICE)
        for start in range(0, n_tr, bs):
            idx = perm[start:start + bs]
            probs = model(tp[idx], tpp[idx], tps[idx], tpoe[idx]).squeeze(1)
            loss = bce(probs, tpl[idx])
            if lambda_kep > 0:
                loss = loss + lambda_kep * kepler_loss(probs, tpv[idx], TOLERANCE)
                loss = loss + LAMBDA_SPARSITY * sparsity_loss(model)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = bce(model(vp, vpp, vps, vpoe).squeeze(1), vpl).item()
        if vl < best_val:
            best_val = vl; best_state = {k: vv.clone() for k, vv in model.state_dict().items()}; wait = 0
        else:
            wait += 1
        if wait >= patience: break
    model.load_state_dict(best_state)
    return model


def evaluate(model, test, thr):
    tp, tpp, tps, tpoe, tpl, _ = test
    model.eval()
    with torch.no_grad():
        probs = model(tp, tpp, tps, tpoe).squeeze(1).cpu()
    preds = (probs > thr).float(); tl = tpl.cpu()
    TP = int(((preds == 1) & (tl == 1)).sum())
    TN = int(((preds == 0) & (tl == 0)).sum())
    FP = int(((preds == 1) & (tl == 0)).sum())
    FN = int(((preds == 0) & (tl == 1)).sum())
    acc = (TP + TN) / len(tl); prec = TP / (TP + FP) if TP + FP else 0.0
    rec = TP / (TP + FN) if TP + FN else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1,
            "TP": TP, "FP": FP, "TN": TN, "FN": FN, "probs": probs}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--asymmetric", action="store_true",
                        help="Only penalize observed > predicted (EB case)")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    mode = "ASYMMETRIC (max(0, log(obs/pred)))" if args.asymmetric else "symmetric (|log(obs/pred)|)"
    print(f"Kepler violation mode: {mode}\n")
    train, val, test = load_splits(asymmetric=args.asymmetric)
    print(f"Train: {len(train[-1])}  Val: {len(val[-1])}  Test: {len(test[-1])}\n")

    # Print training violation distribution by class — separation sanity check
    tv = train[-1].cpu(); tl = train[-2].cpu()
    print(f"Training violation medians — confirmed: {tv[tl==1].median():.3f}  "
          f"FP: {tv[tl==0].median():.3f}\n")

    lambdas = [0.0, 0.3, 1.0, 3.0, 10.0, 30.0]
    print(f"{'lambda':>7} {'acc':>7} {'prec':>7} {'rec':>7} {'F1':>7} "
          f"{'TP':>3} {'FP':>3} {'TN':>3} {'FN':>3} {'A':>7}")
    print("-" * 62)
    for lam in lambdas:
        model = train_once(train, val, lam)
        m = evaluate(model, test, 0.50)
        A = model.taylor_gate.A.item()
        print(f"{lam:>7.1f}  {m['acc']:>6.1%} {m['prec']:>6.1%} "
              f"{m['rec']:>6.1%} {m['f1']:>6.3f}  "
              f"{m['TP']:>3} {m['FP']:>3} {m['TN']:>3} {m['FN']:>3} {A:>7.4f}")


if __name__ == "__main__":
    main()
