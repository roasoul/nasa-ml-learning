"""Experiment 6 — Gate diversity loss.

Adds a diversity penalty that encourages the five gate templates to
stay distinct from each other:

    L_diversity = -(1 / C(5, 2)) * sum_{i<j}  ||g_i - g_j||_2 / 200

Negative sign means minimising the total loss ENCOURAGES large
pairwise differences. Weighted 0.1 in the total loss so it doesn't
dominate BCE.

Background: in V10 λ=0.1 the A1 (planet-U) gate collapses to the
clamp floor — the other gates' shapes are similar enough to the
planet-U at small |x| that A1 isn't pulling its weight. Diversity
pressure should push the templates apart and (hopefully) let A1
develop a distinct role.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import math

import torch
import torch.nn as nn

from src.models.geometry_loss_v2 import InvertedGeometryLoss
from src.models.taylor_cnn_v10 import TaylorCNNv10


DATA_PATH = "data/kepler_tce_v6.pt"
SS_CACHE = "data/ss_flag_cache.csv"
OUT_MODEL = "src/models/taylor_cnn_v10_diversity.pt"
SEED_SPLIT = 42
SEED_INIT = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAMBDA_MAX = 0.1
SNR_PIVOT = 100.0
DIVERSITY_WEIGHT = 0.1


def diversity_loss(model) -> torch.Tensor:
    """L_diversity on the fixed [-pi, pi] grid. Pairwise L2 distances
    between the 5 gate templates, averaged over pairs and sequence
    length. Negated so that minimising the loss pushes the templates
    apart."""
    device = next(model.parameters()).device
    phase = torch.linspace(-math.pi, math.pi, 200, device=device).unsqueeze(0)
    gates = model.gate_bank(phase).squeeze(0)  # (5, 200)
    dists = []
    for i in range(5):
        for j in range(i + 1, 5):
            dists.append(((gates[i] - gates[j]) ** 2).mean())
    total = torch.stack(dists).mean()
    return -total  # encourage separation


def load_snr(names):
    cache = {}
    with open(SS_CACHE, newline="") as f:
        for r in csv.DictReader(f):
            cache[r["kepoi_name"]] = r
    snr = torch.full((len(names),), SNR_PIVOT, dtype=torch.float32)
    for i, n in enumerate(names):
        rec = cache.get(n)
        if rec and rec["snr"] not in ("", "None"):
            snr[i] = float(rec["snr"])
    return snr


def load_and_split():
    d = torch.load(DATA_PATH, weights_only=False)
    y = d["labels"]; names = d["names"]
    snr = load_snr(names)
    torch.manual_seed(SEED_SPLIT)
    conf = (y == 1).nonzero(as_tuple=True)[0]
    fp = (y == 0).nonzero(as_tuple=True)[0]
    conf = conf[torch.randperm(len(conf))]
    fp = fp[torch.randperm(len(fp))]
    def split(idx, tf=0.7, vf=0.15):
        n = len(idx); nt = int(n * tf); nv = int(n * vf)
        return idx[:nt], idx[nt:nt + nv], idx[nt + nv:]
    ct, cv, cte = split(conf); ft, fv, fte = split(fp)
    train_idx = torch.cat([ct, ft]); val_idx = torch.cat([cv, fv]); test_idx = torch.cat([cte, fte])
    def bundle(idx):
        return (d["phases"][idx].to(DEVICE), d["fluxes"][idx].to(DEVICE),
                d["fluxes_secondary"][idx].to(DEVICE), d["fluxes_odd_even"][idx].to(DEVICE),
                y[idx].to(DEVICE), snr[idx].to(DEVICE))
    return bundle(train_idx), bundle(val_idx), bundle(test_idx), names, test_idx.tolist()


def finetune(model, train, val, geo_loss, n_epochs=200, patience=25, batch_size=16):
    tp, tpp, tps, tpoe, tpl, tsnr = train
    vp, vpp, vps, vpoe, vpl, _ = val
    opt = torch.optim.Adam([
        {"params": model.gate_bank.parameters(), "lr": 1e-4},
        {"params": model.cnn.parameters(), "lr": 1e-3},
        {"params": model.classifier.parameters(), "lr": 1e-3},
    ])
    bce = nn.BCELoss()
    n_tr = len(tpl)
    best_val = float("inf"); best_state = None; wait = 0
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n_tr, device=DEVICE)
        for start in range(0, n_tr, batch_size):
            idx = perm[start:start + batch_size]
            pred = model(tp[idx], tpp[idx], tps[idx], tpoe[idx]).squeeze(1)
            t12, auc = model.compute_shape_features(tpp[idx])
            loss = (bce(pred, tpl[idx])
                    + geo_loss(pred, t12, auc, tsnr[idx])
                    + DIVERSITY_WEIGHT * diversity_loss(model))
            opt.zero_grad(); loss.backward(); opt.step()
            model.clamp_amplitudes(0.001)
        model.eval()
        with torch.no_grad():
            v = model(vp, vpp, vps, vpoe).squeeze(1)
            vl = bce(v, vpl).item()
        if vl < best_val:
            best_val = vl
            best_state = {k: vv.clone() for k, vv in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if (epoch + 1) % 50 == 0 or epoch == 0:
            amps = model.gate_bank.amplitudes()
            print(f"    ep {epoch+1:>3}  val={vl:.4f}  wait={wait}  "
                  f"A1={amps['A1']:.4f} A2={amps['A2']:.4f} A3={amps['A3']:.4f} "
                  f"A4={amps['A4']:.4f} A5={amps['A5']:.4f}")
        if wait >= patience:
            print(f"    early stop at ep {epoch+1}")
            break
    model.load_state_dict(best_state)
    return best_val


def evaluate(model, test):
    tp, tpp, tps, tpoe, tpl, _ = test
    model.eval()
    with torch.no_grad():
        probs = model(tp, tpp, tps, tpoe).squeeze(1).cpu()
    preds = (probs > 0.5).float()
    lab = tpl.cpu()
    TP = int(((preds == 1) & (lab == 1)).sum())
    TN = int(((preds == 0) & (lab == 0)).sum())
    FP = int(((preds == 1) & (lab == 0)).sum())
    FN = int(((preds == 0) & (lab == 1)).sum())
    acc = (TP + TN) / len(lab)
    prec = TP / (TP + FP) if TP + FP else 0
    rec = TP / (TP + FN) if TP + FN else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1,
            "TP": TP, "TN": TN, "FP": FP, "FN": FN}


def main():
    print(f"Device: {DEVICE}")
    train, val, test, names, test_idx_list = load_and_split()
    print(f"Train: {len(train[-1])}  Val: {len(val[-1])}  Test: {len(test[-1])}")

    torch.manual_seed(SEED_INIT)
    model = TaylorCNNv10(init_amplitude=0.01).to(DEVICE)
    geo = InvertedGeometryLoss(lambda_min=0.01, lambda_max=LAMBDA_MAX,
                                snr_pivot=SNR_PIVOT).to(DEVICE)
    print(f"\nTraining V10 + diversity (weight {DIVERSITY_WEIGHT}, lambda={LAMBDA_MAX})")
    finetune(model, train, val, geo)
    m = evaluate(model, test)
    amps = model.gate_bank.amplitudes()
    print(f"\nV10+diversity: acc={m['acc']:.1%}  prec={m['prec']:.1%}  "
          f"rec={m['rec']:.1%}  F1={m['f1']:.3f}  "
          f"TP={m['TP']} FP={m['FP']} TN={m['TN']} FN={m['FN']}")
    print(f"Amplitudes: A1={amps['A1']:.4f} A2={amps['A2']:.4f} "
          f"A3={amps['A3']:.4f} A4={amps['A4']:.4f} A5={amps['A5']:.4f}")
    print(f"\nComparison:")
    print(f"  V10 vanilla:     F1 0.861  A1=0.0010 A2=0.0105 A3=0.0086 A4=0.0209 A5=0.0245")
    print(f"  V10 + diversity: F1 {m['f1']:.3f}  A1={amps['A1']:.4f} A2={amps['A2']:.4f} A3={amps['A3']:.4f} A4={amps['A4']:.4f} A5={amps['A5']:.4f}")

    Path("src/models").mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "metrics": m,
        "amplitudes": amps,
        "diversity_weight": DIVERSITY_WEIGHT,
    }, OUT_MODEL)
    print(f"\nSaved -> {OUT_MODEL}")


if __name__ == "__main__":
    main()
