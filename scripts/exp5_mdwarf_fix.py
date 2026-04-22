"""Experiment 5 — M-dwarf depth normalisation.

Hypothesis: V10 misclassifies M-dwarf planets (K00254, K00912) because
their transits look anomalously deep relative to sun-like planets.
Normalising the flux by (R_star / R_sun)^2 should rescale the dip
to sun-like amplitude, letting the learned gate bank match.

Pipeline: multiply fluxes element-wise by (R_star)^2 per sample so
the folded depth becomes (R_planet / R_sun)^2 in absolute units.
Stellar radii are already stored in `data/kepler_tce_v6.pt` under
`stellar_radius`.

Retrain V10 architecture on the normalised dataset at lambda=0.1
(the winning config). Compare final metrics to vanilla V10 and
check whether K00254 / K00912 now classify correctly.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from src.models.geometry_loss_v2 import InvertedGeometryLoss
from src.models.taylor_cnn_v10 import TaylorCNNv10


DATA_PATH = "data/kepler_tce_v6.pt"
SS_CACHE = "data/ss_flag_cache.csv"
OUT_MODEL = "src/models/taylor_cnn_v10_mdwarf.pt"
SEED_SPLIT = 42
SEED_INIT = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAMBDA_MAX = 0.1
SNR_PIVOT = 100.0


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
    y = d["labels"]
    names = d["names"]
    snr = load_snr(names)

    # Normalize depth by (R_star / R_sun)^2
    rstar = d["stellar_radius"].unsqueeze(1).clamp(min=0.1)  # (N, 1)
    # Delta_obs = (Rp/R*)^2. To convert to "sun-equivalent depth" for a
    # given transiter, multiply by R*^2: Delta_sun = Delta_obs * R*^2
    scale = rstar ** 2  # (N, 1)
    print(f"Depth-normalization scale stats: min={float(scale.min()):.3f} "
          f"median={float(scale.median()):.3f} max={float(scale.max()):.3f}")
    phases = d["phases"]
    p = d["fluxes"] * scale
    s = d["fluxes_secondary"] * scale
    oe = d["fluxes_odd_even"] * scale

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
        return (phases[idx].to(DEVICE), p[idx].to(DEVICE),
                s[idx].to(DEVICE), oe[idx].to(DEVICE),
                y[idx].to(DEVICE), snr[idx].to(DEVICE))

    return bundle(train_idx), bundle(val_idx), bundle(test_idx), names, test_idx.tolist(), d


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
            loss = bce(pred, tpl[idx]) + geo_loss(pred, t12, auc, tsnr[idx])
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
                  f"A1={amps['A1']:.4f} A2={amps['A2']:.4f} A5={amps['A5']:.4f}")
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
            "TP": TP, "TN": TN, "FP": FP, "FN": FN, "probs": probs, "preds": preds}


def main():
    print(f"Device: {DEVICE}")
    train, val, test, names, test_idx_list, raw = load_and_split()
    print(f"Train: {len(train[-1])}  Val: {len(val[-1])}  Test: {len(test[-1])}")

    torch.manual_seed(SEED_INIT)
    model = TaylorCNNv10(init_amplitude=0.01).to(DEVICE)
    geo = InvertedGeometryLoss(lambda_min=0.01, lambda_max=LAMBDA_MAX,
                                snr_pivot=SNR_PIVOT).to(DEVICE)
    print(f"\nTraining V10 with R*-normalized depth (lambda={LAMBDA_MAX})")
    finetune(model, train, val, geo)
    m = evaluate(model, test)
    print(f"\nV10+M-dwarf: acc={m['acc']:.1%}  prec={m['prec']:.1%}  "
          f"rec={m['rec']:.1%}  F1={m['f1']:.3f}  "
          f"TP={m['TP']} FP={m['FP']} TN={m['TN']} FN={m['FN']}")

    # Per-TCE M-dwarf check
    tl = test[-2].cpu()
    probs = m["probs"]
    print(f"\nPer-TCE check of the 4 planets V10 missed:")
    target_kois = ["K00013.01", "K00912.01", "K00254.01", "K00183.01"]
    for i, orig in enumerate(test_idx_list):
        if names[orig] in target_kois:
            rstar = float(raw["stellar_radius"][orig])
            depth = float(raw["depths_ppm"][orig])
            truth = "PLNT" if tl[i] == 1 else "FP"
            pred_str = "PLNT" if probs[i] > 0.5 else "FP"
            print(f"  {names[orig]:<12} truth={truth}  R*={rstar:.2f}  "
                  f"depth={depth:.0f}ppm  V10+Mdwarf prob={probs[i]:.3f} -> {pred_str}")

    Path("src/models").mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "metrics": {k: v for k, v in m.items() if k not in ("probs", "preds")},
        "normalization": "flux * (R_star / R_sun)**2",
    }, OUT_MODEL)
    print(f"\nSaved -> {OUT_MODEL}")


if __name__ == "__main__":
    main()
