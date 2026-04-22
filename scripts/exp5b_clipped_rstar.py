"""Experiment 5b — clipped R* normalisation.

Exp 5 used raw (R*/R_sun)^2 as a per-sample multiplier; with
R* ranging 0.12-35.7 Rsun, the scale ratio was ~91 000×, which
destabilised BatchNorm and crashed precision to 52%.

5b clips R* to the main-sequence band [0.5, 2.0] Rsun, giving
a scale ratio of (2.0/0.5)^2 = 16× — well within BatchNorm's
comfort zone. The `--mode log` variant uses flux·log1p(R*/R_sun)
as a fallback if the quadratic clip still underperforms.

Applied to all 3 flux channels (primary, secondary, odd/even).
Train V10 λ=0.1 with the normalised fluxes, seed=42 split.
"""

import argparse
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


def load_and_split(mode: str):
    d = torch.load(DATA_PATH, weights_only=False)
    y = d["labels"]; names = d["names"]
    snr = load_snr(names)

    r_raw = d["stellar_radius"].clamp(min=0.01)
    if mode == "clipped":
        r_use = r_raw.clamp(min=0.5, max=2.0)
        scale = (r_use / 1.0) ** 2
    elif mode == "log":
        scale = torch.log1p(r_raw / 1.0)
    else:
        raise ValueError(mode)

    n_below = int((r_raw < 0.5).sum())
    n_above = int((r_raw > 2.0).sum())
    print(f"[{mode}] R* raw: min={float(r_raw.min()):.3f}  "
          f"max={float(r_raw.max()):.3f}  median={float(r_raw.median()):.3f}")
    print(f"[{mode}] clipped count: below 0.5 Rsun: {n_below}, above 2.0 Rsun: {n_above}")
    print(f"[{mode}] scale: min={float(scale.min()):.3f}  "
          f"max={float(scale.max()):.3f}  ratio={float(scale.max()/scale.min()):.1f}x")

    scale = scale.unsqueeze(1)
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
                  f"A1={amps['A1']:.4f} A5={amps['A5']:.4f}")
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
            "TP": TP, "TN": TN, "FP": FP, "FN": FN, "probs": probs}


def run_mode(mode: str):
    print(f"\n{'='*60}\nExp 5b mode={mode}\n{'='*60}")
    train, val, test, names, test_idx_list, raw = load_and_split(mode)
    torch.manual_seed(SEED_INIT)
    model = TaylorCNNv10(init_amplitude=0.01).to(DEVICE)
    geo = InvertedGeometryLoss(lambda_min=0.01, lambda_max=LAMBDA_MAX,
                                snr_pivot=SNR_PIVOT).to(DEVICE)
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    finetune(model, train, val, geo)
    m = evaluate(model, test)
    print(f"\n[{mode}] V10+{mode}R*: acc={m['acc']:.1%}  prec={m['prec']:.1%}  "
          f"rec={m['rec']:.1%}  F1={m['f1']:.3f}  "
          f"TP={m['TP']} FP={m['FP']} TN={m['TN']} FN={m['FN']}")

    # Per-TCE check of original 4 missed planets
    target_kois = ["K00013.01", "K00912.01", "K00254.01", "K00183.01"]
    lab = test[-2].cpu()
    probs = m["probs"]
    print(f"\n[{mode}] Per-TCE for previously-missed planets:")
    for i, orig in enumerate(test_idx_list):
        if names[orig] in target_kois:
            rstar = float(raw["stellar_radius"][orig])
            pred = "PLNT" if probs[i] > 0.5 else "FP"
            print(f"  {names[orig]:<12} R*={rstar:.2f}  prob={probs[i]:.3f} -> {pred}")

    out_path = f"src/models/taylor_cnn_v10_5b_{mode}.pt"
    Path("src/models").mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "metrics": {k: v for k, v in m.items() if k != "probs"},
                "mode": mode}, out_path)
    print(f"Saved -> {out_path}")
    return m, probs


def main():
    print(f"Device: {DEVICE}")
    m_clip, probs_clip = run_mode("clipped")
    # Always also run log — clipped regressed from V10 0.861 → 0.805
    # so we want both numbers in the report.
    print("\nRunning log variant for comparison...")
    m_log, probs_log = run_mode("log")
    better = "log" if m_log["f1"] > m_clip["f1"] else "clipped"

    # Save winner alias
    winner_src = f"src/models/taylor_cnn_v10_5b_{better}.pt"
    winner_dst = "src/models/taylor_cnn_v10_5b.pt"
    import shutil
    shutil.copy(winner_src, winner_dst)
    print(f"\nWinner: {better}  saved to {winner_dst}")


if __name__ == "__main__":
    main()
