"""V9 DynamicGeometryLoss training + lambda sweep.

Sweep:   lambda_max ∈ {0.1, 0.5, 1.0}
Split:   seed=42, stratified 70/15/15 (identical to V6/V7.5/V8)
Loss:    BCE + 0.01 · |B|.mean() + DynamicGeometryLoss(...)
         Post-step A.clamp(min=0.001) so the gate can't die.

For each lambda we log:
    - final metrics (acc/prec/rec/F1)
    - A, B, t0 trajectory
    - T12/T14 distribution on test set for planets vs FPs

Outputs:
    src/models/taylor_cnn_v9.pt        — winning-lambda model
    data/v9_results.pt                 — probs/preds/features, all lambdas
    data/v9_training.log               — full log
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from src.models.geometry_loss import DynamicGeometryLoss
from src.models.taylor_cnn_v9 import TaylorCNNv9


DATA_PATH = "data/kepler_tce_v6.pt"
SS_CACHE = "data/ss_flag_cache.csv"
OUT_MODEL = "src/models/taylor_cnn_v9.pt"
RESULTS_PATH = "data/v9_results.pt"
SEED_SPLIT = 42
SEED_INIT = 7
LAMBDA_SWEEP = [0.1, 0.5, 1.0]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_snr_per_sample(names: list[str]) -> torch.Tensor:
    """Per-TCE koi_model_snr; missing filled with snr_pivot (12.0)."""
    cache = {}
    with open(SS_CACHE, newline="") as f:
        for r in csv.DictReader(f):
            cache[r["kepoi_name"]] = r
    snr = torch.full((len(names),), 12.0, dtype=torch.float32)
    for i, n in enumerate(names):
        rec = cache.get(n)
        if rec and rec["snr"] not in ("", "None"):
            snr[i] = float(rec["snr"])
    return snr


def load_and_split():
    d = torch.load(DATA_PATH, weights_only=False)
    y = d["labels"]
    names = d["names"]
    snr = load_snr_per_sample(names)

    torch.manual_seed(SEED_SPLIT)
    conf = (y == 1).nonzero(as_tuple=True)[0]
    fp = (y == 0).nonzero(as_tuple=True)[0]
    conf = conf[torch.randperm(len(conf))]
    fp = fp[torch.randperm(len(fp))]

    def split(idx, tf=0.7, vf=0.15):
        n = len(idx); nt = int(n * tf); nv = int(n * vf)
        return idx[:nt], idx[nt:nt + nv], idx[nt + nv:]

    ct, cv, cte = split(conf); ft, fv, fte = split(fp)
    train_idx = torch.cat([ct, ft])
    val_idx = torch.cat([cv, fv])
    test_idx = torch.cat([cte, fte])

    def bundle(idx):
        return (
            d["phases"][idx].to(DEVICE),
            d["fluxes"][idx].to(DEVICE),
            d["fluxes_secondary"][idx].to(DEVICE),
            d["fluxes_odd_even"][idx].to(DEVICE),
            y[idx].to(DEVICE),
            snr[idx].to(DEVICE),
        )

    print(f"Dataset: {len(y)} TCEs  ({int(y.sum())} conf, {int((1-y).sum())} FP)")
    print(f"Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")
    print(f"SNR median: {float(snr.median()):.1f}  |  <12: {int((snr < 12).sum())}/{len(snr)}")
    return bundle(train_idx), bundle(val_idx), bundle(test_idx), names, test_idx.tolist()


def finetune(model, train, val, geo_loss, n_epochs=200, patience=25, batch_size=16,
             lr_gate=1e-4, lr_main=1e-3):
    tp, tpp, tps, tpoe, tpl, tsnr = train
    vp, vpp, vps, vpoe, vpl, vsnr = val
    opt = torch.optim.Adam([
        {"params": model.taylor_gate.parameters(), "lr": lr_gate},
        {"params": model.cnn.parameters(), "lr": lr_main},
        {"params": model.classifier.parameters(), "lr": lr_main},
    ])
    bce = nn.BCELoss()
    n_tr = len(tpl)
    best_val = float("inf"); best_state = None; wait = 0
    A_trace = []

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n_tr, device=DEVICE)
        run_bce = run_geo = run_spar = 0.0
        for start in range(0, n_tr, batch_size):
            idx = perm[start:start + batch_size]
            pred = model(tp[idx], tpp[idx], tps[idx], tpoe[idx]).squeeze(1)
            loss_bce = bce(pred, tpl[idx])
            loss_spar = model.sparsity_loss()
            t12_t14, auc_norm = model.compute_shape_features(tpp[idx])
            loss_geo = geo_loss(pred, t12_t14, auc_norm, tsnr[idx])
            loss = loss_bce + loss_spar + loss_geo
            opt.zero_grad(); loss.backward(); opt.step()
            model.clamp_A(min_val=0.001)
            run_bce += loss_bce.item() * len(idx)
            run_geo += loss_geo.item() * len(idx)
            run_spar += loss_spar.item() * len(idx)

        model.eval()
        with torch.no_grad():
            v = model(vp, vpp, vps, vpoe).squeeze(1)
            vl_bce = bce(v, vpl).item()
        if vl_bce < best_val:
            best_val = vl_bce
            best_state = {k: vv.clone() for k, vv in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        A_trace.append(model.taylor_gate.A.item())
        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(
                f"    ep {epoch+1:>3}  bce={run_bce/n_tr:.4f}  geo={run_geo/n_tr:.4f}  "
                f"spar={run_spar/n_tr:.5f}  val={vl_bce:.4f}  wait={wait}  "
                f"A={model.taylor_gate.A.item():.4f}  "
                f"B={model.taylor_gate.B.item():+.4f}  "
                f"t0={model.taylor_gate.t0.item():+.4f}"
            )
        if wait >= patience:
            print(f"    early stop at ep {epoch+1}")
            break

    model.load_state_dict(best_state)
    return best_val, A_trace


def evaluate(model, test):
    tp, tpp, tps, tpoe, tpl, tsnr = test
    model.eval()
    with torch.no_grad():
        probs = model(tp, tpp, tps, tpoe).squeeze(1).cpu()
        t12_t14, auc_norm = model.compute_shape_features(tpp)
        t12_t14 = t12_t14.cpu(); auc_norm = auc_norm.cpu()
    preds = (probs > 0.5).float()
    tl = tpl.cpu()
    TP = int(((preds == 1) & (tl == 1)).sum())
    TN = int(((preds == 0) & (tl == 0)).sum())
    FP = int(((preds == 1) & (tl == 0)).sum())
    FN = int(((preds == 0) & (tl == 1)).sum())
    acc = (TP + TN) / len(tl)
    prec = TP / (TP + FP) if TP + FP else 0.0
    rec = TP / (TP + FN) if TP + FN else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "A": model.taylor_gate.A.item(),
        "B": model.taylor_gate.B.item(),
        "t0": model.taylor_gate.t0.item(),
        "probs": probs, "preds": preds,
        "t12_t14": t12_t14, "auc_norm": auc_norm,
    }


def param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_lambda(lam_max, train, val, test):
    print(f"\n{'='*60}\nlambda_max = {lam_max}\n{'='*60}")
    torch.manual_seed(SEED_INIT)
    model = TaylorCNNv9(init_amplitude=0.01, init_B=1.0, init_t0=0.0).to(DEVICE)
    geo = DynamicGeometryLoss(lambda_min=0.01, lambda_max=lam_max, snr_pivot=12.0).to(DEVICE)
    print(f"  Trainable params: {param_count(model)}")
    best_val, A_trace = finetune(model, train, val, geo)
    metrics = evaluate(model, test)
    metrics["best_val_loss"] = best_val
    metrics["params"] = param_count(model)
    metrics["A_trace"] = A_trace
    metrics["lambda_max"] = lam_max
    print(
        f"  lambda_max={lam_max}:  acc={metrics['accuracy']:.1%}  "
        f"prec={metrics['precision']:.1%}  rec={metrics['recall']:.1%}  "
        f"F1={metrics['f1']:.3f}  A={metrics['A']:.4f}  "
        f"B={metrics['B']:+.4f}  t0={metrics['t0']:+.4f}"
    )
    return model, metrics


def main():
    print(f"Device: {DEVICE}")
    train, val, test, names, test_idx_list = load_and_split()

    results = {}
    models = {}
    for lam in LAMBDA_SWEEP:
        m, r = run_lambda(lam, train, val, test)
        results[lam] = r
        models[lam] = m

    # Comparison
    print(f"\n{'='*80}")
    print("V9 lambda sweep comparison")
    print(f"{'='*80}")
    print(f"{'lambda_max':>10} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} "
          f"{'A':>7} {'B':>7} {'t0':>7} {'TP':>3} {'FP':>3} {'TN':>3} {'FN':>3}")
    print("-" * 80)
    for lam, r in results.items():
        print(f"{lam:>10.2f}  {r['accuracy']:>6.1%}  {r['precision']:>6.1%}  "
              f"{r['recall']:>6.1%}  {r['f1']:>6.3f}  "
              f"{r['A']:>.4f}  {r['B']:>+.4f}  {r['t0']:>+.4f}  "
              f"{r['TP']:>3} {r['FP']:>3} {r['TN']:>3} {r['FN']:>3}")

    best_lam = max(results.keys(), key=lambda k: (results[k]["f1"], results[k]["precision"]))
    winner = results[best_lam]
    print(f"\nBest lambda_max = {best_lam}  (F1={winner['f1']:.3f}, prec={winner['precision']:.1%})")

    # T12/T14 class separation on test set for the winner
    tl = test[4].cpu()
    t12_w = winner["t12_t14"]
    p_med = float(t12_w[tl == 1].median()); fp_med = float(t12_w[tl == 0].median())
    auc_w = winner["auc_norm"]
    p_auc = float(auc_w[tl == 1].median()); fp_auc = float(auc_w[tl == 0].median())
    print(f"\nT12/T14 class medians (winner):  planets={p_med:.3f}  FPs={fp_med:.3f}  diff={fp_med - p_med:+.3f}")
    print(f"AUC_norm class medians (winner): planets={p_auc:.3f}  FPs={fp_auc:.3f}  diff={fp_auc - p_auc:+.3f}")

    # Gate-alive report per lambda
    print(f"\nGate-alive check (A >= 0.001 at end):")
    for lam, r in results.items():
        status = "ALIVE" if r["A"] >= 0.001 else "DEAD"
        print(f"  lambda={lam:.2f}:  A_final={r['A']:.5f}  ({status})")

    # Precision vs V6 (76.7%)
    print(f"\nPrecision vs V6 (76.7%):")
    for lam, r in results.items():
        delta = r["precision"] - 0.767
        print(f"  lambda={lam:.2f}:  prec={r['precision']:.1%}  (delta {delta:+.1%})")

    # Save
    Path("src/models").mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": models[best_lam].state_dict(),
        "lambda_max": best_lam,
        "metrics": {k: v for k, v in winner.items()
                    if k not in ("probs", "preds", "t12_t14", "auc_norm", "A_trace")},
    }, OUT_MODEL)
    torch.save({
        "sweep": {lam: {k: v for k, v in r.items()
                        if k not in ("probs", "preds", "t12_t14", "auc_norm", "A_trace")}
                  for lam, r in results.items()},
        "per_sample": {lam: {
            "probs": r["probs"], "preds": r["preds"],
            "t12_t14": r["t12_t14"], "auc_norm": r["auc_norm"],
        } for lam, r in results.items()},
        "A_traces": {lam: r["A_trace"] for lam, r in results.items()},
        "test_labels": tl, "test_idx": test_idx_list, "names": names,
        "best_lambda": best_lam,
    }, RESULTS_PATH)
    print(f"\nSaved winner model -> {OUT_MODEL}")
    print(f"Saved sweep results -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
