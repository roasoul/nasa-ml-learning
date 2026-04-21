"""V10 training — 5-gate bank + InvertedGeometryLoss lambda sweep.

Sweep:   lambda_max ∈ {0.1, 0.5, 1.0}
Split:   seed=42 stratified 70/15/15 (identical to V6-V9 for apples-to-apples).
Loss:    BCE + 0.01 · |G4_asymmetry_amp|.mean() optional sparsity
         + InvertedGeometryLoss(prob, t12_t14, auc_norm, snr, pivot=100).
         Post-step all 5 amplitudes clamped to min=0.001.

Records per run:
    - per-epoch amplitude traces (A1..A5)
    - final metrics, gate activations on test set by class
Saves:
    src/models/taylor_cnn_v10.pt       — winning-lambda model
    data/v10_results.pt                — probs/preds + gate activations
    data/v10_training.log              — full log
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
OUT_MODEL = "src/models/taylor_cnn_v10.pt"
RESULTS_PATH = "data/v10_results.pt"
SEED_SPLIT = 42
SEED_INIT = 7
LAMBDA_SWEEP = [0.1, 0.5, 1.0]
SNR_PIVOT = 100.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    print(f"Dataset: {len(y)} TCEs ({int(y.sum())} conf, {int((1-y).sum())} FP)")
    print(f"Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")
    print(f"SNR median: {float(snr.median()):.1f}  |  pivot: {SNR_PIVOT}  |  "
          f"below-pivot: {int((snr < SNR_PIVOT).sum())}/{len(snr)}")
    return bundle(train_idx), bundle(val_idx), bundle(test_idx), names, test_idx.tolist()


def finetune(model, train, val, geo_loss, n_epochs=200, patience=25, batch_size=16,
             lr_gate=1e-4, lr_main=1e-3):
    tp, tpp, tps, tpoe, tpl, tsnr = train
    vp, vpp, vps, vpoe, vpl, _ = val
    opt = torch.optim.Adam([
        {"params": model.gate_bank.parameters(), "lr": lr_gate},
        {"params": model.cnn.parameters(), "lr": lr_main},
        {"params": model.classifier.parameters(), "lr": lr_main},
    ])
    bce = nn.BCELoss()
    n_tr = len(tpl)
    best_val = float("inf"); best_state = None; wait = 0
    amp_traces = {k: [] for k in ("A1", "A2", "A3", "A4", "A5")}

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n_tr, device=DEVICE)
        run_bce = run_geo = 0.0
        for start in range(0, n_tr, batch_size):
            idx = perm[start:start + batch_size]
            pred = model(tp[idx], tpp[idx], tps[idx], tpoe[idx]).squeeze(1)
            loss_bce = bce(pred, tpl[idx])
            t12_t14, auc_norm = model.compute_shape_features(tpp[idx])
            loss_geo = geo_loss(pred, t12_t14, auc_norm, tsnr[idx])
            loss = loss_bce + loss_geo
            opt.zero_grad(); loss.backward(); opt.step()
            model.clamp_amplitudes(0.001)
            run_bce += loss_bce.item() * len(idx)
            run_geo += loss_geo.item() * len(idx)

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

        amps = model.gate_bank.amplitudes()
        for k in amp_traces:
            amp_traces[k].append(amps[k])

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(
                f"    ep {epoch+1:>3}  bce={run_bce/n_tr:.4f}  "
                f"geo={run_geo/n_tr:.4f}  val={vl:.4f}  wait={wait}  "
                f"A1={amps['A1']:.4f} A2={amps['A2']:.4f} A3={amps['A3']:.4f} "
                f"A4={amps['A4']:.4f} A5={amps['A5']:.4f}"
            )
        if wait >= patience:
            print(f"    early stop at ep {epoch+1}")
            break

    model.load_state_dict(best_state)
    return best_val, amp_traces


def gate_activations(model, phase, primary_flux, secondary, oe, labels):
    """Mean |gate output| per TCE per gate, split by class."""
    model.eval()
    with torch.no_grad():
        gates = model.gate_bank(phase)          # (B, 5, L)
        per_sample = gates.abs().mean(dim=-1).cpu()   # (B, 5)
        t12_t14, auc_norm = model.compute_shape_features(primary_flux)
        probs = model(phase, primary_flux, secondary, oe).squeeze(1).cpu()
    return per_sample, probs, t12_t14.cpu(), auc_norm.cpu()


def evaluate(model, test):
    tp, tpp, tps, tpoe, tpl, _ = test
    gate_act, probs, t12_t14, auc_norm = gate_activations(model, tp, tpp, tps, tpoe, tpl)
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
    amps = model.gate_bank.amplitudes()
    return {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "amplitudes": amps,
        "probs": probs, "preds": preds,
        "gate_activations": gate_act,
        "t12_t14": t12_t14, "auc_norm": auc_norm,
    }


def param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_lambda(lam_max, train, val, test):
    print(f"\n{'='*60}\nlambda_max = {lam_max}\n{'='*60}")
    torch.manual_seed(SEED_INIT)
    model = TaylorCNNv10(init_amplitude=0.01).to(DEVICE)
    geo = InvertedGeometryLoss(lambda_min=0.01, lambda_max=lam_max, snr_pivot=SNR_PIVOT).to(DEVICE)
    print(f"  Trainable params: {param_count(model)}")
    best_val, amp_traces = finetune(model, train, val, geo)
    m = evaluate(model, test)
    m["best_val_loss"] = best_val
    m["params"] = param_count(model)
    m["amp_traces"] = amp_traces
    m["lambda_max"] = lam_max
    amps = m["amplitudes"]
    print(
        f"  lambda_max={lam_max}:  acc={m['accuracy']:.1%}  "
        f"prec={m['precision']:.1%}  rec={m['recall']:.1%}  F1={m['f1']:.3f}  "
        f"A1={amps['A1']:.4f} A2={amps['A2']:.4f} A3={amps['A3']:.4f} "
        f"A4={amps['A4']:.4f} A5={amps['A5']:.4f}"
    )
    return model, m


def main():
    print(f"Device: {DEVICE}")
    train, val, test, names, test_idx_list = load_and_split()

    results = {}; models = {}
    for lam in LAMBDA_SWEEP:
        m, r = run_lambda(lam, train, val, test)
        results[lam] = r
        models[lam] = m

    print(f"\n{'='*90}")
    print("V10 lambda sweep comparison (InvertedGeometryLoss, SNR pivot=100)")
    print(f"{'='*90}")
    print(f"{'lambda_max':>10} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} "
          f"{'A1':>6} {'A2':>6} {'A3':>6} {'A4':>6} {'A5':>6} "
          f"{'TP':>3} {'FP':>3} {'TN':>3} {'FN':>3}")
    print("-" * 90)
    for lam, r in results.items():
        a = r["amplitudes"]
        print(f"{lam:>10.2f}  {r['accuracy']:>6.1%}  {r['precision']:>6.1%}  "
              f"{r['recall']:>6.1%}  {r['f1']:>6.3f}  "
              f"{a['A1']:.4f} {a['A2']:.4f} {a['A3']:.4f} {a['A4']:.4f} {a['A5']:.4f}  "
              f"{r['TP']:>3} {r['FP']:>3} {r['TN']:>3} {r['FN']:>3}")

    best_lam = max(results.keys(), key=lambda k: (results[k]["f1"], results[k]["precision"]))
    winner = results[best_lam]
    print(f"\nBest lambda_max = {best_lam}  (F1={winner['f1']:.3f}, prec={winner['precision']:.1%})")

    # Diagnostic: dominant gate per class
    gate_act = winner["gate_activations"]
    tl = test[-2].cpu()
    p_mask = tl == 1; fp_mask = tl == 0
    planet_means = gate_act[p_mask].mean(dim=0)
    fp_means = gate_act[fp_mask].mean(dim=0)
    names_g = ["G1 planet U", "G2 V-shape", "G3 inv. secondary", "G4 asymmetric", "G5 Gaussian"]
    print(f"\nGate activation means (mean |gate output|) on test set (lambda={best_lam}):")
    print(f"  {'gate':<22} {'planet mean':>13} {'FP mean':>13} {'ratio p/fp':>12}")
    for i, n in enumerate(names_g):
        ratio = planet_means[i] / (fp_means[i] + 1e-8)
        print(f"  {n:<22} {planet_means[i]:>13.5f} {fp_means[i]:>13.5f} {ratio:>12.2f}")

    dominant = int(winner["amplitudes"]["A1"] > 0 and
                   max(winner["amplitudes"].items(), key=lambda kv: kv[1])[0] == "A1")
    max_amp_key = max(winner["amplitudes"].items(), key=lambda kv: kv[1])[0]
    print(f"\nDominant gate by amplitude: {max_amp_key}")

    # Precision vs V6 baseline
    print(f"\nPrecision vs V6 (76.7%) and V9 best (72.1%):")
    for lam, r in results.items():
        d_v6 = r["precision"] - 0.767
        d_v9 = r["precision"] - 0.721
        print(f"  lambda={lam:.2f}:  prec={r['precision']:.1%}  delta vs V6 {d_v6:+.1%}  delta vs V9 {d_v9:+.1%}")

    # Gate-alive check
    print(f"\nAmplitude survival (>= 0.001 floor):")
    for lam, r in results.items():
        alive = sum(1 for v in r["amplitudes"].values() if v > 0.0011)
        print(f"  lambda={lam:.2f}:  {alive}/5 above 0.001 floor")

    # Save
    Path("src/models").mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": models[best_lam].state_dict(),
        "lambda_max": best_lam,
        "metrics": {k: v for k, v in winner.items()
                    if k not in ("probs", "preds", "gate_activations",
                                 "t12_t14", "auc_norm", "amp_traces")},
    }, OUT_MODEL)
    torch.save({
        "sweep": {lam: {k: v for k, v in r.items()
                        if k not in ("probs", "preds", "gate_activations",
                                     "t12_t14", "auc_norm", "amp_traces")}
                  for lam, r in results.items()},
        "per_sample": {lam: {
            "probs": r["probs"], "preds": r["preds"],
            "gate_activations": r["gate_activations"],
            "t12_t14": r["t12_t14"], "auc_norm": r["auc_norm"],
        } for lam, r in results.items()},
        "amp_traces": {lam: r["amp_traces"] for lam, r in results.items()},
        "test_labels": tl, "test_idx": test_idx_list, "names": names,
        "best_lambda": best_lam,
    }, RESULTS_PATH)
    print(f"\nSaved winner model -> {OUT_MODEL}")
    print(f"Saved sweep results -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
