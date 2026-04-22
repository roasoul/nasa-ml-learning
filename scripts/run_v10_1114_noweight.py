# ═══════════════════════════════════════════
# PRODUCTION MODEL PROTECTION
# NEVER save to src/models/production/
# Save experiments to src/models/ only
# with descriptive version suffix
# ═══════════════════════════════════════════
"""Exp 4 follow-up — V10 on 1114 Kepler TCEs WITHOUT pos_weight.

Isolates the boundary-shift hypothesis from Exp 4. Architecture, split
(seed=42 stratified 70/15/15), epochs, and geometry-loss settings are
identical to scripts/run_v10_1114.py; only the BCE term changes from
weighted to unweighted (nn.BCELoss()), which matches the original
scripts/run_v10.py loss and the published V10 500-TCE baseline.

If Kepler precision recovers above ~75% → pos_weight was the culprit
and the 1114-TCE data is healthy. If precision stays low → the harder
FP mix from subsample_evenly across 9564 KOIs is the real issue and
the architecture/training recipe needs revision.

Also reports:
  * Original paper-baseline 76-TCE test set from kepler_tce_v6.pt
    (seed=42 split) — direct comparison to V10 500's F1 0.861.
  * TESS 355 zero-shot — direct comparison to Exp 4's F1 0.547.

Records:
    src/models/taylor_cnn_v10_1114_noweight.pt
    data/v10_1114_noweight_results.pt
    data/v10_1114_noweight_training.log
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from src.models.geometry_loss_v2 import InvertedGeometryLoss
from src.models.taylor_cnn_v10 import TaylorCNNv10


DATA_PATH = "data/kepler_tce_2000.pt"
KEPLER_V6_PATH = "data/kepler_tce_v6.pt"
TESS_DATA_PATH = "data/tess_tce_400.pt"
SS_CACHE = "data/ss_flag_cache.csv"
OUT_MODEL = "src/models/taylor_cnn_v10_1114_noweight.pt"
RESULTS_PATH = "data/v10_1114_noweight_results.pt"
SEED_SPLIT = 42
SEED_INIT = 7
LAMBDA = 0.1
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


def stratified_split(labels, seed, tf=0.7, vf=0.15):
    torch.manual_seed(seed)
    conf = (labels == 1).nonzero(as_tuple=True)[0]
    fp = (labels == 0).nonzero(as_tuple=True)[0]
    conf = conf[torch.randperm(len(conf))]
    fp = fp[torch.randperm(len(fp))]
    def part(idx):
        n = len(idx); nt = int(n * tf); nv = int(n * vf)
        return idx[:nt], idx[nt:nt + nv], idx[nt + nv:]
    ct, cv, cte = part(conf); ft, fv, fte = part(fp)
    return torch.cat([ct, ft]), torch.cat([cv, fv]), torch.cat([cte, fte])


def bundle(d, idx, snr):
    return (
        d["phases"][idx].to(DEVICE),
        d["fluxes"][idx].to(DEVICE),
        d["fluxes_secondary"][idx].to(DEVICE),
        d["fluxes_odd_even"][idx].to(DEVICE),
        d["labels"][idx].to(DEVICE),
        snr[idx].to(DEVICE),
    )


def load_and_split():
    d = torch.load(DATA_PATH, weights_only=False)
    snr = load_snr(d["names"])
    train_idx, val_idx, test_idx = stratified_split(d["labels"], SEED_SPLIT)
    print(f"Kepler 1114 — Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")
    print(f"  Loss: nn.BCELoss() (no pos_weight)")
    print(f"  SNR median: {float(snr.median()):.1f}  |  pivot: {SNR_PIVOT}  |  "
          f"below-pivot: {int((snr < SNR_PIVOT).sum())}/{len(snr)}")
    train = bundle(d, train_idx, snr)
    val = bundle(d, val_idx, snr)
    test = bundle(d, test_idx, snr)
    return train, val, test, d["names"], test_idx.tolist()


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

        if (epoch + 1) % 25 == 0 or epoch == 0:
            amps = model.gate_bank.amplitudes()
            print(
                f"  ep {epoch+1:>3}  bce={run_bce/n_tr:.4f}  geo={run_geo/n_tr:.4f}  "
                f"val={vl:.4f}  wait={wait}  "
                f"A1={amps['A1']:.4f} A2={amps['A2']:.4f} A3={amps['A3']:.4f} "
                f"A4={amps['A4']:.4f} A5={amps['A5']:.4f}"
            )
        if wait >= patience:
            print(f"  early stop at ep {epoch+1}")
            break

    model.load_state_dict(best_state)
    return best_val


def forward_probs(model, phase, primary, secondary, oe):
    model.eval()
    with torch.no_grad():
        return model(phase, primary, secondary, oe).squeeze(1).cpu()


def metrics_from(probs, labels, threshold=0.5):
    preds = (probs > threshold).float()
    tl = labels.cpu() if isinstance(labels, torch.Tensor) else torch.tensor(labels)
    TP = int(((preds == 1) & (tl == 1)).sum())
    TN = int(((preds == 0) & (tl == 0)).sum())
    FP = int(((preds == 1) & (tl == 0)).sum())
    FN = int(((preds == 0) & (tl == 1)).sum())
    n = len(tl)
    acc = (TP + TN) / n
    prec = TP / (TP + FP) if TP + FP else 0.0
    rec = TP / (TP + FN) if TP + FN else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "TP": TP, "TN": TN, "FP": FP, "FN": FN, "n": n,
            "probs": probs, "preds": preds}


def eval_on_split(model, d, idx):
    phase = d["phases"][idx].to(DEVICE)
    primary = d["fluxes"][idx].to(DEVICE)
    secondary = d["fluxes_secondary"][idx].to(DEVICE)
    oe = d["fluxes_odd_even"][idx].to(DEVICE)
    probs = forward_probs(model, phase, primary, secondary, oe)
    return metrics_from(probs, d["labels"][idx])


def zero_shot_tess(model):
    t = torch.load(TESS_DATA_PATH, weights_only=False)
    phase = t["phases"].to(DEVICE)
    primary = t["fluxes"].to(DEVICE)
    secondary = t["fluxes_secondary"].to(DEVICE)
    oe = t["fluxes_odd_even"].to(DEVICE)
    probs = forward_probs(model, phase, primary, secondary, oe)
    m = metrics_from(probs, t["labels"])
    m["names"] = t["names"]
    return m


def eval_kepler_76(model):
    d = torch.load(KEPLER_V6_PATH, weights_only=False)
    _, _, test_idx = stratified_split(d["labels"], SEED_SPLIT)
    return eval_on_split(model, d, test_idx), [d["names"][i] for i in test_idx.tolist()]


def fmt(m):
    return (f"acc={m['accuracy']:.1%}  prec={m['precision']:.1%}  "
            f"rec={m['recall']:.1%}  F1={m['f1']:.3f}  "
            f"TP={m['TP']} FP={m['FP']} TN={m['TN']} FN={m['FN']}")


def main():
    print(f"Device: {DEVICE}")
    train, val, test, names, test_idx_list = load_and_split()

    print(f"\n{'='*64}\nTraining V10 lambda={LAMBDA} on Kepler 1114, NO pos_weight\n{'='*64}")
    torch.manual_seed(SEED_INIT)
    model = TaylorCNNv10(init_amplitude=0.01).to(DEVICE)
    geo = InvertedGeometryLoss(lambda_min=0.01, lambda_max=LAMBDA, snr_pivot=SNR_PIVOT).to(DEVICE)
    best_val = finetune(model, train, val, geo)

    # 1114 held-out test
    tp, tpp, tps, tpoe, tpl, _ = test
    probs_k = forward_probs(model, tp, tpp, tps, tpoe)
    kepler_1114_m = metrics_from(probs_k, tpl)
    amps = model.gate_bank.amplitudes()
    print(f"\nKepler 1114 held-out (n={kepler_1114_m['n']}):")
    print(f"  {fmt(kepler_1114_m)}")
    print(f"  Amplitudes  A1={amps['A1']:.4f} A2={amps['A2']:.4f} A3={amps['A3']:.4f} "
          f"A4={amps['A4']:.4f} A5={amps['A5']:.4f}")

    # Paper 76-TCE test set
    kepler_76_m, kepler_76_names = eval_kepler_76(model)
    print(f"\nPaper 76-TCE test set (n={kepler_76_m['n']}):")
    print(f"  {fmt(kepler_76_m)}")

    # TESS 355 zero-shot
    tess_m = zero_shot_tess(model)
    print(f"\nTESS 355 zero-shot:")
    print(f"  {fmt(tess_m)}")

    # Comparison table
    print(f"\n{'='*74}")
    print(f"{'Version':<24} {'Prec':>7} {'Recall':>8} {'F1':>7} {'test_n':>7}")
    print("-" * 74)
    print(f"{'V10 500-TCE baseline':<24} {'82.9%':>7} {'89.5%':>8} {'0.861':>7} {'76':>7}")
    print(f"{'V10 1114 + pos_weight':<24} {'58.2%':>7} {'85.2%':>8} {'0.692':>7} {'170':>7}")
    print(f"{'V10 1114 no pos_weight':<24} "
          f"{kepler_1114_m['precision']:>6.1%} {kepler_1114_m['recall']:>7.1%} "
          f"{kepler_1114_m['f1']:>7.3f} {kepler_1114_m['n']:>7}")
    print(f"{'  (same model, 76-test)':<24} "
          f"{kepler_76_m['precision']:>6.1%} {kepler_76_m['recall']:>7.1%} "
          f"{kepler_76_m['f1']:>7.3f} {kepler_76_m['n']:>7}")
    print(f"{'Kepler -> TESS (Exp 3)':<24} {'80.0%':>7} {'100.0%':>8} {'0.889':>7} {'15':>7}")
    print(f"{'1114 + pos_weight -> TESS':<24} {'53.6%':>7} {'55.8%':>8} {'0.547':>7} {'355':>7}")
    print(f"{'1114 no weight -> TESS':<24} "
          f"{tess_m['precision']:>6.1%} {tess_m['recall']:>7.1%} "
          f"{tess_m['f1']:>7.3f} {tess_m['n']:>7}")

    # Hypothesis verdict
    verdict_kep_1114 = "pos_weight was the culprit" if kepler_1114_m["precision"] >= 0.75 \
        else "harder FPs are the real issue"
    verdict_kep_76 = "matches V10-500 baseline" if kepler_76_m["f1"] >= 0.85 \
        else "below V10-500 baseline"
    verdict_tess = "recall recovers toward 100%" if tess_m["recall"] >= 0.85 \
        else "TESS recall still weak"
    print(f"\nVerdicts:")
    print(f"  Kepler 1114 precision  -> {kepler_1114_m['precision']:.1%}  [{verdict_kep_1114}]")
    print(f"  Paper 76-TCE F1        -> {kepler_76_m['f1']:.3f}  [{verdict_kep_76}]")
    print(f"  TESS 355 recall        -> {tess_m['recall']:.1%}  [{verdict_tess}]")

    # Save
    Path("src/models").mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "lambda_max": LAMBDA,
        "pos_weight": None,
        "metrics_kepler_1114_test": {k: v for k, v in kepler_1114_m.items()
                                     if k not in ("probs", "preds")},
        "metrics_kepler_76_test": {k: v for k, v in kepler_76_m.items()
                                   if k not in ("probs", "preds")},
        "metrics_tess_zero_shot": {k: v for k, v in tess_m.items()
                                   if k not in ("probs", "preds", "names")},
        "amplitudes": amps,
    }, OUT_MODEL)
    torch.save({
        "kepler_1114": {"probs": kepler_1114_m["probs"], "preds": kepler_1114_m["preds"],
                        "labels": tpl.cpu(), "test_idx": test_idx_list},
        "kepler_76": {"probs": kepler_76_m["probs"], "preds": kepler_76_m["preds"],
                      "names": kepler_76_names},
        "tess_355": {"probs": tess_m["probs"], "preds": tess_m["preds"],
                     "names": tess_m["names"]},
        "best_val_loss": best_val,
        "lambda_max": LAMBDA,
    }, RESULTS_PATH)
    print(f"\nSaved model   -> {OUT_MODEL}")
    print(f"Saved results -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
