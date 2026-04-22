"""Exp 4b — V10 trained natively on TESS (355 TCEs).

Same V10 architecture and lambda as Exp 4 (see scripts/run_v10_1114.py)
with two differences:
  * Data source is TESS (tess_tce_400.pt, 199 confirmed / 156 FP), so
    pos_weight = 156/199 = 0.784 (slightly DOWN-weights the positive
    class — planets are the majority in TESS).
  * No Kepler SNR cache for TESS targets; every sample uses
    SNR = SNR_PIVOT (100), putting the geometry loss at mid-strength.
  * Cross-mission evaluation runs against the Kepler 76-TCE test set
    derived from data/kepler_tce_v6.pt with the same seed=42 split used
    in the V10 paper, so the "TESS -> Kepler" number is comparable to
    the paper's "Kepler -> TESS" zero-shot.

Records:
    src/models/taylor_cnn_v10_tess.pt  — model state + metrics
    data/v10_tess_results.pt           — probs/preds on both test sets
    data/v10_tess_training.log         — captured stdout
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from src.models.geometry_loss_v2 import InvertedGeometryLoss
from src.models.taylor_cnn_v10 import TaylorCNNv10


TESS_DATA_PATH = "data/tess_tce_400.pt"
KEPLER_TEST_PATH = "data/kepler_tce_v6.pt"  # 500-TCE V10 baseline source
OUT_MODEL = "src/models/taylor_cnn_v10_tess.pt"
RESULTS_PATH = "data/v10_tess_results.pt"
SEED_SPLIT = 42
SEED_INIT = 7
LAMBDA = 0.1
SNR_PIVOT = 100.0
POS_WEIGHT = 156.0 / 199.0  # 0.784, from dataset class balance
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def stratified_split(labels: torch.Tensor, seed: int, tf=0.7, vf=0.15):
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


def bundle(d: dict, idx, snr: torch.Tensor):
    return (
        d["phases"][idx].to(DEVICE),
        d["fluxes"][idx].to(DEVICE),
        d["fluxes_secondary"][idx].to(DEVICE),
        d["fluxes_odd_even"][idx].to(DEVICE),
        d["labels"][idx].to(DEVICE),
        snr[idx].to(DEVICE),
    )


def load_tess_split():
    d = torch.load(TESS_DATA_PATH, weights_only=False)
    snr = torch.full((len(d["labels"]),), SNR_PIVOT, dtype=torch.float32)
    train_idx, val_idx, test_idx = stratified_split(d["labels"], SEED_SPLIT)
    print(f"TESS 355 — Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")
    print(f"  pos_weight (FP/confirmed) = {POS_WEIGHT:.3f}")
    return (bundle(d, train_idx, snr),
            bundle(d, val_idx, snr),
            bundle(d, test_idx, snr),
            d["names"], test_idx.tolist())


def weighted_bce(pred, target, pos_weight):
    per = nn.functional.binary_cross_entropy(pred, target, reduction="none")
    w = torch.where(target > 0.5, torch.full_like(target, pos_weight),
                    torch.ones_like(target))
    return (per * w).mean()


def finetune(model, train, val, geo_loss, n_epochs=200, patience=25, batch_size=16,
             lr_gate=1e-4, lr_main=1e-3):
    tp, tpp, tps, tpoe, tpl, tsnr = train
    vp, vpp, vps, vpoe, vpl, _ = val
    opt = torch.optim.Adam([
        {"params": model.gate_bank.parameters(), "lr": lr_gate},
        {"params": model.cnn.parameters(), "lr": lr_main},
        {"params": model.classifier.parameters(), "lr": lr_main},
    ])
    n_tr = len(tpl)
    best_val = float("inf"); best_state = None; wait = 0

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n_tr, device=DEVICE)
        run_bce = run_geo = 0.0
        for start in range(0, n_tr, batch_size):
            idx = perm[start:start + batch_size]
            pred = model(tp[idx], tpp[idx], tps[idx], tpoe[idx]).squeeze(1)
            loss_bce = weighted_bce(pred, tpl[idx], POS_WEIGHT)
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
            vl = weighted_bce(v, vpl, POS_WEIGHT).item()
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
    tl = labels.cpu()
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


def zero_shot_kepler_76(model):
    """Evaluate on the 76-TCE Kepler test set (same seed=42 split as V10 paper)."""
    d = torch.load(KEPLER_TEST_PATH, weights_only=False)
    _, _, test_idx = stratified_split(d["labels"], SEED_SPLIT)
    phase = d["phases"][test_idx].to(DEVICE)
    primary = d["fluxes"][test_idx].to(DEVICE)
    secondary = d["fluxes_secondary"][test_idx].to(DEVICE)
    oe = d["fluxes_odd_even"][test_idx].to(DEVICE)
    labels = d["labels"][test_idx]
    probs = forward_probs(model, phase, primary, secondary, oe)
    m = metrics_from(probs, labels)
    m["names"] = [d["names"][i] for i in test_idx.tolist()]
    m["test_idx"] = test_idx.tolist()
    return m


def main():
    print(f"Device: {DEVICE}")
    train, val, test, names, test_idx_list = load_tess_split()

    print(f"\n{'='*64}\nTraining V10 lambda={LAMBDA} on TESS 355, pos_weight={POS_WEIGHT:.3f}\n{'='*64}")
    torch.manual_seed(SEED_INIT)
    model = TaylorCNNv10(init_amplitude=0.01).to(DEVICE)
    geo = InvertedGeometryLoss(lambda_min=0.01, lambda_max=LAMBDA, snr_pivot=SNR_PIVOT).to(DEVICE)
    best_val = finetune(model, train, val, geo)

    # Native TESS test split
    tp, tpp, tps, tpoe, tpl, _ = test
    probs_t = forward_probs(model, tp, tpp, tps, tpoe)
    tess_m = metrics_from(probs_t, tpl)
    amps = model.gate_bank.amplitudes()
    print(f"\nTESS native test (n={tess_m['n']}):")
    print(f"  acc={tess_m['accuracy']:.1%}  prec={tess_m['precision']:.1%}  "
          f"rec={tess_m['recall']:.1%}  F1={tess_m['f1']:.3f}  "
          f"TP={tess_m['TP']} FP={tess_m['FP']} TN={tess_m['TN']} FN={tess_m['FN']}")
    print(f"  Amplitudes  A1={amps['A1']:.4f} A2={amps['A2']:.4f} A3={amps['A3']:.4f} "
          f"A4={amps['A4']:.4f} A5={amps['A5']:.4f}")

    # Kepler 76 zero-shot
    print(f"\nKepler 76-TCE zero-shot (from {KEPLER_TEST_PATH}):")
    kep_m = zero_shot_kepler_76(model)
    print(f"  acc={kep_m['accuracy']:.1%}  prec={kep_m['precision']:.1%}  "
          f"rec={kep_m['recall']:.1%}  F1={kep_m['f1']:.3f}  "
          f"TP={kep_m['TP']} FP={kep_m['FP']} TN={kep_m['TN']} FN={kep_m['FN']}")

    # Comparison
    print(f"\n{'='*72}")
    print(f"{'Version':<22} {'Prec':>7} {'Recall':>8} {'F1':>7} {'test_n':>7}")
    print("-" * 72)
    print(f"{'Kepler -> TESS (Exp 3)':<22} {'80.0%':>7} {'100.0%':>8} {'0.889':>7} {'15':>7}")
    print(f"{'TESS native':<22} "
          f"{tess_m['precision']:>6.1%} {tess_m['recall']:>7.1%} "
          f"{tess_m['f1']:>7.3f} {tess_m['n']:>7}")
    print(f"{'V10 500 native (Kep)':<22} {'82.9%':>7} {'89.5%':>8} {'0.861':>7} {'76':>7}")
    print(f"{'TESS -> Kepler':<22} "
          f"{kep_m['precision']:>6.1%} {kep_m['recall']:>7.1%} "
          f"{kep_m['f1']:>7.3f} {kep_m['n']:>7}")

    # Save
    Path("src/models").mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "lambda_max": LAMBDA,
        "pos_weight": POS_WEIGHT,
        "metrics_tess_native": {k: v for k, v in tess_m.items()
                                if k not in ("probs", "preds")},
        "metrics_kepler_zero_shot": {k: v for k, v in kep_m.items()
                                     if k not in ("probs", "preds", "names", "test_idx")},
        "amplitudes": amps,
    }, OUT_MODEL)
    torch.save({
        "tess_probs": tess_m["probs"],
        "tess_preds": tess_m["preds"],
        "tess_labels": tpl.cpu(),
        "tess_test_idx": test_idx_list,
        "tess_names": names,
        "kepler_probs": kep_m["probs"],
        "kepler_preds": kep_m["preds"],
        "kepler_labels": torch.tensor(
            [torch.load(KEPLER_TEST_PATH, weights_only=False)["labels"][i]
             for i in kep_m["test_idx"]]),
        "kepler_names": kep_m["names"],
        "kepler_test_idx": kep_m["test_idx"],
        "best_val_loss": best_val,
        "lambda_max": LAMBDA,
    }, RESULTS_PATH)
    print(f"\nSaved native model -> {OUT_MODEL}")
    print(f"Saved results     -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
