# ═══════════════════════════════════════════
# PRODUCTION MODEL PROTECTION
# NEVER save to src/models/production/
# Save experiments to src/models/ only
# with descriptive version suffix
# ═══════════════════════════════════════════
"""Exp 4 — V10 retrained on the 1114-TCE Kepler dataset.

Pipeline is identical to scripts/run_v10.py at the winning lambda_max=0.1,
with two additions:
  * Per-sample class weighting via pos_weight = 764/350 = 2.183 applied
    to nn.BCELoss (reduction='none') so the BCE term mirrors
    BCEWithLogitsLoss(pos_weight=...) without touching the sigmoid head.
  * Cross-mission zero-shot evaluation on the TESS 355-TCE dataset.

Records:
    src/models/taylor_cnn_v10_1114.pt    — model state + metrics
    data/v10_1114_results.pt             — probs/preds + gate activations +
                                            cross-mission TESS evaluation
    data/v10_1114_training.log           — captured stdout
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
TESS_DATA_PATH = "data/tess_tce_400.pt"
SS_CACHE = "data/ss_flag_cache.csv"
OUT_MODEL = "src/models/taylor_cnn_v10_1114.pt"
RESULTS_PATH = "data/v10_1114_results.pt"
SEED_SPLIT = 42
SEED_INIT = 7
LAMBDA = 0.1
SNR_PIVOT = 100.0
POS_WEIGHT = 764.0 / 350.0  # class balance from dataset: 2.183
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_snr(names: list[str]) -> torch.Tensor:
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


def load_and_split():
    d = torch.load(DATA_PATH, weights_only=False)
    snr = load_snr(d["names"])
    train_idx, val_idx, test_idx = stratified_split(d["labels"], SEED_SPLIT)
    print(f"Kepler 1114 — Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")
    print(f"  pos_weight (FP/confirmed) = {POS_WEIGHT:.3f}")
    print(f"  SNR median: {float(snr.median()):.1f}  |  pivot: {SNR_PIVOT}  |  "
          f"below-pivot: {int((snr < SNR_PIVOT).sum())}/{len(snr)}")
    train = bundle(d, train_idx, snr)
    val = bundle(d, val_idx, snr)
    test = bundle(d, test_idx, snr)
    return train, val, test, d["names"], test_idx.tolist()


def weighted_bce(pred: torch.Tensor, target: torch.Tensor, pos_weight: float) -> torch.Tensor:
    """BCELoss with per-sample class weight = pos_weight for y=1, 1.0 for y=0.

    Matches the semantics of BCEWithLogitsLoss(pos_weight=...) but operates
    on probabilities so the existing V10 sigmoid head stays unchanged.
    """
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
    amp_traces = {k: [] for k in ("A1", "A2", "A3", "A4", "A5")}

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

        amps = model.gate_bank.amplitudes()
        for k in amp_traces:
            amp_traces[k].append(amps[k])

        if (epoch + 1) % 25 == 0 or epoch == 0:
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
    return best_val, amp_traces


def forward_probs(model, phase, primary, secondary, oe):
    model.eval()
    with torch.no_grad():
        p = model(phase, primary, secondary, oe).squeeze(1).cpu()
    return p


def metrics_from(probs: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5):
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


def zero_shot_tess(model) -> dict:
    """Run the Kepler-trained model on the entire TESS dataset."""
    t = torch.load(TESS_DATA_PATH, weights_only=False)
    phase = t["phases"].to(DEVICE)
    primary = t["fluxes"].to(DEVICE)
    secondary = t["fluxes_secondary"].to(DEVICE)
    oe = t["fluxes_odd_even"].to(DEVICE)
    labels = t["labels"]
    probs = forward_probs(model, phase, primary, secondary, oe)
    m = metrics_from(probs, labels)
    m["names"] = t["names"]
    return m


def main():
    print(f"Device: {DEVICE}")
    train, val, test, names, test_idx_list = load_and_split()

    print(f"\n{'='*64}\nTraining V10 lambda={LAMBDA} on Kepler 1114, pos_weight={POS_WEIGHT:.3f}\n{'='*64}")
    torch.manual_seed(SEED_INIT)
    model = TaylorCNNv10(init_amplitude=0.01).to(DEVICE)
    geo = InvertedGeometryLoss(lambda_min=0.01, lambda_max=LAMBDA, snr_pivot=SNR_PIVOT).to(DEVICE)
    best_val, amp_traces = finetune(model, train, val, geo)

    # Kepler 1114 test split
    tp, tpp, tps, tpoe, tpl, _ = test
    probs_k = forward_probs(model, tp, tpp, tps, tpoe)
    kepler_m = metrics_from(probs_k, tpl)
    amps = model.gate_bank.amplitudes()
    print(f"\nKepler 1114 held-out test (n={kepler_m['n']}):")
    print(f"  acc={kepler_m['accuracy']:.1%}  prec={kepler_m['precision']:.1%}  "
          f"rec={kepler_m['recall']:.1%}  F1={kepler_m['f1']:.3f}  "
          f"TP={kepler_m['TP']} FP={kepler_m['FP']} TN={kepler_m['TN']} FN={kepler_m['FN']}")
    print(f"  Amplitudes  A1={amps['A1']:.4f} A2={amps['A2']:.4f} A3={amps['A3']:.4f} "
          f"A4={amps['A4']:.4f} A5={amps['A5']:.4f}")

    # TESS zero-shot
    print(f"\nTESS 355 zero-shot:")
    tess_m = zero_shot_tess(model)
    print(f"  acc={tess_m['accuracy']:.1%}  prec={tess_m['precision']:.1%}  "
          f"rec={tess_m['recall']:.1%}  F1={tess_m['f1']:.3f}  "
          f"TP={tess_m['TP']} FP={tess_m['FP']} TN={tess_m['TN']} FN={tess_m['FN']}")

    # Comparison table
    print(f"\n{'='*72}")
    print(f"{'Version':<18} {'Prec':>7} {'Recall':>8} {'F1':>7} {'n_train':>9} {'test_n':>7}")
    print("-" * 72)
    print(f"{'V10 500-TCE':<18} {'82.9%':>7} {'89.5%':>8} {'0.861':>7} {'350':>9} {'76':>7}")
    print(f"{'V10 1114-TCE':<18} "
          f"{kepler_m['precision']:>6.1%} {kepler_m['recall']:>7.1%} "
          f"{kepler_m['f1']:>7.3f} {int(0.7*1114):>9} {kepler_m['n']:>7}")
    print(f"{'V10 1114 → TESS':<18} "
          f"{tess_m['precision']:>6.1%} {tess_m['recall']:>7.1%} "
          f"{tess_m['f1']:>7.3f} {'0 (zero)':>9} {tess_m['n']:>7}")

    # Save
    Path("src/models").mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "lambda_max": LAMBDA,
        "pos_weight": POS_WEIGHT,
        "metrics_kepler_1114_test": {k: v for k, v in kepler_m.items()
                                     if k not in ("probs", "preds")},
        "metrics_tess_zero_shot": {k: v for k, v in tess_m.items()
                                   if k not in ("probs", "preds", "names")},
        "amplitudes": amps,
    }, OUT_MODEL)
    torch.save({
        "kepler_probs": kepler_m["probs"],
        "kepler_preds": kepler_m["preds"],
        "kepler_labels": tpl.cpu(),
        "kepler_test_idx": test_idx_list,
        "tess_probs": tess_m["probs"],
        "tess_preds": tess_m["preds"],
        "tess_labels": torch.load(TESS_DATA_PATH, weights_only=False)["labels"],
        "tess_names": tess_m["names"],
        "amp_traces": amp_traces,
        "best_val_loss": best_val,
        "lambda_max": LAMBDA,
    }, RESULTS_PATH)
    print(f"\nSaved winner model -> {OUT_MODEL}")
    print(f"Saved results     -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
