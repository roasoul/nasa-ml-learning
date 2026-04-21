"""V8-style hard Kepler gate applied at inference time.

Approach:
    1. Retrain V6 Config C (BCE-only, from-scratch) — same seed as
       scripts/threshold_sweep_v6.py so results are reproducible.
    2. Compute each test TCE's Kepler violation (symmetric: |log(obs/pred)|).
    3. For each violation threshold in a sweep, apply the hard gate:
           IF violation > threshold → force prediction to "not planet"
           ELSE keep the CNN's prediction.
    4. Report precision, recall, F1, and which KOIs the gate filtered.
    5. Safety check: flag any confirmed planet above the threshold
       (would be an unjust rejection).

Note: saved src/models/taylor_cnn_v6.pt contains Config B (depth-matched
pretrain, F1 winner) rather than Config C. We retrain Config C inline
because the user's question is specifically about its baseline.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn

from src.models.taylor_cnn import TaylorCNN
from src.models.kepler_loss import calculate_kepler_violation


DATA_PATH = "data/kepler_tce_v6.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED_SPLIT = 42
SEED_INIT = 7
PROB_THRESHOLD = 0.50


def load_all():
    d = torch.load(DATA_PATH, weights_only=False)
    phases = d["phases"]; p = d["fluxes"]; s = d["fluxes_secondary"]
    oe = d["fluxes_odd_even"]; y = d["labels"]
    violations = calculate_kepler_violation(
        d["period_days"], d["duration_hours"] / 24.0,
        d["stellar_mass"], d["stellar_radius"],
    )

    torch.manual_seed(SEED_SPLIT)
    conf = (y == 1).nonzero(as_tuple=True)[0]
    fp = (y == 0).nonzero(as_tuple=True)[0]
    conf = conf[torch.randperm(len(conf))]
    fp = fp[torch.randperm(len(fp))]
    def split(idx, tf=0.7, vf=0.15):
        n = len(idx); nt = int(n * tf); nv = int(n * vf)
        return idx[:nt], idx[nt:nt + nv], idx[nt + nv:]
    ct, cv, cte = split(conf); ft, fv, fte = split(fp)
    return {
        "phases": phases, "fluxes_p": p, "fluxes_s": s, "fluxes_oe": oe,
        "labels": y, "violations": violations,
        "names": d["names"], "periods": d["period_days"],
        "durations_h": d["duration_hours"], "smass": d["stellar_mass"],
        "srad": d["stellar_radius"],
        "train_idx": torch.cat([ct, ft]),
        "val_idx":   torch.cat([cv, fv]),
        "test_idx":  torch.cat([cte, fte]),
    }


def train_config_c(D, n_epochs=200, patience=25, batch_size=16):
    def to_dev(idx):
        return (D["phases"][idx].to(DEVICE), D["fluxes_p"][idx].to(DEVICE),
                D["fluxes_s"][idx].to(DEVICE), D["fluxes_oe"][idx].to(DEVICE),
                D["labels"][idx].to(DEVICE))
    train = to_dev(D["train_idx"]); val = to_dev(D["val_idx"])

    torch.manual_seed(SEED_INIT)
    model = TaylorCNN(init_amplitude=0.01).to(DEVICE)
    bce = nn.BCELoss()
    opt = torch.optim.Adam([
        {"params": model.taylor_gate.parameters(), "lr": 1e-4},
        {"params": model.cnn.parameters(), "lr": 1e-3},
        {"params": model.classifier.parameters(), "lr": 1e-3},
    ])
    tp, tpp, tps, tpoe, tpl = train
    vp, vpp, vps, vpoe, vpl = val
    n_tr = len(tpl)
    best_val = float("inf"); best_state = None; wait = 0
    for _ in range(n_epochs):
        model.train()
        perm = torch.randperm(n_tr, device=DEVICE)
        for start in range(0, n_tr, batch_size):
            idx = perm[start:start + batch_size]
            probs = model(tp[idx], tpp[idx], tps[idx], tpoe[idx]).squeeze(1)
            opt.zero_grad(); bce(probs, tpl[idx]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = bce(model(vp, vpp, vps, vpoe).squeeze(1), vpl).item()
        if vl < best_val:
            best_val = vl
            best_state = {k: vv.clone() for k, vv in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience: break
    model.load_state_dict(best_state)
    return model, best_val


def metrics_from(preds, labels):
    TP = int(((preds == 1) & (labels == 1)).sum())
    TN = int(((preds == 0) & (labels == 0)).sum())
    FP = int(((preds == 1) & (labels == 0)).sum())
    FN = int(((preds == 0) & (labels == 1)).sum())
    acc = (TP + TN) / len(labels) if len(labels) else 0
    prec = TP / (TP + FP) if TP + FP else 0.0
    rec = TP / (TP + FN) if TP + FN else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1,
            "TP": TP, "FP": FP, "TN": TN, "FN": FN}


def main():
    print(f"Device: {DEVICE}")
    D = load_all()
    test_idx = D["test_idx"].tolist()
    print(f"Train: {len(D['train_idx'])}  Val: {len(D['val_idx'])}  Test: {len(test_idx)}")

    print("\nRetraining V6 Config C (from-scratch, BCE only)...")
    model, best_val = train_config_c(D)
    print(f"  Done. Best val loss = {best_val:.4f}")

    # Test-time predictions from the CNN
    def stack_test():
        return (
            D["phases"][D["test_idx"]].to(DEVICE),
            D["fluxes_p"][D["test_idx"]].to(DEVICE),
            D["fluxes_s"][D["test_idx"]].to(DEVICE),
            D["fluxes_oe"][D["test_idx"]].to(DEVICE),
        )
    tp, tpp, tps, tpoe = stack_test()
    model.eval()
    with torch.no_grad():
        cnn_probs = model(tp, tpp, tps, tpoe).squeeze(1).cpu()
    cnn_preds = (cnn_probs > PROB_THRESHOLD).float()
    test_labels = D["labels"][D["test_idx"]]
    test_viol = D["violations"][D["test_idx"]]
    test_names = [D["names"][i] for i in test_idx]

    # Baseline (no hard gate)
    base = metrics_from(cnn_preds, test_labels)
    print(f"\nBaseline (no gate, thr=0.50):  "
          f"acc={base['acc']:.1%}  prec={base['prec']:.1%}  "
          f"rec={base['rec']:.1%}  F1={base['f1']:.3f}  "
          f"TP={base['TP']} FP={base['FP']} TN={base['TN']} FN={base['FN']}")

    # Safety check: any confirmed planets with violation > threshold?
    planet_viol = test_viol[test_labels == 1]
    print(f"\nConfirmed-planet violation distribution (test set):")
    print(f"  min={planet_viol.min():.3f}  p50={planet_viol.median():.3f}  "
          f"p95={np.percentile(planet_viol.numpy(), 95):.3f}  "
          f"max={planet_viol.max():.3f}")

    # Sweep
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    print(f"\n{'='*72}")
    print(f"Hard Kepler gate sweep — filter if violation > threshold")
    print(f"{'='*72}")
    print(f"{'thr':>5} {'acc':>7} {'prec':>7} {'rec':>7} {'F1':>7} "
          f"{'TP':>3} {'FP':>3} {'TN':>3} {'FN':>3} {'filtered':>9} "
          f"{'true_pl_blocked':>16}")
    print("-" * 88)

    sweep_results = []
    for thr in thresholds:
        # Apply gate: if violation > thr, force not-planet
        gate_mask = test_viol > thr  # True = filtered
        gated_preds = cnn_preds.clone()
        gated_preds[gate_mask] = 0.0

        m = metrics_from(gated_preds, test_labels)
        n_filtered = int(gate_mask.sum())

        # Count planets wrongly blocked by the gate (they would have been
        # classified correctly by the CNN but violated the hard gate)
        planets_blocked = int(
            ((test_labels == 1) & gate_mask & (cnn_preds == 1)).sum()
        )

        print(
            f"{thr:>5.1f}  {m['acc']:>6.1%} {m['prec']:>6.1%} "
            f"{m['rec']:>6.1%} {m['f1']:>6.3f}  "
            f"{m['TP']:>3} {m['FP']:>3} {m['TN']:>3} {m['FN']:>3} "
            f"{n_filtered:>9} {planets_blocked:>16}"
        )
        sweep_results.append((thr, gate_mask.clone()))

    # Per-threshold filtered KOI list
    print(f"\nKOIs filtered at each threshold (cumulative from strictest):")
    for thr, mask in sweep_results:
        filtered_idx = mask.nonzero(as_tuple=True)[0].tolist()
        if not filtered_idx:
            print(f"\n  thr={thr:.1f}: nothing filtered")
            continue
        print(f"\n  thr={thr:.1f}  (filtered {len(filtered_idx)}):")
        print(f"    {'KOI':<12} {'true':<8} {'viol':<7} {'cnn_prob':<9} {'gate_effect':<15}")
        for pos in filtered_idx:
            truth = "PLANET" if test_labels[pos] == 1 else "FP"
            v = test_viol[pos].item()
            p = cnn_probs[pos].item()
            if test_labels[pos] == 1 and cnn_preds[pos] == 1:
                effect = "WRONG (blocks planet)"
            elif test_labels[pos] == 0 and cnn_preds[pos] == 1:
                effect = "correct (rejects FP)"
            elif test_labels[pos] == 0 and cnn_preds[pos] == 0:
                effect = "no-op (already not planet)"
            else:  # true planet, CNN already said not planet
                effect = "no-op (CNN already missed)"
            print(
                f"    {test_names[pos]:<12} {truth:<8} "
                f"{v:<7.3f} {p:<9.2f} {effect:<15}"
            )

    # Safety assertion
    print(f"\nSafety check — any CONFIRMED planet with violation above threshold?")
    for thr in thresholds:
        above = ((test_labels == 1) & (test_viol > thr)).nonzero(as_tuple=True)[0].tolist()
        if above:
            print(f"  thr={thr:.1f}: YES — these planets would be blocked:")
            for pos in above:
                print(f"    {test_names[pos]} viol={test_viol[pos].item():.3f}")
        else:
            print(f"  thr={thr:.1f}: safe (no planet above threshold)")


if __name__ == "__main__":
    main()
