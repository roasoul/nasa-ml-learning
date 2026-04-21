"""V7 — Kepler's Third Law as a soft physics penalty.

Training loss:
    L_total = L_bce + lambda1 * L_kepler + lambda2 * L_sparsity

Architecture: identical to V6 Config C (from-scratch 4-channel CNN, 914
params). The only change is the loss function.

Compares V7 to V6 Config C on the same 76-TCE test split (seed 42).
Reports results at threshold 0.50 (standard) and 0.40 (user's candidate
recall-recovery threshold).

Saves the trained V7 model to src/models/taylor_cnn_v7.pt.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn

from src.models.taylor_cnn import TaylorCNN
from src.models.kepler_loss import (
    calculate_kepler_violation,
    kepler_loss,
    sparsity_loss,
)


DATA_PATH = "data/kepler_tce_v6.pt"
OUT_PATH = "src/models/taylor_cnn_v7.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED_SPLIT = 42
SEED_INIT = 7

LAMBDA_KEPLER = 0.1
LAMBDA_SPARSITY = 0.01
TOLERANCE = 0.2


def load_splits():
    d = torch.load(DATA_PATH, weights_only=False)
    for k in ("fluxes_odd_even", "period_days", "duration_hours", "stellar_mass"):
        assert k in d, f"V7 needs key '{k}' in dataset. Run retrofit_stellar_params.py."

    phases = d["phases"]
    p = d["fluxes"]
    s = d["fluxes_secondary"]
    oe = d["fluxes_odd_even"]
    y = d["labels"]

    # Precompute Kepler violation per sample
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

    ct, cv, cte = split(conf)
    ft, fv, fte = split(fp)

    def to_dev(idx):
        return (
            phases[idx].to(DEVICE), p[idx].to(DEVICE),
            s[idx].to(DEVICE), oe[idx].to(DEVICE),
            y[idx].to(DEVICE), violations[idx].to(DEVICE),
        )

    train = to_dev(torch.cat([ct, ft]))
    val = to_dev(torch.cat([cv, fv]))
    test = to_dev(torch.cat([cte, fte]))
    test_orig = torch.cat([cte, fte]).tolist()
    return train, val, test, d, test_orig, violations


def train_model(
    train, val,
    use_kepler: bool,
    use_sparsity: bool,
    n_epochs: int = 200,
    patience: int = 25,
    batch_size: int = 16,
):
    torch.manual_seed(SEED_INIT)
    model = TaylorCNN(init_amplitude=0.01).to(DEVICE)
    bce = nn.BCELoss()
    opt = torch.optim.Adam([
        {"params": model.taylor_gate.parameters(), "lr": 1e-4},
        {"params": model.cnn.parameters(), "lr": 1e-3},
        {"params": model.classifier.parameters(), "lr": 1e-3},
    ])
    tp, tpp, tps, tpoe, tpl, tpv = train
    vp, vpp, vps, vpoe, vpl, vpv = val
    n_tr = len(tpl)
    best_val = float("inf"); best_state = None; wait = 0

    print(f"  params: {sum(p.numel() for p in model.parameters())}")
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n_tr, device=DEVICE)
        ep_bce = ep_kep = ep_sp = 0.0
        n_batches = 0
        for start in range(0, n_tr, batch_size):
            idx = perm[start:start + batch_size]
            probs = model(tp[idx], tpp[idx], tps[idx], tpoe[idx]).squeeze(1)
            l_bce = bce(probs, tpl[idx])
            loss = l_bce
            if use_kepler:
                l_kep = kepler_loss(probs, tpv[idx], tolerance=TOLERANCE)
                loss = loss + LAMBDA_KEPLER * l_kep
                ep_kep += float(l_kep) * len(idx)
            if use_sparsity:
                l_sp = sparsity_loss(model)
                loss = loss + LAMBDA_SPARSITY * l_sp
                ep_sp += float(l_sp) * len(idx)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_bce += float(l_bce) * len(idx)
            n_batches += 1

        model.eval()
        with torch.no_grad():
            v_probs = model(vp, vpp, vps, vpoe).squeeze(1)
            vl = bce(v_probs, vpl).item()  # val criterion stays BCE
        if vl < best_val:
            best_val = vl
            best_state = {k: vv.clone() for k, vv in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if (epoch + 1) % 25 == 0:
            extras = []
            if use_kepler: extras.append(f"kep={ep_kep/n_tr:.4f}")
            if use_sparsity: extras.append(f"sp={ep_sp/n_tr:.4f}")
            extra_str = "  " + "  ".join(extras) if extras else ""
            print(
                f"    ep {epoch+1:>3}  bce={ep_bce/n_tr:.4f}  "
                f"val_bce={vl:.4f}  wait={wait}{extra_str}"
            )
        if wait >= patience:
            print(f"    stop at ep {epoch+1}")
            break

    model.load_state_dict(best_state)
    return model, best_val


def evaluate_at(model, test, threshold=0.50):
    tp, tpp, tps, tpoe, tpl, _ = test
    model.eval()
    with torch.no_grad():
        probs = model(tp, tpp, tps, tpoe).squeeze(1).cpu()
    preds = (probs > threshold).float()
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
        "threshold": threshold,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "A": model.taylor_gate.A.item(),
        "probs": probs, "preds": preds,
    }


def summarize(label, m):
    print(
        f"  {label:<40} thr={m['threshold']:.2f}  "
        f"acc={m['accuracy']:.1%}  prec={m['precision']:.1%}  "
        f"rec={m['recall']:.1%}  F1={m['f1']:.3f}  "
        f"TP={m['TP']}/FP={m['FP']}/TN={m['TN']}/FN={m['FN']}"
    )


def main():
    print(f"Device: {DEVICE}")
    train, val, test, dataset_full, test_orig, violations_full = load_splits()
    print(f"Train: {len(train[-1])}  Val: {len(val[-1])}  Test: {len(test[-1])}")

    # Violation stats on the training set — sanity check the separation
    tr_viol = train[-1].cpu()
    tr_lbl = train[-2].cpu()
    print(
        f"Train violation medians: "
        f"confirmed {tr_viol[tr_lbl==1].median():.3f}  "
        f"FP {tr_viol[tr_lbl==0].median():.3f}"
    )

    # --- V6 Config C reference (BCE only) ---
    print("\n=== V6 Config C (BCE only, no physics loss) ===")
    model_v6, _ = train_model(train, val, use_kepler=False, use_sparsity=False)

    # --- V7 (BCE + Kepler + Sparsity) ---
    print("\n=== V7 (BCE + Kepler + Sparsity) ===")
    model_v7, _ = train_model(train, val, use_kepler=True, use_sparsity=True)

    # --- Compare at thresholds 0.50 and 0.40 ---
    print(f"\n{'='*72}")
    print("Comparison at threshold 0.50 and 0.40")
    print(f"{'='*72}")
    v6_50 = evaluate_at(model_v6, test, 0.50)
    v6_40 = evaluate_at(model_v6, test, 0.40)
    v7_50 = evaluate_at(model_v7, test, 0.50)
    v7_40 = evaluate_at(model_v7, test, 0.40)
    summarize("V6 Config C, BCE only",               v6_50)
    summarize("V6 Config C, BCE only",               v6_40)
    summarize("V7 BCE + Kepler + Sparsity",          v7_50)
    summarize("V7 BCE + Kepler + Sparsity",          v7_40)

    # --- Success criterion check ---
    def success(m):
        return m["precision"] > 0.80 and m["recall"] >= 1.0

    print(f"\nSuccess criterion (prec > 80% AND recall = 100%):")
    for label, m in [
        ("V6 @ 0.50", v6_50), ("V6 @ 0.40", v6_40),
        ("V7 @ 0.50", v7_50), ("V7 @ 0.40", v7_40),
    ]:
        mark = "PASS" if success(m) else "FAIL"
        print(f"  {label:<14} {mark}")

    # --- Per-TCE breakdown for V7 @ 0.50 ---
    names = dataset_full["names"]
    tl = test[-2].cpu()
    probs = v7_50["probs"]
    preds = v7_50["preds"]

    print(f"\nV7 surviving FPs / missed planets at threshold 0.50 "
          f"(listing only mistakes):")
    print(f"  {'KOI':<12} {'true':<8} {'prob':<6} {'viol':<7} "
          f"{'period':<7} {'dur_h':<7} {'M_star':<7}")
    for pos, orig in enumerate(test_orig):
        if preds[pos] == tl[pos]:
            continue
        truth = "PLANET" if tl[pos] == 1 else "FP"
        v = violations_full[orig].item()
        P = dataset_full["period_days"][orig].item()
        D = dataset_full["duration_hours"][orig].item()
        M = dataset_full["stellar_mass"][orig].item()
        print(
            f"  {names[orig]:<12} {truth:<8} {probs[pos].item():.2f}  "
            f"{v:.3f}   {P:>6.2f} {D:>7.2f} {M:>6.2f}"
        )

    # Key case study: K00254.01 (stuck planet in V6 Config C)
    for pos, orig in enumerate(test_orig):
        if names[orig] == "K00254.01":
            print(f"\nCase study — K00254.01 (stuck planet in V6 Config C):")
            print(f"  V6 prob: {v6_50['probs'][pos].item():.3f}")
            print(f"  V7 prob: {v7_50['probs'][pos].item():.3f}  "
                  f"(Kepler violation {violations_full[orig]:.3f}, well under tolerance)")

    Path("src/models").mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model_v7.state_dict(),
        "A": model_v7.taylor_gate.A.item(),
        "lambda_kepler": LAMBDA_KEPLER,
        "lambda_sparsity": LAMBDA_SPARSITY,
        "tolerance": TOLERANCE,
        "metrics_thr_050": {k: v for k, v in v7_50.items()
                            if k not in ("probs", "preds")},
        "metrics_thr_040": {k: v for k, v in v7_40.items()
                            if k not in ("probs", "preds")},
        "v6_reference_thr_050": {k: v for k, v in v6_50.items()
                                 if k not in ("probs", "preds")},
    }, OUT_PATH)
    print(f"\nSaved V7 model to {OUT_PATH}")


if __name__ == "__main__":
    main()
