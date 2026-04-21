"""V6 three-way training comparison.

Trains three configurations on the same 500-TCE dataset and reports a
comparison table:

    Config C — from-scratch (V5's winning recipe, extended to 4 channels).
    Config A — synthetic pretrain + BN running-stats reset + finetune.
    Config B — synthetic pretrain with depth distribution matched to real
               Kepler data + finetune (no BN reset).

All three use the same test split (seed 42, stratified 70/15/15).
The best model is saved to src/models/taylor_cnn_v6.pt.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import copy

import numpy as np
import torch
import torch.nn as nn

from src.data.synthetic import make_synthetic_batch
from src.models.taylor_cnn import TaylorCNN


DATA_PATH = "data/kepler_tce_v6.pt"
OUT_PATH = "src/models/taylor_cnn_v6.pt"
SEED_SPLIT = 42
SEED_INIT = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# Data loading & split
# --------------------------------------------------------------------------

def load_and_split():
    d = torch.load(DATA_PATH, weights_only=False)
    assert "fluxes_odd_even" in d, (
        "Dataset is not V6 format. Re-run build_dataset.py."
    )
    phases = d["phases"]
    p = d["fluxes"]
    s = d["fluxes_secondary"]
    oe = d["fluxes_odd_even"]
    y = d["labels"]
    names = d["names"]

    torch.manual_seed(SEED_SPLIT)
    conf_idx = (y == 1).nonzero(as_tuple=True)[0]
    fp_idx = (y == 0).nonzero(as_tuple=True)[0]
    conf_perm = conf_idx[torch.randperm(len(conf_idx))]
    fp_perm = fp_idx[torch.randperm(len(fp_idx))]

    def split(idx, tf=0.7, vf=0.15):
        n = len(idx); nt = int(n * tf); nv = int(n * vf)
        return idx[:nt], idx[nt:nt + nv], idx[nt + nv:]

    ct, cv, cte = split(conf_perm)
    ft, fv, fte = split(fp_perm)

    def to_device(idx):
        return (
            phases[idx].to(DEVICE),
            p[idx].to(DEVICE),
            s[idx].to(DEVICE),
            oe[idx].to(DEVICE),
            y[idx].to(DEVICE),
        )

    train = to_device(torch.cat([ct, ft]))
    val = to_device(torch.cat([cv, fv]))
    test = to_device(torch.cat([cte, fte]))
    test_orig_idx = torch.cat([cte, fte]).tolist()

    print(f"Dataset: {len(y)} TCEs ({int(y.sum())} conf, {int((1-y).sum())} FP)")
    print(f"Train: {len(train[-1])}  Val: {len(val[-1])}  Test: {len(test[-1])}")
    return train, val, test, names, test_orig_idx, p, s, oe, y


# --------------------------------------------------------------------------
# Synthetic batch generators — generic + depth-matched
# --------------------------------------------------------------------------

def synthetic_generic():
    """Generator with generic synthetic depth ranges (V5-style)."""
    ph, p, s, oe, y = make_synthetic_batch(
        n_planets=300,
        n_eclipsing_binaries=100,
        n_eb_doubled=100,
        n_non_transits=100,
        depth_range=(0.005, 0.02),
        eb_secondary_range=(0.003, 0.015),
        eb_odd_even_range=(0.003, 0.015),
        noise_level=0.003,
        seed=123,
    )
    return tuple(t.to(DEVICE) for t in (ph, p, s, oe, y))


def synthetic_depth_matched(p_train, s_train, oe_train, y_train):
    """Generator with depth ranges matched to the real training data.

    Computes the 5th-95th percentile of primary dip depths observed in the
    real training data (separately for confirmed planets and FPs) and uses
    those ranges as the synthetic depth distribution. This fixes the V5
    complaint that synthetic depths (0.5-2%) were much larger than real
    Kepler (median ~0.13%), causing the Taylor gate's A parameter to
    overshoot during pretraining.
    """
    p_cpu = p_train.cpu().numpy()
    s_cpu = s_train.cpu().numpy()
    oe_cpu = oe_train.cpu().numpy()
    y_cpu = y_train.cpu().numpy()

    # Primary depth: most-negative value in bins near phase=0 (95:105)
    center = slice(95, 105)
    primary_depth = -p_cpu[:, center].min(axis=1)  # positive
    sec_depth = -s_cpu[:, center].min(axis=1)
    oe_amp = np.abs(oe_cpu[:, center]).max(axis=1)

    def rng(vals, mask):
        v = vals[mask]
        if len(v) == 0:
            return (1e-4, 1e-3)
        lo, hi = np.percentile(v, [5, 95])
        return float(max(lo, 1e-5)), float(max(hi, 2 * lo, 1e-4))

    depth_all = rng(primary_depth, np.ones_like(y_cpu, dtype=bool))
    sec_fp = rng(sec_depth, y_cpu == 0)
    oe_fp = rng(oe_amp, y_cpu == 0)

    print(
        f"  matched ranges — "
        f"primary: ({depth_all[0]:.4f}, {depth_all[1]:.4f})  "
        f"secondary: ({sec_fp[0]:.4f}, {sec_fp[1]:.4f})  "
        f"odd/even: ({oe_fp[0]:.4f}, {oe_fp[1]:.4f})"
    )

    ph, p, s, oe, y = make_synthetic_batch(
        n_planets=300,
        n_eclipsing_binaries=100,
        n_eb_doubled=100,
        n_non_transits=100,
        depth_range=depth_all,
        eb_secondary_range=sec_fp,
        eb_odd_even_range=oe_fp,
        noise_level=0.0015,
        seed=123,
    )
    return tuple(t.to(DEVICE) for t in (ph, p, s, oe, y))


# --------------------------------------------------------------------------
# Training loops
# --------------------------------------------------------------------------

def pretrain(model, syn, n_epochs=40, lr=1e-3, batch_size=32, verbose_every=10):
    ph, p, s, oe, y = syn
    n = len(y)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCELoss()
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        run_loss = 0.0
        correct = 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            pred = model(ph[idx], p[idx], s[idx], oe[idx]).squeeze(1)
            loss = crit(pred, y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            run_loss += loss.item() * len(idx)
            correct += ((pred > 0.5).float() == y[idx]).sum().item()
        if epoch == 0 or (epoch + 1) % verbose_every == 0:
            print(
                f"    pretrain ep {epoch+1:>3}  loss={run_loss/n:.4f}  "
                f"acc={correct/n:.1%}  A={model.taylor_gate.A.item():.5f}"
            )


def reset_bn_running_stats(model):
    """Reset all BatchNorm running-stats in the model to fresh state."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.reset_running_stats()


def finetune(model, train, val, n_epochs=200, patience=25, batch_size=16):
    tp, tpp, tps, tpoe, tpl = train
    vp, vpp, vps, vpoe, vpl = val
    opt = torch.optim.Adam([
        {"params": model.taylor_gate.parameters(), "lr": 1e-4},
        {"params": model.cnn.parameters(), "lr": 1e-3},
        {"params": model.classifier.parameters(), "lr": 1e-3},
    ])
    crit = nn.BCELoss()
    n_tr = len(tpl)
    best_val = float("inf"); best_state = None; wait = 0
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n_tr, device=DEVICE)
        for start in range(0, n_tr, batch_size):
            idx = perm[start:start + batch_size]
            pred = model(tp[idx], tpp[idx], tps[idx], tpoe[idx]).squeeze(1)
            loss = crit(pred, tpl[idx])
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            v = model(vp, vpp, vps, vpoe).squeeze(1)
            vl = crit(v, vpl).item()
        if vl < best_val:
            best_val = vl
            best_state = {k: vv.clone() for k, vv in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if (epoch + 1) % 25 == 0:
            print(f"    finetune ep {epoch+1:>3}  val_loss={vl:.4f}  wait={wait}")
        if wait >= patience:
            print(f"    stop at ep {epoch+1}")
            break
    model.load_state_dict(best_state)
    return best_val


def evaluate(model, test):
    tp, tpp, tps, tpoe, tpl = test
    model.eval()
    with torch.no_grad():
        probs = model(tp, tpp, tps, tpoe).squeeze(1).cpu()
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
        "probs": probs, "preds": preds,
    }


def run_config(name, train, val, test, pretrain_syn=None, bn_reset=False):
    print(f"\n{'='*60}\nConfig {name}\n{'='*60}")
    torch.manual_seed(SEED_INIT)
    model = TaylorCNN(init_amplitude=0.01).to(DEVICE)
    if pretrain_syn is not None:
        print("  Pretraining on synthetic")
        pretrain(model, pretrain_syn)
        if bn_reset:
            print("  Resetting BatchNorm running stats")
            reset_bn_running_stats(model)
    print("  Fine-tuning on real Kepler")
    best_val = finetune(model, train, val)
    metrics = evaluate(model, test)
    metrics["best_val_loss"] = best_val
    print(
        f"  Result: acc={metrics['accuracy']:.1%}  "
        f"prec={metrics['precision']:.1%}  rec={metrics['recall']:.1%}  "
        f"F1={metrics['f1']:.3f}  A={metrics['A']:.5f}"
    )
    return model, metrics


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    print(f"Device: {DEVICE}")
    train, val, test, names, test_orig_idx, p_full, s_full, oe_full, y_full = load_and_split()

    # Build both synthetic sets (generic + depth-matched).
    syn_generic = synthetic_generic()
    syn_matched = synthetic_depth_matched(train[1], train[2], train[3], train[-1])

    results = {}
    models = {}

    models["C"], results["C"] = run_config(
        "C — from-scratch (no pretraining)", train, val, test
    )
    models["A"], results["A"] = run_config(
        "A — synthetic pretrain + BN reset",
        train, val, test,
        pretrain_syn=syn_generic, bn_reset=True,
    )
    models["B"], results["B"] = run_config(
        "B — depth-matched synthetic pretrain (no BN reset)",
        train, val, test,
        pretrain_syn=syn_matched, bn_reset=False,
    )

    # Comparison table
    print(f"\n{'='*72}")
    print("Three-way comparison")
    print(f"{'='*72}")
    print(f"{'Config':<8} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} "
          f"{'TP':>3} {'FP':>3} {'TN':>3} {'FN':>3} {'A':>7} {'val_loss':>9}")
    print("-" * 72)
    for name, r in results.items():
        print(
            f"{name:<8} {r['accuracy']:>6.1%}  {r['precision']:>6.1%}  "
            f"{r['recall']:>6.1%}  {r['f1']:>6.3f}  "
            f"{r['TP']:>3} {r['FP']:>3} {r['TN']:>3} {r['FN']:>3}  "
            f"{r['A']:>.4f}  {r['best_val_loss']:>9.4f}"
        )

    # Winner = highest F1 (breaks ties on accuracy)
    winner = max(results.keys(), key=lambda k: (results[k]["f1"], results[k]["accuracy"]))
    print(f"\nWinner: Config {winner}")

    winner_metrics = results[winner]
    meets_success = winner_metrics["precision"] > 0.80 and winner_metrics["recall"] >= 1.0
    print(
        f"Success criterion (prec > 80% AND recall = 100%): "
        f"{'PASS' if meets_success else 'FAIL'}"
    )

    # Per-TCE predictions for the winner
    print(f"\nWinner per-TCE predictions:")
    print(f"  {'KOI':<12} {'true':<8} {'prob':<6} {'pred':<8} "
          f"{'p_dip':<8} {'s_dip':<8} {'oe_amp':<7}")
    probs_win = winner_metrics["probs"]
    preds_win = winner_metrics["preds"]
    tl = test[-1].cpu()
    for pos, orig in enumerate(test_orig_idx):
        truth = "PLANET" if tl[pos] == 1 else "FP"
        pred = "PLANET" if preds_win[pos] == 1 else "NOT"
        prob = probs_win[pos].item()
        pd = p_full[orig, 95:105].min().item()
        sd = s_full[orig, 95:105].min().item()
        oe = oe_full[orig, 95:105].abs().max().item()
        mark = "" if preds_win[pos] == tl[pos] else "  <-- wrong"
        print(
            f"  {names[orig]:<12} {truth:<8} {prob:.2f}   "
            f"{pred:<8} {pd:+.4f} {sd:+.4f} {oe:.4f}{mark}"
        )

    # Save winner model
    Path("src/models").mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": models[winner].state_dict(),
        "A": winner_metrics["A"],
        "config": winner,
        "metrics": {
            "accuracy": winner_metrics["accuracy"],
            "precision": winner_metrics["precision"],
            "recall": winner_metrics["recall"],
            "f1": winner_metrics["f1"],
        },
        "all_configs": {k: {kk: vv for kk, vv in v.items()
                            if kk not in ("probs", "preds")}
                        for k, v in results.items()},
    }, OUT_PATH)
    print(f"\nSaved winner (config {winner}) to {OUT_PATH}")


if __name__ == "__main__":
    main()
