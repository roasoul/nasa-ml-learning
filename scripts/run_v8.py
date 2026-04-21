"""V8 + V8.5 training — learnable B plus shape-feature fusion.

Trains two configurations on the same 500-TCE dataset / split as V6:

    V8   — learnable (A, B, t0) gate, no shape-feature fusion.
    V8.5 — same gate plus 5-feature fusion (B, AUC_raw, AUC_norm,
           T12/T14, flat_bottom) concatenated into the classifier head.

Split:
    seed=42, stratified 70/15/15. Matches V6/V7.5 so the test set is
    identical and the numbers are directly comparable.

Loss:
    BCE + 0.01 · |B|.mean() sparsity penalty. Planet/EB class weighting
    is not used; 500-TCE dataset is already balanced 250/250.

Saves:
    src/models/taylor_cnn_v8.pt
    src/models/taylor_cnn_v85.pt
    data/v8_results.pt  — probs, preds, labels, shape features, names
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from src.models.taylor_cnn_v8 import TaylorCNNv8


DATA_PATH = "data/kepler_tce_v6.pt"
OUT_V8 = "src/models/taylor_cnn_v8.pt"
OUT_V85 = "src/models/taylor_cnn_v85.pt"
RESULTS_PATH = "data/v8_results.pt"
SEED_SPLIT = 42
SEED_INIT = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_and_split():
    d = torch.load(DATA_PATH, weights_only=False)
    phases = d["phases"]
    p = d["fluxes"]
    s = d["fluxes_secondary"]
    oe = d["fluxes_odd_even"]
    y = d["labels"]
    names = d["names"]

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
            phases[idx].to(DEVICE),
            p[idx].to(DEVICE),
            s[idx].to(DEVICE),
            oe[idx].to(DEVICE),
            y[idx].to(DEVICE),
        )

    print(f"Dataset: {len(y)} TCEs  ({int(y.sum())} confirmed, {int((1-y).sum())} FP)")
    print(f"Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")
    return bundle(train_idx), bundle(val_idx), bundle(test_idx), names, test_idx.tolist(), d


def finetune(model, train, val, n_epochs=200, patience=25, batch_size=16, lr_gate=1e-4, lr_main=1e-3):
    tp, tpp, tps, tpoe, tpl = train
    vp, vpp, vps, vpoe, vpl = val
    opt = torch.optim.Adam([
        {"params": model.taylor_gate.parameters(), "lr": lr_gate},
        {"params": model.cnn.parameters(), "lr": lr_main},
        {"params": model.classifier.parameters(), "lr": lr_main},
    ])
    crit = nn.BCELoss()
    n_tr = len(tpl)
    best_val = float("inf"); best_state = None; wait = 0
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n_tr, device=DEVICE)
        run_bce = run_spar = 0.0
        for start in range(0, n_tr, batch_size):
            idx = perm[start:start + batch_size]
            pred = model(tp[idx], tpp[idx], tps[idx], tpoe[idx]).squeeze(1)
            bce = crit(pred, tpl[idx])
            spar = model.sparsity_loss()
            loss = bce + spar
            opt.zero_grad(); loss.backward(); opt.step()
            run_bce += bce.item() * len(idx)
            run_spar += spar.item() * len(idx)

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
        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(
                f"    ep {epoch+1:>3}  bce={run_bce/n_tr:.4f}  "
                f"spar={run_spar/n_tr:.5f}  val={vl:.4f}  wait={wait}  "
                f"A={model.taylor_gate.A.item():.4f}  "
                f"B={model.taylor_gate.B.item():.4f}  "
                f"t0={model.taylor_gate.t0.item():+.4f}"
            )
        if wait >= patience:
            print(f"    early stop at ep {epoch+1}")
            break
    model.load_state_dict(best_state)
    return best_val


def evaluate(model, test):
    tp, tpp, tps, tpoe, tpl = test
    model.eval()
    with torch.no_grad():
        probs = model(tp, tpp, tps, tpoe).squeeze(1).cpu()
        shape_feats = (
            model.last_shape_features.cpu().clone()
            if model.last_shape_features is not None
            else None
        )
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
        "shape_features": shape_feats,
    }


def param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run(name, use_shape, train, val, test):
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    torch.manual_seed(SEED_INIT)
    model = TaylorCNNv8(
        init_amplitude=0.01, init_B=1.0, init_t0=0.0,
        use_shape_features=use_shape,
    ).to(DEVICE)
    print(f"  Trainable params: {param_count(model)}")
    best_val = finetune(model, train, val)
    metrics = evaluate(model, test)
    metrics["best_val_loss"] = best_val
    metrics["params"] = param_count(model)
    print(
        f"  {name} result:  acc={metrics['accuracy']:.1%}  "
        f"prec={metrics['precision']:.1%}  rec={metrics['recall']:.1%}  "
        f"F1={metrics['f1']:.3f}  A={metrics['A']:.4f}  B={metrics['B']:+.4f}  "
        f"t0={metrics['t0']:+.4f}"
    )
    return model, metrics


def main():
    print(f"Device: {DEVICE}")
    train, val, test, names, test_idx_list, raw = load_and_split()

    model_v8, res_v8 = run("V8  (learnable B, no shape fusion)", False, train, val, test)
    model_v85, res_v85 = run("V8.5 (B + 5 shape features)", True, train, val, test)

    print(f"\n{'='*72}\nV8 vs V8.5 comparison\n{'='*72}")
    print(f"{'Version':<8} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} "
          f"{'TP':>3} {'FP':>3} {'TN':>3} {'FN':>3} {'A':>7} {'B':>7} "
          f"{'t0':>7} {'params':>7}")
    print("-" * 100)
    for n, r in [("V8", res_v8), ("V8.5", res_v85)]:
        print(
            f"{n:<8} {r['accuracy']:>6.1%}  {r['precision']:>6.1%}  "
            f"{r['recall']:>6.1%}  {r['f1']:>6.3f}  "
            f"{r['TP']:>3} {r['FP']:>3} {r['TN']:>3} {r['FN']:>3}  "
            f"{r['A']:>.4f}  {r['B']:>+.4f}  {r['t0']:>+.4f}  {r['params']:>7}"
        )

    # Per-TCE test predictions (V8.5)
    print(f"\nV8.5 per-TCE test-set predictions:")
    print(f"  {'KOI':<12} {'true':<6} {'prob':<6} {'pred':<6} {'B_shape':<8} "
          f"{'AUC':<7} {'AUC/A':<7} {'T12/T14':<8} {'flat':<6}")
    probs = res_v85["probs"]; preds = res_v85["preds"]
    tl = test[-1].cpu(); sf = res_v85["shape_features"]
    for pos, orig in enumerate(test_idx_list):
        truth = "PLNT" if tl[pos] == 1 else "FP"
        pred = "PLNT" if preds[pos] == 1 else "NOT"
        mark = "" if preds[pos] == tl[pos] else "  <-- wrong"
        row = sf[pos].tolist()
        print(
            f"  {names[orig]:<12} {truth:<6} {probs[pos]:.2f}  {pred:<6} "
            f"{row[0]:+.3f}   {row[1]:.4f}  {row[2]:.4f}  {row[3]:.4f}    "
            f"{row[4]:.3f}{mark}"
        )

    # Save everything
    Path("src/models").mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model_v8.state_dict(),
        "metrics": {k: v for k, v in res_v8.items() if k not in ("probs", "preds", "shape_features")},
    }, OUT_V8)
    torch.save({
        "state_dict": model_v85.state_dict(),
        "metrics": {k: v for k, v in res_v85.items() if k not in ("probs", "preds", "shape_features")},
    }, OUT_V85)
    torch.save({
        "v8":   {"probs": res_v8["probs"], "preds": res_v8["preds"], "shape_features": res_v8["shape_features"]},
        "v85":  {"probs": res_v85["probs"], "preds": res_v85["preds"], "shape_features": res_v85["shape_features"]},
        "test_labels": test[-1].cpu(),
        "test_idx": test_idx_list,
        "names": names,
    }, RESULTS_PATH)
    print(f"\nSaved V8 -> {OUT_V8}")
    print(f"Saved V8.5 -> {OUT_V85}")
    print(f"Saved per-TCE results -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
