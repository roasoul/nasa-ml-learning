# ═══════════════════════════════════════════
# PRODUCTION MODEL PROTECTION
# NEVER save to src/models/production/
# Save experiments to src/models/ only
# with descriptive version suffix
# ═══════════════════════════════════════════
"""Curriculum experiment — fine-tune V10 production on the harder 1114-TCE set.

Hypothesis: Exp 4 (F1 0.692 with pos_weight) and Exp 4 no-weight (F1 0.667)
both trained from scratch on the harder distribution and each solved only
one side of the precision / recall tradeoff. If the V10-500 solution is a
genuinely good local optimum, a gentle fine-tune starting from its weights
should stay near that optimum while absorbing the extra 614 samples'
information — keeping recall from collapsing while improving precision
on the broader FP mix.

Setup:
    * Start from src/models/production/v10_f1861.pt (V10 λ=0.1, F1 0.861).
    * Fine-tune on data/kepler_tce_2000.pt (1114 TCEs, seed=42 stratified
      70/15/15 = 778/166/170 split — same split used in Exp 4).
    * lr = 1e-5 for both gate bank and CNN/classifier (10x below original).
    * No pos_weight. Plain nn.BCELoss on the sigmoid head.
    * patience = 50 (2x original) to let the fine-tune settle.
    * epochs up to 200.
    * InvertedGeometryLoss with lambda_max = 0.1 (unchanged from V10).

Output:
    src/models/taylor_cnn_v10_curriculum.pt  (NOT production)
    data/v10_curriculum_results.pt
    data/v10_curriculum_training.log

Report:
    * Metrics on the 170-TCE 1114-held-out split
    * Metrics on the 76-TCE paper test split (seed=42 kepler_tce_v6.pt)
    * Triple OR ensemble at threshold 0.4:
        original:    V6b + V10 production   + V10 log-R*
        curriculum:  V6b + V10 curriculum   + V10 log-R*
      Does 100% recall hold? Does precision improve?
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from src.models.geometry_loss_v2 import InvertedGeometryLoss
from src.models.taylor_cnn import TaylorCNN
from src.models.taylor_cnn_v10 import TaylorCNNv10


DATA_PATH = "data/kepler_tce_2000.pt"
KEPLER_V6_PATH = "data/kepler_tce_v6.pt"
SS_CACHE = "data/ss_flag_cache.csv"
PROD_V10 = "src/models/production/v10_f1861.pt"
PROD_V6B = "src/models/production/v6b_recall947.pt"
PROD_V10_LOG = "src/models/production/v10_log_mdwarf.pt"
OUT_MODEL = "src/models/taylor_cnn_v10_curriculum.pt"
RESULTS_PATH = "data/v10_curriculum_results.pt"

SEED_SPLIT = 42
LAMBDA = 0.1
SNR_PIVOT = 100.0
LR = 1e-5  # 10x below original (lr_main=1e-3, lr_gate=1e-4)
N_EPOCHS = 200
PATIENCE = 50
BATCH_SIZE = 16
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


def load_production_v10():
    """Instantiate TaylorCNNv10 and load the F1=0.861 production weights."""
    blob = torch.load(PROD_V10, weights_only=False, map_location=DEVICE)
    state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    model = TaylorCNNv10(init_amplitude=0.01).to(DEVICE)
    model.load_state_dict(state)
    return model


def load_cls_model(cls, path):
    blob = torch.load(path, weights_only=False, map_location=DEVICE)
    state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    m = cls(init_amplitude=0.01).to(DEVICE)
    m.load_state_dict(state)
    m.eval()
    return m


def finetune(model, train, val, geo_loss):
    tp, tpp, tps, tpoe, tpl, tsnr = train
    vp, vpp, vps, vpoe, vpl, _ = val
    # Single LR across all parameter groups — no group-specific ratio so the
    # fine-tune nudges everything equally from the pretrained optimum.
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    bce = nn.BCELoss()
    n_tr = len(tpl)
    best_val = float("inf"); best_state = None; wait = 0

    for epoch in range(N_EPOCHS):
        model.train()
        perm = torch.randperm(n_tr, device=DEVICE)
        run_bce = run_geo = 0.0
        for start in range(0, n_tr, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
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
        if wait >= PATIENCE:
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


def eval_on_idx(model, d, idx):
    phase = d["phases"][idx].to(DEVICE)
    primary = d["fluxes"][idx].to(DEVICE)
    secondary = d["fluxes_secondary"][idx].to(DEVICE)
    oe = d["fluxes_odd_even"][idx].to(DEVICE)
    probs = forward_probs(model, phase, primary, secondary, oe)
    return metrics_from(probs, d["labels"][idx]), probs


def eval_log_on_idx(model, d, idx):
    """For V10 log-R*: multiply all flux channels by log1p(R*/R_sun) first."""
    r = d["stellar_radius"][idx].clamp(min=0.01).unsqueeze(1)
    scale = torch.log1p(r).to(DEVICE)
    phase = d["phases"][idx].to(DEVICE)
    primary = (d["fluxes"][idx].to(DEVICE)) * scale
    secondary = (d["fluxes_secondary"][idx].to(DEVICE)) * scale
    oe = (d["fluxes_odd_even"][idx].to(DEVICE)) * scale
    probs = forward_probs(model, phase, primary, secondary, oe)
    return metrics_from(probs, d["labels"][idx]), probs


def fmt(m):
    return (f"acc={m['accuracy']:.1%}  prec={m['precision']:.1%}  "
            f"rec={m['recall']:.1%}  F1={m['f1']:.3f}  "
            f"TP={m['TP']} FP={m['FP']} TN={m['TN']} FN={m['FN']}")


def triple_or_metrics(p_v6b, p_v10_variant, p_log, labels, threshold=0.4):
    """V6b OR V10-variant OR V10-log at a shared threshold."""
    preds = ((p_v6b > threshold) | (p_v10_variant > threshold) | (p_log > threshold)).float()
    return metrics_from(preds, labels, threshold=0.5)


def main():
    print(f"Device: {DEVICE}")

    # Load 1114 data + splits
    d = torch.load(DATA_PATH, weights_only=False)
    snr = load_snr(d["names"])
    train_idx, val_idx, test_idx = stratified_split(d["labels"], SEED_SPLIT)
    print(f"Kepler 1114 — Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")
    print(f"  lr={LR:g}  patience={PATIENCE}  epochs={N_EPOCHS}  no pos_weight")

    train = bundle(d, train_idx, snr)
    val = bundle(d, val_idx, snr)

    # Curriculum fine-tune
    print(f"\n{'='*64}\nCurriculum fine-tune starting from {PROD_V10}\n{'='*64}")
    model = load_production_v10()
    amps_init = model.gate_bank.amplitudes()
    print(f"  Start amplitudes  A1={amps_init['A1']:.4f} A2={amps_init['A2']:.4f} "
          f"A3={amps_init['A3']:.4f} A4={amps_init['A4']:.4f} A5={amps_init['A5']:.4f}")
    geo = InvertedGeometryLoss(lambda_min=0.01, lambda_max=LAMBDA, snr_pivot=SNR_PIVOT).to(DEVICE)
    best_val = finetune(model, train, val, geo)
    amps_final = model.gate_bank.amplitudes()
    print(f"  Final amplitudes  A1={amps_final['A1']:.4f} A2={amps_final['A2']:.4f} "
          f"A3={amps_final['A3']:.4f} A4={amps_final['A4']:.4f} A5={amps_final['A5']:.4f}")

    # Evaluate curriculum model on 170-TCE held-out
    kep_1114_m, probs_1114 = eval_on_idx(model, d, test_idx)
    print(f"\nKepler 1114 held-out (n={kep_1114_m['n']}):")
    print(f"  {fmt(kep_1114_m)}")

    # Evaluate curriculum model on 76-TCE paper test
    d76 = torch.load(KEPLER_V6_PATH, weights_only=False)
    _, _, test_idx_76 = stratified_split(d76["labels"], SEED_SPLIT)
    kep_76_curr_m, probs_76_curr = eval_on_idx(model, d76, test_idx_76)
    print(f"\nPaper 76-TCE test (n={kep_76_curr_m['n']}):")
    print(f"  {fmt(kep_76_curr_m)}")

    # Comparison table
    print(f"\n{'='*72}")
    print(f"{'Version':<22} {'Prec':>7} {'Recall':>8} {'F1':>7} {'test_n':>7}")
    print("-" * 72)
    print(f"{'V10 production':<22} {'82.9%':>7} {'89.5%':>8} {'0.861':>7} {'76':>7}")
    print(f"{'V10 1114 noweight':<22} {'76.2%':>7} {'59.3%':>8} {'0.667':>7} {'170':>7}")
    print(f"{'V10 curriculum (170)':<22} "
          f"{kep_1114_m['precision']:>6.1%} {kep_1114_m['recall']:>7.1%} "
          f"{kep_1114_m['f1']:>7.3f} {kep_1114_m['n']:>7}")
    print(f"{'V10 curriculum (76)':<22} "
          f"{kep_76_curr_m['precision']:>6.1%} {kep_76_curr_m['recall']:>7.1%} "
          f"{kep_76_curr_m['f1']:>7.3f} {kep_76_curr_m['n']:>7}")

    # Triple OR ensembles on 76-TCE paper test
    print(f"\n{'='*72}")
    print("Triple OR ensemble @ threshold 0.4 on 76-TCE paper test")
    print(f"{'='*72}")

    v6b = load_cls_model(TaylorCNN, PROD_V6B)
    v10_prod = load_cls_model(TaylorCNNv10, PROD_V10)
    v10_log = load_cls_model(TaylorCNNv10, PROD_V10_LOG)

    _, p76_v6b = eval_on_idx(v6b, d76, test_idx_76)
    _, p76_v10 = eval_on_idx(v10_prod, d76, test_idx_76)
    _, p76_log = eval_log_on_idx(v10_log, d76, test_idx_76)

    labels76 = d76["labels"][test_idx_76]
    m_original = triple_or_metrics(p76_v6b, p76_v10, p76_log, labels76)
    m_curriculum = triple_or_metrics(p76_v6b, probs_76_curr, p76_log, labels76)

    print(f"  Original (V6b + V10 production + V10 log)")
    print(f"    {fmt(m_original)}")
    print(f"  Curriculum (V6b + V10 curriculum + V10 log)")
    print(f"    {fmt(m_curriculum)}")

    print(f"\n{'Version':<30} {'Prec':>7} {'Recall':>8} {'F1':>7}")
    print("-" * 60)
    print(f"{'Original triple OR':<30} "
          f"{m_original['precision']:>6.1%} {m_original['recall']:>7.1%} "
          f"{m_original['f1']:>7.3f}")
    print(f"{'Curriculum triple OR':<30} "
          f"{m_curriculum['precision']:>6.1%} {m_curriculum['recall']:>7.1%} "
          f"{m_curriculum['f1']:>7.3f}")

    verdict_recall = "YES — 100% recall holds" if m_curriculum["recall"] >= 0.999 \
        else f"NO — recall drops to {m_curriculum['recall']:.1%}"
    verdict_prec = "YES — precision improves" if m_curriculum["precision"] > m_original["precision"] \
        else f"NO — precision unchanged/worse ({m_curriculum['precision']:.1%} vs {m_original['precision']:.1%})"
    print(f"\nVerdicts:")
    print(f"  100% recall holds?   {verdict_recall}")
    print(f"  Precision improves?  {verdict_prec}")

    # Save
    Path("src/models").mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "lambda_max": LAMBDA,
        "pos_weight": None,
        "lr": LR,
        "start_weights": PROD_V10,
        "best_val_loss": best_val,
        "metrics_1114_test": {k: v for k, v in kep_1114_m.items()
                              if k not in ("probs", "preds")},
        "metrics_76_test": {k: v for k, v in kep_76_curr_m.items()
                            if k not in ("probs", "preds")},
        "amplitudes_final": amps_final,
    }, OUT_MODEL)
    torch.save({
        "kepler_1114": {"probs": probs_1114, "preds": kep_1114_m["preds"],
                        "labels": d["labels"][test_idx].cpu(),
                        "test_idx": test_idx.tolist()},
        "kepler_76": {"probs": probs_76_curr, "preds": kep_76_curr_m["preds"],
                      "labels": labels76.cpu(),
                      "names": [d76["names"][i] for i in test_idx_76.tolist()]},
        "triple_or_original": {k: v for k, v in m_original.items()
                               if k not in ("probs", "preds")},
        "triple_or_curriculum": {k: v for k, v in m_curriculum.items()
                                 if k not in ("probs", "preds")},
        "best_val_loss": best_val,
        "lr": LR,
    }, RESULTS_PATH)
    print(f"\nSaved model   -> {OUT_MODEL}")
    print(f"Saved results -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
