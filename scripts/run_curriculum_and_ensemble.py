"""V6b + V10 curriculum AND ensemble — can the high-precision curriculum
head lift the AND ensemble above the current F1=0.872 session best?

Scope:
    1. AND at default threshold 0.5 on both models.
    2. Sweep V10_curriculum threshold in {0.40, 0.45, 0.50, 0.55, 0.60}
       with V6b fixed at 0.5.
    3. Triple OR at threshold 0.4:
         V6b + V10_curriculum + V10 log-R*
       (Prior experiment showed V6b + V10 production + V10 log = 66.7%
       prec, 100% recall. Repeat with curriculum slotted in.)
    4. Per-TCE check on K00254.01 (M-dwarf, V10 production blind spot):
       does the stricter curriculum head move p_curr further below 0.5
       or does the recalibration accidentally help?

Outputs the table the task spec requested; promotion to production/
is left to the caller depending on whether F1 clears 0.872.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.models.taylor_cnn import TaylorCNN
from src.models.taylor_cnn_v10 import TaylorCNNv10


KEPLER_V6_PATH = "data/kepler_tce_v6.pt"
PROD_V6B = "src/models/production/v6b_recall947.pt"
PROD_V10 = "src/models/production/v10_f1861.pt"
PROD_V10_LOG = "src/models/production/v10_log_mdwarf.pt"
CURRICULUM_MODEL = "src/models/taylor_cnn_v10_curriculum.pt"

SEED_SPLIT = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    return torch.cat([cte, fte])


def load_model(cls, path):
    blob = torch.load(path, weights_only=False, map_location=DEVICE)
    state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    m = cls(init_amplitude=0.01).to(DEVICE)
    m.load_state_dict(state)
    m.eval()
    return m


def forward(m, phase, primary, secondary, oe):
    m.eval()
    with torch.no_grad():
        return m(phase, primary, secondary, oe).squeeze(1).cpu()


def metrics_from_bool(pred_bool, labels):
    preds = pred_bool.long()
    tl = labels.long().cpu()
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
            "TP": TP, "TN": TN, "FP": FP, "FN": FN, "n": n}


def fmt(m):
    return (f"acc={m['accuracy']:.1%}  prec={m['precision']:.1%}  "
            f"rec={m['recall']:.1%}  F1={m['f1']:.3f}  "
            f"TP={m['TP']} FP={m['FP']} TN={m['TN']} FN={m['FN']}")


def main():
    print(f"Device: {DEVICE}")
    d = torch.load(KEPLER_V6_PATH, weights_only=False)
    test_idx = stratified_split(d["labels"], SEED_SPLIT)
    n_test = len(test_idx)
    labels = d["labels"][test_idx]
    print(f"76-TCE paper test set: n={n_test} (conf={int(labels.sum())}, FP={int((1-labels).sum())})")

    phase = d["phases"][test_idx].to(DEVICE)
    primary = d["fluxes"][test_idx].to(DEVICE)
    secondary = d["fluxes_secondary"][test_idx].to(DEVICE)
    odd_even = d["fluxes_odd_even"][test_idx].to(DEVICE)

    # Log-R* scaled inputs for the log model
    r = d["stellar_radius"][test_idx].clamp(min=0.01).unsqueeze(1)
    scale = torch.log1p(r).to(DEVICE)
    primary_log = primary * scale
    secondary_log = secondary * scale
    odd_even_log = odd_even * scale

    # Load all four models
    v6b = load_model(TaylorCNN, PROD_V6B)
    v10 = load_model(TaylorCNNv10, PROD_V10)
    v10_log = load_model(TaylorCNNv10, PROD_V10_LOG)
    v10_curr = load_model(TaylorCNNv10, CURRICULUM_MODEL)

    # Forward passes
    p_v6b = forward(v6b, phase, primary, secondary, odd_even)
    p_v10 = forward(v10, phase, primary, secondary, odd_even)
    p_log = forward(v10_log, phase, primary_log, secondary_log, odd_even_log)
    p_curr = forward(v10_curr, phase, primary, secondary, odd_even)

    # --- Default AND ensembles ---
    print(f"\n{'='*72}")
    print("Default AND ensembles (threshold 0.5 on both models)")
    print(f"{'='*72}")
    m_v6b_v10 = metrics_from_bool((p_v6b > 0.5) & (p_v10 > 0.5), labels)
    m_v6b_curr = metrics_from_bool((p_v6b > 0.5) & (p_curr > 0.5), labels)

    print(f"  V6b + V10 production AND   : {fmt(m_v6b_v10)}")
    print(f"  V6b + V10 curriculum AND   : {fmt(m_v6b_curr)}")

    # --- Threshold sweep on V10_curriculum ---
    print(f"\n{'='*72}")
    print("V6b (thr 0.5) + V10 curriculum (swept threshold) AND")
    print(f"{'='*72}")
    print(f"{'curr_thr':>9} {'Prec':>7} {'Recall':>8} {'F1':>7} "
          f"{'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4}")
    sweep = {}
    for thr in (0.40, 0.45, 0.50, 0.55, 0.60):
        m = metrics_from_bool((p_v6b > 0.5) & (p_curr > thr), labels)
        sweep[thr] = m
        print(f"{thr:>9.2f} {m['precision']:>6.1%} {m['recall']:>7.1%} "
              f"{m['f1']:>7.3f} {m['TP']:>4} {m['FP']:>4} {m['TN']:>4} {m['FN']:>4}")

    best_thr = max(sweep, key=lambda t: (sweep[t]["f1"], sweep[t]["precision"]))
    best = sweep[best_thr]
    print(f"\n  Best sweep point: curr_thr={best_thr:.2f}  -> F1 {best['f1']:.3f}")

    # --- Task-spec comparison table ---
    print(f"\n{'='*72}")
    print("Task-spec comparison")
    print(f"{'='*72}")
    print(f"{'Ensemble':<32} {'Prec':>7} {'Recall':>8} {'F1':>7}")
    print("-" * 64)
    print(f"{'V6b + V10 AND (current best)':<32} "
          f"{'85.0%':>7} {'89.5%':>8} {'0.872':>7}")
    print(f"{'V6b + V10_curr AND (thr 0.5)':<32} "
          f"{m_v6b_curr['precision']:>6.1%} {m_v6b_curr['recall']:>7.1%} "
          f"{m_v6b_curr['f1']:>7.3f}")
    print(f"{'V6b + V10_curr AND (best thr)':<32} "
          f"{best['precision']:>6.1%} {best['recall']:>7.1%} "
          f"{best['f1']:>7.3f}")

    # --- Triple OR at 0.4 ---
    print(f"\n{'='*72}")
    print("Triple OR @ threshold 0.4")
    print(f"{'='*72}")
    m_trip_orig = metrics_from_bool((p_v6b > 0.4) | (p_v10 > 0.4) | (p_log > 0.4), labels)
    m_trip_curr = metrics_from_bool((p_v6b > 0.4) | (p_curr > 0.4) | (p_log > 0.4), labels)
    print(f"  V6b + V10 production + V10 log  : {fmt(m_trip_orig)}")
    print(f"  V6b + V10 curriculum + V10 log  : {fmt(m_trip_curr)}")

    # --- K00254.01 showcase ---
    print(f"\n{'='*72}")
    print("Per-TCE: K00254.01 (M-dwarf, V10 production blind spot)")
    print(f"{'='*72}")
    i = d["names"].index("K00254.01")
    ph_i = d["phases"][i:i+1].to(DEVICE)
    pr_i = d["fluxes"][i:i+1].to(DEVICE)
    se_i = d["fluxes_secondary"][i:i+1].to(DEVICE)
    oe_i = d["fluxes_odd_even"][i:i+1].to(DEVICE)
    r_i = float(d["stellar_radius"][i])
    s_i = float(torch.log1p(torch.tensor(max(r_i, 0.01))))
    pr_log_i = pr_i * s_i
    se_log_i = se_i * s_i
    oe_log_i = oe_i * s_i

    with torch.no_grad():
        p6 = v6b(ph_i, pr_i, se_i, oe_i).item()
        p10 = v10(ph_i, pr_i, se_i, oe_i).item()
        plog = v10_log(ph_i, pr_log_i, se_log_i, oe_log_i).item()
        pc = v10_curr(ph_i, pr_i, se_i, oe_i).item()
    print(f"  true label       : PLANET")
    print(f"  stellar_radius   : {r_i:.3f} R_sun")
    print(f"  p_v6b            : {p6:.3f}")
    print(f"  p_v10 production : {p10:.3f}")
    print(f"  p_v10 curriculum : {pc:.3f}   (stricter head — does it help or hurt?)")
    print(f"  p_v10 log-R*     : {plog:.3f}")

    # --- Verdict ---
    new_best = m_v6b_curr["f1"] > 0.872 or best["f1"] > 0.872
    print(f"\n{'='*72}")
    if new_best:
        print(f"NEW BEST! V6b + V10_curriculum AND F1 = {max(m_v6b_curr['f1'], best['f1']):.3f}")
        print("  Promote to production: see task step 4 (manual promotion + README update).")
    else:
        print(f"No new best. V6b + V10 production AND remains F1 0.872.")
        print(f"  Curriculum AND (thr 0.5): F1 {m_v6b_curr['f1']:.3f}")
        print(f"  Curriculum AND (best sweep thr {best_thr:.2f}): F1 {best['f1']:.3f}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
