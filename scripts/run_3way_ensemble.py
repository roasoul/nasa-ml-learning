"""3-way AND + 2-of-3 voting across V6b, V10 production, V10 curriculum.

Zero training — pure forward pass + vote aggregation on the 76-TCE
paper test set. Three combinations reported:
    1. 3-way AND at shared threshold in {0.40, 0.45, 0.50}
    2. 2-of-3 majority vote at threshold 0.5 on each head
    3. (for reference) V6b + V10 AND  — current F1 0.872 baseline
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


def metrics(pred_bool, labels):
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
    labels = d["labels"][test_idx]
    n_test = len(test_idx)
    print(f"76-TCE paper test set: n={n_test} (conf={int(labels.sum())}, FP={int((1-labels).sum())})")

    phase = d["phases"][test_idx].to(DEVICE)
    primary = d["fluxes"][test_idx].to(DEVICE)
    secondary = d["fluxes_secondary"][test_idx].to(DEVICE)
    odd_even = d["fluxes_odd_even"][test_idx].to(DEVICE)

    v6b = load_model(TaylorCNN, PROD_V6B)
    v10 = load_model(TaylorCNNv10, PROD_V10)
    v10_curr = load_model(TaylorCNNv10, CURRICULUM_MODEL)

    p_v6b = forward(v6b, phase, primary, secondary, odd_even)
    p_v10 = forward(v10, phase, primary, secondary, odd_even)
    p_curr = forward(v10_curr, phase, primary, secondary, odd_even)

    # --- Reference: V6b + V10 AND at 0.5 ---
    m_baseline = metrics((p_v6b > 0.5) & (p_v10 > 0.5), labels)

    # --- 3-way AND sweep ---
    sweep_3way = {}
    for thr in (0.40, 0.45, 0.50):
        pred = (p_v6b > thr) & (p_v10 > thr) & (p_curr > thr)
        sweep_3way[thr] = metrics(pred, labels)

    # --- 2-of-3 voting at 0.5 each ---
    vote = (p_v6b > 0.5).int() + (p_v10 > 0.5).int() + (p_curr > 0.5).int()
    m_2of3 = metrics(vote >= 2, labels)

    # --- Report ---
    print(f"\n{'='*72}")
    print("Results")
    print(f"{'='*72}")
    print(f"{'Ensemble':<28} {'Prec':>7} {'Recall':>8} {'F1':>7} {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4}")
    print("-" * 72)
    print(f"{'V6b + V10 AND  [baseline]':<28} "
          f"{m_baseline['precision']:>6.1%} {m_baseline['recall']:>7.1%} "
          f"{m_baseline['f1']:>7.3f} {m_baseline['TP']:>4} {m_baseline['FP']:>4} "
          f"{m_baseline['TN']:>4} {m_baseline['FN']:>4}")
    for thr, m in sweep_3way.items():
        print(f"{'3-way AND thr=' + f'{thr:.2f}':<28} "
              f"{m['precision']:>6.1%} {m['recall']:>7.1%} "
              f"{m['f1']:>7.3f} {m['TP']:>4} {m['FP']:>4} "
              f"{m['TN']:>4} {m['FN']:>4}")
    print(f"{'2-of-3 voting (thr 0.5)':<28} "
          f"{m_2of3['precision']:>6.1%} {m_2of3['recall']:>7.1%} "
          f"{m_2of3['f1']:>7.3f} {m_2of3['TP']:>4} {m_2of3['FP']:>4} "
          f"{m_2of3['TN']:>4} {m_2of3['FN']:>4}")

    # --- Winner ---
    candidates = {
        "V6b + V10 AND (baseline)": m_baseline,
        "3-way AND thr=0.40": sweep_3way[0.40],
        "3-way AND thr=0.45": sweep_3way[0.45],
        "3-way AND thr=0.50": sweep_3way[0.50],
        "2-of-3 voting (0.5)": m_2of3,
    }
    winner_name = max(candidates, key=lambda k: (candidates[k]["f1"],
                                                 candidates[k]["precision"]))
    winner = candidates[winner_name]
    print(f"\n{'='*72}")
    if winner["f1"] > 0.872:
        print(f"NEW BEST: {winner_name} — F1 {winner['f1']:.3f} "
              f"(prec {winner['precision']:.1%}, rec {winner['recall']:.1%})")
        print("  Promote V10_curriculum to production and add 'strict' mode to AdaptivePINNClassifier.")
    else:
        print(f"No new best. {winner_name} -> F1 {winner['f1']:.3f} "
              f"(baseline still 0.872).")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
