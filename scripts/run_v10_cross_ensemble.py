"""Exp 4c — Cross-mission AND ensemble of Kepler-native and TESS-native V10.

Runs two AND ensembles on two held-out test sets and reports whether
the combined probability both-above-0.5 decision improves precision
over each constituent model without sacrificing too much recall.

Test sets:
  * Kepler 76-TCE test split (seed=42 on kepler_tce_v6.pt) — the same
    test set used throughout the V10 paper.
  * TESS native test split (seed=42 on tess_tce_400.pt, ~54 TCEs).

Model pairings:
  * Kepler 76 test set:   V10 500-TCE native (taylor_cnn_v10.pt)  AND  TESS-native V10
  * TESS test split:      V10 1114-TCE (taylor_cnn_v10_1114.pt)   AND  TESS-native V10

The first pair is the honest "Kepler-native + TESS-zero-shot" ensemble
(neither model was trained on the 76-TCE holdout). The second is the
"Kepler-zero-shot + TESS-native" ensemble (V10_1114 was trained on a
superset of the 500-TCE dataset but NOT on any TESS sample).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.models.taylor_cnn_v10 import TaylorCNNv10


KEPLER_V6_PATH = "data/kepler_tce_v6.pt"
TESS_PATH = "data/tess_tce_400.pt"

MODEL_V10_500 = "src/models/taylor_cnn_v10.pt"          # native Kepler 500
MODEL_V10_1114 = "src/models/taylor_cnn_v10_1114.pt"    # native Kepler 1114
MODEL_V10_TESS = "src/models/taylor_cnn_v10_tess.pt"    # native TESS

RESULTS_PATH = "data/v10_cross_ensemble_results.pt"
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
    return torch.cat([ct, ft]), torch.cat([cv, fv]), torch.cat([cte, fte])


def load_model(path: str) -> TaylorCNNv10:
    blob = torch.load(path, weights_only=False, map_location=DEVICE)
    state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    model = TaylorCNNv10(init_amplitude=0.01).to(DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def forward(model, phase, primary, secondary, oe):
    with torch.no_grad():
        return model(phase, primary, secondary, oe).squeeze(1).cpu()


def metrics(probs, labels, threshold=0.5):
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
            "TP": TP, "TN": TN, "FP": FP, "FN": FN, "n": n}


def fmt(m: dict) -> str:
    return (f"acc={m['accuracy']:.1%}  prec={m['precision']:.1%}  "
            f"rec={m['recall']:.1%}  F1={m['f1']:.3f}  "
            f"TP={m['TP']} FP={m['FP']} TN={m['TN']} FN={m['FN']}")


def load_kepler_76():
    d = torch.load(KEPLER_V6_PATH, weights_only=False)
    _, _, test_idx = stratified_split(d["labels"], SEED_SPLIT)
    return (
        d["phases"][test_idx].to(DEVICE),
        d["fluxes"][test_idx].to(DEVICE),
        d["fluxes_secondary"][test_idx].to(DEVICE),
        d["fluxes_odd_even"][test_idx].to(DEVICE),
        d["labels"][test_idx],
        [d["names"][i] for i in test_idx.tolist()],
    )


def load_tess_test():
    d = torch.load(TESS_PATH, weights_only=False)
    _, _, test_idx = stratified_split(d["labels"], SEED_SPLIT)
    return (
        d["phases"][test_idx].to(DEVICE),
        d["fluxes"][test_idx].to(DEVICE),
        d["fluxes_secondary"][test_idx].to(DEVICE),
        d["fluxes_odd_even"][test_idx].to(DEVICE),
        d["labels"][test_idx],
        [d["names"][i] for i in test_idx.tolist()],
    )


def and_ensemble(probs_a, probs_b, labels, th_a=0.5, th_b=0.5):
    """Both models must agree on planet for the ensemble to predict planet."""
    preds_a = (probs_a > th_a)
    preds_b = (probs_b > th_b)
    combined = (preds_a & preds_b).float()
    # metrics with the combined preds as 'probs' above 0.5 convention
    m = metrics(combined, labels, threshold=0.5)
    return m, combined


def evaluate_pair(name_left: str, model_left: TaylorCNNv10,
                  name_right: str, model_right: TaylorCNNv10,
                  test_set, labels) -> dict:
    phase, primary, secondary, oe = test_set[:4]
    p_left = forward(model_left, phase, primary, secondary, oe)
    p_right = forward(model_right, phase, primary, secondary, oe)

    m_left = metrics(p_left, labels)
    m_right = metrics(p_right, labels)
    m_ens, _ = and_ensemble(p_left, p_right, labels)

    print(f"  {name_left:<28}: {fmt(m_left)}")
    print(f"  {name_right:<28}: {fmt(m_right)}")
    print(f"  {'AND ensemble':<28}: {fmt(m_ens)}")
    return {
        "left_name": name_left, "right_name": name_right,
        "left_probs": p_left, "right_probs": p_right,
        "left_metrics": m_left, "right_metrics": m_right,
        "ensemble_metrics": m_ens,
    }


def main():
    print(f"Device: {DEVICE}")
    print("Loading models...")
    m500 = load_model(MODEL_V10_500)
    m1114 = load_model(MODEL_V10_1114)
    mtess = load_model(MODEL_V10_TESS)

    print("\n" + "=" * 76)
    print("Kepler 76-TCE test set (V10 paper baseline split)")
    print("=" * 76)
    kep_test = load_kepler_76()
    kep_labels = kep_test[4]
    kep_result = evaluate_pair(
        "V10 500 (Kepler native)", m500,
        "V10 TESS-native (zero-shot)", mtess,
        kep_test, kep_labels,
    )
    kep_result["test_set"] = "kepler_76"
    kep_result["names"] = kep_test[5]
    kep_result["labels"] = kep_labels.cpu()

    print("\n" + "=" * 76)
    print("TESS test split (seed=42 on tess_tce_400.pt)")
    print("=" * 76)
    tess_test = load_tess_test()
    tess_labels = tess_test[4]
    tess_result = evaluate_pair(
        "V10 1114 (Kepler, zero-shot)", m1114,
        "V10 TESS-native", mtess,
        tess_test, tess_labels,
    )
    tess_result["test_set"] = "tess_split"
    tess_result["names"] = tess_test[5]
    tess_result["labels"] = tess_labels.cpu()

    print("\n" + "=" * 76)
    print("Summary: does cross-mission AND beat the best constituent?")
    print("=" * 76)
    for label, r in [("Kepler-76", kep_result), ("TESS-split", tess_result)]:
        best_single = max(r["left_metrics"]["f1"], r["right_metrics"]["f1"])
        delta = r["ensemble_metrics"]["f1"] - best_single
        prec_left = r["left_metrics"]["precision"]
        prec_right = r["right_metrics"]["precision"]
        prec_ens = r["ensemble_metrics"]["precision"]
        print(f"  {label:<12}  best-single F1={best_single:.3f}  "
              f"ens F1={r['ensemble_metrics']['f1']:.3f}  "
              f"delta={delta:+.3f}  |  "
              f"max-single prec={max(prec_left, prec_right):.1%}  ens prec={prec_ens:.1%}")

    torch.save({"kepler_76": kep_result, "tess_split": tess_result}, RESULTS_PATH)
    print(f"\nSaved cross-ensemble results -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
