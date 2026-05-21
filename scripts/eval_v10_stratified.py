"""Evaluate V10 (original) vs V10.2 (stratified) across all test distributions.

Sets evaluated:
  A) Curated 76-TCE test  (kepler_tce_v6.pt, seed=42 held-out)
  B) DR25 Batch 1         (kepler_dr25_1k.pt)
  C) DR25 Batch 2         (kepler_dr25_2k.pt)
  D) DR25 Batch 3         (kepler_dr25_3k.pt)

For each set: V10 original, V10 stratified, AND ensemble (V6b AND V10_stratified),
weighted ensemble (0.7 * p_v10_strat + 0.3 * p_v6b, thr=0.40).
V6b scores loaded from existing CSV files.
"""

import sys, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pandas as pd

from src.models.taylor_cnn_v10 import TaylorCNNv10
from src.models.taylor_cnn import TaylorCNN

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SNR_PIVOT  = 100.0
SS_CACHE   = "data/ss_flag_cache.csv"

# ── load SNR ─────────────────────────────────────────────────────────────────
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


# ── model loading ─────────────────────────────────────────────────────────────
def load_v10(path):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    m  = TaylorCNNv10(init_amplitude=0.01).to(DEVICE)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    m.load_state_dict(sd)
    m.eval()
    return m


def load_v6b():
    ck = torch.load("src/models/production/v6b_recall947.pt",
                    map_location=DEVICE, weights_only=False)
    m  = TaylorCNN().to(DEVICE)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    m.load_state_dict(sd)
    m.eval()
    return m


# ── inference ────────────────────────────────────────────────────────────────
@torch.no_grad()
def score_v10(model, d, idx):
    phase = d["phases"][idx].to(DEVICE)
    flux  = d["fluxes"][idx].to(DEVICE)
    sec   = d["fluxes_secondary"][idx].to(DEVICE)
    oe    = d["fluxes_odd_even"][idx].to(DEVICE)
    return model(phase, flux, sec, oe).squeeze(1).cpu()


@torch.no_grad()
def score_v6b(model, d, idx):
    phase = d["phases"][idx].to(DEVICE)
    flux  = d["fluxes"][idx].to(DEVICE)
    sec   = d["fluxes_secondary"][idx].to(DEVICE)
    oe    = d["fluxes_odd_even"][idx].to(DEVICE)
    return model(phase, flux, sec, oe).squeeze(1).cpu()


# ── metrics ──────────────────────────────────────────────────────────────────
def metrics(probs, labels, thr=0.5):
    preds = (probs > thr).float()
    tl    = labels.float()
    TP = int(((preds == 1) & (tl == 1)).sum())
    TN = int(((preds == 0) & (tl == 0)).sum())
    FP = int(((preds == 1) & (tl == 0)).sum())
    FN = int(((preds == 0) & (tl == 1)).sum())
    prec = TP / (TP + FP) if TP + FP else 0.0
    rec  = TP / (TP + FN) if TP + FN else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc  = (TP + TN) / len(tl)
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1,
            "TP": TP, "TN": TN, "FP": FP, "FN": FN}


def row(name, m):
    return (f"  {name:<30} acc={m['acc']:.1%}  prec={m['prec']:.1%}  "
            f"rec={m['rec']:.1%}  F1={m['f1']:.3f}  "
            f"TP={m['TP']} FP={m['FP']} TN={m['TN']} FN={m['FN']}")


# ── curated 76-TCE test (replicate seed=42 split on v6 dataset) ──────────────
def eval_curated(v10_orig, v10_strat, v6b):
    print("=" * 65)
    print("A) Curated 76-TCE held-out test (kepler_tce_v6.pt, seed=42)")
    print("=" * 65)
    d = torch.load("data/kepler_tce_v6.pt", weights_only=False)
    y = d["labels"]
    torch.manual_seed(42)
    conf = (y == 1).nonzero(as_tuple=True)[0]
    fp   = (y == 0).nonzero(as_tuple=True)[0]
    conf = conf[torch.randperm(len(conf))]
    fp   = fp[torch.randperm(len(fp))]
    def split(idx, tf=0.7, vf=0.15):
        n = len(idx); nt = int(n*tf); nv = int(n*vf)
        return idx[:nt], idx[nt:nt+nv], idx[nt+nv:]
    _, _, cte = split(conf)
    _, _, fte = split(fp)
    test_idx  = torch.cat([cte, fte])
    labels    = y[test_idx]

    p_v10o  = score_v10(v10_orig,  d, test_idx)
    p_v10s  = score_v10(v10_strat, d, test_idx)
    p_v6b_  = score_v6b(v6b,       d, test_idx)

    and_orig  = torch.minimum(p_v10o,  p_v6b_)   # AND ~ min(p_v10, p_v6b)
    and_strat = torch.minimum(p_v10s,  p_v6b_)
    w_strat   = 0.7 * p_v10s + 0.3 * p_v6b_

    print(row("V10 original  (thr=0.50)", metrics(p_v10o,  labels)))
    print(row("V10 stratified(thr=0.50)", metrics(p_v10s,  labels)))
    print(row("V6b           (thr=0.50)", metrics(p_v6b_,  labels)))
    print(row("AND(V6b,V10_orig)(thr=0.50)",  metrics(and_orig,  labels)))
    print(row("AND(V6b,V10_strat)(thr=0.50)", metrics(and_strat, labels)))
    print(row("Weighted(0.7s+0.3b,thr=0.40)", metrics(w_strat,   labels, thr=0.40)))
    return {"p_v10s": p_v10s, "p_v6b": p_v6b_, "labels": labels}


# ── DR25 batch evaluation ─────────────────────────────────────────────────────
def eval_batch(label, pt_path, csv_path, v10_orig, v10_strat, v6b):
    print()
    print("=" * 65)
    print(f"{label}  ({pt_path})")
    print("=" * 65)

    d      = torch.load(pt_path, weights_only=False)
    labels = d["labels"]
    all_idx = torch.arange(len(labels))

    p_v10o  = score_v10(v10_orig,  d, all_idx)
    p_v10s  = score_v10(v10_strat, d, all_idx)
    p_v6b_  = score_v6b(v6b,       d, all_idx)

    # Cross-check: load stored V6b scores from CSV (sanity check)
    df_csv   = pd.read_csv(csv_path)
    names_pt = d["names"]
    csv_map  = dict(zip(df_csv["koi_id"].tolist(), df_csv["p_v6b"].tolist()))
    p_v6b_stored = torch.tensor(
        [csv_map.get(n, float("nan")) for n in names_pt], dtype=torch.float32)
    valid = ~p_v6b_stored.isnan()
    if valid.sum() > 0:
        corr = float(torch.corrcoef(
            torch.stack([p_v6b_[valid], p_v6b_stored[valid]]))[0, 1])
        print(f"  V6b live vs stored CSV corr: {corr:.4f}  "
              f"(n={valid.sum()}, should be ~1.0)")

    and_orig  = torch.minimum(p_v10o,  p_v6b_)
    and_strat = torch.minimum(p_v10s,  p_v6b_)
    w_strat   = 0.7 * p_v10s + 0.3 * p_v6b_

    print(row("V10 original  (thr=0.50)", metrics(p_v10o,  labels)))
    print(row("V10 stratified(thr=0.50)", metrics(p_v10s,  labels)))
    print(row("AND(V6b,V10_orig)(thr=0.50)",  metrics(and_orig,  labels)))
    print(row("AND(V6b,V10_strat)(thr=0.50)", metrics(and_strat, labels)))
    print(row("Weighted(0.7s+0.3b,thr=0.40)", metrics(w_strat,   labels, thr=0.40)))

    return {
        "p_v10o": p_v10o, "p_v10s": p_v10s,
        "p_v6b":  p_v6b_, "labels": labels,
    }


def main():
    print(f"Device: {DEVICE}")
    v10_orig  = load_v10("src/models/production/v10_f1861.pt")
    v10_strat = load_v10("src/models/taylor_cnn_v10_stratified.pt")
    v6b       = load_v6b()
    print("Models loaded.")
    print()

    eval_curated(v10_orig, v10_strat, v6b)

    eval_batch("B) DR25 Batch 1",
               "data/kepler_dr25_1k.pt",
               "data/kepler_dr25_1k_scores.csv",
               v10_orig, v10_strat, v6b)

    eval_batch("C) DR25 Batch 2",
               "data/kepler_dr25_2k.pt",
               "data/kepler_dr25_2k_scores.csv",
               v10_orig, v10_strat, v6b)

    eval_batch("D) DR25 Batch 3",
               "data/kepler_dr25_3k.pt",
               "data/batch3_candidates_verified.csv",
               v10_orig, v10_strat, v6b)

    # Summary table
    print()
    print("=" * 65)
    print("SUMMARY (F1 comparison)")
    print("=" * 65)
    print("See per-set output above for full precision/recall breakdown.")


if __name__ == "__main__":
    main()
