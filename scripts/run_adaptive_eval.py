"""Evaluate AdaptivePINNClassifier on the 76-TCE paper test set.

Reproduces the per-mode ablation table from the task spec and prints
`predict_with_report` output for the paper's showcase samples:
  K00254.01 — M-dwarf confirmed planet, SNR 1341.6, R* 0.568
  K01091.01 — 78-hour FP, SNR 56.0, R* 1.086
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.models.adaptive_classifier import AdaptivePINNClassifier


DATA_PATH = "data/kepler_tce_v6.pt"
SS_CACHE = "data/ss_flag_cache.csv"
SEED_SPLIT = 42


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


def load_snr(names):
    cache = {}
    with open(SS_CACHE, newline="") as f:
        for r in csv.DictReader(f):
            cache[r["kepoi_name"]] = r
    snr = torch.zeros(len(names))
    for i, n in enumerate(names):
        rec = cache.get(n)
        if rec and rec["snr"] not in ("", "None"):
            snr[i] = float(rec["snr"])
    return snr


def metrics(preds, labels):
    preds = preds.long()
    labels = labels.long()
    TP = int(((preds == 1) & (labels == 1)).sum())
    TN = int(((preds == 0) & (labels == 0)).sum())
    FP = int(((preds == 1) & (labels == 0)).sum())
    FN = int(((preds == 0) & (labels == 1)).sum())
    n = len(labels)
    acc = (TP + TN) / n
    prec = TP / (TP + FP) if TP + FP else 0.0
    rec = TP / (TP + FN) if TP + FN else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "TP": TP, "TN": TN, "FP": FP, "FN": FN, "n": n}


def main():
    d = torch.load(DATA_PATH, weights_only=False)
    test_idx = stratified_split(d["labels"], SEED_SPLIT)
    names = [d["names"][i] for i in test_idx.tolist()]
    snr_all = load_snr(d["names"])
    r_all = d["stellar_radius"]

    phase = d["phases"][test_idx]
    flux = d["fluxes"][test_idx]
    secondary = d["fluxes_secondary"][test_idx]
    odd_even = d["fluxes_odd_even"][test_idx]
    labels = d["labels"][test_idx]
    snr = snr_all[test_idx]
    r = r_all[test_idx]

    print(f"Loaded {len(test_idx)} test TCEs (conf={int(labels.sum())}, FP={int((1-labels).sum())})")
    clf = AdaptivePINNClassifier()

    print("\n" + "=" * 64)
    print("Per-mode evaluation on 76-TCE paper test set")
    print("=" * 64)
    print(f"{'Mode':<14}{'Prec':>8}{'Recall':>9}{'F1':>8}{'TP':>5}{'FP':>5}{'TN':>5}{'FN':>5}")
    print("-" * 64)

    rows = []
    for mode in ("discovery", "balanced", "lightweight", "auto"):
        out = clf.predict_batch(phase, flux, secondary, odd_even,
                                snr=snr, mode=mode, stellar_radius=r)
        m = metrics(out["preds"], labels)
        rows.append((mode, m))
        print(f"{mode:<14}{m['precision']:>7.1%} {m['recall']:>8.1%} "
              f"{m['f1']:>7.3f} {m['TP']:>4}{m['FP']:>5}{m['TN']:>5}{m['FN']:>5}")

    # Routing histogram for auto
    out_auto = clf.predict_batch(phase, flux, secondary, odd_even,
                                 snr=snr, mode="auto", stellar_radius=r)
    from collections import Counter
    ctr = Counter(out_auto["modes_used"])
    print(f"\nAuto routing histogram (n={len(test_idx)}): "
          f"discovery={ctr.get('discovery', 0)}  "
          f"lightweight={ctr.get('lightweight', 0)}  "
          f"balanced={ctr.get('balanced', 0)}")

    # Showcase reports — pull from the FULL dataset (indices by name)
    print("\n" + "=" * 64)
    print("Showcase: predict_with_report()")
    print("=" * 64)
    for name in ("K00254.01", "K01091.01"):
        i = d["names"].index(name)
        rep = clf.predict_with_report(
            d["phases"][i], d["fluxes"][i],
            d["fluxes_secondary"][i], d["fluxes_odd_even"][i],
            snr=float(snr_all[i]),
            stellar_radius=float(d["stellar_radius"][i]),
        )
        label_str = "PLANET" if int(d["labels"][i]) == 1 else "FP"
        print(f"\n{name}  (true label: {label_str}):")
        for k, v in rep.items():
            print(f"  {k:<18}: {v}")


if __name__ == "__main__":
    main()
