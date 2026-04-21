"""V7.5 package: safe Kepler-gate model + analysis figures.

Produces:
    src/models/taylor_cnn_v75.pt
        V6 Config C weights (retrained, seed=7) plus the hard-gate
        threshold = 1.0 and evaluation metrics.

    notebooks/figures/lambda_sweep.png
        Precision, recall, F1 vs lambda_kepler, for both symmetric
        and asymmetric violation modes. Parsed from v7_*_sweep.log.

    notebooks/figures/duration_ratio_histogram.png
        log(obs / predicted) distribution split by disposition and
        fpflag_ss — confirmed planets vs EB-FPs vs non-EB FPs. Shows
        the lack of duration separation that explains V7's null result.

    notebooks/figures/violation_distribution.png
        |log(obs/predicted)| — symmetric Kepler violation — by class.
        Same story from the other angle.
"""

import csv
import io
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.models.kepler_loss import calculate_kepler_violation
from src.models.taylor_cnn import TaylorCNN


DATA_PATH = "data/kepler_tce_v6.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED_SPLIT = 42
SEED_INIT = 7
GATE_THRESHOLD = 1.0
FIGURES_DIR = Path("notebooks/figures")
MODEL_PATH = "src/models/taylor_cnn_v75.pt"


# -----------------------------------------------------------------------
# V6 Config C retraining (matches scripts/threshold_sweep_v6.py)
# -----------------------------------------------------------------------

def load_v6():
    d = torch.load(DATA_PATH, weights_only=False)
    violations = calculate_kepler_violation(
        d["period_days"], d["duration_hours"] / 24.0,
        d["stellar_mass"], d["stellar_radius"],
    )
    torch.manual_seed(SEED_SPLIT)
    conf = (d["labels"] == 1).nonzero(as_tuple=True)[0]
    fp = (d["labels"] == 0).nonzero(as_tuple=True)[0]
    conf = conf[torch.randperm(len(conf))]
    fp = fp[torch.randperm(len(fp))]
    def split(idx, tf=0.7, vf=0.15):
        n = len(idx); nt = int(n * tf); nv = int(n * vf)
        return idx[:nt], idx[nt:nt + nv], idx[nt + nv:]
    ct, cv, cte = split(conf); ft, fv, fte = split(fp)
    return d, violations, torch.cat([ct, ft]), torch.cat([cv, fv]), torch.cat([cte, fte])


def train_config_c(d, train_idx, val_idx, n_epochs=200, patience=25, batch_size=16):
    def to_dev(idx):
        return (d["phases"][idx].to(DEVICE), d["fluxes"][idx].to(DEVICE),
                d["fluxes_secondary"][idx].to(DEVICE), d["fluxes_odd_even"][idx].to(DEVICE),
                d["labels"][idx].to(DEVICE))
    tp, tpp, tps, tpoe, tpl = to_dev(train_idx)
    vp, vpp, vps, vpoe, vpl = to_dev(val_idx)

    torch.manual_seed(SEED_INIT)
    model = TaylorCNN(init_amplitude=0.01).to(DEVICE)
    bce = nn.BCELoss()
    opt = torch.optim.Adam([
        {"params": model.taylor_gate.parameters(), "lr": 1e-4},
        {"params": model.cnn.parameters(), "lr": 1e-3},
        {"params": model.classifier.parameters(), "lr": 1e-3},
    ])
    n_tr = len(tpl); best_val = float("inf"); best_state = None; wait = 0
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
    return model


def metrics_from(preds, labels):
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
            "TP": TP, "FP": FP, "TN": TN, "FN": FN}


def build_v75_model():
    print("=== V7.5: retrain V6 Config C + apply safe hard gate ===")
    d, violations, train_idx, val_idx, test_idx = load_v6()
    model = train_config_c(d, train_idx, val_idx)

    # Evaluate without / with gate
    def stack(idx):
        return (d["phases"][idx].to(DEVICE), d["fluxes"][idx].to(DEVICE),
                d["fluxes_secondary"][idx].to(DEVICE), d["fluxes_odd_even"][idx].to(DEVICE))
    tp, tpp, tps, tpoe = stack(test_idx)
    model.eval()
    with torch.no_grad():
        probs = model(tp, tpp, tps, tpoe).squeeze(1).cpu()
    labels_test = d["labels"][test_idx]
    viol_test = violations[test_idx]

    preds_no_gate = (probs > 0.5).float()
    m_before = metrics_from(preds_no_gate, labels_test)

    gate_mask = viol_test > GATE_THRESHOLD
    preds_gated = preds_no_gate.clone()
    preds_gated[gate_mask] = 0.0
    m_after = metrics_from(preds_gated, labels_test)

    print(f"  Before gate (V6 C baseline): "
          f"acc={m_before['accuracy']:.1%}  prec={m_before['precision']:.1%}  "
          f"rec={m_before['recall']:.1%}  F1={m_before['f1']:.3f}")
    print(f"  After gate (V7.5, thr={GATE_THRESHOLD}): "
          f"acc={m_after['accuracy']:.1%}  prec={m_after['precision']:.1%}  "
          f"rec={m_after['recall']:.1%}  F1={m_after['f1']:.3f}")
    n_gated = int(gate_mask.sum())
    planets_blocked = int((gate_mask & (labels_test == 1)).sum())
    print(f"  Gate filtered {n_gated} TCE(s); blocked {planets_blocked} real planet(s)")

    names_test = [d["names"][int(i)] for i in test_idx.tolist()]
    gated_names = [names_test[i] for i, g in enumerate(gate_mask.tolist()) if g]
    print(f"  Gated KOIs: {gated_names}")

    Path("src/models").mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "A": model.taylor_gate.A.item(),
        "kepler_gate_threshold": GATE_THRESHOLD,
        "violation_mode": "symmetric |log(obs/pred)|",
        "base_config": "V6 Config C — from-scratch BCE, seed_init=7",
        "metrics_before_gate": m_before,
        "metrics_after_gate": m_after,
        "gated_kois": gated_names,
    }, MODEL_PATH)
    print(f"  Saved V7.5 model to {MODEL_PATH}\n")

    return d, violations


# -----------------------------------------------------------------------
# Figure 1 — lambda sweep
# -----------------------------------------------------------------------

_SWEEP_ROW = re.compile(
    r"^\s*([\d.]+)\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)"
)


def parse_sweep_log(path):
    lambdas, acc, prec, rec, f1 = [], [], [], [], []
    # tee captured the console output, which may contain non-ASCII bytes on
    # Windows. Use errors="replace" so parsing never fails on encoding issues.
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.strip().startswith("lambda") or line.startswith("-"):
                continue
            m = _SWEEP_ROW.match(line)
            if not m:
                continue
            lam = float(m.group(1))
            if lam == 0.0 and lambdas and lambdas[-1] == 0.0:
                continue
            lambdas.append(lam)
            acc.append(float(m.group(2)))
            prec.append(float(m.group(3)))
            rec.append(float(m.group(4)))
            f1.append(float(m.group(5)) * 100)
    return lambdas, acc, prec, rec, f1


def plot_lambda_sweep():
    print("=== Figure 1: lambda_sweep.png ===")
    sym = parse_sweep_log("data/v7_lambda_sweep.log")
    asym = parse_sweep_log("data/v7_asymmetric_sweep.log")
    print(f"  symmetric:  {len(sym[0])} rows: {sym[0]}")
    print(f"  asymmetric: {len(asym[0])} rows: {asym[0]}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, data, title in [
        (axes[0], sym, "Symmetric  |log(obs/pred)|"),
        (axes[1], asym, "Asymmetric  max(0, log(obs/pred))"),
    ]:
        lambdas, acc, prec, rec, f1 = data
        ax.plot(lambdas, prec, "o-", label="precision", color="#d62728")
        ax.plot(lambdas, rec, "s-", label="recall", color="#2ca02c")
        ax.plot(lambdas, f1, "^-", label="F1", color="#1f77b4")
        ax.set_xscale("symlog", linthresh=0.1)
        ax.set_xlabel("lambda_kepler")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.axhline(80, color="k", lw=0.8, linestyle="--", alpha=0.4)
        ax.text(0.02, 82, "80% target", fontsize=8, alpha=0.6)
        ax.legend(loc="lower left")
    axes[0].set_ylabel("percent")
    fig.suptitle("V7 Kepler-loss lambda sweep — both modes (no lambda helps)")
    fig.tight_layout()
    path = FIGURES_DIR / "lambda_sweep.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}\n")


# -----------------------------------------------------------------------
# Duration/violation histograms — need fpflag_ss
# -----------------------------------------------------------------------

def fetch_flags():
    url = (
        "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"
        "?table=cumulative&select=kepoi_name,koi_fpflag_ss,koi_fpflag_nt&format=csv"
    )
    with urllib.request.urlopen(url, timeout=60) as resp:
        text = resp.read().decode("utf-8")
    table = {}
    for row in csv.DictReader(io.StringIO(text)):
        def _f(k):
            v = row.get(k, "")
            return int(v) if v not in ("", "NaN", "nan") else 0
        table[row["kepoi_name"]] = {
            "ss": _f("koi_fpflag_ss"),
            "nt": _f("koi_fpflag_nt"),
        }
    return table


def plot_duration_histogram(d, violations):
    print("=== Figure 2: duration_ratio_histogram.png ===")
    names = d["names"]
    dur_h = d["duration_hours"].numpy()
    period = d["period_days"].numpy()
    smass = d["stellar_mass"].numpy()
    srad = d["stellar_radius"].numpy()
    labels = d["labels"].numpy()

    # Predicted duration from Kepler III — reuse utilities
    from src.models.kepler_loss import calculate_predicted_duration
    pred_h = (
        calculate_predicted_duration(
            torch.tensor(period), torch.tensor(smass), torch.tensor(srad)
        ).numpy()
        * 24.0
    )
    log_ratio = np.log(np.clip(dur_h / pred_h, 1e-8, None))

    flags = fetch_flags()
    is_eb = np.array([flags.get(nm, {}).get("ss", 0) == 1 for nm in names])
    is_nt = np.array([flags.get(nm, {}).get("nt", 0) == 1 for nm in names])

    planets = log_ratio[labels == 1]
    eb_fp = log_ratio[(labels == 0) & is_eb]
    nt_fp = log_ratio[(labels == 0) & is_nt]
    nonebnon_fp = log_ratio[(labels == 0) & ~is_eb & ~is_nt]

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(-3, 3, 61)
    ax.hist(planets, bins=bins, alpha=0.55, label=f"Confirmed planets (n={len(planets)})",
            color="#1f77b4", edgecolor="white", linewidth=0.5)
    ax.hist(eb_fp, bins=bins, alpha=0.55, label=f"FP  fpflag_ss=1 (n={len(eb_fp)})",
            color="#d62728", edgecolor="white", linewidth=0.5)
    ax.hist(nt_fp, bins=bins, alpha=0.8, label=f"FP  fpflag_nt=1 (n={len(nt_fp)})",
            color="#ff7f0e", edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="k", lw=0.8, linestyle="--", alpha=0.5)
    ax.axvline(np.log(2), color="gray", lw=0.5, alpha=0.4)
    ax.axvline(-np.log(2), color="gray", lw=0.5, alpha=0.4)
    ax.text(np.log(2), ax.get_ylim()[1] * 0.9, " 2x too long", fontsize=8, alpha=0.6)
    ax.text(-np.log(2), ax.get_ylim()[1] * 0.9, " 2x too short", fontsize=8, ha="right", alpha=0.6)
    ax.set_xlabel("log(observed duration / Kepler-III predicted duration)")
    ax.set_ylabel("count")
    ax.set_title("Duration ratio by class — EB FPs look planet-like\n"
                 "Only NT-flagged (not-transit-like) FPs show extreme durations")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = FIGURES_DIR / "duration_ratio_histogram.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    print(f"  planets median: {np.median(planets):.3f}    "
          f"EB median: {np.median(eb_fp):.3f}    "
          f"NT median: {np.median(nt_fp):.3f}\n")


def plot_violation_distribution(d, violations):
    print("=== Figure 3: violation_distribution.png ===")
    labels = d["labels"].numpy()
    v = violations.numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, 3, 61)
    ax.hist(v[labels == 1], bins=bins, alpha=0.6,
            label=f"Confirmed planets (n={(labels==1).sum()})",
            color="#1f77b4", edgecolor="white", linewidth=0.5)
    ax.hist(v[labels == 0], bins=bins, alpha=0.6,
            label=f"FP (all) (n={(labels==0).sum()})",
            color="#d62728", edgecolor="white", linewidth=0.5)
    for thr in (0.2, 1.0):
        ax.axvline(thr, color="k", lw=0.8, linestyle="--", alpha=0.4)
    ax.text(0.2, ax.get_ylim()[1] * 0.92, "  V7 tolerance (0.2)", fontsize=8, alpha=0.6)
    ax.text(1.0, ax.get_ylim()[1] * 0.92, "  V7.5 safe gate (1.0)", fontsize=8, alpha=0.6)
    ax.set_xlabel("Kepler violation  |log(obs / predicted)|")
    ax.set_ylabel("count")
    ax.set_title("Kepler's Third Law violation distribution — weak class separation\n"
                 "Most planets and FPs cluster below 0.5; only extreme-violation FPs are filterable")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = FIGURES_DIR / "violation_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    pm, pfp = float(np.median(v[labels == 1])), float(np.median(v[labels == 0]))
    print(f"  planet median violation: {pm:.3f}    FP median: {pfp:.3f}\n")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    d, violations = build_v75_model()
    plot_lambda_sweep()
    plot_duration_histogram(d, violations)
    plot_violation_distribution(d, violations)
    print("V7.5 package complete.")


if __name__ == "__main__":
    main()
