"""SS-flag validation v2 — per-sample (A, B) via scipy.optimize.curve_fit.

Fits y_i ≈ taylor_template(x, A, B) independently for each TCE's
primary_flux with bounds  A ∈ [0, 0.1], B ∈ [-2, 2]. Plots the
per-sample B distribution split by koi_fpflag_ss.

Compare against scripts/v8_ss_validation.py which used a closed-form
least-squares fit. The closed-form path treats A and AB as linear
parameters; curve_fit optimises A and B directly under explicit
bounds. Either way the result is an independent estimate of per-sample
U-vs-V-shape from photometry alone.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import curve_fit


DATA_PATH = "data/kepler_tce_v6.pt"
SS_CACHE = "data/ss_flag_cache.csv"
FIGDIR = Path("notebooks/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)


def taylor_poly(x, A, B):
    """-A * (1 - x^2/2 + B * x^4/24). Smooth polynomial — no min(0, ...)
    because the hard clamp breaks gradient-based fitting. We call this
    only on dip-region samples (flux < 0) where the clamp is inactive."""
    return -A * (1.0 - x ** 2 / 2.0 + B * (x ** 4) / 24.0)


def main():
    d = torch.load(DATA_PATH, weights_only=False)
    phases = d["phases"].cpu().numpy()
    fluxes = d["fluxes"].cpu().numpy()
    names = d["names"]

    # Archive SS flag
    cache = {}
    with open(SS_CACHE, newline="") as f:
        for r in csv.DictReader(f):
            cache[r["kepoi_name"]] = r

    B_ss0, B_ss1 = [], []
    n_fit_ok = n_fit_fail = n_missing_ss = 0
    for i, name in enumerate(names):
        rec = cache.get(name)
        if rec is None or rec["ss"] in ("", "None"):
            n_missing_ss += 1
            continue
        ss = int(rec["ss"])
        # Only fit inside the dip — baseline points carry no info
        # about B, and the polynomial without the min(0) clamp would
        # predict spurious positive values outside |x| > sqrt(2).
        mask = fluxes[i] < 0
        if mask.sum() < 8:
            n_fit_fail += 1
            continue
        x_dip = phases[i][mask]
        y_dip = fluxes[i][mask]
        try:
            popt, _ = curve_fit(
                taylor_poly, x_dip, y_dip,
                p0=[0.01, 0.5],
                bounds=([0.0, -2.0], [0.1, 2.0]),
                maxfev=3000,
            )
            B_val = float(popt[1])
            n_fit_ok += 1
        except Exception:
            n_fit_fail += 1
            continue
        if ss == 1:
            B_ss1.append(B_val)
        else:
            B_ss0.append(B_val)

    print(f"Fits OK: {n_fit_ok}  |  failed: {n_fit_fail}  |  missing SS: {n_missing_ss}")
    print(f"SS=0 (not EB): n={len(B_ss0)}, median B={np.median(B_ss0):+.3f}")
    print(f"SS=1 (EB flag): n={len(B_ss1)}, median B={np.median(B_ss1):+.3f}")
    sep = abs(np.median(B_ss0) - np.median(B_ss1))
    print(f"Separation gap: {sep:.3f}")
    if sep > 0.2:
        print("B rediscovered SS Flag from photometry alone!")
    else:
        print("Weak separation — SS flag not recoverable from per-sample B")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(B_ss0, bins=25, alpha=0.7, color="#10b981",
                 label=f"SS=0 not-EB (n={len(B_ss0)})")
    axes[0].hist(B_ss1, bins=25, alpha=0.7, color="#ef4444",
                 label=f"SS=1 EB flag (n={len(B_ss1)})")
    axes[0].axvline(np.median(B_ss0), color="#10b981", ls="--",
                    label=f"median={np.median(B_ss0):+.2f}")
    axes[0].axvline(np.median(B_ss1), color="#ef4444", ls="--",
                    label=f"median={np.median(B_ss1):+.2f}")
    axes[0].set_xlabel("Per-sample B (scipy curve_fit, bounded)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Did bounded per-sample B rediscover the SS Flag?")
    axes[0].legend(fontsize=9)

    axes[1].bar(["SS=0 (not EB)", "SS=1 (EB flag)"],
                [np.median(B_ss0), np.median(B_ss1)],
                color=["#10b981", "#ef4444"], alpha=0.85)
    axes[1].set_ylabel("Median per-sample B")
    axes[1].set_title(f"Separation gap: {sep:.3f}")

    plt.suptitle("B Parameter vs SS Flag — scipy.curve_fit variant (Figure 7v2)",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGDIR / "B_vs_SS_flag_v2.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {FIGDIR / 'B_vs_SS_flag_v2.png'}")


if __name__ == "__main__":
    main()
