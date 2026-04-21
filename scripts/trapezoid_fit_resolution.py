"""Resolution sweep for the V7 trapezoidal fit feasibility test.

V7 fit at 200 phase bins did not separate planets from EB FPs
(separation gap = -0.237). Question: does higher phase-bin resolution
sharpen the fit enough to close the gap?

For the same 5 planets and 5 EB FPs selected in V7:
    1. Re-download + preprocess each light curve.
    2. Refold and bin at 200, 500, and 1000 phase bins.
    3. Fit trapezoidal model at each resolution.
    4. Compute Kepler's-Third-Law violation from the fitted T14.
    5. Report median violations and the separation gap per resolution.
    6. Plot K03745.01 (the deep EB) at all three resolutions so we can
       eyeball whether ingress/egress sharpens visually.

Success criterion: separation gap grows with resolution
    -> a Kepler-loss path may still be viable for V8.
Failure criterion: gap stays flat or shrinks
    -> confirms the V7 null result is not a binning artefact; the
       shape-based V8 direction stands.
"""

from __future__ import annotations

import csv
import io
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import curve_fit

from src.data.kepler import _fold_and_bin, download_lightcurve
from src.models.kepler_loss import calculate_predicted_duration


# TCEs identified by the V7 feasibility run (data/trapezoid_feasibility.log).
# Hard-coded here so this script reproduces the exact V7 selection without
# having to re-run the EB-flag fetch + sort by depth on the test split.
PLANETS = ["K00254.01", "K00183.01", "K01320.01", "K00840.01", "K00017.01"]
EBS = ["K03745.01", "K01541.01", "K07192.01", "K07343.01", "K07955.01"]
DEEP_EB = "K03745.01"

RESOLUTIONS = [200, 500, 1000]

FIGURE_OUT = Path("notebooks/figures/trapezoid_fit_resolution_k03745.png")


# -----------------------------------------------------------------------
# Trapezoid model + fit (identical to V7 feasibility script)
# -----------------------------------------------------------------------

def trapezoid(phase, depth, T14, T12, t0=0.0):
    abs_t = np.abs(phase - t0)
    half_T14 = T14 / 2.0
    half_T23 = max(0.0, (T14 - 2.0 * T12) / 2.0)
    flux = np.zeros_like(phase, dtype=float)
    in_flat = abs_t <= half_T23
    in_slope = (abs_t > half_T23) & (abs_t <= half_T14)
    flux[in_flat] = -depth
    if T12 > 0:
        flux[in_slope] = -depth * (half_T14 - abs_t[in_slope]) / T12
    return flux


def fit_trapezoid(phase_np: np.ndarray, flux_np: np.ndarray) -> dict:
    depth0 = max(abs(flux_np.min()), 1e-4)
    T14_0 = 0.3
    T12_0 = T14_0 / 4.0
    popt, pcov = curve_fit(
        trapezoid,
        phase_np,
        flux_np,
        p0=[depth0, T14_0, T12_0],
        bounds=([0.0, 0.02, 0.005], [1.0, np.pi, np.pi / 2.0]),
        maxfev=5000,
    )
    depth, T14, T12 = popt
    resid = flux_np - trapezoid(phase_np, *popt)
    return {
        "depth": float(depth),
        "T14_rad": float(T14),
        "T12_rad": float(T12),
        "rms": float(np.sqrt(np.mean(resid ** 2))),
    }


# -----------------------------------------------------------------------
# Archive metadata fetch — need kepid + epoch for lightkurve download
# -----------------------------------------------------------------------

def fetch_archive_meta(koi_names: set[str]) -> dict[str, dict]:
    """Return per-KOI metadata needed for download + Kepler's-law check."""
    url = (
        "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"
        "?table=cumulative"
        "&select=kepid,kepoi_name,koi_period,koi_time0bk,koi_duration,"
        "koi_smass,koi_srad"
        "&format=csv"
    )
    with urllib.request.urlopen(url, timeout=60) as resp:
        text = resp.read().decode("utf-8")

    def pf(row: dict, key: str, default: float = 0.0) -> float:
        v = row.get(key, "")
        if v in ("", "NaN", "nan", None):
            return default
        try:
            return float(v)
        except ValueError:
            return default

    meta = {}
    for row in csv.DictReader(io.StringIO(text)):
        name = row.get("kepoi_name", "")
        if name not in koi_names:
            continue
        meta[name] = {
            "kepid": int(pf(row, "kepid")),
            "period": pf(row, "koi_period"),
            "epoch": pf(row, "koi_time0bk"),
            "duration_h": pf(row, "koi_duration"),
            "smass": pf(row, "koi_smass", 1.0),
            "srad": pf(row, "koi_srad", 1.0),
        }
    return meta


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    koi_all = PLANETS + EBS
    print("Fetching archive metadata for 10 test TCEs...")
    meta = fetch_archive_meta(set(koi_all))
    missing = [k for k in koi_all if k not in meta]
    if missing:
        print(f"  WARN: missing archive rows for {missing}")

    # Download + flatten + normalize once per target, reuse across resolutions.
    print("\nDownloading + preprocessing light curves (lightkurve cache "
          "should be warm from V6 build)...")
    curves = {}
    for koi in koi_all:
        if koi not in meta:
            continue
        m = meta[koi]
        t0 = time.time()
        print(f"  {koi} (KIC {m['kepid']}) ...", end="", flush=True)
        try:
            lc = download_lightcurve(m["kepid"])
            lc_flat = lc.flatten(window_length=401).normalize()
            curves[koi] = lc_flat
            print(f" OK ({time.time() - t0:.0f}s)")
        except Exception as e:
            curves[koi] = None
            print(f" FAIL ({time.time() - t0:.0f}s): {e}")

    # Fit at each resolution
    rows_by_res: dict[int, list[dict]] = defaultdict(list)
    deep_eb_curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for koi in koi_all:
        if curves.get(koi) is None:
            continue
        lc_flat = curves[koi]
        m = meta[koi]
        label = "PLANET" if koi in PLANETS else "EB_FP"
        pred_days = calculate_predicted_duration(m["period"], m["smass"], m["srad"])
        pred_h = pred_days * 24.0
        arch_h = m["duration_h"]
        arch_viol = abs(np.log(max(arch_h / pred_h, 1e-8))) if pred_h > 0 else float("nan")

        for n in RESOLUTIONS:
            try:
                phase_t, flux_t = _fold_and_bin(
                    lc_flat,
                    period=m["period"],
                    epoch_time=m["epoch"],
                    n_points=n,
                )
                phase_np = phase_t.numpy()
                flux_np = flux_t.numpy()
                if koi == DEEP_EB:
                    deep_eb_curves[n] = (phase_np.copy(), flux_np.copy())

                fit = fit_trapezoid(phase_np, flux_np)
                fit_h = fit["T14_rad"] * m["period"] / (2 * np.pi) * 24.0
                fit_viol = abs(np.log(max(fit_h / pred_h, 1e-8)))
                rows_by_res[n].append({
                    "koi": koi,
                    "label": label,
                    "arch_h": arch_h,
                    "fit_h": fit_h,
                    "pred_h": pred_h,
                    "arch_viol": arch_viol,
                    "fit_viol": fit_viol,
                    "depth_fit": fit["depth"],
                    "T12_rad": fit["T12_rad"],
                    "T14_rad": fit["T14_rad"],
                    "rms": fit["rms"],
                    "ok": True,
                })
            except Exception as e:
                print(f"  {koi} @ {n} bins fit failed: {e}")
                rows_by_res[n].append({
                    "koi": koi, "label": label, "ok": False,
                })

    # -------------------------------------------------------------------
    # Per-resolution detail
    # -------------------------------------------------------------------
    for n in RESOLUTIONS:
        print(f"\n=== Resolution: {n} bins ===")
        header = (
            f"{'class':<7} {'KOI':<12} {'arch_h':<7} {'fit_h':<7} "
            f"{'pred_h':<7} {'arch_viol':<10} {'fit_viol':<9} "
            f"{'depth_fit':<10} {'rms':<7}"
        )
        print(header)
        print("-" * len(header))
        for r in rows_by_res[n]:
            if not r["ok"]:
                print(f"{r['label']:<7} {r['koi']:<12} FAIL")
                continue
            print(
                f"{r['label']:<7} {r['koi']:<12} "
                f"{r['arch_h']:<7.2f} {r['fit_h']:<7.2f} {r['pred_h']:<7.2f} "
                f"{r['arch_viol']:<10.3f} {r['fit_viol']:<9.3f} "
                f"{r['depth_fit']:<10.4f} {r['rms']:<7.4f}"
            )

    # -------------------------------------------------------------------
    # Separation summary (this is the scientific question)
    # -------------------------------------------------------------------
    print("\n=== Separation summary (fit violation) ===")
    header = (
        f"{'Resolution':<12} {'planet_med':<12} {'EB_med':<10} "
        f"{'planet_max':<12} {'EB_min':<10} {'gap':<10}"
    )
    print(header)
    print("-" * len(header))
    for n in RESOLUTIONS:
        P = [r["fit_viol"] for r in rows_by_res[n] if r["ok"] and r["label"] == "PLANET"]
        E = [r["fit_viol"] for r in rows_by_res[n] if r["ok"] and r["label"] == "EB_FP"]
        if not P or not E:
            print(f"{n:<12}  (insufficient data)")
            continue
        gap = min(E) - max(P)
        print(
            f"{n:<12} {np.median(P):<12.3f} {np.median(E):<10.3f} "
            f"{max(P):<12.3f} {min(E):<10.3f} {gap:<10.3f}"
        )

    # Per-KOI fit_h across resolutions — useful to spot targets whose
    # fitted duration is converging vs. diverging with resolution.
    print("\n=== Fitted T14 (hours) across resolutions, per KOI ===")
    all_kois = [r["koi"] for r in rows_by_res[RESOLUTIONS[0]] if r["ok"]]
    cols = "   ".join(f"{n}b" for n in RESOLUTIONS)
    print(f"{'label':<7} {'KOI':<12} {'arch_h':<8} {cols}")
    for koi in all_kois:
        lab = "PLANET" if koi in PLANETS else "EB_FP"
        by_n = {}
        arch_h = None
        for n in RESOLUTIONS:
            for r in rows_by_res[n]:
                if r["koi"] == koi and r["ok"]:
                    by_n[n] = r["fit_h"]
                    arch_h = r["arch_h"]
        vals = "  ".join(f"{by_n.get(n, float('nan')):<5.2f}" for n in RESOLUTIONS)
        print(f"{lab:<7} {koi:<12} {arch_h:<8.2f} {vals}")

    # -------------------------------------------------------------------
    # Plot K03745.01 at all three resolutions
    # -------------------------------------------------------------------
    if len(deep_eb_curves) == len(RESOLUTIONS):
        fig, axes = plt.subplots(1, len(RESOLUTIONS), figsize=(15, 4), sharey=True)
        for ax, n in zip(axes, RESOLUTIONS):
            phase_np, flux_np = deep_eb_curves[n]
            ax.plot(phase_np, flux_np, linewidth=0.9, color="C0")
            ax.axvline(0.0, color="red", linestyle="--", linewidth=0.6, label="fold center")
            # Overlay fitted trapezoid
            fit_row = next(
                (r for r in rows_by_res[n] if r["koi"] == DEEP_EB and r["ok"]),
                None,
            )
            if fit_row is not None:
                model = trapezoid(
                    phase_np, fit_row["depth_fit"],
                    fit_row["T14_rad"], fit_row["T12_rad"],
                )
                ax.plot(phase_np, model, color="orange", linewidth=1.1,
                        label="trapezoid fit")
            ax.set_xlim(-0.6, 0.6)
            ax.set_title(f"K03745.01 (deep EB) — {n} bins")
            ax.set_xlabel("phase (rad)")
            if ax is axes[0]:
                ax.set_ylabel("flux - 1")
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(alpha=0.2)
        plt.tight_layout()
        FIGURE_OUT.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(FIGURE_OUT, dpi=150)
        plt.close()
        print(f"\nSaved plot: {FIGURE_OUT}")
    else:
        print("\nSkipping K03745.01 plot — not all resolutions produced a curve.")


if __name__ == "__main__":
    main()
