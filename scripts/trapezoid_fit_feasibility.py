"""Feasibility: fit a trapezoidal transit model directly to folded light
curves and see if fitted durations separate planets from EBs better than
archive koi_duration.

Hypothesis: archive koi_duration is biased by the Kepler pipeline's
planet-shaped Mandel-Agol fit, so EBs get planet-like durations that
make Kepler's-Third-Law violations ambiguous. A fit run ON the actual
photometry, without any planet prior, should recover the *true* eclipse
duration — long for EBs, short for planets.

Trapezoid transit model (Seager & Mallen-Ornelas 2003 approximation):

                              _________________
                             /                 \\    ← flat bottom (T23)
           _________________/                   \\_________________
                           |← T12 →|        |← T12 →|
                          |←———————— T14 ——————————→|

    Parameters:
        depth (positive, the dip amplitude)
        T14   (total first-to-last-contact duration, radians of phase)
        T12   (ingress duration, radians of phase; egress is equal)

    T23 (flat-bottom duration) = T14 - 2·T12

For each test TCE we:
    1. Pull the primary-view folded flux from kepler_tce_v6.pt.
    2. Fit the trapezoid with scipy.optimize.curve_fit.
    3. Convert fitted T14 from radians of phase → days → hours.
    4. Compare to archive koi_duration.
    5. Compute Kepler violation using fitted T14 (and, for reference,
       archive T14).

We pick 5 confirmed planets with the deepest dips (strong signal) and
5 EB-flagged FPs (koi_fpflag_ss=1) also sorted by depth, so both sets
have enough SNR for a meaningful fit.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import csv
import io
import urllib.request

import numpy as np
import torch
from scipy.optimize import curve_fit

from src.models.kepler_loss import (
    calculate_predicted_duration,
    calculate_kepler_violation,
)


DATA_PATH = "data/kepler_tce_v6.pt"
SEED_SPLIT = 42


def trapezoid(phase, depth, T14, T12, t0=0.0):
    """Trapezoidal transit in phase space (flux units: baseline-at-0).

    Args:
        phase: 1D array of phase values (radians).
        depth: positive — magnitude of the dip.
        T14:   total transit duration (radians).
        T12:   ingress (= egress) duration (radians). Must be <= T14/2.
        t0:    transit center (radians). Defaults to 0.

    Returns:
        Model flux with the dip (negative values in transit).
    """
    abs_t = np.abs(phase - t0)
    half_T14 = T14 / 2.0
    half_T23 = (T14 - 2.0 * T12) / 2.0
    if half_T23 < 0:
        half_T23 = 0.0

    flux = np.zeros_like(phase, dtype=float)
    in_flat = abs_t <= half_T23
    in_slope = (abs_t > half_T23) & (abs_t <= half_T14)
    flux[in_flat] = -depth
    if T12 > 0:
        # Linear ramp: 0 at abs_t=half_T14, -depth at abs_t=half_T23
        flux[in_slope] = -depth * (half_T14 - abs_t[in_slope]) / T12
    return flux


def fit_trapezoid(phase_np, flux_np):
    """Fit a trapezoid to a folded, binned light curve.

    Returns a dict with fitted parameters, converted to physical units
    if a period is known later. Raises RuntimeError on fit failure.
    """
    # Initial guess: depth = minimum observed dip, T14 = 10% of phase,
    # T12 = T14/4 (ingress ~ 25% of total — typical for small planets).
    depth0 = max(abs(flux_np.min()), 1e-4)
    T14_0 = 0.3                  # rad of phase
    T12_0 = T14_0 / 4.0

    # Bounds: depth in [0, 1], T14 in [0.02, pi], T12 in [0.005, T14/2]
    popt, pcov = curve_fit(
        trapezoid,
        phase_np,
        flux_np,
        p0=[depth0, T14_0, T12_0],
        bounds=(
            [0.0,    0.02,  0.005],
            [1.0,    np.pi, np.pi / 2.0],
        ),
        maxfev=5000,
    )
    depth, T14, T12 = popt
    perr = np.sqrt(np.diag(pcov))
    # Residual RMS for fit quality
    resid = flux_np - trapezoid(phase_np, *popt)
    rms = float(np.sqrt(np.mean(resid ** 2)))
    return {
        "depth": float(depth),
        "T14_rad": float(T14),
        "T12_rad": float(T12),
        "T14_err_rad": float(perr[1]),
        "rms": rms,
    }


def fetch_fp_flags() -> dict:
    """Return dict: KOI name → fpflag_ss (0 or 1)."""
    url = (
        "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"
        "?table=cumulative&select=kepoi_name,koi_fpflag_ss&format=csv"
    )
    with urllib.request.urlopen(url, timeout=60) as resp:
        text = resp.read().decode("utf-8")
    out = {}
    for row in csv.DictReader(io.StringIO(text)):
        val = row.get("koi_fpflag_ss", "")
        out[row["kepoi_name"]] = int(val) if val not in ("", "NaN", "nan") else 0
    return out


def get_test_indices(y):
    torch.manual_seed(SEED_SPLIT)
    conf = (y == 1).nonzero(as_tuple=True)[0]
    fp = (y == 0).nonzero(as_tuple=True)[0]
    conf = conf[torch.randperm(len(conf))]
    fp = fp[torch.randperm(len(fp))]
    def split(idx, tf=0.7, vf=0.15):
        n = len(idx); nt = int(n * tf); nv = int(n * vf)
        return idx[nt + nv:]
    return split(conf).tolist(), split(fp).tolist()


def main():
    print("Loading data...")
    d = torch.load(DATA_PATH, weights_only=False)
    names = d["names"]
    phases = d["phases"].numpy()
    fluxes = d["fluxes"].numpy()
    period_d = d["period_days"].numpy()
    archive_dur_h = d["duration_hours"].numpy()
    smass = d["stellar_mass"].numpy()
    srad = d["stellar_radius"].numpy()
    y = d["labels"]

    # Narrow to test set
    conf_test, fp_test = get_test_indices(y)

    # Need EB flags to pick EB FPs
    flags = fetch_fp_flags()
    fp_test_eb = [i for i in fp_test if flags.get(names[i], 0) == 1]
    print(f"Test-set FPs total: {len(fp_test)}")
    print(f"Test-set FPs flagged as EB (fpflag_ss=1): {len(fp_test_eb)}")

    # Pick 5 deepest planets and 5 deepest EB FPs
    def depth_of(i):
        return -fluxes[i, 95:105].min()  # positive
    conf_test.sort(key=lambda i: -depth_of(i))
    fp_test_eb.sort(key=lambda i: -depth_of(i))
    planets = conf_test[:5]
    ebs = fp_test_eb[:5]

    print(f"\nSelected 5 deepest planets and 5 deepest EB-flagged FPs.\n")

    def run(idx, label):
        ph = phases[idx]
        fl = fluxes[idx]
        P = period_d[idx]
        M = smass[idx]
        R = srad[idx]
        arch_h = archive_dur_h[idx]
        name = names[idx]

        # Fit
        try:
            fit = fit_trapezoid(ph, fl)
            T14_rad = fit["T14_rad"]
            fitted_h = T14_rad * P / (2 * np.pi) * 24.0
            rms = fit["rms"]
            depth_fit = fit["depth"]
            ok = True
        except Exception as e:
            fitted_h = float("nan"); rms = float("nan"); depth_fit = float("nan")
            ok = False
            print(f"  {name} fit failed: {e}")

        # Predicted duration from Kepler's Third Law
        pred_h = calculate_predicted_duration(P, M, R) * 24.0

        # Violations: archive vs fitted
        arch_ratio = arch_h / pred_h
        fit_ratio = fitted_h / pred_h if ok else float("nan")
        arch_viol = abs(np.log(max(arch_ratio, 1e-8)))
        fit_viol = abs(np.log(max(fit_ratio, 1e-8))) if ok else float("nan")

        return {
            "name": name, "P": P, "M": M, "R": R,
            "archive_h": arch_h, "fitted_h": fitted_h,
            "predicted_h": pred_h,
            "depth_fit": depth_fit, "rms": rms,
            "archive_viol": arch_viol, "fitted_viol": fit_viol,
            "ok": ok, "label": label,
        }

    rows = []
    for i in planets:
        rows.append(run(i, "PLANET"))
    for i in ebs:
        rows.append(run(i, "EB_FP"))

    print(
        f"{'class':<7} {'KOI':<12} {'P_d':<6} {'arch_h':<7} {'fit_h':<7} "
        f"{'pred_h':<7} {'arch_viol':<10} {'fit_viol':<9} {'depth_fit':<10} "
        f"{'rms':<7}"
    )
    print("-" * 88)
    for r in rows:
        fit_h = f"{r['fitted_h']:.2f}" if r["ok"] else "FAIL"
        fit_v = f"{r['fitted_viol']:.3f}" if r["ok"] else "---"
        print(
            f"{r['label']:<7} {r['name']:<12} {r['P']:<6.2f} "
            f"{r['archive_h']:<7.2f} {fit_h:<7} {r['predicted_h']:<7.2f} "
            f"{r['archive_viol']:<10.3f} {fit_v:<9} {r['depth_fit']:<10.4f} "
            f"{r['rms']:<7.4f}"
        )

    # Separation analysis
    print("\n--- Separation analysis ---")
    planet_rows = [r for r in rows if r["label"] == "PLANET" and r["ok"]]
    eb_rows = [r for r in rows if r["label"] == "EB_FP" and r["ok"]]

    def stats(group, key):
        vals = [r[key] for r in group]
        if not vals: return (float("nan"),) * 3
        return min(vals), float(np.median(vals)), max(vals)

    p_arch = stats(planet_rows, "archive_viol")
    e_arch = stats(eb_rows, "archive_viol")
    p_fit = stats(planet_rows, "fitted_viol")
    e_fit = stats(eb_rows, "fitted_viol")

    print(f"  Archive violation:  planets {p_arch[0]:.3f}/{p_arch[1]:.3f}/{p_arch[2]:.3f}    "
          f"EBs {e_arch[0]:.3f}/{e_arch[1]:.3f}/{e_arch[2]:.3f}   (min/median/max)")
    print(f"  Fitted violation:   planets {p_fit[0]:.3f}/{p_fit[1]:.3f}/{p_fit[2]:.3f}    "
          f"EBs {e_fit[0]:.3f}/{e_fit[1]:.3f}/{e_fit[2]:.3f}")

    # Separation metric: do EB violations sit above planet violations in each mode?
    def separation(planets_vals, ebs_vals):
        if not planets_vals or not ebs_vals:
            return None
        p_max = max(planets_vals); e_min = min(ebs_vals)
        return e_min - p_max

    sep_arch = separation(
        [r["archive_viol"] for r in planet_rows],
        [r["archive_viol"] for r in eb_rows],
    )
    sep_fit = separation(
        [r["fitted_viol"] for r in planet_rows],
        [r["fitted_viol"] for r in eb_rows],
    )
    print(f"  Separation gap (EB_min - planet_max): archive={sep_arch}, fit={sep_fit}")
    if sep_fit is not None and sep_arch is not None and sep_fit > sep_arch + 0.1:
        print("  CONCLUSION: Fitted violations separate classes better.")
        print("  V8 direction: direct transit-model fitting + Kepler gate is promising.")
    elif sep_fit is not None and sep_arch is not None and abs(sep_fit - sep_arch) < 0.1:
        print("  CONCLUSION: Fitted and archive violations are similarly useful (or useless).")
        print("  V8 direction: direct fitting doesn't add much; pursue other V8 ideas.")
    else:
        print("  CONCLUSION: Fitted violations are not clearly better than archive. "
              "V7 null result stands; try other V8 directions.")


if __name__ == "__main__":
    main()
