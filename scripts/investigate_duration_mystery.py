"""Investigate why FP observed durations are systematically shorter than
Kepler's Third Law predictions.

Method:
    1. Fetch koi_fpflag_ss (stellar eclipse / EB flag) for every TCE in
       our dataset. FPs with fpflag_ss=1 are flagged as confirmed
       eclipsing binaries by the Kepler pipeline.
    2. Split the 500 TCEs into three groups:
           (a) confirmed planets  (dispo=CONFIRMED)
           (b) EB FPs             (dispo=FP, fpflag_ss=1)
           (c) non-EB FPs         (dispo=FP, fpflag_ss=0)
    3. For each group, compute obs_duration / predicted_duration and
       report the distribution.
    4. If EB FPs show observed > predicted (the expected "too long"
       signature), the FP duration-short signal in the full FP class
       is an artifact of non-EB FPs dominating the sample.
    5. If EB FPs also show observed < predicted, the issue is
       a measurement-convention mismatch.

Also compares 5 named confirmed planets (Kepler-5b through Kepler-12b,
etc.) to archive durations to sanity-check our formula.
"""

import csv
import io
import urllib.request
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from src.models.kepler_loss import (
    calculate_predicted_duration,
    calculate_kepler_violation,
)


def fetch_fp_flags() -> dict:
    """Fetch koi_disposition and koi_fpflag_ss (stellar eclipse / EB flag)
    for every KOI from the cumulative archive table."""
    url = (
        "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"
        "?table=cumulative"
        "&select=kepoi_name,koi_disposition,koi_fpflag_ss,koi_fpflag_co,"
        "koi_fpflag_nt,koi_fpflag_ec"
        "&format=csv"
    )
    print("Fetching FP flags from archive...")
    with urllib.request.urlopen(url, timeout=60) as resp:
        text = resp.read().decode("utf-8")
    table = {}
    for row in csv.DictReader(io.StringIO(text)):
        name = row["kepoi_name"]
        def _f(k):
            v = row.get(k, "")
            return int(v) if v not in ("", "NaN", "nan") else 0
        table[name] = {
            "disposition": row.get("koi_disposition", ""),
            "fpflag_ss":   _f("koi_fpflag_ss"),
            "fpflag_co":   _f("koi_fpflag_co"),
            "fpflag_nt":   _f("koi_fpflag_nt"),
            "fpflag_ec":   _f("koi_fpflag_ec"),
        }
    return table


def describe(name: str, vals: np.ndarray, fmt: str = "{:.3f}"):
    if len(vals) == 0:
        print(f"  {name:<30} (n=0, no data)")
        return
    p5, p25, p50, p75, p95 = np.percentile(vals, [5, 25, 50, 75, 95])
    print(
        f"  {name:<30} n={len(vals):<4} "
        f"p5={fmt.format(p5)}  p25={fmt.format(p25)}  "
        f"p50={fmt.format(p50)}  p75={fmt.format(p75)}  p95={fmt.format(p95)}"
    )


def main():
    d = torch.load("data/kepler_tce_v6.pt", weights_only=False)
    names = d["names"]
    period = d["period_days"].numpy()
    dur_h = d["duration_hours"].numpy()
    smass = d["stellar_mass"].numpy()
    srad = d["stellar_radius"].numpy()
    labels = d["labels"].numpy()
    print(f"Dataset: {len(names)} TCEs "
          f"({int(labels.sum())} confirmed, {int((1-labels).sum())} FP)")

    flags_table = fetch_fp_flags()
    flags = np.zeros((len(names), 4), dtype=int)
    disposition = []
    for i, nm in enumerate(names):
        row = flags_table.get(nm, {})
        flags[i, 0] = row.get("fpflag_ss", 0)
        flags[i, 1] = row.get("fpflag_co", 0)
        flags[i, 2] = row.get("fpflag_nt", 0)
        flags[i, 3] = row.get("fpflag_ec", 0)
        disposition.append(row.get("disposition", ""))
    disposition = np.array(disposition)

    # Compute predicted duration from Kepler's third law (b=0)
    pred_dur_d = calculate_predicted_duration(
        torch.tensor(period), torch.tensor(smass), torch.tensor(srad)
    ).numpy()
    pred_dur_h = pred_dur_d * 24.0
    ratio = dur_h / pred_dur_h
    log_ratio = np.log(ratio)

    # Split into groups
    is_conf = labels == 1
    is_fp = labels == 0
    is_eb = is_fp & (flags[:, 0] == 1)  # FP + stellar eclipse flag
    is_co = is_fp & (flags[:, 1] == 1)  # FP + centroid offset
    is_nt = is_fp & (flags[:, 2] == 1)  # FP + not transit-like
    is_ec = is_fp & (flags[:, 3] == 1)  # FP + ephemeris-match contamination
    is_non_eb = is_fp & (flags[:, 0] == 0)  # FP without stellar eclipse flag

    print(f"\nFP breakdown by flag:")
    print(f"  Stellar eclipse (EB):           {int(is_eb.sum())}")
    print(f"  Centroid offset (background):   {int(is_co.sum())}")
    print(f"  Not transit-like:               {int(is_nt.sum())}")
    print(f"  Ephemeris match contamination:  {int(is_ec.sum())}")
    print(f"  FPs WITHOUT EB flag:            {int(is_non_eb.sum())}")
    print(f"  FPs with multiple flags are counted in each row.")

    print(f"\nDuration ratio obs/pred (p5, p25, p50, p75, p95):")
    describe("Confirmed planets",      ratio[is_conf])
    describe("FP — all",               ratio[is_fp])
    describe("FP — stellar eclipse",   ratio[is_eb])
    describe("FP — NOT stellar eclip", ratio[is_non_eb])
    describe("FP — centroid offset",   ratio[is_co])
    describe("FP — not transit-like",  ratio[is_nt])

    print(f"\nLog ratio log(obs/pred):")
    describe("Confirmed planets",      log_ratio[is_conf])
    describe("FP — stellar eclipse",   log_ratio[is_eb])
    describe("FP — NOT stellar eclip", log_ratio[is_non_eb])

    # If we have enough EBs, show a few specific examples
    if is_eb.sum() >= 5:
        print(f"\nFirst 5 EB-flagged FPs — do their durations look too long?")
        eb_idx = np.where(is_eb)[0][:5]
        print(f"  {'KOI':<12} {'period_d':<9} {'obs_h':<7} {'pred_h':<7} "
              f"{'ratio':<6} {'M':<6} {'R':<6}")
        for i in eb_idx:
            print(
                f"  {names[i]:<12} {period[i]:<9.3f} {dur_h[i]:<7.2f} "
                f"{pred_dur_h[i]:<7.2f} {ratio[i]:<6.3f} "
                f"{smass[i]:<6.2f} {srad[i]:<6.2f}"
            )

    # Cross-check on well-known confirmed planets (convention sanity)
    known = [
        ("K00018.01", "Kepler-5b",  3.548465, 1.374, 1.793),  # Archive values
        ("K00017.01", "Kepler-6b",  3.234699, 1.209, 1.391),
        ("K00097.01", "Kepler-7b",  4.885489, 1.347, 1.843),
        ("K00010.01", "Kepler-8b",  3.522498, 1.213, 1.486),
        ("K00020.01", "Kepler-12b", 4.437963, 1.166, 1.483),
    ]
    print(f"\nKnown-planet sanity check (Kepler-5b through Kepler-12b):")
    print(f"  {'name':<11} {'P':<6} {'archive_dur':<13} {'pred_dur':<10} {'ratio':<6}")
    for kepoi, pl_name, p_exp, m_exp, r_exp in known:
        if kepoi in names:
            i = names.index(kepoi)
            p = calculate_predicted_duration(
                torch.tensor(period[i]), torch.tensor(smass[i]), torch.tensor(srad[i])
            ).item() * 24.0
            r = dur_h[i] / p
            print(
                f"  {pl_name:<11} {period[i]:<6.2f} "
                f"{dur_h[i]:<13.2f} {p:<10.2f} {r:<6.3f}"
            )

    # Summary interpretation
    print("\n--- Interpretation ---")
    eb_median = float(np.median(ratio[is_eb])) if is_eb.sum() > 0 else float("nan")
    conf_median = float(np.median(ratio[is_conf]))
    non_eb_median = float(np.median(ratio[is_non_eb])) if is_non_eb.sum() > 0 else float("nan")
    print(f"  Confirmed planets ratio median: {conf_median:.3f}")
    print(f"  EB FPs ratio median:            {eb_median:.3f}")
    print(f"  Non-EB FPs ratio median:        {non_eb_median:.3f}")

    if not np.isnan(eb_median) and eb_median > conf_median + 0.1:
        print("  -> EB FPs have LONGER observed durations than planets. Physics signal is real.")
        print("     V7 soft Kepler loss should work on EB-type FPs with a cleaner dataset.")
    elif not np.isnan(eb_median) and abs(eb_median - conf_median) < 0.1:
        print("  -> EB FPs look duration-similar to planets.")
        print("     Archive koi_duration may use a convention that normalizes out the EB difference,")
        print("     OR the Kepler pipeline's duration fit constrains EB models to planet-like durations.")
    else:
        print("  -> Signal is ambiguous or non-EB FPs dominate.")


if __name__ == "__main__":
    main()
