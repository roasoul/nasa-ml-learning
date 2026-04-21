"""Add stellar parameters (duration, M_star, R_star) to an existing dataset.

V7 needs the Kepler's Third Law check:
    t_dur_predicted = f(period, M_star, R_star)

The current kepler_tce_v6.pt has only depth/name — this script fetches
koi_duration (hours), koi_srad (R_sun), koi_smass (M_sun) from NASA's
Exoplanet Archive for each saved KOI and merges them into the .pt file.
One query total — avoids re-downloading 500 light curves.

Missing values fall back to solar defaults (M=1, R=1) so downstream code
can always do the Kepler calculation.
"""

import argparse
import csv
import io
import sys
import urllib.request
from pathlib import Path

import torch
import numpy as np


ARCHIVE_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"
    "?table=cumulative"
    "&select=kepoi_name,koi_duration,koi_srad,koi_smass"
    "&format=csv"
)


def fetch_stellar_table() -> dict:
    """Fetch stellar params for every KOI in the archive. Returns dict
    keyed by KOI name, values with keys duration_h, srad, smass."""
    print(f"Fetching stellar params from NASA Exoplanet Archive...")
    with urllib.request.urlopen(ARCHIVE_URL, timeout=60) as resp:
        text = resp.read().decode("utf-8")
    table = {}
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        try:
            name = row["kepoi_name"]
            table[name] = {
                "duration_h": _to_float(row.get("koi_duration")),
                "srad":      _to_float(row.get("koi_srad")),
                "smass":     _to_float(row.get("koi_smass")),
            }
        except KeyError:
            continue
    print(f"  Fetched params for {len(table)} KOIs")
    return table


def _to_float(val):
    if val is None or val == "" or val.lower() == "nan":
        return float("nan")
    try:
        return float(val)
    except (ValueError, TypeError):
        return float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/kepler_tce_v6.pt")
    parser.add_argument("--output", default="data/kepler_tce_v6.pt")
    args = parser.parse_args()

    d = torch.load(args.input, weights_only=False)
    print(f"Loaded {args.input}: {len(d['names'])} TCEs")

    archive = fetch_stellar_table()

    durations = []
    smasses = []
    srads = []
    n_missing = 0
    for name in d["names"]:
        row = archive.get(name)
        if row is None:
            n_missing += 1
            dur = float("nan"); m = float("nan"); r = float("nan")
        else:
            dur = row["duration_h"]
            m = row["smass"]
            r = row["srad"]
        durations.append(dur)
        smasses.append(m)
        srads.append(r)

    # Fall back to solar defaults for missing values
    def fill_solar(values, label, default):
        arr = np.array(values, dtype=float)
        n_nan = int(np.isnan(arr).sum())
        arr = np.where(np.isnan(arr), default, arr)
        print(f"  {label}: {n_nan} missing, filled with {default}")
        return arr

    duration_arr = fill_solar(durations, "koi_duration", 5.0)  # 5 hr fallback
    smass_arr = fill_solar(smasses, "koi_smass (M_sun)", 1.0)
    srad_arr = fill_solar(srads, "koi_srad (R_sun)", 1.0)

    d["duration_hours"] = torch.tensor(duration_arr, dtype=torch.float32)
    d["stellar_mass"] = torch.tensor(smass_arr, dtype=torch.float32)
    d["stellar_radius"] = torch.tensor(srad_arr, dtype=torch.float32)

    # Derive TCE period from the archive too — we need it for Kepler's
    # Third Law. Fetch separately because the existing dataset dropped it.
    period_arr = _fetch_periods(d["names"])
    d["period_days"] = torch.tensor(period_arr, dtype=torch.float32)

    torch.save(d, args.output)
    print(f"\nSaved retrofit dataset to {args.output}")
    print(f"  New keys: duration_hours, period_days, stellar_mass, stellar_radius")
    print(f"  KOIs not found in archive: {n_missing}")
    print(f"  Duration summary (hours):  "
          f"min={duration_arr.min():.2f}  median={np.median(duration_arr):.2f}  "
          f"max={duration_arr.max():.2f}")
    print(f"  Period summary (days):     "
          f"min={period_arr.min():.3f}  median={np.median(period_arr):.3f}  "
          f"max={period_arr.max():.3f}")


def _fetch_periods(names):
    """Fetch koi_period for each KOI in the dataset."""
    url = (
        "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"
        "?table=cumulative&select=kepoi_name,koi_period&format=csv"
    )
    print("Fetching periods...")
    with urllib.request.urlopen(url, timeout=60) as resp:
        text = resp.read().decode("utf-8")
    periods = {}
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        periods[row["kepoi_name"]] = _to_float(row.get("koi_period"))

    out = []
    n_missing = 0
    for name in names:
        p = periods.get(name, float("nan"))
        if np.isnan(p):
            n_missing += 1
            p = 1.0  # rare — solar-ish fallback
        out.append(p)
    print(f"  Periods: {n_missing} missing, filled with 1.0 day")
    return np.array(out, dtype=float)


if __name__ == "__main__":
    main()
