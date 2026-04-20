"""Download and cache a real Kepler TCE dataset for training.

Fetches light curves from MAST, preprocesses them using the CLAUDE.md
pipeline, and saves the result as a single .pt file for fast loading.

Usage:
    python -m src.data.build_dataset --n-per-class 100 --output data/kepler_tce.pt
"""

import argparse
import csv
import io
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

from src.data.kepler import download_lightcurve, preprocess_lightcurve


# NASA Exoplanet Archive TAP query for confirmed planets and false positives
# Filters: depth 1000-50000 ppm, period 0.5-30 days (enough transits in Kepler baseline)
_ARCHIVE_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"
    "?table=cumulative"
    "&select=kepid,kepoi_name,koi_period,koi_time0bk,koi_depth,koi_disposition"
    "&where=koi_disposition+like+'{disposition}'"  # caller must URL-encode spaces
    "+and+koi_depth+between+1000+and+50000"
    "+and+koi_period+between+0.5+and+30"
    "&order=koi_depth+desc"
    "&format=csv"
)


def fetch_targets(disposition: str) -> list[dict]:
    """Fetch target list from NASA Exoplanet Archive.

    Args:
        disposition: 'CONFIRMED' or 'FALSE POSITIVE'.

    Returns:
        List of dicts with keys: kepid, kepoi, period, epoch, depth_ppm.
    """
    url = _ARCHIVE_URL.format(disposition=disposition.replace(" ", "+"))
    with urllib.request.urlopen(url, timeout=30) as resp:
        text = resp.read().decode("utf-8")

    targets = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        targets.append({
            "kepid": int(row["kepid"]),
            "kepoi": row["kepoi_name"],
            "period": float(row["koi_period"]),
            "epoch": float(row["koi_time0bk"]),
            "depth_ppm": float(row["koi_depth"]),
        })
    return targets


def subsample_evenly(targets: list[dict], n: int) -> list[dict]:
    """Subsample n targets evenly spaced across the list.

    Since the list is sorted by depth, this gives coverage from deepest
    to shallowest transits rather than clustering at one end.
    """
    if n >= len(targets):
        return targets
    indices = np.linspace(0, len(targets) - 1, n, dtype=int)
    return [targets[i] for i in indices]


def build_dataset(
    n_per_class: int = 100,
    n_points: int = 200,
    output_path: str = "data/kepler_tce.pt",
) -> None:
    """Download, preprocess, and save a Kepler TCE dataset.

    Args:
        n_per_class: Number of targets per class (confirmed / FP).
        n_points: Phase bins per light curve.
        output_path: Where to save the .pt file.
    """
    print("Fetching target lists from NASA Exoplanet Archive...")
    confirmed_all = fetch_targets("CONFIRMED")
    fp_all = fetch_targets("FALSE POSITIVE")
    print(f"  Archive has {len(confirmed_all)} confirmed, {len(fp_all)} FP")

    # Subsample evenly across depth range
    confirmed = subsample_evenly(confirmed_all, n_per_class)
    fp = subsample_evenly(fp_all, n_per_class)
    print(f"  Selected {len(confirmed)} confirmed, {len(fp)} FP")

    # Download and preprocess
    all_phases = []
    all_fluxes = []
    all_fluxes_secondary = []
    all_labels = []
    all_names = []
    all_depths = []

    targets = [(t, 1) for t in confirmed] + [(t, 0) for t in fp]
    n_total = len(targets)
    n_success = 0
    n_fail = 0

    for i, (target, label) in enumerate(targets):
        label_str = "CONF" if label == 1 else "FP  "
        name = target["kepoi"]
        print(
            f"  [{i+1:>3}/{n_total}] {label_str} {name:<12} "
            f"(depth={target['depth_ppm']:>7.0f} ppm) ... ",
            end="", flush=True,
        )

        t0 = time.time()
        try:
            lc = download_lightcurve(target["kepid"])
            phase, primary_flux, secondary_flux = preprocess_lightcurve(
                lc, target["period"], target["epoch"], n_points
            )

            all_phases.append(phase)
            all_fluxes.append(primary_flux)
            all_fluxes_secondary.append(secondary_flux)
            all_labels.append(label)
            all_names.append(name)
            all_depths.append(target["depth_ppm"])
            n_success += 1

            elapsed = time.time() - t0
            print(
                f"OK  ({elapsed:.0f}s, "
                f"p_dip={primary_flux.min():.4f}, "
                f"s_dip={secondary_flux.min():.4f})"
            )

        except Exception as e:
            elapsed = time.time() - t0
            n_fail += 1
            print(f"FAIL ({elapsed:.0f}s): {e}")

    print(f"\nCompleted: {n_success} success, {n_fail} failed")

    # Stack into tensors
    phases = torch.stack(all_phases)                     # (N, n_points)
    fluxes = torch.stack(all_fluxes)                     # (N, n_points)
    fluxes_secondary = torch.stack(all_fluxes_secondary) # (N, n_points)
    labels = torch.tensor(all_labels, dtype=torch.float32)

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    dataset = {
        "phases": phases,
        "fluxes": fluxes,
        "fluxes_secondary": fluxes_secondary,
        "labels": labels,
        "names": all_names,
        "depths_ppm": all_depths,
        "n_points": n_points,
    }
    torch.save(dataset, output_path)
    print(f"Saved dataset to {output_path}")
    print(f"  Phases shape:    {phases.shape}")
    print(f"  Primary shape:   {fluxes.shape}")
    print(f"  Secondary shape: {fluxes_secondary.shape}")
    print(f"  Confirmed: {int(labels.sum())}, FP: {int((1 - labels).sum())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Kepler TCE dataset")
    parser.add_argument("--n-per-class", type=int, default=100)
    parser.add_argument("--n-points", type=int, default=200)
    parser.add_argument("--output", type=str, default="data/kepler_tce.pt")
    args = parser.parse_args()

    build_dataset(args.n_per_class, args.n_points, args.output)
