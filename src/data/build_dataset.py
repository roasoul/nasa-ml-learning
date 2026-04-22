"""Download and cache a real Kepler or TESS TCE dataset for training.

Fetches light curves from MAST, preprocesses them using the CLAUDE.md
pipeline, and saves the result as a single .pt file for fast loading.

Usage:
    # Fresh Kepler download
    python -m src.data.build_dataset --n-per-class 1000 --output data/kepler_tce_2000.pt

    # Resume an interrupted download — skips TCEs already present in the file
    python -m src.data.build_dataset --n-per-class 1000 \
        --output data/kepler_tce_2000.pt --resume data/kepler_tce_2000.pt

    # TESS 2-min SPOC download
    python -m src.data.build_dataset --n-per-class 200 \
        --output data/tess_tce_400.pt --mission TESS
"""

import argparse
import csv
import io
import time
import urllib.parse
import urllib.request
from pathlib import Path

import lightkurve as lk
import numpy as np
import torch

from src.data.kepler import download_lightcurve, preprocess_lightcurve


# NASA Exoplanet Archive — Kepler KOI cumulative table
# Filters: depth 1000-50000 ppm, period 0.5-30 days (enough transits in Kepler baseline)
_KEPLER_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"
    "?table=cumulative"
    "&select=kepid,kepoi_name,koi_period,koi_time0bk,koi_depth,koi_disposition,"
    "koi_duration,koi_srad,koi_smass"
    "&where=koi_disposition+like+'{disposition}'"
    "+and+koi_depth+between+1000+and+50000"
    "+and+koi_period+between+0.5+and+30"
    "&order=koi_depth+desc"
    "&format=csv"
)

# NASA Exoplanet Archive TAP service — TESS TOI table.
# tfopwg_disp codes: CP=confirmed, KP=known planet, FP=false positive, PC=candidate.
# The `toi` table has no st_mass column (only st_rad, st_teff, st_logg), so
# stellar mass is defaulted to 1.0 for TESS. pl_tranmid is in full BJD
# (~2459000); SPOC lightkurve.time is BTJD (= BJD - 2457000), so we convert.
_TESS_TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
_TESS_QUERY = (
    "select tid,toi,tfopwg_disp,pl_orbper,pl_tranmid,"
    "pl_trandep,pl_trandurh,st_rad "
    "from toi "
    "where tfopwg_disp in ({dispositions}) "
    "and pl_trandep between 1000 and 50000 "
    "and pl_orbper between 0.5 and 30 "
    "order by pl_trandep desc"
)
_BJD_TO_BTJD = 2457000.0  # SPOC TESS time reference


def _parse_float(val, default: float) -> float:
    """Parse an archive CSV cell; fall back to `default` on NaN/empty."""
    if val is None or val == "" or str(val).lower() == "nan":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def fetch_kepler_targets(disposition: str) -> list[dict]:
    """Fetch Kepler KOI targets from NASA Exoplanet Archive.

    Args:
        disposition: 'CONFIRMED' or 'FALSE POSITIVE'.
    """
    url = _KEPLER_URL.format(disposition=disposition.replace(" ", "+"))
    with urllib.request.urlopen(url, timeout=30) as resp:
        text = resp.read().decode("utf-8")

    targets = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        targets.append({
            "mission": "Kepler",
            "id": int(row["kepid"]),
            "name": row["kepoi_name"],
            "period": float(row["koi_period"]),
            "epoch": float(row["koi_time0bk"]),
            "depth_ppm": float(row["koi_depth"]),
            "duration_hours": _parse_float(row.get("koi_duration"), default=5.0),
            "stellar_mass": _parse_float(row.get("koi_smass"), default=1.0),
            "stellar_radius": _parse_float(row.get("koi_srad"), default=1.0),
        })
    return targets


def fetch_tess_targets(dispositions: list[str]) -> list[dict]:
    """Fetch TESS TOI targets from NASA Exoplanet Archive TAP service.

    Args:
        dispositions: TFOPWG codes, e.g. ['CP', 'KP'] for confirmed,
            ['FP'] for false positives.
    """
    disp_list = ",".join(f"'{d}'" for d in dispositions)
    query = _TESS_QUERY.format(dispositions=disp_list)
    url = f"{_TESS_TAP_URL}?query={urllib.parse.quote(query)}&format=csv"
    with urllib.request.urlopen(url, timeout=60) as resp:
        text = resp.read().decode("utf-8")

    targets = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        period = _parse_float(row.get("pl_orbper"), default=0.0)
        epoch_bjd = _parse_float(row.get("pl_tranmid"), default=0.0)
        if period <= 0 or epoch_bjd <= 0:
            continue  # skip rows missing ephemeris
        # Convert full BJD to BTJD if needed; SPOC lightkurve.time is BTJD.
        epoch = epoch_bjd - _BJD_TO_BTJD if epoch_bjd > 2_400_000 else epoch_bjd
        targets.append({
            "mission": "TESS",
            "id": int(float(row["tid"])),
            "name": f"TOI-{row['toi']}",
            "period": period,
            "epoch": epoch,
            "depth_ppm": _parse_float(row.get("pl_trandep"), default=0.0),
            "duration_hours": _parse_float(row.get("pl_trandurh"), default=5.0),
            "stellar_mass": 1.0,  # not present in toi table
            "stellar_radius": _parse_float(row.get("st_rad"), default=1.0),
        })
    return targets


def download_tess_lightcurve(tic_id: int) -> lk.LightCurve:
    """Download and stitch all TESS SPOC 2-min sectors for a target.

    Matches the Kepler download pipeline: stitch all sectors, remove NaNs,
    clip upward outliers only (downward clipping would delete transit dips).
    """
    search = lk.search_lightcurve(
        f"TIC {tic_id}", mission="TESS", author="SPOC", exptime=120,
    )
    if len(search) == 0:
        raise ValueError(f"No TESS SPOC 2-min light curves for TIC {tic_id}")
    lc = search.download_all().stitch()
    return lc.remove_nans().remove_outliers(sigma_lower=float("inf"), sigma_upper=5)


def fetch_targets(mission: str, label: str) -> list[dict]:
    """Dispatch catalog fetch to the right mission.

    Args:
        mission: 'Kepler' or 'TESS'.
        label: 'confirmed' or 'fp'.
    """
    if mission == "Kepler":
        disp = "CONFIRMED" if label == "confirmed" else "FALSE POSITIVE"
        return fetch_kepler_targets(disp)
    if mission == "TESS":
        dispositions = ["CP", "KP"] if label == "confirmed" else ["FP"]
        return fetch_tess_targets(dispositions)
    raise ValueError(f"Unknown mission: {mission!r}")


def download_for_mission(mission: str, target_id: int) -> lk.LightCurve:
    """Dispatch LC download to the right mission."""
    if mission == "Kepler":
        return download_lightcurve(target_id)
    if mission == "TESS":
        return download_tess_lightcurve(target_id)
    raise ValueError(f"Unknown mission: {mission!r}")


def subsample_evenly(targets: list[dict], n: int) -> list[dict]:
    """Subsample n targets evenly spaced across the list.

    Since the list is sorted by depth, this gives coverage from deepest
    to shallowest transits rather than clustering at one end.
    """
    if n >= len(targets):
        return targets
    indices = np.linspace(0, len(targets) - 1, n, dtype=int)
    return [targets[i] for i in indices]


def _load_resume(path: str, n_points: int) -> tuple[dict | None, set[str]]:
    """Load an existing dataset for resume, or return (None, empty) if absent.

    Raises ValueError if the existing file's n_points doesn't match the
    requested n_points — mixing bin counts would corrupt the tensors.
    """
    if not Path(path).exists():
        print(f"  Resume file {path} does not exist — starting fresh.")
        return None, set()
    existing = torch.load(path, weights_only=False)
    existing_np = existing.get("n_points", n_points)
    if existing_np != n_points:
        raise ValueError(
            f"n_points mismatch: existing file has {existing_np}, "
            f"requested {n_points}. Refuse to mix."
        )
    names = set(existing["names"])
    print(f"  Resuming from {path}: {len(names)} TCEs already downloaded.")
    return existing, names


def build_dataset(
    n_per_class: int = 100,
    n_points: int = 200,
    output_path: str = "data/kepler_tce.pt",
    mission: str = "Kepler",
    seed: int = 42,
    resume: str | None = None,
) -> None:
    """Download, preprocess, and save a Kepler or TESS TCE dataset.

    Args:
        n_per_class: Number of targets per class (confirmed / FP).
        n_points: Phase bins per light curve.
        output_path: Where to save the .pt file.
        mission: 'Kepler' (default) or 'TESS'.
        seed: Random seed for reproducibility.
        resume: Optional path to an existing .pt dataset to extend.
            Targets whose names are already present are skipped; new
            downloads are appended and the full dataset is re-saved.
    """
    np.random.seed(seed)

    existing, existing_names = (None, set())
    if resume is not None:
        existing, existing_names = _load_resume(resume, n_points)

    print(f"Fetching {mission} target lists from NASA Exoplanet Archive...")
    confirmed_all = fetch_targets(mission, "confirmed")
    fp_all = fetch_targets(mission, "fp")
    print(f"  Archive has {len(confirmed_all)} confirmed, {len(fp_all)} FP")

    confirmed = subsample_evenly(confirmed_all, n_per_class)
    fp = subsample_evenly(fp_all, n_per_class)
    print(f"  Selected {len(confirmed)} confirmed, {len(fp)} FP")

    targets = [(t, 1) for t in confirmed] + [(t, 0) for t in fp]
    remaining = [(t, lbl) for (t, lbl) in targets if t["name"] not in existing_names]
    if existing_names:
        print(
            f"  Skipping {len(targets) - len(remaining)} already-downloaded; "
            f"{len(remaining)} remaining."
        )

    all_phases, all_fluxes, all_fluxes_secondary, all_fluxes_odd_even = [], [], [], []
    all_labels, all_names, all_depths, all_durations = [], [], [], []
    all_periods, all_stellar_mass, all_stellar_radius = [], [], []

    n_total_remain = len(remaining)
    n_success = 0
    n_fail = 0

    for i, (target, label) in enumerate(remaining):
        label_str = "CONF" if label == 1 else "FP  "
        name = target["name"]
        print(
            f"  [{i+1:>3}/{n_total_remain}] {label_str} {name:<14} "
            f"(depth={target['depth_ppm']:>7.0f} ppm) ... ",
            end="", flush=True,
        )

        t0 = time.time()
        try:
            lc = download_for_mission(mission, target["id"])
            phase, primary_flux, secondary_flux, odd_even_diff = preprocess_lightcurve(
                lc, target["period"], target["epoch"], n_points
            )

            all_phases.append(phase)
            all_fluxes.append(primary_flux)
            all_fluxes_secondary.append(secondary_flux)
            all_fluxes_odd_even.append(odd_even_diff)
            all_labels.append(label)
            all_names.append(name)
            all_depths.append(target["depth_ppm"])
            all_durations.append(target["duration_hours"])
            all_periods.append(target["period"])
            all_stellar_mass.append(target["stellar_mass"])
            all_stellar_radius.append(target["stellar_radius"])
            n_success += 1

            elapsed = time.time() - t0
            oe_amp = float(odd_even_diff[95:105].abs().max())
            print(
                f"OK  ({elapsed:.0f}s, "
                f"p_dip={primary_flux.min():.4f}, "
                f"s_dip={secondary_flux.min():.4f}, "
                f"oe_amp={oe_amp:.4f})"
            )

        except Exception as e:
            elapsed = time.time() - t0
            n_fail += 1
            print(f"FAIL ({elapsed:.0f}s): {e}")

    print(f"\nDownload round: {n_success} success, {n_fail} failed")

    # Stack this round (may be empty if everything was already present)
    if n_success > 0:
        new_phases = torch.stack(all_phases)
        new_fluxes = torch.stack(all_fluxes)
        new_fluxes_secondary = torch.stack(all_fluxes_secondary)
        new_fluxes_odd_even = torch.stack(all_fluxes_odd_even)
        new_labels = torch.tensor(all_labels, dtype=torch.float32)
        new_durations = torch.tensor(all_durations, dtype=torch.float32)
        new_periods = torch.tensor(all_periods, dtype=torch.float32)
        new_stellar_mass = torch.tensor(all_stellar_mass, dtype=torch.float32)
        new_stellar_radius = torch.tensor(all_stellar_radius, dtype=torch.float32)

    if existing is not None and n_success > 0:
        phases = torch.cat([existing["phases"], new_phases])
        fluxes = torch.cat([existing["fluxes"], new_fluxes])
        fluxes_secondary = torch.cat([existing["fluxes_secondary"], new_fluxes_secondary])
        fluxes_odd_even = torch.cat([existing["fluxes_odd_even"], new_fluxes_odd_even])
        labels = torch.cat([existing["labels"], new_labels])
        names = list(existing["names"]) + all_names
        depths = list(existing["depths_ppm"]) + all_depths
        durations = torch.cat([existing["duration_hours"], new_durations])
        periods = torch.cat([existing["period_days"], new_periods])
        stellar_mass = torch.cat([existing["stellar_mass"], new_stellar_mass])
        stellar_radius = torch.cat([existing["stellar_radius"], new_stellar_radius])
    elif existing is not None:
        # Resume requested but no new successes — re-save existing as-is
        phases = existing["phases"]
        fluxes = existing["fluxes"]
        fluxes_secondary = existing["fluxes_secondary"]
        fluxes_odd_even = existing["fluxes_odd_even"]
        labels = existing["labels"]
        names = list(existing["names"])
        depths = list(existing["depths_ppm"])
        durations = existing["duration_hours"]
        periods = existing["period_days"]
        stellar_mass = existing["stellar_mass"]
        stellar_radius = existing["stellar_radius"]
    else:
        if n_success == 0:
            print("No successful downloads and no existing file — nothing to save.")
            return
        phases = new_phases
        fluxes = new_fluxes
        fluxes_secondary = new_fluxes_secondary
        fluxes_odd_even = new_fluxes_odd_even
        labels = new_labels
        names = all_names
        depths = all_depths
        durations = new_durations
        periods = new_periods
        stellar_mass = new_stellar_mass
        stellar_radius = new_stellar_radius

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    dataset = {
        "phases": phases,
        "fluxes": fluxes,
        "fluxes_secondary": fluxes_secondary,
        "fluxes_odd_even": fluxes_odd_even,
        "labels": labels,
        "names": names,
        "depths_ppm": depths,
        "duration_hours": durations,
        "period_days": periods,
        "stellar_mass": stellar_mass,
        "stellar_radius": stellar_radius,
        "n_points": n_points,
        "mission": mission,
    }
    torch.save(dataset, output_path)
    print(f"Saved dataset to {output_path}")
    print(f"  Phases shape:    {phases.shape}")
    print(f"  Primary shape:   {fluxes.shape}")
    print(f"  Secondary shape: {fluxes_secondary.shape}")
    print(f"  Odd/even shape:  {fluxes_odd_even.shape}")

    # Class imbalance report — always print after download completes.
    n_total = len(labels)
    n_conf = int(labels.sum().item())
    n_fp = n_total - n_conf
    pos_weight = n_fp / max(n_conf, 1)
    conf_pct = 100.0 * n_conf / max(n_total, 1)
    fp_pct = 100.0 * n_fp / max(n_total, 1)
    print("\nClass balance:")
    print(f"  Total TCEs:         {n_total}")
    print(f"  Confirmed planets:  {n_conf} ({conf_pct:.1f}%)")
    print(f"  False positives:    {n_fp} ({fp_pct:.1f}%)")
    print(f"  Recommended pos_weight (FP/confirmed): {pos_weight:.3f}")
    print("    Use with nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a Kepler or TESS TCE dataset")
    parser.add_argument("--n-per-class", type=int, default=100,
                        help="Targets per class (confirmed / FP).")
    parser.add_argument("--n-points", type=int, default=200,
                        help="Phase bins after folding.")
    parser.add_argument("--output", type=str, default="data/kepler_tce.pt",
                        help="Output .pt path.")
    parser.add_argument("--mission", type=str, default="Kepler",
                        choices=["Kepler", "TESS"],
                        help="Mission catalog and LC source.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Optional existing .pt to extend; already-downloaded "
                             "TCEs are skipped and new ones are appended.")
    args = parser.parse_args()

    build_dataset(
        n_per_class=args.n_per_class,
        n_points=args.n_points,
        output_path=args.output,
        mission=args.mission,
        seed=args.seed,
        resume=args.resume,
    )
