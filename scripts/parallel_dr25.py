# ═══════════════════════════════════════════
# PRODUCTION MODEL PROTECTION
# NEVER save to src/models/production/
# Save experiments to src/models/ only
# with descriptive version suffix
# ═══════════════════════════════════════════

"""Parallel DR25 download: 5 worker processes, each downloads a shard.

Usage:
    .\\nasaml\\Scripts\\python.exe scripts\\parallel_dr25.py

Writes:
    data/dr25_shard_0.pt ... data/dr25_shard_4.pt   (worker outputs)
    data/kepler_dr25_1k.pt                           (merged dataset)
    data/kepler_dr25_1k_scores.csv                   (ensemble scores)
"""

import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from src.data.build_dataset import (
    fetch_targets,
    subsample_evenly,
    download_for_mission,
)
from src.data.kepler import preprocess_lightcurve

N_PER_CLASS  = 500
SEED         = 99
N_WORKERS    = 5
N_POINTS     = 200
OUTPUT_PT    = "data/kepler_dr25_1k.pt"
OUTPUT_CSV   = "data/kepler_dr25_1k_scores.csv"
SHARD_TMPL   = "data/dr25_shard_{}.pt"
THRESHOLD    = 0.5


# ── Worker (must be module-level for Windows multiprocessing spawn) ────────────

def download_shard(shard_id: int, targets_labels: list, n_points: int, out_path: str):
    """Download one shard of TCEs and save to a .pt file."""
    import sys, time, torch
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.data.build_dataset import download_for_mission
    from src.data.kepler import preprocess_lightcurve

    all_phases, all_fluxes, all_sec, all_oe = [], [], [], []
    all_labels, all_names, all_depths = [], [], []
    all_durations, all_periods, all_smass, all_srad = [], [], [], []

    n = len(targets_labels)
    n_ok = n_fail = 0

    for i, (target, label) in enumerate(targets_labels):
        name = target["name"]
        label_str = "CONF" if label else "FP  "
        print(
            f"[W{shard_id}] [{i+1:>3}/{n}] {label_str} {name:<14} "
            f"(depth={target['depth_ppm']:>7.0f} ppm) ... ",
            end="", flush=True,
        )
        t0 = time.time()
        try:
            lc = download_for_mission(target["mission"], target["id"])
            phase, primary, secondary, odd_even = preprocess_lightcurve(
                lc, target["period"], target["epoch"], n_points
            )
            all_phases.append(phase)
            all_fluxes.append(primary)
            all_sec.append(secondary)
            all_oe.append(odd_even)
            all_labels.append(label)
            all_names.append(name)
            all_depths.append(target["depth_ppm"])
            all_durations.append(target["duration_hours"])
            all_periods.append(target["period_days"] if "period_days" in target else target["period"])
            all_smass.append(target["stellar_mass"])
            all_srad.append(target["stellar_radius"])
            n_ok += 1
            print(f"OK  ({time.time()-t0:.0f}s)")
        except Exception as e:
            n_fail += 1
            print(f"FAIL ({time.time()-t0:.0f}s): {e}")

    if n_ok == 0:
        print(f"[W{shard_id}] No successful downloads.")
        return

    dataset = {
        "phases":            torch.stack(all_phases),
        "fluxes":            torch.stack(all_fluxes),
        "fluxes_secondary":  torch.stack(all_sec),
        "fluxes_odd_even":   torch.stack(all_oe),
        "labels":            torch.tensor(all_labels, dtype=torch.float32),
        "names":             all_names,
        "depths_ppm":        all_depths,
        "duration_hours":    torch.tensor(all_durations, dtype=torch.float32),
        "period_days":       torch.tensor(all_periods,   dtype=torch.float32),
        "stellar_mass":      torch.tensor(all_smass,     dtype=torch.float32),
        "stellar_radius":    torch.tensor(all_srad,      dtype=torch.float32),
        "n_points":          n_points,
        "mission":           "Kepler",
    }
    torch.save(dataset, out_path)
    print(f"[W{shard_id}] Saved {n_ok} TCEs -> {out_path}  ({n_fail} failed)")


# ── Merge shards ──────────────────────────────────────────────────────────────

def merge_shards(shard_paths: list[str], output_path: str) -> dict:
    """Concatenate all shard .pt files into one dataset."""
    shards = [torch.load(p, weights_only=False)
               for p in shard_paths if Path(p).exists()]
    if not shards:
        raise RuntimeError("No shard files found to merge.")

    def cat(key, is_list=False):
        if is_list:
            out = []
            for s in shards:
                out.extend(s[key])
            return out
        return torch.cat([s[key] for s in shards])

    merged = {
        "phases":           cat("phases"),
        "fluxes":           cat("fluxes"),
        "fluxes_secondary": cat("fluxes_secondary"),
        "fluxes_odd_even":  cat("fluxes_odd_even"),
        "labels":           cat("labels"),
        "names":            cat("names",        is_list=True),
        "depths_ppm":       cat("depths_ppm",   is_list=True),
        "duration_hours":   cat("duration_hours"),
        "period_days":      cat("period_days"),
        "stellar_mass":     cat("stellar_mass"),
        "stellar_radius":   cat("stellar_radius"),
        "n_points":         shards[0]["n_points"],
        "mission":          "Kepler",
    }
    torch.save(merged, output_path)
    print(f"Merged {len(merged['names'])} TCEs -> {output_path}")
    return merged


# ── Score ─────────────────────────────────────────────────────────────────────

def score_dataset(d: dict, output_csv: str):
    from src.models.adaptive_classifier import AdaptivePINNClassifier

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Scoring {len(d['names'])} TCEs on {device}...")
    model = AdaptivePINNClassifier(model_dir="src/models/production", device=device)

    phases    = d["phases"]
    primary   = d["fluxes"]
    secondary = d["fluxes_secondary"]
    odd_even  = d["fluxes_odd_even"]
    labels    = d["labels"].numpy().astype(int)
    names     = d["names"]
    depths    = np.array(d["depths_ppm"], dtype=float)
    stellar_r = d["stellar_radius"].numpy()

    results = []
    for i in range(len(names)):
        ph = phases[i].unsqueeze(0).to(device)
        pr = primary[i].unsqueeze(0).to(device)
        sc = secondary[i].unsqueeze(0).to(device)
        oe = odd_even[i].unsqueeze(0).to(device)

        with torch.no_grad():
            p_v10 = float(model.v10(ph, pr, sc, oe).item())
            p_v6b = float(model.v6b(ph, pr, sc, oe).item())

        pred  = int(p_v10 > THRESHOLD and p_v6b > THRESHOLD)
        truth = int(labels[i])
        r_s   = float(stellar_r[i]) if not np.isnan(stellar_r[i]) else 1.0

        if truth == 0 and pred == 1:
            case = "MISSED_PLANET_CANDIDATE"
        elif truth == 1 and pred == 0:
            case = "SUSPICIOUS_CONFIRMATION"
        elif truth == 1 and p_v10 > 0.85 and p_v6b > 0.85:
            case = "high_confidence_planet"
        elif truth == 0 and p_v10 < 0.15 and p_v6b < 0.15:
            case = "high_confidence_fp"
        else:
            case = "agreement"

        results.append({
            "koi_id":        names[i],
            "depth_ppm":     round(float(depths[i]), 1),
            "stellar_r":     round(r_s, 3),
            "truth":         truth,
            "p_v10":         round(p_v10, 4),
            "p_v6b":         round(p_v6b, 4),
            "ensemble_pred": pred,
            "case":          case,
        })

    df = pd.DataFrame(results)
    total  = len(df)
    agree  = (df["truth"] == df["ensemble_pred"]).sum()
    tp     = ((df["truth"] == 1) & (df["ensemble_pred"] == 1)).sum()
    tn     = ((df["truth"] == 0) & (df["ensemble_pred"] == 0)).sum()
    fp_cnt = ((df["truth"] == 0) & (df["ensemble_pred"] == 1)).sum()
    fn     = ((df["truth"] == 1) & (df["ensemble_pred"] == 0)).sum()
    prec   = tp / (tp + fp_cnt) if (tp + fp_cnt) > 0 else 0.0
    rec    = tp / (tp + fn)     if (tp + fn)     > 0 else 0.0
    f1     = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0

    print("=" * 60)
    print("V10+V6b AND ensemble results")
    print("=" * 60)
    print(f"Total: {total}  Acc: {agree/total:.1%}  "
          f"TP={tp} TN={tn} FP={fp_cnt} FN={fn}")
    print(f"Precision: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}")

    for label, key, note in [
        ("MISSED PLANET CANDIDATES", "MISSED_PLANET_CANDIDATE",
         "(archive=FP, model=PLANET)"),
        ("SUSPICIOUS CONFIRMATIONS", "SUSPICIOUS_CONFIRMATION",
         "(archive=CONFIRMED, model=FP)"),
    ]:
        sub = df[df["case"] == key]
        print(f"\n{label}: {len(sub)}  {note}")
        if len(sub) > 0:
            print(sub[["koi_id", "depth_ppm", "stellar_r",
                        "p_v10", "p_v6b"]].to_string(index=False))

    hi_p = (df["case"] == "high_confidence_planet").sum()
    hi_f = (df["case"] == "high_confidence_fp").sum()
    print(f"\nHigh-confidence planets (both >0.85): {hi_p}")
    print(f"High-confidence FPs    (both <0.15): {hi_f}")

    df.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    t0 = time.time()
    print("=" * 60)
    print(f"Parallel DR25 download — {N_WORKERS} workers")
    print("=" * 60)

    # Check for existing shards (resume support)
    shard_paths = [SHARD_TMPL.format(i) for i in range(N_WORKERS)]
    existing_names: set[str] = set()
    for sp in shard_paths:
        if Path(sp).exists():
            try:
                s = torch.load(sp, weights_only=False)
                existing_names.update(s["names"])
            except Exception:
                pass
    if existing_names:
        print(f"Resume: {len(existing_names)} TCEs already in shard files, skipping.")

    # Fetch full catalog
    print("Fetching Kepler catalog from NASA Exoplanet Archive...")
    np.random.seed(SEED)
    confirmed_all = fetch_targets("Kepler", "confirmed")
    fp_all        = fetch_targets("Kepler", "fp")
    print(f"  Archive: {len(confirmed_all)} confirmed, {len(fp_all)} FP")

    confirmed = subsample_evenly(confirmed_all, N_PER_CLASS)
    fp_list   = subsample_evenly(fp_all,        N_PER_CLASS)
    all_targets = [(t, 1) for t in confirmed] + [(t, 0) for t in fp_list]

    # Filter out already-downloaded names
    remaining = [(t, lbl) for t, lbl in all_targets
                 if t["name"] not in existing_names]
    print(f"  Targets: {len(all_targets)} total, {len(remaining)} to download")

    if remaining:
        # Split into N_WORKERS shards
        shards = [remaining[i::N_WORKERS] for i in range(N_WORKERS)]
        jobs = [
            (i, shards[i], N_POINTS, shard_paths[i])
            for i in range(N_WORKERS)
            if shards[i]
        ]
        print(f"  Shard sizes: {[len(s) for s in shards]}")
        print(f"  Launching {len(jobs)} worker processes...")

        with mp.Pool(processes=N_WORKERS) as pool:
            pool.starmap(download_shard, jobs)

    # Merge
    print("\nMerging shards...")
    merged = merge_shards(shard_paths, OUTPUT_PT)

    n_conf = int(merged["labels"].sum().item())
    n_fp   = len(merged["labels"]) - n_conf
    print(f"  Confirmed: {n_conf}  FP: {n_fp}  "
          f"pos_weight: {n_fp/max(n_conf,1):.3f}")

    # Score
    print()
    score_dataset(merged, OUTPUT_CSV)

    print(f"\nTotal wall time: {(time.time()-t0)/60:.1f} min")
