# ═══════════════════════════════════════════
# PRODUCTION MODEL PROTECTION
# NEVER save to src/models/production/
# Save experiments to src/models/ only
# with descriptive version suffix
# ═══════════════════════════════════════════

"""Batch 2: next 1000 Kepler DR25 TCEs — expanded depth/period filters.

Expands beyond batch 1's narrow filters (1000-50000 ppm, 0.5-30d) to:
  depth  100-200000 ppm   (adds shallow transits and deep EBs)
  period 0.5-400 days     (adds Jupiter-analog long-period candidates)

Skips the 850 TCEs already in data/kepler_dr25_1k.pt.
5 parallel workers, 500 confirmed + 500 FP = 1000 new TCEs.

Usage:
    .\\nasaml\\Scripts\\python.exe scripts\\parallel_dr25_b2.py
"""

import csv
import io
import multiprocessing as mp
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from src.data.build_dataset import subsample_evenly, download_for_mission
from src.data.kepler import preprocess_lightcurve

# ── Config ────────────────────────────────────────────────────────────────────
N_PER_CLASS  = 500
SEED         = 42               # different from batch 1 (seed=99)
N_WORKERS    = 5
N_POINTS     = 200
RESUME_PT    = "data/kepler_dr25_1k.pt"          # skip already-downloaded
OUTPUT_PT    = "data/kepler_dr25_2k.pt"
OUTPUT_CSV   = "data/kepler_dr25_2k_scores.csv"
SHARD_TMPL   = "data/dr25_b2_shard_{}.pt"
THRESHOLD    = 0.5

# Expanded catalog URL — wider depth and period than batch 1
_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"
    "?table=cumulative"
    "&select=kepid,kepoi_name,koi_period,koi_time0bk,koi_depth,koi_disposition,"
    "koi_duration,koi_srad,koi_smass"
    "&where=koi_disposition+like+'{disp}'"
    "+and+koi_depth+between+100+and+200000"
    "+and+koi_period+between+0.5+and+400"
    "&order=koi_depth+desc"
    "&format=csv"
)


def _parse_float(val, default: float) -> float:
    if val is None or val == "" or str(val).lower() == "nan":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def fetch_kepler_targets_expanded(disposition: str) -> list[dict]:
    url = _URL.format(disp=disposition.replace(" ", "+"))
    with urllib.request.urlopen(url, timeout=30) as resp:
        text = resp.read().decode("utf-8")
    targets = []
    for row in csv.DictReader(io.StringIO(text)):
        targets.append({
            "mission":        "Kepler",
            "id":             int(row["kepid"]),
            "name":           row["kepoi_name"],
            "period":         float(row["koi_period"]),
            "epoch":          float(row["koi_time0bk"]),
            "depth_ppm":      float(row["koi_depth"]),
            "duration_hours": _parse_float(row.get("koi_duration"), 5.0),
            "stellar_mass":   _parse_float(row.get("koi_smass"),    1.0),
            "stellar_radius": _parse_float(row.get("koi_srad"),     1.0),
        })
    return targets


# ── Worker ────────────────────────────────────────────────────────────────────

def download_shard(shard_id: int, targets_labels: list, n_points: int, out_path: str):
    import sys, time, torch
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.data.build_dataset import download_for_mission
    from src.data.kepler import preprocess_lightcurve

    all_phases, all_fluxes, all_sec, all_oe = [], [], [], []
    all_labels, all_names, all_depths      = [], [], []
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
            all_phases.append(phase);  all_fluxes.append(primary)
            all_sec.append(secondary); all_oe.append(odd_even)
            all_labels.append(label);  all_names.append(name)
            all_depths.append(target["depth_ppm"])
            all_durations.append(target["duration_hours"])
            all_periods.append(target["period"])
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

    torch.save({
        "phases":           torch.stack(all_phases),
        "fluxes":           torch.stack(all_fluxes),
        "fluxes_secondary": torch.stack(all_sec),
        "fluxes_odd_even":  torch.stack(all_oe),
        "labels":           torch.tensor(all_labels, dtype=torch.float32),
        "names":            all_names,
        "depths_ppm":       all_depths,
        "duration_hours":   torch.tensor(all_durations, dtype=torch.float32),
        "period_days":      torch.tensor(all_periods,   dtype=torch.float32),
        "stellar_mass":     torch.tensor(all_smass,     dtype=torch.float32),
        "stellar_radius":   torch.tensor(all_srad,      dtype=torch.float32),
        "n_points":         n_points,
        "mission":          "Kepler",
    }, out_path)
    print(f"[W{shard_id}] Saved {n_ok} TCEs -> {out_path}  ({n_fail} failed)")


# ── Merge ─────────────────────────────────────────────────────────────────────

def merge_shards(shard_paths: list[str], output_path: str) -> dict:
    shards = [torch.load(p, weights_only=False)
              for p in shard_paths if Path(p).exists()]
    if not shards:
        raise RuntimeError("No shard files found.")

    def cat(key, is_list=False):
        if is_list:
            out = []
            for s in shards: out.extend(s[key])
            return out
        return torch.cat([s[key] for s in shards])

    merged = {
        "phases":           cat("phases"),
        "fluxes":           cat("fluxes"),
        "fluxes_secondary": cat("fluxes_secondary"),
        "fluxes_odd_even":  cat("fluxes_odd_even"),
        "labels":           cat("labels"),
        "names":            cat("names",      is_list=True),
        "depths_ppm":       cat("depths_ppm", is_list=True),
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

    phases = d["phases"]; primary = d["fluxes"]
    secondary = d["fluxes_secondary"]; odd_even = d["fluxes_odd_even"]
    labels = d["labels"].numpy().astype(int); names = d["names"]
    depths = np.array(d["depths_ppm"], dtype=float)
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
        if   truth == 0 and pred == 1:              case = "MISSED_PLANET_CANDIDATE"
        elif truth == 1 and pred == 0:              case = "SUSPICIOUS_CONFIRMATION"
        elif truth == 1 and p_v10 > 0.85 and p_v6b > 0.85: case = "high_confidence_planet"
        elif truth == 0 and p_v10 < 0.15 and p_v6b < 0.15: case = "high_confidence_fp"
        else:                                       case = "agreement"
        results.append({"koi_id": names[i], "depth_ppm": round(float(depths[i]),1),
                         "stellar_r": round(r_s,3), "truth": truth,
                         "p_v10": round(p_v10,4), "p_v6b": round(p_v6b,4),
                         "ensemble_pred": pred, "case": case})

    df = pd.DataFrame(results)
    total = len(df)
    tp = ((df["truth"]==1)&(df["ensemble_pred"]==1)).sum()
    tn = ((df["truth"]==0)&(df["ensemble_pred"]==0)).sum()
    fp_c = ((df["truth"]==0)&(df["ensemble_pred"]==1)).sum()
    fn = ((df["truth"]==1)&(df["ensemble_pred"]==0)).sum()
    prec = tp/(tp+fp_c) if (tp+fp_c)>0 else 0.0
    rec  = tp/(tp+fn)   if (tp+fn)>0   else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    agree = (df["truth"]==df["ensemble_pred"]).sum()

    print("=" * 60)
    print("V10+V6b AND ensemble — batch 2 results")
    print("=" * 60)
    print(f"Total: {total}  Acc: {agree/total:.1%}  TP={tp} TN={tn} FP={fp_c} FN={fn}")
    print(f"Precision: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}")

    missed = df[df["case"]=="MISSED_PLANET_CANDIDATE"]
    susp   = df[df["case"]=="SUSPICIOUS_CONFIRMATION"]
    print(f"\nMISSED PLANET CANDIDATES: {len(missed)}  (archive=FP, model=PLANET)")
    print(f"SUSPICIOUS CONFIRMATIONS: {len(susp)}  (archive=CONFIRMED, model=FP)")
    print(f"High-confidence planets (both >0.85): {(df['case']=='high_confidence_planet').sum()}")
    print(f"High-confidence FPs    (both <0.15): {(df['case']=='high_confidence_fp').sum()}")

    print("\nTop 20 missed planet candidates (by p_v10):")
    top20 = missed.sort_values("p_v10", ascending=False).head(20)
    print(top20[["koi_id","depth_ppm","stellar_r","p_v10","p_v6b"]].to_string(index=False))

    df.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    t0 = time.time()

    print("=" * 60)
    print(f"Parallel DR25 batch 2 — {N_WORKERS} workers, expanded filters")
    print("  depth: 100-200000 ppm | period: 0.5-400 days")
    print("=" * 60)

    # Load already-downloaded names to skip
    skip_names: set[str] = set()
    shard_paths = [SHARD_TMPL.format(i) for i in range(N_WORKERS)]
    if Path(RESUME_PT).exists():
        prev = torch.load(RESUME_PT, weights_only=False)
        skip_names.update(prev["names"])
        print(f"Skipping {len(skip_names)} TCEs already in {RESUME_PT}")
    for sp in shard_paths:
        if Path(sp).exists():
            try:
                s = torch.load(sp, weights_only=False)
                skip_names.update(s["names"])
            except Exception:
                pass

    print("Fetching expanded Kepler catalog...")
    np.random.seed(SEED)
    confirmed_all = fetch_kepler_targets_expanded("CONFIRMED")
    fp_all        = fetch_kepler_targets_expanded("FALSE POSITIVE")
    print(f"  Archive (expanded): {len(confirmed_all)} confirmed, {len(fp_all)} FP")

    confirmed = subsample_evenly(confirmed_all, N_PER_CLASS)
    fp_list   = subsample_evenly(fp_all,        N_PER_CLASS)
    all_targets = [(t, 1) for t in confirmed] + [(t, 0) for t in fp_list]

    remaining = [(t, lbl) for t, lbl in all_targets if t["name"] not in skip_names]
    print(f"  Selected {len(all_targets)} targets, {len(remaining)} new to download")

    if remaining:
        shards = [remaining[i::N_WORKERS] for i in range(N_WORKERS)]
        jobs = [(i, shards[i], N_POINTS, shard_paths[i])
                for i in range(N_WORKERS) if shards[i]]
        print(f"  Shard sizes: {[len(s) for s in shards]}")
        print(f"  Launching {len(jobs)} workers...")
        with mp.Pool(processes=N_WORKERS) as pool:
            pool.starmap(download_shard, jobs)

    print("\nMerging shards...")
    merged = merge_shards(shard_paths, OUTPUT_PT)
    n_conf = int(merged["labels"].sum().item())
    n_fp   = len(merged["labels"]) - n_conf
    print(f"  Confirmed: {n_conf}  FP: {n_fp}  pos_weight: {n_fp/max(n_conf,1):.3f}")

    print()
    score_dataset(merged, OUTPUT_CSV)
    print(f"\nTotal wall time: {(time.time()-t0)/60:.1f} min")
