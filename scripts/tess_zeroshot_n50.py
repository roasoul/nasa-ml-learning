"""N=50 TESS zero-shot validation of V10 in discovery mode.

Expands the paper's Exp 3 (N=8, 100% recall, 80% prec) to ~50 targets
before enabling discovery mode on unknown FFI stars.

Mode: discovery — predict PLANET if p_v10 > 0.4
Model: src/models/production/v10_f1861.pt  (READ-ONLY production)
Pipeline: identical to Exp 3 (flatten sigma_upper=4 sigma_lower=10 ->
normalize -> fold -> scale [-pi, pi] -> subtract 1.0 -> 200-bin resample).

Target list:
    Confirmed planets — from user prompt; epochs auto-fetched from
        NASA Archive TAP `toi` table (matched by host + period).
        TOI-620.01 skipped (malformed period in prompt).
    Known FPs / EBs — Prsa et al. 2022 TESS EB catalog, J/ApJS/258/16
        via Vizier (replaces the user's J/AJ/156/234 which is the
        Kirk Kepler EB catalog, not TESS).

Outputs:
    data/tess_zeroshot_n50.csv           per-target table
    notebooks/figures/tess_zeroshot_n50.png
"""

# ════════════════════════════════════════════════════════
# PRODUCTION MODEL PROTECTION
# Loads READ-ONLY from src/models/production/v10_f1861.pt
# No model weights are saved by this script.
# ════════════════════════════════════════════════════════

import csv
import io
import math
import sys
import urllib.parse
import urllib.request
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import torch
import matplotlib.pyplot as plt

import lightkurve as lk
from astroquery.vizier import Vizier

from src.models.taylor_cnn_v10 import TaylorCNNv10


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "src/models/production/v10_f1861.pt"
THRESHOLD = 0.40

FIGDIR = Path("notebooks/figures")
DATADIR = Path("data")
FIGDIR.mkdir(parents=True, exist_ok=True)
DATADIR.mkdir(parents=True, exist_ok=True)

TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
BJD_TO_BTJD = 2457000.0


# Requested TOI hosts (user's period hints were unreliable, so we take
# ALL CP/KP rows per host from the TAP `toi` table). TOI-620 omitted
# (malformed period in prompt).
REQUESTED_HOSTS = [
    "132", "824", "1431", "560", "172", "125", "136",
    "260", "261", "270", "431", "451", "519",
    "700", "736", "776",
]


def _parse_float(val, default=float("nan")):
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def fetch_toi_table() -> list[dict]:
    """Pull CP/KP/PC/APC rows from the NASA Archive TAP `toi` table."""
    query = (
        "select toi, tid, tfopwg_disp, pl_orbper, pl_tranmid, "
        "pl_trandep, st_rad from toi "
        "where tfopwg_disp in ('CP','KP','PC','APC')"
    )
    url = f"{TAP_URL}?query={urllib.parse.quote(query)}&format=csv"
    with urllib.request.urlopen(url, timeout=90) as r:
        text = r.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def planets_for_host(toi_rows, host: str) -> list[dict]:
    """Return every CP/KP row under a TOI host (one entry per planet)."""
    out = []
    for r in toi_rows:
        if r["toi"].split(".")[0] != host:
            continue
        if r["tfopwg_disp"] not in ("CP", "KP"):
            continue
        period = _parse_float(r["pl_orbper"])
        epoch = _parse_float(r["pl_tranmid"])
        tid = _parse_float(r["tid"])
        if math.isnan(period) or math.isnan(epoch) or math.isnan(tid):
            continue
        if period < 0.3 or period > 30:
            continue
        if epoch > 2_400_000:
            epoch -= BJD_TO_BTJD
        out.append({
            "toi": r["toi"],
            "period": period,
            "epoch": epoch,
            "tid": int(tid),
            "depth_ppm": _parse_float(r["pl_trandep"], 0.0),
        })
    return out


def fetch_tess_ebs_prsa(n_target: int = 30) -> list[dict]:
    """Fetch TESS EBs from Prsa+2022 (J/ApJS/258/16) via Vizier.

    The Kirk 2018 catalog (J/AJ/156/234) the user referenced is a
    Kepler EB catalog with no TESS coverage. Prsa+2022 is the
    canonical TESS EB catalog (~4584 systems from Sectors 1–26).
    """
    Vizier.ROW_LIMIT = -1
    v = Vizier(row_limit=-1)
    # TESS EB catalog — columns vary by Vizier edition;
    # we fetch all then resolve.
    cats = v.get_catalogs("J/ApJS/258/16")
    if not cats:
        return []
    tbl = cats[0]
    colnames = list(tbl.colnames)
    # Typical columns: TIC, Per, BJD0 (epoch), morph, ...
    tic_col = next((c for c in colnames if c.upper() == "TIC"), None)
    per_col = next((c for c in ["Per", "Period", "P0"] if c in colnames), None)
    ep_col = next(
        (c for c in ["BJD0", "T0", "Epoch", "BJDref"] if c in colnames), None,
    )
    print(f"  Prsa catalog columns: {colnames[:15]}{'...' if len(colnames) > 15 else ''}")
    print(f"  Using: TIC={tic_col!r}  Per={per_col!r}  Epoch={ep_col!r}")
    if not all([tic_col, per_col, ep_col]):
        return []

    ebs = []
    for row in tbl:
        try:
            tic = int(row[tic_col])
            period = float(row[per_col])
            epoch = float(row[ep_col])
        except (ValueError, TypeError, KeyError):
            continue
        if period <= 0.5 or period > 30:
            continue  # same range as V10 training distribution
        if epoch > 2_400_000:
            epoch -= BJD_TO_BTJD
        ebs.append({"tic": tic, "period": period, "epoch": epoch})
    return ebs[:n_target]


def download_tess_lc(tic_id: int) -> lk.LightCurve:
    search = lk.search_lightcurve(
        f"TIC {tic_id}", mission="TESS", author="SPOC", exptime=120,
    )
    if len(search) == 0:
        raise RuntimeError(f"no SPOC 2-min LC for TIC {tic_id}")
    lc = search.download_all().stitch()
    lc = lc.remove_nans()
    lc = lc.remove_outliers(sigma_upper=4, sigma_lower=10)
    lc = lc.flatten(window_length=401)
    return lc.normalize()


def fold_resample(lc, period: float, epoch: float, n: int = 200) -> np.ndarray:
    folded = lc.fold(period=period, epoch_time=epoch)
    ph = folded.phase.value.astype(np.float32)
    fl = folded.flux.value.astype(np.float32) - 1.0
    order = np.argsort(ph)
    ph = ph[order]
    fl = fl[order]
    ph_scaled = ph * 2 * math.pi
    bins = np.linspace(-math.pi, math.pi, n + 1)
    digit = np.digitize(ph_scaled, bins) - 1
    out = np.zeros(n, dtype=np.float32)
    cnt = np.zeros(n, dtype=np.int32)
    for p, f in zip(digit, fl):
        if 0 <= p < n:
            out[p] += f
            cnt[p] += 1
    mask = cnt > 0
    out[mask] /= cnt[mask]
    if not mask.all():
        idx = np.arange(n)
        out = np.interp(idx, idx[mask], out[mask])
    return out.astype(np.float32)


def preprocess(tic_id: int, period: float, epoch: float):
    lc = download_tess_lc(tic_id)
    primary = fold_resample(lc, period, epoch)
    secondary = fold_resample(lc, period, epoch + period / 2)
    double = fold_resample(lc, period * 2, epoch)
    half = 100
    oe_half = double[:half] - double[half:]
    oe = np.concatenate([oe_half, oe_half]).astype(np.float32)
    phase = np.linspace(-math.pi, math.pi, 200, dtype=np.float32)
    return phase, primary, secondary, oe


def compute_snr(primary: np.ndarray) -> float:
    """Depth / out-of-transit noise std."""
    depth = float(np.abs(np.min(primary)))
    phase = np.linspace(-math.pi, math.pi, 200)
    oot = np.abs(phase) > 0.6 * math.pi
    noise = float(np.std(primary[oot])) if oot.any() else float(np.std(primary))
    return 0.0 if noise < 1e-6 else depth / noise


def infer(model, phase, primary, secondary, oe) -> float:
    with torch.no_grad():
        t = lambda a: torch.tensor(a).unsqueeze(0).to(DEVICE)
        return float(
            model(t(phase), t(primary), t(secondary), t(oe)).squeeze().cpu().item()
        )


def main():
    print("="*90)
    print("TESS zero-shot N=50 — V10 discovery mode (p > 0.40)")
    print(f"Model: {MODEL_PATH}")
    print("="*90)

    # 1. Resolve planet epochs via TAP
    print("\nStep 1/4 — Fetch TOI table from NASA Archive TAP...")
    toi_rows = fetch_toi_table()
    print(f"  {len(toi_rows)} TOI rows fetched")

    planets = []
    for host in REQUESTED_HOSTS:
        rows = planets_for_host(toi_rows, host)
        if not rows:
            print(f"  [no CP/KP rows] TOI-{host}")
            continue
        for p in rows:
            planets.append({
                "name": f"TOI-{p['toi']}",
                "tic": p["tid"],
                "period": p["period"],
                "epoch": p["epoch"],
                "truth": "PLNT",
            })
    print(f"  Resolved {len(planets)} planet rows from "
          f"{len(REQUESTED_HOSTS)} hosts")

    # 2. Fetch TESS EB catalog from Vizier
    print("\nStep 2/4 — Fetch TESS EB catalog (Prsa+2022) from Vizier...")
    try:
        eb_rows = fetch_tess_ebs_prsa(n_target=30)
        print(f"  {len(eb_rows)} EBs selected")
    except Exception as e:
        print(f"  Vizier fetch failed: {e}")
        eb_rows = []

    ebs = [
        {"name": f"EB TIC {eb['tic']}", "tic": eb["tic"],
         "period": eb["period"], "epoch": eb["epoch"], "truth": "FP"}
        for eb in eb_rows
    ]
    targets = planets + ebs
    print(f"\nTotal targets: {len(targets)} "
          f"({len(planets)} planets + {len(ebs)} EBs)")

    # 3. Load V10
    print("\nStep 3/4 — Load V10...")
    model = TaylorCNNv10(init_amplitude=0.01).to(DEVICE)
    ck = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    print(f"  Loaded  (device={DEVICE})")

    # 4. Run inference
    print("\nStep 4/4 — Download + preprocess + infer")
    print("-"*90)
    header = f"  {'Name':<22} {'TIC':>11} {'Period':>8} {'Depth':>9} {'SNR':>6} {'p_v10':>6} {'Pred':>5} {'Truth':>5}"
    print(header)
    print("-"*90)

    rows = []
    for i, t in enumerate(targets, 1):
        name = t["name"]
        tic = t["tic"]
        period = t["period"]
        epoch = t["epoch"]
        truth = t["truth"]
        try:
            phase, primary, secondary, oe = preprocess(tic, period, epoch)
        except Exception as e:
            msg = f"{type(e).__name__}: {str(e)[:50]}"
            print(f"  [{i:>2}/{len(targets)} SKIP] {name:<20}  {msg}")
            rows.append({
                "name": name, "tic": tic, "period": period, "epoch": epoch,
                "truth": truth, "skipped": True, "reason": msg,
            })
            continue
        snr = compute_snr(primary)
        depth_ppm = float(np.abs(np.min(primary))) * 1e6
        if snr < 5:
            print(f"  [{i:>2}/{len(targets)} SKIP] {name:<20}  SNR={snr:.1f} < 5")
            rows.append({
                "name": name, "tic": tic, "period": period, "epoch": epoch,
                "truth": truth, "snr": snr, "depth_ppm": depth_ppm,
                "skipped": True, "reason": f"low SNR {snr:.1f}",
            })
            continue
        prob = infer(model, phase, primary, secondary, oe)
        pred = "PLNT" if prob > THRESHOLD else "FP"
        ok = pred == truth
        tag = "OK" if ok else "!"
        print(f"  {name:<22} {tic:>11} {period:>7.3f}d {depth_ppm:>7.0f}ppm {snr:>6.1f} {prob:>6.3f} {pred:>5} {truth:>5}  {tag}")
        rows.append({
            "name": name, "tic": tic, "period": period, "epoch": epoch,
            "depth_ppm": depth_ppm, "snr": snr,
            "prob": prob, "pred": pred, "truth": truth, "ok": ok,
            "skipped": False,
        })

    # 5. Save CSV
    csv_path = DATADIR / "tess_zeroshot_n50.csv"
    fieldnames = [
        "name", "tic", "period", "epoch", "depth_ppm", "snr",
        "prob", "pred", "truth", "ok", "skipped", "reason",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nSaved {csv_path}")

    # 6. Metrics
    processed = [r for r in rows if not r.get("skipped")]
    plist = [r for r in processed if r["truth"] == "PLNT"]
    flist = [r for r in processed if r["truth"] == "FP"]
    tp = sum(1 for r in plist if r["ok"])
    tn = sum(1 for r in flist if r["ok"])
    fn = len(plist) - tp
    fp_ct = len(flist) - tn
    recall = tp / len(plist) if plist else 0.0
    prec = tp / (tp + fp_ct) if (tp + fp_ct) else 0.0
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) else 0.0

    print("\n" + "="*90)
    print("SUMMARY")
    print("="*90)
    print(f"N processed : {len(processed)}  "
          f"(planets {len(plist)}, FPs {len(flist)}, "
          f"skipped {len(rows) - len(processed)})")
    print(f"Confusion   : TP={tp}  TN={tn}  FP={fp_ct}  FN={fn}")
    print(f"Recall      : {tp}/{len(plist)} = {recall:.1%}  "
          f"(paper Exp 3 N=8: 100%)")
    print(f"Precision   : {prec:.1%}  (paper Exp 3 N=8: 80%)")
    print(f"F1          : {f1:.3f}  (paper Exp 3 N=8: 0.889)")

    caught = [r["snr"] for r in plist if r["ok"]]
    missed = [(r["name"], r["snr"], r["prob"]) for r in plist if not r["ok"]]
    if caught:
        print(f"SNR caught  : median={np.median(caught):.1f}  "
              f"range [{min(caught):.1f}, {max(caught):.1f}]")
    if missed:
        print("Missed planets (FN):")
        for nm, snr, pr in missed:
            print(f"    {nm:<22} SNR={snr:.1f}  p_v10={pr:.3f}")
    snipped = [r["snr"] for r in flist if not r["ok"]]
    if snipped:
        print(f"FP SNRs (leaked through): {[f'{s:.1f}' for s in snipped]}")

    # 7. Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    bins = np.linspace(0, 1, 21)
    ax1.hist([r["prob"] for r in plist], bins=bins, color="#10b981",
             alpha=0.7, label=f"Planet (N={len(plist)})")
    ax1.hist([r["prob"] for r in flist], bins=bins, color="#ef4444",
             alpha=0.7, label=f"FP (N={len(flist)})")
    ax1.axvline(THRESHOLD, color="black", linestyle="--",
                label=f"threshold={THRESHOLD}")
    ax1.set_xlabel("V10 probability")
    ax1.set_ylabel("count")
    ax1.set_title(f"N={len(processed)}  recall={recall:.1%}  "
                  f"prec={prec:.1%}  F1={f1:.3f}")
    ax1.legend()

    for r in processed:
        color = "#10b981" if r["truth"] == "PLNT" else "#ef4444"
        marker = "o" if r["ok"] else "x"
        ax2.scatter(max(r["snr"], 0.1), r["prob"], c=color,
                    marker=marker, s=60,
                    edgecolors="black", linewidths=0.5)
    ax2.axhline(THRESHOLD, color="black", linestyle="--", alpha=0.5)
    ax2.set_xscale("log")
    ax2.set_xlabel("SNR (log scale)")
    ax2.set_ylabel("V10 probability")
    ax2.set_title("SNR vs V10 prob  (x = misclassified)")
    fig.suptitle(
        f"TESS zero-shot N={len(processed)} — V10 @ threshold {THRESHOLD}",
        fontweight="bold",
    )
    fig.tight_layout()
    fig_path = FIGDIR / "tess_zeroshot_n50.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
