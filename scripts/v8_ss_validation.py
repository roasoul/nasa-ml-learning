"""V8.5 SS-flag validation (Figure 7) + mask-overlap / SNR robustness.

Fetches koi_fpflag_ss and koi_model_snr from the NASA Exoplanet Archive
for every KOI in the 500-TCE V6 dataset, caches to
data/ss_flag_cache.csv, then:

    Figure 7 — per-sample B (closed-form fit) split by SS flag.
               Did the learned shape parameter rediscover the
               centroid/spectroscopic EB flag from photometry alone?

    Mask-overlap check — T12/T14 distribution for
               koi_model_snr < 12 vs ≥ 12. Confirms the normalized
               soft masks don't drift to 0.5 at low SNR.
"""

import csv
import io
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.v8_figures import fit_B_closed_form


FIGDIR = Path("notebooks/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = "data/kepler_tce_v6.pt"
RESULTS_PATH = "data/v8_results.pt"
CACHE_PATH = Path("data/ss_flag_cache.csv")

TAP_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
    "query=select+kepoi_name,koi_fpflag_ss,koi_model_snr+"
    "from+cumulative&format=csv"
)


def fetch_and_cache() -> dict[str, dict]:
    """Return {kepoi_name → {ss: int|None, snr: float|None}}."""
    if CACHE_PATH.exists():
        print(f"Loading cached archive metadata from {CACHE_PATH}")
        out = {}
        with open(CACHE_PATH, newline="") as f:
            for row in csv.DictReader(f):
                out[row["kepoi_name"]] = {
                    "ss": int(row["ss"]) if row["ss"] not in ("", "None") else None,
                    "snr": float(row["snr"]) if row["snr"] not in ("", "None") else None,
                }
        return out

    print(f"Fetching koi_fpflag_ss + koi_model_snr from NASA Archive")
    with urllib.request.urlopen(TAP_URL, timeout=60) as resp:
        text = resp.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    out = {}
    for row in reader:
        name = row["kepoi_name"]
        ss_raw = row.get("koi_fpflag_ss", "")
        snr_raw = row.get("koi_model_snr", "")
        out[name] = {
            "ss": int(ss_raw) if ss_raw not in ("", "None") else None,
            "snr": float(snr_raw) if snr_raw not in ("", "None") else None,
        }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kepoi_name", "ss", "snr"])
        for name, rec in out.items():
            w.writerow([name, rec["ss"] if rec["ss"] is not None else "",
                        rec["snr"] if rec["snr"] is not None else ""])
    print(f"Cached {len(out)} KOIs to {CACHE_PATH}")
    return out


def match(archive: dict, dataset_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Align archive records to dataset order. Names in the dataset are
    like 'K00001.01' — the archive uses 'K00001.01' too (kepoi_name).
    """
    ss = np.full(len(dataset_names), -1, dtype=int)   # -1 = unknown
    snr = np.full(len(dataset_names), np.nan)
    for i, name in enumerate(dataset_names):
        rec = archive.get(name)
        if rec is None:
            continue
        if rec["ss"] is not None:
            ss[i] = rec["ss"]
        if rec["snr"] is not None:
            snr[i] = rec["snr"]
    return ss, snr


def main() -> None:
    d = torch.load(DATA_PATH, weights_only=False)
    r = torch.load(RESULTS_PATH, weights_only=False)
    test_idx = r["test_idx"]
    labels = r["test_labels"].cpu().numpy()
    shape_feats = r["v85"]["shape_features"].cpu().numpy()

    archive = fetch_and_cache()
    names = d["names"]
    ss_full, snr_full = match(archive, names)

    # Subset to test split
    ss_test = ss_full[np.array(test_idx)]
    snr_test = snr_full[np.array(test_idx)]

    # Per-sample B (closed-form fit on primary flux)
    phases_test = d["phases"][torch.tensor(test_idx)]
    fluxes_test = d["fluxes"][torch.tensor(test_idx)]
    B_per_sample = np.zeros(len(test_idx))
    for i in range(len(test_idx)):
        _, B_per_sample[i] = fit_B_closed_form(
            phases_test[i].cpu().numpy(), fluxes_test[i].cpu().numpy()
        )

    # -------- Figure 7: B split by SS flag --------
    B_ss0 = B_per_sample[ss_test == 0]
    B_ss1 = B_per_sample[ss_test == 1]
    unknown = (ss_test == -1).sum()
    print(f"Test set SS tags: ss=0 n={len(B_ss0)}, ss=1 n={len(B_ss1)}, unknown n={unknown}")

    # Clip for plotting — closed-form fits on pure-noise flux can explode
    B_ss0_clip = np.clip(B_ss0, -5, 5)
    B_ss1_clip = np.clip(B_ss1, -5, 5)
    med_ss0 = float(np.median(B_ss0)) if len(B_ss0) else np.nan
    med_ss1 = float(np.median(B_ss1)) if len(B_ss1) else np.nan
    sep = abs(med_ss0 - med_ss1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(B_ss0_clip, bins=20, alpha=0.7, color="#10b981",
                 label=f"SS=0 not-EB (n={len(B_ss0)})")
    axes[0].hist(B_ss1_clip, bins=20, alpha=0.7, color="#ef4444",
                 label=f"SS=1 EB flag (n={len(B_ss1)})")
    axes[0].axvline(med_ss0, color="#10b981", linestyle="--",
                    label=f"median={med_ss0:+.2f}")
    axes[0].axvline(med_ss1, color="#ef4444", linestyle="--",
                    label=f"median={med_ss1:+.2f}")
    axes[0].set_xlabel("Per-sample B  (U-shape ≈ 1, V-shape ≈ 0 or negative)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Did B rediscover the SS Flag?")
    axes[0].legend(fontsize=9)

    axes[1].bar(["SS=0 (not EB)", "SS=1 (EB flag)"], [med_ss0, med_ss1],
                color=["#10b981", "#ef4444"], alpha=0.85)
    axes[1].set_ylabel("Median per-sample B")
    axes[1].set_title(f"Separation gap: {sep:.3f}")

    plt.suptitle("B Parameter vs SS Flag Validation (Figure 7)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGDIR / "B_vs_SS_flag.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Median B (SS=0): {med_ss0:+.3f}")
    print(f"Median B (SS=1): {med_ss1:+.3f}")
    print(f"Separation gap:  {sep:.3f}")
    if sep > 0.2:
        print("B rediscovered SS Flag from photometry!")
    else:
        print("Weak separation — SS flag not recovered from per-sample B alone")

    # -------- Mask-overlap check: T12/T14 at low vs high SNR --------
    t12_t14 = shape_feats[:, 3]
    valid_snr = ~np.isnan(snr_test)
    low = valid_snr & (snr_test < 12.0)
    high = valid_snr & (snr_test >= 12.0)
    print(f"\nSNR split: low (<12) n={low.sum()}, high (>=12) n={high.sum()}, "
          f"unknown n={(~valid_snr).sum()}")

    fig, ax = plt.subplots(figsize=(8, 5))
    if low.sum():
        ax.hist(t12_t14[low], bins=20, alpha=0.65, color="#f59e0b",
                label=f"koi_model_snr < 12 (n={low.sum()})")
    if high.sum():
        ax.hist(t12_t14[high], bins=20, alpha=0.65, color="#0ea5e9",
                label=f"koi_model_snr >= 12 (n={high.sum()})")
    ax.axvline(0.5, color="black", linestyle=":", label="0.5 degeneracy line")
    ax.set_xlabel("T12/T14  (normalized ingress fraction)")
    ax.set_ylabel("Count")
    ax.set_title("Mask-overlap check — T12/T14 by SNR\n"
                 "Normalized soft masks should NOT collapse to 0.5 at low SNR")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(FIGDIR / "v8_T12T14_by_SNR.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {FIGDIR / 'v8_T12T14_by_SNR.png'}")

    if low.sum():
        print(f"Low-SNR  T12/T14: median={np.median(t12_t14[low]):.3f}  "
              f"|  fraction within [0.40, 0.60]: {((t12_t14[low] > 0.4) & (t12_t14[low] < 0.6)).mean():.2%}")
    if high.sum():
        print(f"High-SNR T12/T14: median={np.median(t12_t14[high]):.3f}  "
              f"|  fraction within [0.40, 0.60]: {((t12_t14[high] > 0.4) & (t12_t14[high] < 0.6)).mean():.2%}")


if __name__ == "__main__":
    main()
