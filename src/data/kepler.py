"""Download and preprocess real Kepler light curves for the Taylor-CNN model.

Uses the lightkurve package to download PDC-SAP (Pre-search Data
Conditioning) light curves from MAST, then applies the preprocessing
pipeline specified in CLAUDE.md:

    1. flatten()    — remove stellar variability / instrumental trends
    2. normalize()  — center flux at 1.0
    3. fold()       — fold on known orbital period and epoch
    4. Scale phase to [-π, π]
    5. Subtract 1.0 from flux — baseline = 0, transit dip is negative
    6. Phase-bin to fixed number of points (200 by default)

The output matches the format expected by the TaylorCNN model:
    phase: tensor of shape (n_points,) in [-π, π]
    flux:  tensor of shape (n_points,) centered at 0
"""

import numpy as np
import torch
import lightkurve as lk


# -----------------------------------------------------------------------
# Target catalog — parameters from NASA Exoplanet Archive cumulative KOI
# table (DR25). Epochs are in BKJD (BJD - 2454833).
# -----------------------------------------------------------------------

CONFIRMED_PLANETS = [
    {
        "name": "Kepler-5b",
        "kepoi": "K00018.01",
        "kepid": 8191672,
        "period": 3.548465,
        "epoch": 122.9014,
        "depth_ppm": 7252,
    },
    {
        "name": "Kepler-6b",
        "kepoi": "K00017.01",
        "kepid": 10874614,
        "period": 3.234699,
        "epoch": 121.4866,
        "depth_ppm": 10559,
    },
    {
        "name": "Kepler-7b",
        "kepoi": "K00097.01",
        "kepid": 5780885,
        "period": 4.885489,
        "epoch": 134.2769,
        "depth_ppm": 7458,
    },
    {
        "name": "Kepler-8b",
        "kepoi": "K00010.01",
        "kepid": 6922244,
        "period": 3.522498,
        "epoch": 121.1194,
        "depth_ppm": 9146,
    },
    {
        "name": "Kepler-12b",
        "kepoi": "K00020.01",
        "kepid": 11804465,
        "period": 4.437963,
        "epoch": 171.0091,
        "depth_ppm": 16388,
    },
    {
        "name": "Kepler-15b",
        "kepoi": "K00128.01",
        "kepid": 11359879,
        "period": 4.942783,
        "epoch": 136.3294,
        "depth_ppm": 11145,
    },
    {
        "name": "Kepler-17b",
        "kepoi": "K00203.01",
        "kepid": 10619192,
        "period": 1.485711,
        "epoch": 132.7936,
        "depth_ppm": 20824,
    },
    {
        "name": "Kepler-43b",
        "kepoi": "K00135.01",
        "kepid": 9818381,
        "period": 3.024093,
        "epoch": 132.4169,
        "depth_ppm": 7765,
    },
]

FALSE_POSITIVES = [
    {
        "name": "KOI-6235.01",
        "kepoi": "K06235.01",
        "kepid": 11147460,
        "period": 2.053866,
        "epoch": 132.9441,
        "depth_ppm": 5021,
    },
    {
        "name": "KOI-3622.01",
        "kepoi": "K03622.01",
        "kepid": 6677267,
        "period": 3.125821,
        "epoch": 134.4797,
        "depth_ppm": 5025,
    },
    {
        "name": "KOI-1285.01",
        "kepoi": "K01285.01",
        "kepid": 10599397,
        "period": 0.937406,
        "epoch": 134.7265,
        "depth_ppm": 5080,
    },
    {
        "name": "KOI-772.01",
        "kepoi": "K00772.01",
        "kepid": 11493732,
        "period": 61.256363,
        "epoch": 173.8383,
        "depth_ppm": 5123,
    },
    {
        "name": "KOI-4089.01",
        "kepoi": "K04089.01",
        "kepid": 11972872,
        "period": 0.531107,
        "epoch": 131.9065,
        "depth_ppm": 5134,
    },
    {
        "name": "KOI-136.01",
        "kepoi": "K00136.01",
        "kepid": 7601633,
        "period": 15.663887,
        "epoch": 147.3957,
        "depth_ppm": 5144,
    },
    {
        "name": "KOI-3293.01",
        "kepoi": "K03293.01",
        "kepid": 3858851,
        "period": 25.951988,
        "epoch": 154.8810,
        "depth_ppm": 5103,
    },
    {
        "name": "KOI-3891.01",
        "kepoi": "K03891.01",
        "kepid": 5479689,
        "period": 8.249789,
        "epoch": 137.3960,
        "depth_ppm": 5300,
    },
]


def download_lightcurve(kepid: int) -> lk.LightCurve:
    """Download and stitch all Kepler quarters for a target.

    Args:
        kepid: Kepler Input Catalog ID.

    Returns:
        Stitched light curve with NaNs and outliers removed.

    Raises:
        ValueError: If no light curves found for the target.
    """
    search = lk.search_lightcurve(
        f"KIC {kepid}",
        mission="Kepler",
        author="Kepler",
        cadence="long",
    )
    if len(search) == 0:
        raise ValueError(f"No Kepler light curves found for KIC {kepid}")

    # Download all quarters and stitch into one continuous light curve
    lc_collection = search.download_all()
    lc = lc_collection.stitch()

    # Remove bad data points.
    # IMPORTANT: only clip UPWARD outliers (cosmic rays, instrument glitches).
    # Downward outliers include real transit dips — clipping them destroys
    # the transit signal! sigma_lower=inf means "never clip downward."
    lc = lc.remove_nans().remove_outliers(sigma_lower=float("inf"), sigma_upper=5)

    return lc


def preprocess_lightcurve(
    lc: lk.LightCurve,
    period: float,
    epoch: float,
    n_points: int = 200,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the full CLAUDE.md preprocessing pipeline.

    Pipeline:
        1. flatten()    — remove stellar variability
        2. normalize()  — center flux at 1.0
        3. fold()       — fold on known period/epoch
        4. phase → [-π, π]
        5. flux -= 1.0  — baseline at 0
        6. phase-bin to n_points

    Args:
        lc: Raw lightkurve LightCurve object.
        period: Orbital period in days.
        epoch: Transit epoch in BKJD (same time system as light curve).
        n_points: Number of output phase bins.

    Returns:
        Tuple of (phase, flux) tensors, each shape (n_points,).
    """
    # Step 1: Flatten — remove long-term stellar variability and
    # instrumental trends using a Savitzky-Golay filter. window_length
    # must be odd; 401 cadences ≈ 8 days for long-cadence (30-min) data.
    lc_flat = lc.flatten(window_length=401)

    # Step 2: Normalize — divide by median flux so baseline ≈ 1.0
    lc_norm = lc_flat.normalize()

    # Step 3: Fold — phase-fold on the known orbital period and epoch.
    # After folding, the transit is centered at phase = 0.
    folded = lc_norm.fold(period=period, epoch_time=epoch)

    # Step 4: Scale phase to [-π, π].
    # folded.time gives phase in units of days (range: -period/2 to +period/2).
    # Dividing by period gives [-0.5, 0.5], then × 2π gives [-π, π].
    phase_days = folded.time.value  # days
    phase_scaled = (phase_days / period) * 2.0 * np.pi  # [-π, π]

    # Step 5: Subtract 1.0 from flux so baseline = 0 and dip is negative
    flux_centered = folded.flux.value - 1.0

    # Step 6: Phase-bin to n_points using median in each bin.
    # Median is more robust to outliers than mean.
    phase_binned, flux_binned = phase_bin(
        phase_scaled, flux_centered, n_points
    )

    phase_tensor = torch.tensor(phase_binned, dtype=torch.float32)
    flux_tensor = torch.tensor(flux_binned, dtype=torch.float32)

    return phase_tensor, flux_tensor


def phase_bin(
    phase: np.ndarray,
    flux: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin a phase-folded light curve into equal-width phase bins.

    Args:
        phase: Phase values (any range, but typically [-π, π]).
        flux: Corresponding flux values.
        n_bins: Number of output bins.

    Returns:
        Tuple of (bin_centers, binned_flux). Empty bins are filled with 0.
    """
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    binned_flux = np.zeros(n_bins)

    # Assign each data point to a bin
    bin_indices = np.digitize(phase, bin_edges) - 1  # 0-indexed

    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            binned_flux[i] = np.median(flux[mask])
        # else: remains 0 (baseline)

    return bin_centers, binned_flux


def download_and_preprocess(
    target: dict,
    n_points: int = 200,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Download and preprocess a single target end-to-end.

    Args:
        target: Dict with keys: kepid, period, epoch, name.
        n_points: Number of output phase bins.

    Returns:
        Tuple of (phase, flux) tensors, each shape (n_points,).
    """
    lc = download_lightcurve(target["kepid"])
    phase, flux = preprocess_lightcurve(
        lc, target["period"], target["epoch"], n_points
    )
    return phase, flux
