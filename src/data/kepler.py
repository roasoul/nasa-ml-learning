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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the full CLAUDE.md preprocessing pipeline with 4 output views.

    Pipeline:
        1. flatten()   — remove stellar variability
        2. normalize() — center flux at 1.0
        3a. Primary        fold at epoch              → transit dip at phase=0
        3b. Secondary      fold at epoch + period/2   → secondary eclipse at phase=0
        3c. Odd/even diff  fold odd and even transits separately, take diff
        4. phase → [-π, π]
        5. flux -= 1.0  — baseline at 0
        6. phase-bin each view to n_points

    View 3c (new in V6) catches period-doubled EBs: if the TCE detection
    pipeline picked a period of 2·P_true (mistaking primary AND secondary
    eclipses as the same "transit"), odd-indexed transits show the primary
    eclipse depth and even-indexed transits show the secondary. The diff
    peaks at phase=0 for EBs; for real planets odd/even are consistent
    and the diff is flat noise.

    Args:
        lc: Raw lightkurve LightCurve object.
        period: Orbital period in days.
        epoch: Transit epoch in BKJD (same time system as light curve).
        n_points: Number of output phase bins per view.

    Returns:
        Tuple of (phase, primary_flux, secondary_flux, odd_even_diff)
        tensors, each shape (n_points,). Phase in [-π, π]; fluxes
        baseline-at-0. odd_even_diff = odd_fold - even_fold.
    """
    lc_flat = lc.flatten(window_length=401)
    lc_norm = lc_flat.normalize()

    primary_phase, primary_flux = _fold_and_bin(
        lc_norm, period=period, epoch_time=epoch, n_points=n_points
    )

    _, secondary_flux = _fold_and_bin(
        lc_norm, period=period, epoch_time=epoch + period / 2.0, n_points=n_points
    )

    odd_even_diff = _fold_odd_even_diff(
        lc_norm, period=period, epoch=epoch, n_points=n_points
    )

    return primary_phase, primary_flux, secondary_flux, odd_even_diff


def _fold_odd_even_diff(
    lc_norm: lk.LightCurve,
    period: float,
    epoch: float,
    n_points: int,
) -> torch.Tensor:
    """Compute odd-transit-fold minus even-transit-fold, phase-binned.

    For each data point, determine which transit it belongs to via
        N = round((t - epoch) / period)
    then split into "odd" (N even — 0th, 2nd, 4th... transit) and
    "even" (N odd — 1st, 3rd, 5th...) subsets. Fold each subset at the
    TCE epoch and phase-bin to n_points. Return odd - even.

    Why this catches period-doubled EBs: if the pipeline fit period =
    2·P_true, then transits 0, 2, 4... are actually primary eclipses
    at the correct physical period while transits 1, 3, 5... are
    secondary eclipses. Their depths differ, so odd - even shows a
    residual dip (or bump) at phase=0.

    For real planets, odd and even folds overlap within noise → diff
    is flat. For EBs correctly fit at period = P_true, both odd and
    even contain the same primary eclipse → diff is also flat (this
    case is caught by the secondary-view channel instead).

    Edge case: if the LC has only transits of one parity (rare for
    Kepler's 4-year baseline), returns zeros so the channel stays
    well-defined but carries no information.
    """
    time_arr = np.asarray(lc_norm.time.value)
    transit_number = np.round((time_arr - epoch) / period).astype(int)
    is_odd_fold = (transit_number % 2) == 0  # N=0 (first transit) counts as "odd"

    lc_odd = lc_norm[is_odd_fold]
    lc_even = lc_norm[~is_odd_fold]

    if len(lc_odd) == 0 or len(lc_even) == 0:
        return torch.zeros(n_points, dtype=torch.float32)

    _, odd_flux = _fold_and_bin(lc_odd, period=period, epoch_time=epoch, n_points=n_points)
    _, even_flux = _fold_and_bin(lc_even, period=period, epoch_time=epoch, n_points=n_points)

    return odd_flux - even_flux


def _fold_and_bin(
    lc_norm: lk.LightCurve,
    period: float,
    epoch_time: float,
    n_points: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold a normalized LC at a given epoch and return (phase, flux) tensors.

    Handles steps 3–6 of the preprocessing pipeline. Factored out so the
    primary and secondary views use identical math (only epoch differs).
    """
    folded = lc_norm.fold(period=period, epoch_time=epoch_time)
    phase_days = folded.time.value
    phase_scaled = (phase_days / period) * 2.0 * np.pi  # [-π, π]
    flux_centered = folded.flux.value - 1.0  # baseline at 0

    phase_binned, flux_binned = phase_bin(phase_scaled, flux_centered, n_points)
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Download and preprocess a single target end-to-end.

    Args:
        target: Dict with keys: kepid, period, epoch, name.
        n_points: Number of output phase bins.

    Returns:
        Tuple of (phase, primary_flux, secondary_flux, odd_even_diff)
        tensors, each shape (n_points,).
    """
    lc = download_lightcurve(target["kepid"])
    return preprocess_lightcurve(
        lc, target["period"], target["epoch"], n_points
    )
