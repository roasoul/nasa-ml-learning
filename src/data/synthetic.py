"""Synthetic light curve generator for testing the Taylor gate.

Generates fake transit light curves with known parameters so we can
verify the gate works before touching real Kepler data. This is standard
practice in ML: test on synthetic data with known ground truth first.

A synthetic transit is modeled as a sine-shaped dip:
    flux = 1.0 + depth · max(0, sin(phase))   (inside transit window)
    flux = 1.0                                  (outside transit window)

Then Gaussian noise is added to simulate photon noise / stellar variability.
The preprocessing pipeline (normalize, fold, scale, subtract baseline) is
applied to match what we'd do with real lightkurve data.
"""

import numpy as np
import torch


def _half_sine_dip(
    phase: np.ndarray, depth: float, duration_fraction: float
) -> np.ndarray:
    """Build a flat baseline with a half-sine dip centered at phase=0.

    Returns an array the same shape as `phase` with a negative dip of
    magnitude `depth` spanning ±π·duration_fraction. No noise added.
    """
    flux = np.zeros_like(phase)
    if depth <= 0.0:
        return flux
    half_width = np.pi * duration_fraction
    in_transit = np.abs(phase) < half_width
    transit_phase = (phase[in_transit] + half_width) / (2 * half_width) * np.pi
    flux[in_transit] = -depth * np.sin(transit_phase)
    return flux


def make_synthetic_transit(
    n_points: int = 200,
    depth: float = 0.01,
    noise_level: float = 0.005,
    duration_fraction: float = 0.1,
    secondary_depth: float = 0.0,
    seed: int | None = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Generate a synthetic light curve with primary + secondary views.

    Produces two flux arrays on a shared phase grid, matching the V5
    two-view data layout:
        - Primary view: dip of `depth` centered at phase=0.
        - Secondary view: dip of `secondary_depth` centered at phase=0.
          This simulates an eclipsing binary's secondary eclipse seen
          after phase-folding at `epoch + period/2`. Use 0 for planets
          and non-transits; use a non-zero value for EB-like signals.

    Args:
        n_points: Number of phase-folded data points.
        depth: Fractional primary transit depth (0.01 = 1% dip). Set to
            0 for a non-transit noise-only example.
        noise_level: Standard deviation of Gaussian noise (0.005 = 0.5%).
        duration_fraction: Fraction of the full phase that the transit
            occupies. 0.1 means the dip spans 10% of the orbital period.
        secondary_depth: Fractional depth of the secondary-view dip.
            0 for planets/non-transits, non-zero for EBs.
        seed: Random seed for reproducibility. None for random.

    Returns:
        Tuple of (phase, primary_flux, secondary_flux, metadata):
            phase: Shape (n_points,), values in [-π, π].
            primary_flux: Shape (n_points,), dip of `depth` at phase=0.
            secondary_flux: Shape (n_points,), dip of `secondary_depth`
                at phase=0 (flat noise if secondary_depth == 0).
            metadata: Dict with generation parameters.
    """
    if seed is not None:
        np.random.seed(seed)

    phase = np.linspace(-np.pi, np.pi, n_points)

    primary = _half_sine_dip(phase, depth, duration_fraction)
    secondary = _half_sine_dip(phase, secondary_depth, duration_fraction)

    # Independent noise per view — they come from different fold-epochs of
    # the same underlying data in the real pipeline, so the noise realizations
    # differ even though the stellar noise spectrum is the same.
    primary += np.random.normal(0, noise_level, n_points)
    secondary += np.random.normal(0, noise_level, n_points)

    phase_t = torch.tensor(phase, dtype=torch.float32)
    primary_t = torch.tensor(primary, dtype=torch.float32)
    secondary_t = torch.tensor(secondary, dtype=torch.float32)

    metadata = {
        "n_points": n_points,
        "depth": depth,
        "secondary_depth": secondary_depth,
        "noise_level": noise_level,
        "duration_fraction": duration_fraction,
        "seed": seed,
        "snr": depth / noise_level if noise_level > 0 else float("inf"),
    }

    return phase_t, primary_t, secondary_t, metadata


def make_synthetic_batch(
    n_planets: int = 32,
    n_eclipsing_binaries: int = 16,
    n_non_transits: int = 16,
    n_points: int = 200,
    depth_range: tuple[float, float] = (0.005, 0.02),
    eb_secondary_range: tuple[float, float] = (0.003, 0.015),
    noise_level: float = 0.005,
    seed: int | None = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a batch of synthetic two-view light curves with labels.

    V5 distinguishes three underlying classes but uses binary labels
    because the downstream task is "planet vs. not-a-planet":
        - Planet (label=1): primary dip, flat secondary view
        - Eclipsing binary (label=0): primary dip AND secondary dip
        - Non-transit (label=0): noise in both views

    EB primary depths are drawn from the same range as planets (they
    often look indistinguishable in the primary view alone — which is
    exactly why V4 confused them). The secondary channel is what the
    model must learn to exploit.

    Args:
        n_planets: Number of planet examples (label=1).
        n_eclipsing_binaries: Number of EB examples (label=0, primary +
            secondary dip).
        n_non_transits: Number of noise-only examples (label=0).
        n_points: Points per light curve.
        depth_range: (min, max) for primary-dip depth.
        eb_secondary_range: (min, max) for EB secondary-dip depth.
        noise_level: Gaussian noise standard deviation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (phases, primary_fluxes, secondary_fluxes, labels):
            phases:             (n_total, n_points)
            primary_fluxes:     (n_total, n_points)
            secondary_fluxes:   (n_total, n_points)
            labels:             (n_total,)  — 1 for planet, else 0
    """
    if seed is not None:
        np.random.seed(seed)

    phases: list[torch.Tensor] = []
    primaries: list[torch.Tensor] = []
    secondaries: list[torch.Tensor] = []
    labels: list[int] = []

    # Planets: primary dip only
    for _ in range(n_planets):
        depth = np.random.uniform(*depth_range)
        ph, p, s, _ = make_synthetic_transit(
            n_points=n_points,
            depth=depth,
            secondary_depth=0.0,
            noise_level=noise_level,
            seed=None,
        )
        phases.append(ph)
        primaries.append(p)
        secondaries.append(s)
        labels.append(1)

    # Eclipsing binaries: primary dip + secondary dip
    for _ in range(n_eclipsing_binaries):
        depth = np.random.uniform(*depth_range)
        sec_depth = np.random.uniform(*eb_secondary_range)
        ph, p, s, _ = make_synthetic_transit(
            n_points=n_points,
            depth=depth,
            secondary_depth=sec_depth,
            noise_level=noise_level,
            seed=None,
        )
        phases.append(ph)
        primaries.append(p)
        secondaries.append(s)
        labels.append(0)

    # Non-transits: noise in both views
    for _ in range(n_non_transits):
        ph, p, s, _ = make_synthetic_transit(
            n_points=n_points,
            depth=0.0,
            secondary_depth=0.0,
            noise_level=noise_level,
            seed=None,
        )
        phases.append(ph)
        primaries.append(p)
        secondaries.append(s)
        labels.append(0)

    phases_t = torch.stack(phases)
    primary_t = torch.stack(primaries)
    secondary_t = torch.stack(secondaries)
    labels_t = torch.tensor(labels, dtype=torch.float32)

    return phases_t, primary_t, secondary_t, labels_t
