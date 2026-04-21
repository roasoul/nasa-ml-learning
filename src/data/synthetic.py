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
    odd_even_diff_depth: float = 0.0,
    seed: int | None = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Generate a synthetic light curve with primary + secondary + odd/even views.

    Produces three flux arrays on a shared phase grid, matching the V6
    four-view data layout:
        - Primary view:        dip of `depth` at phase=0.
        - Secondary view:      dip of `secondary_depth` at phase=0.
                               Simulates an EB's secondary eclipse after
                               folding at epoch + period/2.
        - Odd/even diff view:  dip of `odd_even_diff_depth` at phase=0.
                               Simulates a period-doubled EB where odd and
                               even transits have different depths —
                               real-data diff = odd_fold - even_fold.
                               Zero for planets.

    Args:
        n_points: Number of phase-folded data points.
        depth: Primary transit depth (0.01 = 1% dip). 0 for non-transit.
        noise_level: Gaussian noise std (0.005 = 0.5%).
        duration_fraction: Fraction of phase the transit occupies.
        secondary_depth: Depth of secondary-view dip. 0 for planets.
        odd_even_diff_depth: Depth of odd/even-diff-view dip.
            0 for planets and correct-period EBs; non-zero for
            period-doubled EBs.
        seed: Random seed for reproducibility. None for random.

    Returns:
        (phase, primary_flux, secondary_flux, odd_even_diff, metadata)
    """
    if seed is not None:
        np.random.seed(seed)

    phase = np.linspace(-np.pi, np.pi, n_points)

    primary = _half_sine_dip(phase, depth, duration_fraction)
    secondary = _half_sine_dip(phase, secondary_depth, duration_fraction)
    odd_even = _half_sine_dip(phase, odd_even_diff_depth, duration_fraction)

    # Independent noise per view — in the real pipeline they come from
    # different foldings of the same underlying data, so noise realizations
    # differ. For the odd/even diff view, noise is sqrt(2)x larger because
    # it's a difference of two independent folds; scale accordingly.
    primary += np.random.normal(0, noise_level, n_points)
    secondary += np.random.normal(0, noise_level, n_points)
    odd_even += np.random.normal(0, noise_level * np.sqrt(2), n_points)

    phase_t = torch.tensor(phase, dtype=torch.float32)
    primary_t = torch.tensor(primary, dtype=torch.float32)
    secondary_t = torch.tensor(secondary, dtype=torch.float32)
    odd_even_t = torch.tensor(odd_even, dtype=torch.float32)

    metadata = {
        "n_points": n_points,
        "depth": depth,
        "secondary_depth": secondary_depth,
        "odd_even_diff_depth": odd_even_diff_depth,
        "noise_level": noise_level,
        "duration_fraction": duration_fraction,
        "seed": seed,
        "snr": depth / noise_level if noise_level > 0 else float("inf"),
    }

    return phase_t, primary_t, secondary_t, odd_even_t, metadata


def make_synthetic_batch(
    n_planets: int = 32,
    n_eclipsing_binaries: int = 16,
    n_eb_doubled: int = 16,
    n_non_transits: int = 16,
    n_points: int = 200,
    depth_range: tuple[float, float] = (0.005, 0.02),
    eb_secondary_range: tuple[float, float] = (0.003, 0.015),
    eb_odd_even_range: tuple[float, float] = (0.003, 0.015),
    noise_level: float = 0.005,
    seed: int | None = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a batch of synthetic four-view light curves with labels.

    V6 distinguishes four underlying classes but uses binary labels
    (planet vs. not-a-planet):
        - Planet (label=1):          primary dip, flat secondary, flat odd/even
        - EB correct period (0):     primary dip + secondary dip, flat odd/even
        - EB doubled period (0):     primary dip, flat secondary, odd/even dip
        - Non-transit (0):           noise in all views

    The EB-doubled class targets the V5 failure mode: TCE detection
    misfits the period as 2·P_true, so every "transit" alternates
    between the true primary and secondary eclipses — invisible in the
    secondary channel (it sees the same averaged dip) but visible in
    odd_fold - even_fold.

    Args:
        n_planets:             Planet examples (label=1).
        n_eclipsing_binaries:  EB-correct-period examples (label=0).
        n_eb_doubled:          EB-doubled-period examples (label=0, new in V6).
        n_non_transits:        Noise-only examples (label=0).
        n_points:              Points per light curve.
        depth_range:           (min, max) for primary depth.
        eb_secondary_range:    (min, max) for EB secondary depth.
        eb_odd_even_range:     (min, max) for odd/even diff depth.
        noise_level:           Gaussian noise std.
        seed:                  Random seed.

    Returns:
        (phases, primary_fluxes, secondary_fluxes, odd_even_fluxes, labels)
    """
    if seed is not None:
        np.random.seed(seed)

    phases: list[torch.Tensor] = []
    primaries: list[torch.Tensor] = []
    secondaries: list[torch.Tensor] = []
    odd_evens: list[torch.Tensor] = []
    labels: list[int] = []

    def _append(ph, p, s, oe, lbl):
        phases.append(ph); primaries.append(p); secondaries.append(s)
        odd_evens.append(oe); labels.append(lbl)

    # Planets
    for _ in range(n_planets):
        depth = np.random.uniform(*depth_range)
        ph, p, s, oe, _ = make_synthetic_transit(
            n_points=n_points, depth=depth,
            secondary_depth=0.0, odd_even_diff_depth=0.0,
            noise_level=noise_level, seed=None,
        )
        _append(ph, p, s, oe, 1)

    # Correct-period EBs: secondary dip, flat odd/even
    for _ in range(n_eclipsing_binaries):
        depth = np.random.uniform(*depth_range)
        sec = np.random.uniform(*eb_secondary_range)
        ph, p, s, oe, _ = make_synthetic_transit(
            n_points=n_points, depth=depth,
            secondary_depth=sec, odd_even_diff_depth=0.0,
            noise_level=noise_level, seed=None,
        )
        _append(ph, p, s, oe, 0)

    # Doubled-period EBs: flat secondary, odd/even dip
    for _ in range(n_eb_doubled):
        depth = np.random.uniform(*depth_range)
        oe_d = np.random.uniform(*eb_odd_even_range)
        ph, p, s, oe, _ = make_synthetic_transit(
            n_points=n_points, depth=depth,
            secondary_depth=0.0, odd_even_diff_depth=oe_d,
            noise_level=noise_level, seed=None,
        )
        _append(ph, p, s, oe, 0)

    # Non-transits: noise in all views
    for _ in range(n_non_transits):
        ph, p, s, oe, _ = make_synthetic_transit(
            n_points=n_points, depth=0.0,
            secondary_depth=0.0, odd_even_diff_depth=0.0,
            noise_level=noise_level, seed=None,
        )
        _append(ph, p, s, oe, 0)

    return (
        torch.stack(phases),
        torch.stack(primaries),
        torch.stack(secondaries),
        torch.stack(odd_evens),
        torch.tensor(labels, dtype=torch.float32),
    )
