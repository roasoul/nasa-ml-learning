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


def make_synthetic_transit(
    n_points: int = 200,
    depth: float = 0.01,
    noise_level: float = 0.005,
    duration_fraction: float = 0.1,
    seed: int | None = 42,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Generate a single synthetic transit light curve.

    The light curve is returned in the preprocessed form expected by the
    Taylor gate: phase in [-π, π], flux centered at 0 with dip negative.

    Args:
        n_points: Number of phase-folded data points.
        depth: Fractional transit depth (0.01 = 1% dip).
        noise_level: Standard deviation of Gaussian noise (0.005 = 0.5%).
        duration_fraction: Fraction of the full phase that the transit
            occupies. 0.1 means the dip spans 10% of the orbital period.
        seed: Random seed for reproducibility. None for random.

    Returns:
        Tuple of (phase, flux, metadata):
            phase: Shape (n_points,), values in [-π, π].
            flux: Shape (n_points,), baseline ≈ 0, dip is negative.
            metadata: Dict with generation parameters for reference.
    """
    if seed is not None:
        np.random.seed(seed)

    # Phase grid from -π to π (one full orbit, folded)
    phase = np.linspace(-np.pi, np.pi, n_points)

    # Start with flat baseline at 0.0
    # (In real data, we subtract 1.0 after normalizing, so baseline = 0)
    flux = np.zeros(n_points)

    # Add transit dip: a smooth sine-shaped dip centered at phase = 0
    # The transit window is where |phase| < π · duration_fraction
    half_width = np.pi * duration_fraction
    in_transit = np.abs(phase) < half_width

    # Inside the transit window, create a sine-shaped dip:
    # Map the transit region to [0, π] for a smooth half-sine shape
    # Then scale by -depth so the dip goes negative
    transit_phase = (phase[in_transit] + half_width) / (2 * half_width) * np.pi
    flux[in_transit] = -depth * np.sin(transit_phase)

    # Add Gaussian noise everywhere (photon noise + stellar variability)
    noise = np.random.normal(0, noise_level, n_points)
    flux_noisy = flux + noise

    # Convert to PyTorch tensors
    phase_tensor = torch.tensor(phase, dtype=torch.float32)
    flux_tensor = torch.tensor(flux_noisy, dtype=torch.float32)

    metadata = {
        "n_points": n_points,
        "depth": depth,
        "noise_level": noise_level,
        "duration_fraction": duration_fraction,
        "seed": seed,
        "snr": depth / noise_level,  # signal-to-noise ratio
    }

    return phase_tensor, flux_tensor, metadata


def make_synthetic_batch(
    n_transits: int = 32,
    n_non_transits: int = 32,
    n_points: int = 200,
    depth_range: tuple[float, float] = (0.005, 0.02),
    noise_level: float = 0.005,
    seed: int | None = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a batch of synthetic light curves with labels.

    Creates a balanced dataset of transit and non-transit examples for
    training/testing the classifier.

    Args:
        n_transits: Number of transit examples (label = 1).
        n_non_transits: Number of non-transit examples (label = 0).
        n_points: Points per light curve.
        depth_range: (min_depth, max_depth) for random transit depths.
        noise_level: Gaussian noise standard deviation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (phases, fluxes, labels):
            phases: Shape (n_total, n_points), phase values.
            fluxes: Shape (n_total, n_points), flux values.
            labels: Shape (n_total,), 1 for transit, 0 for non-transit.
    """
    if seed is not None:
        np.random.seed(seed)

    all_phases = []
    all_fluxes = []
    all_labels = []

    # Generate transit examples
    for i in range(n_transits):
        depth = np.random.uniform(*depth_range)
        phase, flux, _ = make_synthetic_transit(
            n_points=n_points,
            depth=depth,
            noise_level=noise_level,
            seed=None,  # already seeded above
        )
        all_phases.append(phase)
        all_fluxes.append(flux)
        all_labels.append(1)

    # Generate non-transit examples (pure noise, no dip)
    for i in range(n_non_transits):
        phase, flux, _ = make_synthetic_transit(
            n_points=n_points,
            depth=0.0,  # no transit — just noise
            noise_level=noise_level,
            seed=None,
        )
        all_phases.append(phase)
        all_fluxes.append(flux)
        all_labels.append(0)

    # Stack into batch tensors
    phases = torch.stack(all_phases)    # (n_total, n_points)
    fluxes = torch.stack(all_fluxes)    # (n_total, n_points)
    labels = torch.tensor(all_labels, dtype=torch.float32)  # (n_total,)

    return phases, fluxes, labels
