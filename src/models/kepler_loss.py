"""Kepler's Third Law as a soft physics penalty for the V7 classifier.

For a circular orbit of a planet crossing a star at zero impact parameter:
    T_dur_predicted = P × R_star / (π × a)
    a = (G × M_star × P² / 4π²)^(1/3)          (Kepler's Third Law)

A real planet's (period, duration) pair is consistent with its host star's
(M_star, R_star). An eclipsing binary with a stellar companion, or a
false-positive signal from stellar variability, generally is not — the
observed duration is too long (EBs) or too short (noise spikes) for the
inferred period.

Violation metric:
    violation = |log(observed_duration / predicted_duration)|

Zero means perfect physical consistency. 0.69 means a factor of 2 off.

Training-time Kepler loss:
    L_kepler = mean( pred_planet × violation² )

When the model already predicts low probability of planet, this term is
~0 (soft: it doesn't force predictions, only pushes them DOWN when a
non-physical duration is paired with a planet-like prediction).

References
----------
- Seager & Mallen-Ornelas 2003: transit geometry formulas.
- Mass-radius relation for main-sequence stars: R ∝ M^0.8 (rough fit).
"""

import math

import torch


# Physical constants in SI
_G = 6.674e-11          # m^3 kg^-1 s^-2
_M_SUN = 1.989e30       # kg
_R_SUN = 6.96e8         # m
_DAY_SEC = 86400.0      # seconds


def calculate_predicted_duration(
    period_days: torch.Tensor | float,
    stellar_mass: torch.Tensor | float,
    stellar_radius: torch.Tensor | float | None = None,
) -> torch.Tensor | float:
    """Central-transit duration for a circular orbit, b=0.

    Args:
        period_days:    Orbital period in days.
        stellar_mass:   Stellar mass in solar masses.
        stellar_radius: Stellar radius in solar radii. If None, uses the
                        main-sequence mass-radius relation R ≈ M^0.8.

    Returns:
        Predicted transit duration in days (same type as inputs —
        torch tensor in/out, float in/out).
    """
    if stellar_radius is None:
        # Main-sequence M-R relation; works well near 1 M_sun, rough at
        # extremes. Good enough for a violation-detection heuristic.
        stellar_radius = stellar_mass ** 0.8

    if isinstance(period_days, torch.Tensor):
        P_sec = period_days * _DAY_SEC
        M_kg = stellar_mass * _M_SUN
        R_m = stellar_radius * _R_SUN
        pi = torch.tensor(math.pi, dtype=P_sec.dtype, device=P_sec.device)
        a_m = (_G * M_kg * P_sec ** 2 / (4 * pi ** 2)) ** (1.0 / 3.0)
        t_dur_sec = (R_m / pi) * (P_sec / a_m)
        return t_dur_sec / _DAY_SEC

    # Scalar path
    P_sec = period_days * _DAY_SEC
    M_kg = stellar_mass * _M_SUN
    R_m = stellar_radius * _R_SUN
    a_m = (_G * M_kg * P_sec ** 2 / (4 * math.pi ** 2)) ** (1.0 / 3.0)
    t_dur_sec = (R_m / math.pi) * (P_sec / a_m)
    return t_dur_sec / _DAY_SEC


def calculate_kepler_violation(
    period: torch.Tensor | float,
    duration: torch.Tensor | float,
    stellar_mass: torch.Tensor | float,
    stellar_radius: torch.Tensor | float | None = None,
    asymmetric: bool = False,
) -> torch.Tensor | float:
    """Dimensionless violation of Kepler's Third Law.

    Units follow the input types: if `duration` is in days, predicted
    duration is in days; if `duration` is in hours, convert to days
    before calling.

    Args:
        period:         Orbital period (days).
        duration:       Observed transit duration (days).
        stellar_mass:   Stellar mass (M_sun).
        stellar_radius: Stellar radius (R_sun), optional.
        asymmetric:     If True, only penalize *too-long* durations
                        (observed > predicted) — the "EB too long"
                        failure mode. Planets with shorter observed
                        durations (non-zero impact parameter, limb
                        darkening, eccentricity) get violation=0.
                        If False (default), symmetric |log(ratio)|.

    Returns:
        Symmetric mode:  violation = |log(observed / predicted)|
        Asymmetric mode: violation = max(0, log(observed / predicted))

        0 means acceptable; larger = more unphysical.
        0.69 ≈ factor-of-2 discrepancy.
    """
    predicted = calculate_predicted_duration(
        period, stellar_mass, stellar_radius
    )
    eps = 1e-8
    if isinstance(predicted, torch.Tensor):
        ratio = torch.clamp(duration / (predicted + eps), min=eps)
        log_ratio = torch.log(ratio)
        if asymmetric:
            return torch.clamp(log_ratio, min=0.0)
        return torch.abs(log_ratio)
    ratio = max(duration / (predicted + eps), eps)
    log_ratio = math.log(ratio)
    if asymmetric:
        return max(0.0, log_ratio)
    return abs(log_ratio)


def kepler_loss(
    probs: torch.Tensor,
    violations: torch.Tensor,
    tolerance: float = 0.2,
) -> torch.Tensor:
    """Soft Kepler's-law penalty for a training batch.

    Args:
        probs:      Model's planet probabilities, shape (batch,), in [0, 1].
        violations: Precomputed |log(obs/pred)| per sample, shape (batch,).
                    A value of 0.2 corresponds to roughly ±20% duration error,
                    which we allow for free (stellar-parameter uncertainty
                    and non-zero impact parameter can move durations by this
                    much). Beyond 0.2 the penalty grows quadratically.
        tolerance:  Violation magnitude below which no penalty is applied.
                    Default 0.2 ≈ ±22% duration mismatch tolerance.

    Returns:
        Scalar loss. 0 when every prediction is already low or every
        sample is physics-consistent. Positive and growing when the model
        assigns high planet probability to samples whose duration conflicts
        with Kepler's Third Law.
    """
    excess = torch.clamp(violations - tolerance, min=0.0)
    return (probs * excess ** 2).mean()


def sparsity_loss(model: torch.nn.Module) -> torch.Tensor:
    """L1 norm on CNN Conv1d weights (encourages a minimal correction
    on top of the physics-informed Taylor gate).

    Small L1 penalties push redundant filter weights toward zero, so the
    learned CNN tends to use a few strong filters rather than spreading
    signal across many — a physics-informed network should mostly be
    the Taylor gate plus small residual corrections.
    """
    total = 0.0
    for m in model.modules():
        if isinstance(m, torch.nn.Conv1d):
            total = total + m.weight.abs().mean()
    if not isinstance(total, torch.Tensor):
        return torch.tensor(0.0)
    return total
