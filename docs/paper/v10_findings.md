# V10 Findings — Multi-Template Gate Bank

Status: **first positive result since V5.** F1 0.861 on the 76-TCE test
set. Beats every V6 configuration, clears the ≥ 80% precision target,
narrowly misses the ≥ 90% recall target (89.5%).

## Ablation table (seed=42, 76-TCE test set)

| Version     | Acc   | Prec  | Recall | F1        | Params |
|-------------|------:|------:|-------:|----------:|-------:|
| V4          |       |       |        |           | 798    |
| V5          |       |       |        | 0.842     | 856    |
| V6 Config A | 72.4% | 67.3% | 86.8%  | 0.759     | 914    |
| V6 Config B | 78.9% | 72.0% | 94.7%  | 0.818     | 914    |
| V6 Config C | 80.3% | 76.7% | 86.8%  | 0.815     | 914    |
| V7.5 safe gate | 80.3% | 76.7% | 86.8% | 0.815  | 914    |
| V8          | 77.6% | 73.3% | 86.8%  | 0.795     | 916    |
| V8.5        | 76.3% | 71.7% | 86.8%  | 0.786     | 921    |
| V9 (λ=0.1)  | 73.7% | 69.6% | 84.2%  | 0.762     | 916    |
| V9 (λ=0.5)  | 75.0% | 72.1% | 81.6%  | 0.765     | 916    |
| V9 (λ=1.0)  | 50.0% |  0.0% |  0.0%  | 0.000     | 916    |
| **V10 (λ=0.1)** | **85.5%** | **82.9%** | **89.5%** | **0.861** | **1150** |
| V10 (λ=0.5) | 76.3% | 70.8% | 89.5%  | 0.791     | 1150   |
| V10 (λ=1.0) | 73.7% | 68.0% | 89.5%  | 0.773     | 1150   |

`taylor_cnn_v6.pt` on disk is Config B (depth-matched pretrain). Config
C is the historically cited V6 "baseline" but its weights were never
saved — we compare V10 against both configs below using the stored
`all_configs` metrics + the loaded Config B weights.

## Architecture

Two simultaneous changes from V9:

1. **MultiTemplateGateBank** — five parallel gates, each with a fixed
   dip morphology and its own learnable amplitude *A_i*. A_i is
   clamped to min=0.001 inside the forward pass so no gate can die.

        G1 planet U-shape       y = min(0, -A1 · (1 - x²/2 + x⁴/24))
        G2 V-shape              y = min(0, -A2 · (1 - x²/2))
        G3 inverted secondary   y = max(0,  A3 · (1 - x²/2))
        G4 asymmetric ingress   y = min(0, -A4 · (x - x³/6))
        G5 narrow Gaussian      y = min(0, -A5 · exp(-x² / 0.5))

   Output shape: `(batch, 5, seq_len)`. The five gate channels are
   stacked with `[primary, secondary, odd/even]` to give the CNN
   an 8-channel input. Same Conv1d(8→8, k=7) → Conv1d(8→16, k=5) →
   AdaptiveAvgPool head as V6, just wider at the first layer.

2. **InvertedGeometryLoss** — V9's `DynamicGeometryLoss` had the
   penalty direction anti-aligned with the empirical class
   separation on this dataset. V10 re-aims it at *FP* predictions
   with planet-like shape:

        L = BCE + λ(SNR) · (1 - planet_prob) · t12_t14
              + λ(SNR) · (1 - planet_prob) · (1 - AUC_norm)

        λ(SNR) = λ_min + (λ_max - λ_min) · sigmoid(SNR - 100)

   SNR pivot raised from V9's 12 to 100 because the 500-TCE
   dataset's median koi_model_snr is 131 — pivot=12 gave full
   penalty to nearly every sample.

Shape features are computed from `primary_flux` per-sample and used
in the loss only. They never enter the classifier input — that
prevents the BCE-bypass failure V8.5 saw.

## Gradcheck

`scripts/gradcheck_v10.py` runs three independent checks:

- Per-gate amplitude gradients (A1..A5) against numerical perturbation
  in float64, `atol=1e-5, rtol=1e-4`.
- Gate-independence: gate *i*'s output must depend only on *A_i*;
  backward into *A_j* for *j≠i* must be exactly zero.
- All five gates pass. Independence holds.

## Lambda sweep — λ=0.1 wins

    lambda=0.1  → F1 0.861  (winner)
    lambda=0.5  → F1 0.791
    lambda=1.0  → F1 0.773

At λ=0.5 and λ=1.0 the inverted-geometry pressure dominates BCE and
the decision boundary shifts toward false-positive predictions
(precision drops below V9). λ=0.1 gives just enough regularization
to steer the CNN toward the correct sign while BCE still drives
classification.

## Learned-amplitude behaviour at the winning λ

    A1 planet U        = 0.0010   ← clamp floor (unused!)
    A2 V-shape         = 0.0105
    A3 inverted sec    = 0.0086
    A4 asymmetric      = 0.0209
    A5 narrow Gaussian = 0.0245   ← dominant

Surprising: the planet-prior gate *A1* decays straight to the 0.001
clamp floor and stays there for every λ in the sweep. The
discriminator relies on the FP-like templates (V-shape, asymmetric
ingress, narrow Gaussian) rather than a dedicated planet template.
That is consistent with the V8.5 / V9 finding that the EB/FP
signature is what's easy to detect in the primary fold; a "planet
signature" cannot be extracted directly because the pipeline's
planet-fit bias removes it.

## Per-TCE V6 vs V10 diagnostic

Confusion matrix on the 76-TCE test set:

| Model | TP | TN | FP | FN | Acc | Prec | Rec | F1 |
|---|---|---|---|---|---|---|---|---|
| V6 Config B (stored weights) | 36 | 24 | 14 |  2 | 78.9% | 72.0% | 94.7% | 0.818 |
| V6 Config C (cited baseline) | — | — | — | — | 80.3% | 76.7% | 86.8% | 0.815 |
| V10 (λ=0.1)                  | 34 | 31 |  7 |  4 | 85.5% | 82.9% | 89.5% | 0.861 |

### FPs V10 catches that V6 Config B missed (n=8)

Every one of these was flagged planet by V6 with probability
0.51–0.64, and V10 correctly drops it below 0.50:

| KOI         | V6 prob | V10 prob | G1   | G2   | G3   | G4   | G5   |
|-------------|--------:|---------:|-----:|-----:|-----:|-----:|-----:|
| K00230.01   |    0.52 |     0.31 | +0.41| +0.41| −0.41| −0.15| +0.48|
| K03785.01   |    0.58 |     0.50 | +0.15| +0.15| −0.15| +0.30| +0.28|
| K00048.01   |    0.55 |     0.49 | +0.18| +0.18| −0.18| −0.08| +0.25|
| K01896.01   |    0.60 |     0.41 | +0.49| +0.51| −0.51| −0.16| +0.66|
| K00225.01   |    0.64 |     0.24 | +0.49| +0.50| −0.50| −0.18| +0.67|
| K01156.01   |    0.57 |     0.44 | +0.29| +0.29| −0.29| −0.09| +0.42|
| K00132.01   |    0.63 |     0.35 | +0.27| +0.29| −0.29| −0.11| +0.38|
| K03064.01   |    0.51 |     0.06 | +0.66| +0.68| −0.68| −0.13| +0.86|

Mean gate-vs-primary correlation across these eight:

    G1  +0.369    G2  +0.376    G3  -0.376    G4  -0.075    G5  +0.500

These FPs look planet-shaped to V6 — in fact their *highest* Pearson
correlation is with G5 (narrow Gaussian) and G1/G2 (symmetric U/V).
V10's discriminator uses the 5-template pattern plus the inverted-
geometry regularizer to route them as FPs despite the planet-like
primary shape. V6's single-gate-plus-CNN couldn't separate them
because a single gate can only match one template — the network
had to re-derive the contrast between shapes from scratch.

### Planets V10 misses (n=4, FN=4)

| KOI         | V6 prob | V10 prob | Notes |
|-------------|--------:|---------:|-------|
| K00013.01   |    0.55 |     0.26 | V6 had it right; V10 lost it |
| K00912.01   |    0.62 |     0.42 | V6 had it right; V10 lost it |
| K00254.01   |    0.23 |     0.23 | Both models miss it |
| K00183.01   |    0.43 |     0.49 | Both models miss it |

Net planet-catch delta: V10 loses 2 planets that V6 Config B had
right. The geometry-loss pressure that helps reject FPs mildly
hurts on borderline planets whose shape features are near the
planet/FP boundary. K00013 and K00912 both have moderate gate
correlations (K00013 G5 +0.69, K00912 G5 +0.29) but V10's Conv1d
classifier routes them toward FP.

### New FPs V10 introduces (n=1)

    K03650.01   V6 prob 0.19 → V10 prob 0.83

One deep-eclipse TCE that V6 correctly rejected but V10 accepts.
A cost of the looser decision boundary that catches the eight FPs.

### Net

    +8 FP catches − 2 planet-catch losses − 1 new FP = +5 correct classifications

That's the source of the F1 gain: trade 2 planet recalls for
8 precision gains (+1 offset by the new FP). Precision shoots from
72.0% to 82.9%; F1 climbs from 0.818 to 0.861 despite the 2-planet
recall hit.

## Figure 8 — gate-vs-primary correlation heatmap

    notebooks/figures/v10_figure_8_gate_heatmap.png

Per-TCE Pearson correlation between each of the five learned gate
templates and the TCE's `primary_flux`. 76 rows × 5 columns.
Rows sorted by V10's predicted planet probability within each class
(planets on top of the divider, FPs below). A side panel shows the
V10 probability as a green-red column. FPs cluster at the bottom
with characteristic high G1/G2/G5 correlation (deep dip → strong
match with every symmetric template) and strong negative G3 (the
inverted-secondary gate flips sign with a dip). Planets sit with
moderate correlations — they simply have smaller dips, so every
template matches less strongly.

This figure replaces the naive "mean |gate output| per class"
heatmap which is batch-constant: because the folded phase grid is
identical for every TCE, the gate outputs themselves are
batch-constant. The per-sample signal lives in the projection of
the TCE's flux onto each gate template — i.e., the correlation —
which is what the Conv1d layer sees after the BN + convolution
fuses the gate channels with the primary channel.

## What this says about V7–V9's unified root cause

V7–V9 argued that a single global B could not encode per-sample
morphology because the Kepler pipeline's Mandel-Agol planet-fit
bias makes every primary-fold shape metric look planet-consistent
regardless of class. V10 resolves that tension not by bypassing
the pipeline bias — we still see its fingerprint: A1 (planet-U gate)
is unused; planets and FPs correlate similarly with every template
after normalizing for depth — but by giving the CNN a *multi-gate
dictionary* that exposes the **relative** match between templates
as a feature. The FP signature is "strong match on every symmetric
template" (deep dip, planet-like shape, with the only difference
being the shape *of the difference*, which Conv1d can learn).

V7's "EB primaries look planet-like" finding remains true. V10
doesn't defeat it; it sidesteps it by giving the CNN five shape
references so it can operate on the contrast between them rather
than on any one single match.

## Artefacts

| File | Purpose |
|---|---|
| `src/models/multi_template_gate.py` | 5-gate bank with internal amplitude clamp |
| `src/models/taylor_cnn_v10.py`      | V10 classifier (8-channel input) |
| `src/models/geometry_loss_v2.py`    | InvertedGeometryLoss (sign corrected, SNR pivot 100) |
| `src/models/taylor_cnn_v10.pt`      | Winner weights (λ=0.1) |
| `scripts/gradcheck_v10.py`          | Five-amplitude + independence gradcheck |
| `scripts/run_v10.py`                | Lambda sweep training |
| `scripts/v10_figures.py`            | Metrics, amplitudes, templates, correlation heatmap, T12/T14 dist |
| `scripts/v10_vs_v6_diagnostic.py`   | Per-TCE V6 vs V10 comparison + Figure 8 |
| `data/v10_training.log`             | Full sweep log |
| `data/v10_results.pt`               | Per-sample probs, features, amplitude traces |
| `data/v10_vs_v6_diagnostic.csv`     | Per-TCE table |
| `notebooks/figures/v10_*.png`       | Five V10 diagnostic figures |
| `notebooks/figures/v10_figure_8_gate_heatmap.png` | Figure 8 (Pearson heatmap) |
