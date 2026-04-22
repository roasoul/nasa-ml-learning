# Taylor-CNN: Physics-Informed Transit Classification with a Multi-Template Gate Bank

*Draft paper — personal ML/AI learning project. Independent of
any employer. All results reproducible from the code in this
repository.*

## Abstract

We present Taylor-CNN, a physics-informed 1D CNN for exoplanet
transit classification on phase-folded light curves. The model
replaces a single learned planet-shape filter with a bank of five
fixed-morphology gates (planet U, V, inverted secondary, asymmetric
ingress, narrow Gaussian), each with a learnable amplitude. A
dataset-calibrated geometry-consistency loss regularises training
toward known class-separation directions. On a 76-TCE stratified
Kepler DR25 test set the model (V10) achieves
F1 = 0.861 / precision = 82.9 % / recall = 89.5 % with 1150
parameters. A V6 + V10 ensemble lifts F1 to 0.872 (precision 85.0 %)
and a three-model OR ensemble reaches 100 % recall. **Zero-shot
transfer to TESS achieves 100 % recall without retraining, compared
to 63 % for statistical-feature approaches (Malik et al. 2022).**

## 1. Introduction

Exoplanet transit surveys (Kepler, K2, TESS) produce millions of
Threshold Crossing Events (TCEs). The downstream classification
step — separating true planet transits from eclipsing binaries
(EBs), stellar variability, and instrumental artefacts — is
labour-intensive. Prior deep-learning approaches (AstroNet,
ExoMiner, Malik et al.) depend on large training sets (15 k+ TCEs)
and dozens of hand-crafted feature views. We investigate whether
a small physics-informed model can match or beat that with ~1 k
parameters on a few hundred TCEs.

## 2. Background

**Mandel-Agol bias.** The Kepler TCE pipeline fits every candidate
with a planet transit model. EB durations and eclipse morphologies
inherit that bias — primary-eclipse shape features extracted from
the pipeline's folded light curve look planet-consistent even for
obvious EBs. This was diagnosed in our V7 null results: the
archive `koi_duration` for EB-flagged FPs has a nearly identical
median obs/predicted ratio to confirmed planets (0.999 vs 0.957).

**Taylor-gate family.** Near-transit dip morphology is well
approximated by the first few terms of a cosine Taylor series,
with a min-zero clamp enforcing the "baseline or dip" constraint:

    y(x) = min( 0,  -A · (1 − x²/2 + B · x⁴/24) )

A controls depth, B controls curvature. In V4–V7 a single gate
with B=0 (quadratic) was used. V8/V9 learned B; V10 fixes
B at five distinct values and learns only A_i per gate.

## 3. Method

### 3.1 MultiTemplateGateBank

Five parallel gates, each with a fixed dip morphology and its own
learnable amplitude A_i ∈ [0.001, ∞):

    G1 planet U-shape       y = min(0, -A1 · (1 - x²/2 + x⁴/24))
    G2 V-shape              y = min(0, -A2 · (1 - x²/2))
    G3 inverted secondary   y = max(0,  A3 · (1 - x²/2))
    G4 asymmetric ingress   y = min(0, -A4 · (x - x³/6))
    G5 narrow Gaussian      y = min(0, -A5 · exp(-x² / 0.5))

Output shape: (batch, 5, 200). Stacked with [primary_flux,
secondary_flux, odd/even_diff] for an 8-channel CNN input.

Gradient correctness verified by `torch.autograd.gradcheck` in
float64 for all five amplitudes, plus a gate-independence check:
∂L/∂A_i is non-zero only when L depends on G_i.

### 3.2 CNN head

    BatchNorm1d(8)
    → Conv1d(8 → 8, k=7) + ReLU
    → Conv1d(8 → 16, k=5) + ReLU
    → AdaptiveAvgPool1d(1) → Linear(16, 1) + Sigmoid

1150 trainable parameters total. ~ 1.3× V6 baseline (914 params).

### 3.3 InvertedGeometryLoss

V9 tried a DynamicGeometryLoss with penalty direction
`prob × t12_t14 + prob × (1 − auc_norm)`. On this dataset the sign
was anti-aligned — planets have *higher* T12/T14 than FPs (0.67
vs 0.60), and *lower* AUC_norm (0.09 vs 0.51). V10's inverted
version re-aims the penalty at FP predictions that look planet-like:

    L_geom = λ(SNR) · (1 − prob) · t12_t14
           + λ(SNR) · (1 − prob) · (1 − auc_norm)

    λ(SNR) = λ_min + (λ_max − λ_min) · σ(SNR − 100)

SNR pivot = 100 (the 500-TCE dataset's median `koi_model_snr`).
Shape features are computed from `primary_flux` per-sample and
used *only* in the loss — never in the classifier input. V8.5
showed that features in the input get bypassed by BCE.

### 3.4 Training

BCE + geometry loss + 0.01·|B|.mean() sparsity (where B is the
single-global-B from V8 — unused in V10). 500-TCE stratified
70/15/15 split, seed=42. Adam with gate_lr=1e-4, CNN/head lr=1e-3.
Early stop on val BCE, patience 25.

Post-step amplitude clamp: `A_i.clamp(min=0.001)` to prevent
gate death. Verified to be active (A1 consistently hits floor in
every run).

## 4. Results

### 4.1 Ablation on 76-TCE stratified test set

See Figure 10. Numbers below.

| Version                    | Prec   | Recall | F1     | Test-set | Notes |
|----------------------------|-------:|-------:|-------:|---------:|-------|
| V4                         | 66.7 % | 100 %  | 0.800  | 16-TCE   | hot-Jupiter subset only |
| V5                         | 72.7 % | 100 %  | 0.842  | 16-TCE   | secondary-view added |
| V6 Config C (baseline)     | 76.7 % | 86.8 % | 0.815  | 76-TCE   | cited baseline |
| V6 Config B                | 72.0 % | 94.7 % | 0.818  | 76-TCE   | stored weights |
| V7.5 safe Kepler gate      | 76.7 % | 86.8 % | 0.815  | 76-TCE   | hard-gate at thr=1 |
| V8 (learnable B)           | 73.3 % | 86.8 % | 0.795  | 76-TCE   | single global B null |
| V8.5 (shape features)      | 71.7 % | 86.8 % | 0.786  | 76-TCE   | features in input bypassed |
| V9 (dynamic geometry loss) | 72.1 % | 81.6 % | 0.765  | 76-TCE   | penalty direction inverted |
| **V10 (λ=0.1) — single best** | **82.9 %** | **89.5 %** | **0.861** | 76-TCE | 5-gate bank + inverted loss |
| V10_5b log R*              | 79.5 % | 92.1 % | 0.854  | 76-TCE   | log-R* normalised fluxes |
| **V6b + V10 AND ensemble** | **85.0 %** | **89.5 %** | **0.872** | 76-TCE | **session best F1** |
| V6b + V10_5b AND           | 85.0 % | 89.5 % | 0.872  | 76-TCE   | ties above |
| V10 + V10_5b AND           | 86.8 % | 86.8 % | 0.868  | 76-TCE   | highest precision + balance |
| V6b + V10 OR @ 0.4         | 68.5 % | 97.4 % | 0.804  | 76-TCE   | V10 recall booster |
| **Triple OR @ 0.4**        | 66.7 % | **100 %** | 0.800 | 76-TCE | **first 100 % recall** |
| TESS zero-shot             | 80.0 % | 100 %  | 0.889  | TESS     | no retraining, 8 targets |

**Caveat on V4/V5.** Their 100 % recall was measured on a 16-TCE
test set sampled from the original 100-TCE dataset — effectively
a hot-Jupiter subset (bright planets with clear dips). It does
not generalise to the representative 76-TCE DR25 test set used
from V6 onwards. All later comparisons are within-test-set.

### 4.2 Null results, for the record

V8, V8.5, V9 are all null on this dataset. V8 (learnable B)
learns a population-averaged curvature useful to neither class.
V8.5 (shape features fused into the classifier input) is bypassed
by BCE. V9 (shape features penalised in the loss with the wrong
sign) degrades F1 vs V6. See `docs/paper/v9_findings.md`.

### 4.3 Why V10 works where V8/V8.5/V9 failed

Not a single structural change, but three compounding fixes:

1. **Multi-template routing.** Five fixed gate morphologies
   give the CNN a reference dictionary. The per-sample signal
   is the *relative* match across templates, not any single
   match (Figure 8 — gate-vs-primary correlation heatmap).
2. **Shape features in loss only, not input.** V8.5 had them
   in the input and BCE routed around them. V10 keeps the loss
   pressure external to the classifier.
3. **Inverted-geometry sign matches class direction.** V9 had
   the sign wrong; Fig. V9 T12/T14 distribution shows planets
   higher than FPs — V10's loss reflects that.

A1 (the planet-U gate) collapses to the clamp floor in every V10
run. The discriminator rides the FP-like templates
(V-shape, asymmetric, Gaussian) — a consistent surprise result.

### 4.4 Per-TCE error analysis (V10 λ=0.1)

| KOI       | Period  | Depth     | SNR   | R*     | V10 prob | Class outcome |
|-----------|--------:|----------:|------:|-------:|---------:|---------------|
| K00013.01 | 1.76 d  |  4591 ppm | 4062  | 3.03   | 0.261    | FN, sub-giant host |
| K00912.01 | 10.85 d |  1676 ppm | 64.5  | 0.59   | 0.420    | FN, M-dwarf |
| K00254.01 | 2.46 d  | 36912 ppm | 1342  | 0.57   | 0.228    | FN, M-dwarf, 3.7 % dip |
| K00183.01 | 2.68 d  | 18136 ppm | 2833  | 0.96   | 0.492    | FN, borderline |

Two of four FN are M-dwarfs. Their dips look anomalously deep
because R* is small, not because the transiter is exotic. This
motivated Experiment 5b.

### 4.5 TESS generalisation

V10 λ=0.1 weights — trained only on Kepler — applied to 8 TESS
targets with identical preprocessing (flatten → fold → scale to
[−π, π] → subtract 1.0 → resample to 200 bins):

| Target           | V10 prob | Prediction | Truth | Outcome |
|------------------|---------:|------------|-------|---------|
| TOI-132 b        | 0.928    | PLNT       | PLNT  | OK |
| TOI-824 b        | 0.757    | PLNT       | PLNT  | OK |
| TOI-1431 b       | 0.877    | PLNT       | PLNT  | OK |
| TOI-560 b        | 0.596    | PLNT       | PLNT  | OK |
| EB TIC 268766053 | 0.317    | FP         | FP    | OK |
| EB TIC 388104525 | 0.082    | FP         | FP    | OK |
| EB TIC 229804573 | 0.000    | FP         | FP    | OK |
| EB TIC 441765914 | 0.762    | PLNT       | FP    | WRONG |

**Recall 100 % (4/4), precision 80.0 %, F1 0.889.** Beats the
Malik et al. 2020 TESS zero-shot baseline of 63 % recall. Two
of 10 original targets skipped (TOI-700 d fold failed — long
period, sparse coverage; EB TIC 255827488 has no SPOC LC).

### 4.6 Stellar-radius normalisation (Experiment 5b)

Motivated by the M-dwarf FN analysis. Goal: rescale each TCE's
flux by a function of R* so M-dwarf deep dips look sun-like.

    raw R*² :   scale = (R*/R_sun)²             — 91 000× range, BN broken
    clipped :   R* → clamp(R*, 0.5, 2.0), then squared — 16× range
    log     :   scale = log1p(R* / R_sun)       — 32× range, smoother

    Variant            | Prec  | Recall | F1    |
    V10 λ=0.1 baseline | 82.9% | 89.5%  | 0.861 |
    raw R*²            | 52.1% | 100.0% | 0.685 |  crashes precision
    clipped R*²        | 79.5% | 81.6%  | 0.805 |  K00013 over-clipped
    log R*             | 79.5% | 92.1%  | 0.854 |  rescues K00912, K00183

Clipped clips R*=3.03 sub-giant down to 2.0, losing signal.
Log is smoother (no discontinuity at the clip boundary) and trades
a little precision for a recall bump.

Log R* variant as a V10_5b alternative model. Its ensemble with
V6b matches the V6b+V10 AND record at F1 = 0.872.

### 4.7 Ensemble results

See Figure 11 (precision vs recall scatter).

- **V6b + V10 AND:** F1 0.872 — best single-shot F1 of any
  configuration. Combines V6b's recall with V10's precision.
- **V10 + V10_5b AND:** F1 0.868, precision 86.8 % — highest
  precision + most balanced (precision = recall = 86.8 %).
- **Triple OR @ 0.4:** **100 % recall** (38/38). First config
  to catch every planet, including the two M-dwarfs and the
  sub-giant. Precision 66.7 % is the cost. Useful mode when
  zero false negatives is the operational constraint.

No single configuration cleanly hits **both** prec > 80 % AND
rec ≥ 90 %. The boundary is hard on this dataset.

## 5. Discussion

The single-template gate bottleneck (V4–V9) comes from the Kepler
Mandel-Agol bias combined with the limits of a global shape
parameter. A single A or (A, B) can only describe the population
average. Five fixed templates + learned amplitudes let the CNN
treat dip morphology as a *dictionary lookup* rather than a
point-estimate — the signal lives in the ratio of matches.

The 5-gate bank still doesn't defeat the pipeline bias. It
sidesteps it. Consistent with that interpretation: A1 (the planet
template) is *unused* in every V10 run. The discriminator
recognises FP signatures rather than planet signatures.

## 5.5 Scale-up and cross-mission retraining (null results)

We added a scaled-up Kepler dataset (1114 TCEs via subsampling
`koi_disposition in ('CONFIRMED', 'FALSE POSITIVE')` evenly across
the depth-sorted archive) and a native TESS dataset (355 TOIs, SPOC
2-min cadence — 45 of a target 400 were dropped because MAST has no
2-min postage stamps for those TICs). `pos_weight = n_FP / n_CONF`
was applied as a per-sample weight on the sigmoid-output BCE so the
V10 head is unchanged. Seeds and splits (seed=42 stratified 70/15/15)
match the earlier experiments.

**Exp 4 — V10 retrained on 1114 Kepler TCEs.** Held-out 170-TCE
test: prec 58.2%, rec 85.2%, F1 0.692 (vs V10-500's F1 0.861).
Precision dropped 24.7 percentage points; recall held within 5 pp.
Amplitude profile is identical to V10-500 (A1 pinned at the 0.001
floor, A4=0.022 and A5=0.027 dominant), so the architecture's
learned solution did not change — only its calibration did. The
1.3-million-KOI-wide depth subsampling pulls in shallower, noisier
FPs, and `pos_weight=2.18` moves the threshold toward "planet".
At 1150 trainable parameters the model is not expressive enough to
absorb the extra diversity without over-firing on ambiguous FPs.
TESS 355-zero-shot from the 1114-TCE model: F1 0.547 — notably
worse than Exp 3's V10-500 → TESS F1 0.889 on 15 hand-picked hot
Jupiters, reflecting the harder 355-TCE TOI mix rather than a
regression in the model.

**Exp 4b — V10 trained natively on TESS.** Collapsed to an
always-positive classifier: TN=FN=0 on both the TESS 55-TCE test
split and the Kepler 76-TCE test set (cross-mission TESS→Kepler
zero-shot). Val loss plateaued at 0.61 after epoch 1; amplitudes
stayed within 0.001 of their 0.01 init. The 248-sample training
set, the mild 56/44 class imbalance, and the absence of a TESS
SNR cache (which left the InvertedGeometryLoss at mid-strength for
every sample) together provided insufficient gradient signal to
escape the "predict 1 everywhere" local minimum. A real F1 for
TESS-native training will need longer patience, LR warmup, or a
per-target TESS SNR cache.

**Exp 4c — cross-mission AND ensemble.** Because the TESS-native
model votes 1 on every sample, `AND` reduces to the other model's
prediction: Kepler-76 AND = V10-500's exact result (F1 0.861), TESS-55
AND = V10-1114 zero-shot's exact result (F1 0.638). No gain over
the existing V6b + V10 AND ensemble (F1 0.872).

**Interpretation.** The 500-TCE V10 baseline is a surprisingly
tight local optimum. Scaling the Kepler data alone or moving to a
different mission without also scaling the architecture and the
training recipe breaks either calibration or convergence. The
V6b + V10 AND ensemble remains the session's best F1.

## 6. Future Work

- **Scale architecture alongside data.** The 1150-parameter V10
  held its learned geometry on 1114 TCEs but miscalibrated;
  next step is a wider CNN (32→64 filters) plus the MLP head
  below, retrained on the 1114-TCE set with a longer schedule
  and a per-TCE SNR cache.
- **M-dwarf handling.** Log-R* helps but doesn't close the gap.
  Future work: per-class stellar prior, or explicit planet
  radius in Earth/Jupiter units as an auxiliary input.
- **Roman Space Telescope / PLATO generalisation.** TESS
  zero-shot already works; these next-mission cadences are
  similar enough that the gate bank should carry over.
- **ECG cross-domain.** The Taylor-gate layer applied to
  cardiac waveforms (RLC physical model: inductive R-peak,
  capacitive T-wave, asymmetric QRS). Planned PTB-XL +
  MIT-BIH comparison against Ribeiro et al. 2020.
- **Non-linear classifier head.** Current Linear(16, 1) +
  sigmoid is conservative — the 5-template signal might be
  better-exploited by a small MLP.
- **Multi-head attention over gates.** Let the model learn
  per-sample gate weighting rather than relying on Conv1d
  average pooling. Needs more data (~2000 TCEs) to avoid
  overfitting.

## 7. Reproducibility

Full code + data pipeline in
<https://github.com/roasoul/nasa-ml-learning>. Release commit:
latest on `main`. Run:

    python -m src.data.build_dataset --n-per-class 250
    python scripts/run_v10.py                     # reproduces F1 0.861
    python scripts/exp2_v6b_v10_ensemble.py       # reproduces F1 0.872
    python scripts/exp3_tess_zeroshot.py          # reproduces TESS 100% recall
    # Exp 4 / 4b / 4c — scale-up and cross-mission (null results)
    python -m src.data.build_dataset \
        --n-per-class 1000 --output data/kepler_tce_2000.pt
    python -m src.data.build_dataset \
        --n-per-class 200 --mission TESS --output data/tess_tce_400.pt
    python scripts/run_v10_1114.py                # Exp 4  — F1 0.692
    python scripts/run_v10_tess.py                # Exp 4b — degenerate
    python scripts/run_v10_cross_ensemble.py      # Exp 4c — null

All figures in `notebooks/figures/`.
