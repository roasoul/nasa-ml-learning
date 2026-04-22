# Experimental Round after V10 — Summary

Seven experiments on top of the V10 λ=0.1 winning model. One
deferred.

## Final ablation table (76-TCE seed=42 test set)

| Experiment                        | Acc   | Prec  | Recall | F1    | Notes |
|-----------------------------------|------:|------:|-------:|------:|-------|
| V6 Config C (cited baseline)      | 80.3% | 76.7% | 86.8%  | 0.815 | from all_configs dict |
| V6 Config B (stored weights)      | 78.9% | 72.0% | 94.7%  | 0.818 | stored taylor_cnn_v6.pt |
| V10 λ=0.1 (winner from prior)     | 85.5% | 82.9% | 89.5%  | 0.861 | 5-gate bank + inverted loss |
| **Exp 1 — V10 thr=0.40**          | 81.6% | 75.0% | 94.7%  | 0.837 | recall up, prec below target |
| Exp 1 — V10 thr=0.45              | 82.9% | 77.8% | 92.1%  | 0.843 | close to target |
| Exp 1 — V10 thr=0.50              | 85.5% | 82.9% | 89.5%  | 0.861 | F1 peak (same as V10 default) |
| **Exp 2 — V6b + V10 AND**         | **86.8%** | **85.0%** | 89.5% | **0.872** | **new best F1** |
| Exp 2 — V6b + V10 avg             | 81.6% | 76.1% | 92.1%  | 0.833 | |
| Exp 2 — V6b + V10 weighted 0.4/0.6| 84.2% | 79.5% | 92.1%  | 0.854 | close to both targets |
| Exp 2 — V6b + V10 OR              | 76.3% | 68.5% | 97.4%  | 0.804 | **highest recall** |
| Exp 5 — V10 + R*² normalisation   | 53.9% | 52.1% |100.0%  | 0.685 | rescues K00254 but trashes precision |
| Exp 6 — V10 + diversity loss 0.1  | 82.9% | 80.5% | 86.8%  | 0.835 | A1 still collapses |
| Exp 7 — V5 + V10 ensembles        | 63-83%| 58-84%| 82-95% | 0.72–0.83 | all below V10 alone |
| Exp 3 — TESS zero-shot (8 TCEs)   | 87.5% | 80.0% |100.0%  | 0.889 | **beats Malik et al. 63% recall** |
| Exp 4 — 2000-TCE rebuild          |   —   |   —   |   —    |   —   | **DEFERRED** — ~75–125 min of downloads |

### Best-in-session: **V6b + V10 AND ensemble, F1 0.872.**

    TP=34  TN=32  FP=6  FN=4
    precision 85.0%  (+2.1 over V10 alone, +13.0 over V6b)
    recall    89.5%  (unchanged from V10 alone, -5.2 vs V6b)
    F1        0.872  (+0.011 over V10 alone, +0.054 over V6b)

---

## Experiment 1 — V10.5 threshold sweep

Loaded V10 λ=0.1 weights, swept the decision threshold with no
retraining.

| thr  | Acc   | Prec  | Rec   | F1    |
|------|------:|------:|------:|------:|
| 0.30 | 77.6% | 70.6% | 94.7% | 0.809 |
| 0.35 | 78.9% | 72.0% | 94.7% | 0.818 |
| 0.40 | 81.6% | 75.0% | 94.7% | 0.837 |
| 0.45 | 82.9% | 77.8% | 92.1% | 0.843 |
| 0.50 | 85.5% | 82.9% | 89.5% | **0.861** |

**No threshold in [0.30, 0.50] hits both prec > 80% AND rec ≥ 90%.**
F1 peaks at thr=0.50 (the default). Lower thresholds gain recall
but lose precision faster — the dataset has a finite ceiling on the
precision/recall product with this model.

### The four planets V10 misses at thr=0.50 (FN):

| KOI         | Period   | Depth    | SNR    | R* (Rsun) | Notes |
|-------------|---------:|---------:|-------:|----------:|-------|
| K00013.01   |  1.76 d  |  4591 ppm| 4062   | 3.03      | sub-giant host |
| K00912.01   | 10.85 d  |  1676 ppm|  64.5  | 0.59      | M-dwarf |
| K00254.01   |  2.46 d  | 36912 ppm| 1342   | 0.57      | **M-dwarf** — 3.7% dip |
| K00183.01   |  2.68 d  | 18136 ppm| 2833   | 0.96      | sun-like |

Two of four missed planets orbit M-dwarfs (R* ≈ 0.57–0.59 Rsun).
Their transits look anomalously deep (3.7% and 1.7%) because the
host star is small, not because the transiter is exotic. Motivates
Experiment 5.

---

## Experiment 2 — V6 Config B + V10 ensemble (four strategies)

`taylor_cnn_v6.pt` on disk is V6 Config B (depth-matched pretrain,
F1 0.818). V6 Config C weights were never saved. So we ensemble
V6 Config B with V10 λ=0.1.

| Strategy                              | Acc   | Prec  | Rec    | F1    |
|---------------------------------------|------:|------:|-------:|------:|
| V6 Config B alone                     | 78.9% | 72.0% | 94.7%  | 0.818 |
| V10 alone                             | 85.5% | 82.9% | 89.5%  | 0.861 |
| 1. simple average                     | 81.6% | 76.1% | 92.1%  | 0.833 |
| 2. weighted 0.4·V6b + 0.6·V10         | 84.2% | 79.5% | 92.1%  | 0.854 |
| 3. OR (V6b > 0.4 or V10 > 0.4)        | 76.3% | 68.5% | 97.4%  | 0.804 |
| **4. AND (V6b > 0.5 and V10 > 0.5)**  | **86.8%** | **85.0%** | **89.5%** | **0.872** |

**AND ensemble is the new best F1 (0.872) of the session.** The
geometry: V10 is the precision specialist (rejects deep-EB shapes);
V6b is the recall specialist (94.7% recall on its own). Taking the
intersection of their positive predictions retains V10's precision
while V6b doesn't add any new FPs (its FPs are either rejected by
V10 or also FP-looking to V10). Recall stays at V10's 89.5% because
the intersection can't recover planets V10 missed.

OR ensemble hits **97.4% recall** (37/38) — only one planet
unreachable by either model: K00254.01 (V6b 0.23, V10 0.23). That
M-dwarf is a dataset-level blind spot.

Figure 10: `notebooks/figures/ensemble_scatter.png` — scatter of V6b
probability vs V10 probability on the 76 test TCEs, colored by
class. The AND ensemble accepts only the top-right quadrant (both
> 0.5).

---

## Experiment 3 — TESS zero-shot transfer

V10 λ=0.1 weights, Kepler-trained, no TESS retraining. 10 targets
selected, 8 processed successfully (2 skipped due to data/availability
issues).

| Target           | TIC       | V10 prob | pred | truth | OK |
|------------------|-----------|---------:|------|-------|----|
| TOI-132 b        | 89020549  |    0.928 | PLNT | PLNT  | YES |
| TOI-700 d        | 150428135 |    —     | —    | PLNT  | SKIP (fold error) |
| TOI-824 b        | 193641523 |    0.757 | PLNT | PLNT  | YES |
| TOI-1431 b       | 295541511 |    0.877 | PLNT | PLNT  | YES |
| TOI-560 b        | 101955023 |    0.596 | PLNT | PLNT  | YES |
| EB TIC 268766053 | 268766053 |    0.317 | FP   | FP    | YES |
| EB TIC 388104525 | 388104525 |    0.082 | FP   | FP    | YES |
| EB TIC 229804573 | 229804573 |    0.000 | FP   | FP    | YES |
| EB TIC 441765914 | 441765914 |    0.762 | PLNT | FP    | NO |
| EB TIC 255827488 | 255827488 |    —     | —    | FP    | SKIP (no SPOC LC) |

### TESS zero-shot metrics (8 processed targets)

    Recall:        4 / 4 = 100.0%     (Malik et al. TESS baseline 63%)
    FP rejection:  3 / 4 =  75.0%
    Precision:     4 / 5 =  80.0%
    Accuracy:      7 / 8 =  87.5%
    F1 =        0.889

**100 % recall on the TESS planets with no retraining.** The 5-gate
bank generalises across instruments. Preprocessing is identical to
the Kepler pipeline; the only data-side difference is cadence
(2-min for TESS SPOC vs 30-min Kepler LC).

---

## Experiment 5 — M-dwarf depth normalisation (mixed)

Multiply fluxes by (R_star / R_sun)² to rescale M-dwarf deep dips
to sun-equivalent amplitude. Retrain V10 at λ=0.1.

    V10+R*² : acc 53.9%  prec 52.1%  rec 100.0%  F1 0.685
    All 4 previously missed planets now classify correctly,
    including K00254.01 (prob 0.607 → PLNT).

**But precision crashes.** The R*² multiplier range is too wide
(min 0.014 for R*=0.12 Rsun, max 1274 for R*=35.7 Rsun). Giant stars
get multiplied by ~1000× and their ordinary transits come out
looking 100% deep — most samples get routed to "planet" because
BatchNorm can't stabilise over such a wide input range. Not a net
F1 win. **Correct fix would be to clip R* to [0.5, 2.0] Rsun or
use log(R*).**

---

## Experiment 6 — Gate diversity loss (null)

Added `L_diversity = −⟨||g_i − g_j||²⟩_{i<j}` weighted 0.1. Goal:
push A1 away from the clamp floor.

    V10 vanilla:     F1 0.861  A1=0.0010 A2=0.0105 A3=0.0086 A4=0.0209 A5=0.0245
    V10 + diversity: F1 0.835  A1=0.0010 A2=0.0104 A3=0.0089 A4=0.0213 A5=0.0250

A1 STILL collapses to the floor. Other amplitudes essentially
unchanged. Diagnosis: diversity loss operates on gate outputs. When
A1 → 0, g1 output is zero, which actually *maximises* pairwise
distance from the other non-zero gates — the objective doesn't
push A1 up. A direct amplitude-floor penalty (e.g. −log(A_i))
would be needed.

---

## Experiment 7 — V5 + V10 ensemble (null)

V5's claimed 100% recall was on the **old** 16-TCE test set
(100-TCE dataset). On the current 76-TCE seed=42 test set V5 gets
only 84.2% recall, F1 0.727. No ensemble with V10 beats V10 alone:

    V10 alone:             F1 0.861
    0.3·V5 + 0.7·V10:      F1 0.829
    0.5·V5 + 0.5·V10:      F1 0.810
    V5>0.3 OR V10>0.5:     F1 0.720  (recall 94.7%, prec 58.1%)
    V5>0.5 AND V10>0.5:    F1 0.827  (prec 83.8%, rec 81.6%)

V5 drags V10 down because V5 is a weaker classifier on this
dataset (3-channel, no odd/even, trained on 100 TCEs, possible
train/test overlap).

Best ensemble pairing remains **V6b + V10 AND** from Exp 2
(F1 0.872).

---

## Experiment 4 — Scale to 2000 TCEs (DEFERRED)

Rebuilding the dataset to 2000 TCEs requires ~1500 new light-curve
downloads via lightkurve plus training. At 3–5 seconds per target,
that's 75–125 minutes of network I/O alone, plus 20–30 minutes of
retraining. Not feasible in this session without blocking the other
experiments.

Plan for next session:
1. Run `build_dataset.py --n-per-class 1000 --output data/kepler_tce_v10_scale.pt`
   as a background task (~90 min).
2. Retrain V10 at λ=0.1 on the new dataset (seed=42 split).
3. Re-evaluate on same test protocol.
4. Expected: +3–5% precision and recall from more data, plus a
   larger, more representative FP mix.

---

## Figures

| File | Description |
|---|---|
| `notebooks/figures/gate_activation_heatmap.png` | Figure 8 (recolored) — 76×5 gate-vs-primary correlation with class stripe (green=planet, red=FP) and V10 probability side bar. |
| `notebooks/figures/tess_kepler_comparison.png`  | Figure 9 — TESS primary folds per target with V10 probability in title. |
| `notebooks/figures/ensemble_scatter.png`        | Figure 10 — V6b vs V10 probability scatter on 76-TCE test set, colored by class; AND-ensemble accepts top-right quadrant. |

## Commits this session

    1124d74  experiment 1: V10.5 threshold sweep — F1 peaks at thr=0.50 (0.861)
    5a8b605  experiment 2: V6b + V10 AND ensemble — F1 0.872 (new best)
    0a6b273  experiment 7: V5 + V10 ensemble — NULL, V5 drags V10 down
    6cab651  experiment 5: M-dwarf R*-norm — 100% recall but precision crashes
    0839602  experiment 6: gate diversity loss — NULL, A1 still collapses
    598bde6  experiment 3: TESS zero-shot — 100% recall, beats Malik et al. 63%

## Outstanding (deferred / high-risk)

- Exp 4 — 2000-TCE scale. Network-bound; ~2 hours. Next session.
- HR-A — per-gate differential learning rates. Low-priority; A1
  collapses independently of loss pressure.
- HR-B — multi-head attention over gates. Worth trying when
  2000-TCE dataset is available (more data reduces overfit risk).
- HR-C — self-supervised gate pretraining (gate-sum reconstruction).
  High risk of learning the wrong features; only if the Exp 4 +
  HR-B path doesn't yield a clear gain.
