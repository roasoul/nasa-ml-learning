# CLAUDE.md — NASA ML Learning Project

## Project Context
This is a personal ML/AI learning project. All code here is independent of my
employer (MathWorks). I retain full IP ownership of everything in this repo.

**Goal:** Build ML/AI skills targeting a NASA Force AI/ML or Data role.
**Current phase:** V10 + all experiments complete. Best single model
V10 λ=0.1 F1=0.861. Best ensemble V6b+V10 AND F1=0.872. Triple-OR
ensemble hits **100% recall** (38/38). TESS zero-shot 100% recall,
80% precision. Exp 5b log-R* normalisation: F1 0.854 (rescues 2
of 4 M-dwarf/radius-mismatch FNs, ties V6b+V10 AND in ensembles).
Full paper draft at `docs/paper/paper_draft.md`. Exp 4/4b/4c run to
completion: Kepler 1114-TCE scale-up hurt precision (F1 0.692),
TESS-native collapsed to always-positive (F1 0.721 but TN=FN=0), and
the cross-mission AND ensemble reproduces the non-degenerate constituent
— no gain over V10 500-TCE F1 0.861 / V6b+V10 AND F1 0.872.
**Degree:** MS AI Engineering at Quantic (starting June 2025)

---

## Project Structure
```
C:\Users\skapa\Projects\ml-learning\nasa-ml-learning\
├── CLAUDE.md
├── requirements.txt
├── nasaml\                   # Virtual environment — never edit
├── notebooks\
│   ├── 00_installation_test.ipynb
│   ├── 01_taylor_gate_verification.ipynb   ✅ complete
│   ├── 02_synthetic_training.ipynb          ✅ complete
│   ├── 03_real_kepler_data.ipynb            ✅ complete
│   ├── 04_real_data_training.ipynb          ✅ complete
│   └── 05_secondary_view.ipynb              ✅ complete (V5)
├── src\
│   ├── data\
│   │   ├── synthetic.py        ✅ 3-class generator (planet/EB/non-transit)
│   │   ├── kepler.py           ✅ primary + secondary fold pipeline
│   │   └── build_dataset.py    ✅ saves fluxes_secondary
│   ├── models\
│   │   ├── taylor_layer.py     ✅ TaylorGateFunction + TaylorGateLayer
│   │   ├── taylor_cnn.py       ✅ 3-channel TaylorCNN (V5, 856 params)
│   │   └── taylor_cnn_v5.pt    ✅ V5 trained weights
│   └── utils\
├── scripts\
│   └── run_v5.py               ✅ V5 training/eval script
└── data\                     # NOT committed — rebuild with build_dataset.py
    └── kepler_tce.pt          100 TCEs w/ primary + secondary flux
```

---

## Environment Setup (Windows)
```bash
cd C:\Users\skapa\Projects\ml-learning\nasa-ml-learning
nasaml\Scripts\activate
python -c "import torch; print(torch.cuda.is_available())"

# Install packages — ALWAYS use PyPI explicitly (rogue Artifactory config)
pip install <package> --index-url https://pypi.org/simple/

# PyTorch — RTX 5060 Blackwell (sm_120) needs nightly cu128
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip freeze > requirements.txt
```

---

## Hardware
- **GPU:** NVIDIA GeForce RTX 5060 Laptop (Blackwell, sm_120), 8GB VRAM
- **CUDA:** 12.8 / Driver: 591.74
- **PyTorch:** 2.12.0.dev nightly — required for sm_120 support

---

## Installed Libraries
- torch 2.12.0 / torchvision / torchaudio (CUDA 12.8 nightly)
- numpy 2.4.3 / pandas 2.3.3 / matplotlib 3.10.8 / scikit-learn 1.8.0
- lightkurve 2.6.0 / astropy 7.2.0
- jupyter / transformers / sentence-transformers

---

## Known Issues & Fixes

### pip
- Artifactory intercepts pip — always use `--index-url https://pypi.org/simple/`
- lightkurve oktopus warning on import is harmless

### Preprocessing (CRITICAL)
```python
# CORRECT order — always follow this exactly
lc.flatten(sigma_upper=4, sigma_lower=10)  # asymmetric — protects dip
.normalize()
.fold(period, epoch_time)
# then scale phase to [-pi, pi] and subtract 1.0 so baseline=0, dip=negative
```
- Default sigma clipping DELETES transit dips — use sigma_lower=10
- Apply outlier removal AFTER folding, never before

### Taylor Gate (CRITICAL)
```
sin(x) dip at pi/2 = 1.57 rad  →  WRONG (misaligned with fold center)
cos(x) dip at 0                 →  CORRECT (fold centers transit at 0)

Forward:  y = min(0, -A * (1 - x**2/2))
Backward: grad_A = -(1 - x**2/2) * mask
Mask:     1 in dip (output < 0), 0 at baseline
```

### Training
- BatchNorm essential before Conv1d — transit depths ~1% produce tiny gate
  outputs. Without BN, gradients ~0.001, accuracy stuck at 50%.
- Always fine-tune on real Kepler TCEs — synthetic-only model gets ~50% on real data
- EB false positives are expected — binary classifier needs secondary eclipse
  view to distinguish planets from eclipsing binaries

### V5 Known Issues
- **Synthetic pretraining BN running-stats mismatch.** Pretraining on synthetic
  3-class data then fine-tuning on real Kepler causes fine-tune loss to start
  at ~1.9 (worse than chance) and never recover. BatchNorm running stats
  computed on synthetic data don't match real-data distribution. Taylor A also
  overshoots to ~0.08 during pretrain (tuned for synthetic 1-2% depths) and
  can't anneal back to real-data optimum (~0.015).
  **Fix for V6:** reset BN running stats with `model.cnn[0].reset_running_stats()`
  between pretrain and finetune, OR match synthetic depth distribution to
  real Kepler depth histogram (median ~1300 ppm, not 1-2%).
- **3 surviving FPs on 16-TCE test are non-EB.** K05999.01, K00839.01,
  K01829.01 have flat secondary channels (~0 dip at phase=0) — they are FPs
  for reasons unrelated to eclipsing binaries (stellar activity, centroid
  offsets, V-shape grazing transits, instrumental artifacts). Secondary-view
  classification cannot catch them.
  **Fix for V6:** add centroid-offset check, odd/even transit depth comparison,
  or V-shape transit analysis as additional channels.

### Post-V10 experiment round: `docs/paper/experiments_final.md`
Seven experiments on top of V10 λ=0.1. Full ablation table:

| Experiment                    | Acc   | Prec  | Recall | F1    |
| V6 Config C (baseline)        | 80.3% | 76.7% | 86.8%  | 0.815 |
| V6 Config B (stored)          | 78.9% | 72.0% | 94.7%  | 0.818 |
| V10 λ=0.1                     | 85.5% | 82.9% | 89.5%  | 0.861 |
| Exp 1 — V10 thr=0.40          | 81.6% | 75.0% | 94.7%  | 0.837 |
| **Exp 2 — V6b+V10 AND**       | **86.8%** | **85.0%** | 89.5% | **0.872** |
| Exp 2 — V6b+V10 OR            | 76.3% | 68.5% | 97.4%  | 0.804 |
| Exp 3 — TESS zero-shot        | 87.5% | 80.0% | 100%   | 0.889 |
| Exp 5 — R*² norm              | 53.9% | 52.1% | 100%   | 0.685 |
| Exp 6 — diversity loss        | 82.9% | 80.5% | 86.8%  | 0.835 |
| Exp 7 — V5+V10 best           | 82.9% | 83.8% | 81.6%  | 0.827 |
| Exp 4 — V10 on 1114 Kepler   | 75.9% | 58.2% | 85.2%  | 0.692 |
| Exp 4 — 1114 no weight (170) | 81.2% | 76.2% | 59.3%  | 0.667 |
| Exp 4 — 1114 no weight (76)  | 69.7% | 89.5% | 44.7%  | 0.596 |
| Exp 4 — 1114 → TESS (zero)   | 48.2% | 53.6% | 55.8%  | 0.547 |
| Exp 4 — 1114 no wt → TESS    | 49.3% | 60.0% | 28.6%  | 0.388 |
| Exp 4b — TESS native (degen) | 56.4% | 56.4% | 100%   | 0.721 |
| Exp 4b — TESS→Kepler (degen) | 50.0% | 50.0% | 100%   | 0.667 |
| Exp 4c — AND ensemble Kep 76 | 85.5% | 82.9% | 89.5%  | 0.861 |
| Exp 4c — AND ensemble TESS55 | 54.5% | 57.9% | 71.0%  | 0.638 |

**Key new findings:**
- **V6b+V10 AND ensemble is the new session best.** Precision
  jumps from 82.9% to 85.0% by intersecting with V6b's predictions;
  V6b's FPs are all either rejected by V10 or also suspect.
- **TESS zero-shot: 100% recall (4/4), 80% precision.** V10 weights
  trained only on Kepler generalise to TESS without retraining.
  Beats Malik et al. 2020 TESS baseline (63% recall).
- **M-dwarf confirmation:** 2 of 4 V10 FN are M-dwarfs (K00254 R*=0.57,
  K00912 R*=0.59). R*² normalisation rescues them → 100% recall
  but destabilises training (precision 52%). Fix: clip R* or use
  log(R*) instead.
- **K00254.01 is a dataset-level blind spot.** No ensemble / no
  threshold catches it; both V6 and V10 give it prob ≈ 0.23.
  Only the R*²-normalised retrain gets it right.
- **A1 collapse is structural.** Diversity loss didn't help because
  the penalty operates on gate outputs, and A1 → 0 *helps* pairwise
  distance. Needs direct amplitude penalty if we want A1 alive.

**Session commits:**
- 1124d74 V10.5 threshold sweep
- 5a8b605 V6b+V10 AND ensemble → new best F1 0.872
- 0a6b273 V5+V10 ensemble → null
- 6cab651 M-dwarf R*² normalisation
- 0839602 gate diversity loss → null
- 598bde6 TESS zero-shot → 100% recall
- 2f20d70 final report + Fig 8 + Fig 10
- 3d5dacb Exp 5b clipped + log R* + triple OR 100% recall
- 1d0cb13 build_dataset: --resume, --mission TESS, class imbalance log
- 0e25f09 Exp 4: V10 on 1114 TCEs → null (F1 0.692, precision collapse)
- 3a2c16b Exp 4b: TESS-native V10 → null (degenerate always-planet)
- 3707646 Exp 4c: cross-mission AND → null (equals best constituent)
- caece6a Exp 4 no-weight: precision +18pp but recall halves, still null

### Exp 4 / 4b / 4c (scale-up + cross-mission) — null results
- **Exp 4 — V10 on 1114 Kepler TCEs.** Class weighting pos_weight=2.18
  (764/350) applied as weighted BCE on the sigmoid head. Seed=42
  stratified 70/15/15 = 778/166/170 split. Early stop at ep 154.
  Kepler held-out: acc 75.9%, **prec 58.2%**, rec 85.2%, F1 0.692
  (vs V10 500's 0.861). Precision drop = 24.7pp; recall drop = 4.3pp.
  Amplitude pattern matches V10 500 (A1 at 0.001 floor, A4=0.022 and
  A5=0.027 dominant). The larger training set + shallow-FP tail from
  subsample_evenly across 9564 KOIs + boundary shift from pos_weight
  combined to drag precision. Scaling data without scaling the
  1150-param architecture did not help. TESS 355 zero-shot from this
  model: F1 0.547 (vs Exp 3 V10-500's F1 0.889 on the 15-TCE hand-
  picked hot-Jupiter set — the 355 mix is a much harder distribution).
- **Exp 4 no-weight — isolates pos_weight from data difficulty.** Same
  1114 setup, plain `nn.BCELoss()`. Precision recovered to 76.2% on
  the 170-TCE held-out (and 89.5% on the paper 76-TCE test set) — so
  pos_weight was part of the problem. But recall halved: 85.2% →
  59.3% on the 170 test, 44.7% on the 76 test. F1 still worse than
  V10-500 (0.667 on 170, 0.596 on 76). TESS 355 zero-shot degrades
  further (F1 0.388, recall 28.6%). Amplitudes also weaker than V10-
  500 — A4 0.012 vs 0.021, A5 0.023 vs 0.025. **Verdict: the 1114-TCE
  distribution is genuinely harder than the curated 500, and
  pos_weight only slides the precision/recall tradeoff along a curve
  that never reaches V10-500's F1.** Architecture scaling is needed
  alongside the larger dataset.
- **Exp 4b — TESS-native V10.** 355 TCEs (199 conf / 156 FP),
  pos_weight=0.784 (156/199), SNR defaulted to pivot=100 for every
  sample (no TESS SNR cache). Seed=42 split = 248/52/55. Val loss
  plateaued at 0.61 after ep 1, gate amplitudes barely moved from
  init (A1..A5 ≈ 0.01 vs init 0.01). Model collapsed to always-
  planet (TN=0, FN=0 on both TESS test split and Kepler 76-TCE
  zero-shot). Degenerate F1 0.721 TESS-native, 0.667 TESS→Kepler
  — not a real F1. Fix candidates: longer patience, LR warmup,
  real SNR cache, stronger class weighting.
- **Exp 4c — cross-mission AND ensemble.** Because TESS-native
  predicts 1 for every sample, `(p_TESS > 0.5) AND (p_other > 0.5)`
  reduces to `p_other > 0.5`. On Kepler 76 the AND F1 is identically
  V10 500's 0.861; on TESS 55 the AND F1 is identically V10 1114's
  zero-shot 0.638. No gain — V6b+V10 AND F1 0.872 remains session best.

### Data infrastructure — build_dataset.py (now multi-mission)
`src/data/build_dataset.py` supports: `--mission Kepler|TESS`, `--resume
path.pt` (skips already-downloaded TCEs by name, appends new ones), `--seed`,
and always prints the class-balance report + recommended `pos_weight`
(use with `nn.BCEWithLogitsLoss(pos_weight=torch.tensor(...))`). TESS fetch
uses the TAP `toi` table (no `st_mass` column — defaulted to 1.0) and
converts BJD→BTJD (−2457000) to align with SPOC `lightkurve.time`.
Datasets now on disk:
- `data/kepler_tce_2000.pt` — 1114 TCEs (350/764), pos_weight 2.183
- `data/tess_tce_400.pt` — 355 TCEs (199/156), pos_weight 0.784 (45 of
  400 targets dropped because MAST has no SPOC 2-min data for those TICs)

### Paper draft v1: `docs/paper/paper_draft.md`
End-to-end write-up — abstract, method, ablation (V4→V10+ensembles),
error analysis, TESS generalisation, stellar-R* normalisation,
future work. Ready for external review.

### V10 paper draft (prior): `docs/paper/v10_findings.md`
Full write-up with ablation table (V4→V10), per-TCE V6 vs V10
diagnostic identifying the 8 FPs V10 catches and the 2 planets it
loses, Figure 8 (gate-vs-primary Pearson heatmap), and the
architectural-root-cause note explaining why multi-template routing
works where single-global-B (V8–V9) did not.

### V10 — Multi-template gate bank + InvertedGeometryLoss (FIRST GAIN since V5)
- **Winning config:** 5 parallel fixed-morphology gates, 8-channel CNN
  (5 gates + primary + secondary + odd/even), InvertedGeometryLoss
  at lambda_max=0.1, SNR pivot=100.
- **Metrics on 76-TCE test set:**
    | Version | Acc | Prec | Recall | F1 | Params |
    | V6 C   | 80.3% | 76.7% | 86.8% | 0.815 | 914 |
    | V7.5   | 80.3% | 76.7% | 86.8% | 0.815 | 914 |
    | V8     | 77.6% | 73.3% | 86.8% | 0.795 | 916 |
    | V8.5   | 76.3% | 71.7% | 86.8% | 0.786 | 921 |
    | V9 best | 75.0% | 72.1% | 81.6% | 0.765 | 916 |
    | **V10** | **85.5%** | **82.9%** | **89.5%** | **0.861** | **1150** |
  Precision target (>80%) MET. Recall target (>90%) narrowly missed
  by one TCE (34/38 = 89.5%). F1 jumps 0.046 over V6.
- **Lambda sweep** (InvertedGeometryLoss):
    - lambda=0.1 → F1 0.861  (best, used in winner model)
    - lambda=0.5 → F1 0.791
    - lambda=1.0 → F1 0.773
  Stronger penalty degrades — at higher lambda the "push FP-predictions
  up when shape is planet-like" term dominates BCE and shifts the
  decision boundary. lambda=0.1 gives just enough pressure to help
  without overriding BCE.
- **Why V10 works where V8/V8.5/V9 failed:**
  - Single global B (V8-V9) learns a population average — useless to
    both classes. Five fixed morphologies cover the morphology space:
    U-shape, V-shape, inverted (secondary bump), asymmetric ingress
    (sin-Taylor), narrow Gaussian spike. The CNN learns which
    *combination* of gate signals distinguishes a TCE.
  - Shape features in the LOSS only (never in the CNN input) means
    BCE cannot trivially bypass them. V8.5 had features in the input
    and the classifier routed around them.
  - InvertedGeometryLoss's sign matches the empirical class-separation
    direction: planets have HIGH T12/T14 and LOW AUC_norm on this
    dataset. V9's DynamicGeometryLoss had the sign wrong.
- **Gate amplitude behaviour (best lambda=0.1):**
    A1 (U-shape)      = 0.0010 (floor — unused)
    A2 (V-shape)      = 0.0105
    A3 (inverted sec) = 0.0086
    A4 (asymmetric)   = 0.0209
    A5 (Gaussian)     = 0.0245 ← dominant by amplitude
  Surprising finding: A1, the planet-prior U-shape gate, is pinned
  at the 0.001 clamp floor. The Gaussian + asymmetric gates carry
  most of the signal. The CNN's discriminator relies on the *FP-
  like* templates (V-shape, asymmetric ingress, Gaussian spike) to
  pick out non-planets, not a dedicated planet template.
- **Gate-vs-primary correlation diagnostic:** per-TCE Pearson
  correlation of each gate template against primary_flux (since the
  gates are phase-only, naive mean-|gate| is batch-constant — the
  per-sample signal is the correlation):
    | Gate | Planet mean | FP mean | diff |
    | G1 | +0.20 | +0.40 | −0.20 |
    | G2 | +0.21 | +0.41 | −0.20 |
    | G3 | −0.21 | −0.41 | +0.20 |
    | G5 | +0.30 | +0.53 | −0.23 |
  FPs correlate ~2x more strongly with every template because they
  have deeper dips — amplitude dominates correlation. Discrimination
  lives in the *ratio* between gate correlations, which is what the
  Conv1d layer learns.
- **Per-TCE V6 vs V10 diagnostic** (stored V6 = Config B, not Config C;
  Config C weights were never saved):
    - V6 Config B on test set: TP=36 TN=24 FP=14 FN=2 (prec 72.0%,
      rec 94.7%, F1 0.818).
    - V10 lambda=0.1: TP=34 TN=31 FP=7 FN=4 (prec 82.9%, rec 89.5%,
      F1 0.861).
    - 8 FPs V10 newly catches (V6 prob 0.51–0.64, all flipped below
      0.50 by V10): K00230, K03785, K00048, K01896, K00225, K01156,
      K00132, K03064. Mean gate-vs-primary correlation on these 8:
      G1 +0.37, G2 +0.38, G3 −0.38, G4 −0.08, G5 +0.50. All look
      planet-shaped by any single template — the discriminator
      rides the *relative* match across gates.
    - 2 planets V10 loses that V6 had right: K00013 (V6 0.55, V10
      0.26), K00912 (V6 0.62, V10 0.42).
    - 1 new FP V10 introduces: K03650 (V6 0.19, V10 0.83).
    - Net: +8 FP catches − 2 planet losses − 1 new FP = +5 correct
      classifications.
- **Figure 8** (`notebooks/figures/v10_figure_8_gate_heatmap.png`)
  is the 76×5 Pearson correlation heatmap between each gate
  template and each TCE's primary_flux, with a V10-probability
  side panel and class divider.
- **V10 artefacts:**
  - `src/models/multi_template_gate.py` — MultiTemplateGateBank (5
    gates, internal .clamp(min=0.001) on each amplitude).
  - `src/models/taylor_cnn_v10.py` — TaylorCNNv10, 8-channel CNN.
  - `src/models/geometry_loss_v2.py` — InvertedGeometryLoss (sign-
    corrected, SNR pivot=100).
  - `src/models/taylor_cnn_v10.pt` — winner weights (lambda=0.1).
  - `scripts/gradcheck_v10.py` — gradcheck (5 amplitudes + gate
    independence check). All PASS.
  - `scripts/run_v10.py` — lambda-sweep training.
  - `scripts/v10_figures.py` — metrics sweep, amplitude traces,
    learned-template viz, gate-vs-primary correlation heatmap,
    T12/T14 distribution.
  - `scripts/v10_vs_v6_diagnostic.py` — per-TCE V6 vs V10
    comparison + Figure 8.
  - `data/v10_training.log`, `data/v10_results.pt`,
    `data/v10_vs_v6_diagnostic.csv`.
  - `notebooks/figures/v10_metrics_vs_lambda.png`,
    `v10_amplitude_traces.png`, `v10_learned_templates.png`,
    `v10_gate_primary_corr.png`, `v10_T12T14_distribution.png`,
    `v10_figure_8_gate_heatmap.png`.

### V8/V8.5/V9 paper draft: `docs/paper/v9_findings.md`
Full narrative with unifying root-cause section (Kepler-pipeline Mandel-
Agol bias contaminates every primary-fold shape metric), aggregate
feature-separation table, both SS-flag validation variants (v1 closed-
form, v2 scipy bounded), V8 gradcheck analysis, and V10 direction.

### V9 Null Result — DynamicGeometryLoss hurts because penalty is inverted for this data
- **Sweep:** lambda_max ∈ {0.1, 0.5, 1.0}. BCE + 0.01·|B| +
  DynamicGeometryLoss (SNR pivot 12.0). 4-channel CNN unchanged
  from V6; shape features only in the loss, not the input.
  A clamped post-step to min=0.001 so the gate can't die.
- **Results (seed=42 76-TCE test set):**
    | lambda_max | Acc   | Prec  | Rec   | F1    | A      |
    | 0.1        | 73.7% | 69.6% | 84.2% | 0.762 | 0.0010 |
    | 0.5        | 75.0% | 72.1% | 81.6% | 0.765 | 0.0010 |
    | 1.0        | 50.0% |  0.0% |  0.0% | 0.000 | 0.0085 |
  All three worse than V6 Config C (F1 0.815, prec 76.7%).
  lambda=1.0 catastrophic — geometry term overwhelms BCE and the
  classifier predicts "not-planet" for everything.
- **Gate collapse despite clamp.** A hit the 0.001 floor within ~25
  epochs for lambda ≤ 0.5. The clamp kept the gate "alive" in name
  but gate_out ≈ 0 in practice.
- **Why the geometry loss hurts this dataset:** the penalty encodes
  "U-shape planet / V-shape EB" → `prob · t12_t14` punishes high
  ingress fraction. But on this data the direction is reversed:
    - Planets  T12/T14 median 0.673  |  FPs  0.604
    - Planets  AUC_norm 0.090        |  FPs  0.506
  Planets have *higher* T12/T14 (more ingress-like) and *lower*
  AUC_norm because real planet dips are narrow relative to the full
  phase. Normalized soft masks working on `depth_norm = depth/A`
  route deep FP eclipses into the flat_bot bucket (depth_norm > 0.8),
  leaving planets with a relatively larger ingress fraction. The
  penalty direction hard-coded into DynamicGeometryLoss is therefore
  anti-aligned with the class-separation direction — pushing planets
  down and pulling EBs up.
- **Fixing the sign would likely help** (use `prob·(1-t12_t14)` and
  `prob·auc_norm` instead), but that's a new experiment — and the
  broader lesson is that a single global B cannot encode a per-sample
  morphology signal. The next step is structural, not loss-tuning.
- **Root cause (unifying V7 → V9):** the Kepler TCE pipeline fits a
  Mandel-Agol *planet* model to every candidate including EBs.
  Archive `koi_duration` (V7), archive folded-flux shape (V8, V8.5),
  shape penalties derived from that flux (V9), and per-sample B fits
  on the primary fold (SS-flag v1 and v2) all inherit that bias:
  EB primaries look planet-consistent in both duration and morphology
  when viewed through the pipeline. The EB signal the pipeline
  removed, the downstream model cannot recover — regardless of how
  we repackage it. The remaining signal lives in secondary-eclipse
  depth, odd/even transit depth, and centroid motion (pixel-level
  data, not the folded LC).
- **SS-flag v2 (scipy curve_fit, bounded):** Per-sample B fit on
  dip-only samples with bounds B ∈ [−2, 2], A ∈ [0, 0.1]. Result:
    - SS=0 (not EB): n=285, median B = +1.212
    - SS=1 (EB flag): n=215, median B = +1.177
    - Separation gap = 0.034 → NULL
  Matches V8's closed-form fit (0.012 gap). Primary-eclipse shape
  alone cannot recover the SS flag regardless of fit method.
- **V9 artefacts:**
  - src/models/taylor_cnn_v9.py — TaylorCNNv9, shape features in
    loss only; `compute_shape_features(primary_flux)` method.
  - src/models/taylor_cnn_v9.pt — winning lambda (0.5) weights.
  - scripts/run_v9.py, scripts/v9_figures.py,
    scripts/v9_ss_validation.py.
  - data/v9_training.log, data/v9_results.pt.
  - notebooks/figures/v9_metrics_vs_lambda.png,
    v9_A_trace.png, v9_T12T14_distribution.png,
    B_vs_SS_flag_v2.png.

### V8 + V8.5 Null Result — Shape features insufficient as simple fusion
- **V8** (learnable A, B, t0 Taylor gate, y = min(0, -A·(1 - x²/2 +
  B·x⁴/24))): acc 77.6%, prec 73.3%, rec 86.8%, F1 0.795, 916 params.
  Slightly worse than V6 Config C (F1 0.815). A drifted up to 0.026,
  B drifted down to +0.58, t0 stayed ≈ 0.
- **V8.5** (V8 + 5 shape-features fused into classifier): acc 76.3%,
  prec 71.7%, rec 86.8%, F1 0.786, 921 params. Gate's A collapsed
  to -0.0013 — i.e. the gate channel got zeroed out because the shape
  features provided an easier path to the BCE minimum. The 5 features
  are (B_global, AUC_raw, AUC_norm=AUC/A, T12/T14, flat_bottom)
  extracted per-sample from primary_flux, not from gate_output (see
  "degenerate gate features" note below).
- **Degenerate gate features (V8.5 v1 bug, now fixed).** The first pass
  computed shape features from gate_output. Because every TCE in the
  dataset is folded to the same fixed phase grid, gate_output is
  byte-identical across samples — the 5 "per-sample" features were
  batch-constant. V8.5-v2 extracts features from primary_flux instead,
  keeping only B as the global prior (constant by design).
- **Features ARE discriminative in aggregate** (test set n=76):
    - Planets median AUC_norm 0.07   |   FPs median 0.39 (5.6x)
    - Planets median flat_bottom 0.010 | FPs median 0.050 (5x)
    - Planets median T12/T14 0.688    | FPs 0.598 (weak)
  The simple Linear(16+5, 1) fusion head didn't convert this into an
  F1 gain. Either a non-linear head or a per-sample gate is needed.
- **SS-flag validation: NULL** (Figure 7). Per-sample B from closed-
  form least-squares fit on primary_flux: SS=0 median B=+1.156,
  SS=1 median B=+1.168, separation gap = 0.012. Primary-eclipse
  morphology alone cannot recover koi_fpflag_ss — EB primary eclipses
  genuinely look planet-like when fit as single events (consistent
  with V7's trapezoid-fit finding that EB primaries are single-event
  indistinguishable from planets). The SS flag lives in secondary
  eclipse / centroid data.
- **Mask-overlap check PASSED** (Figure v8_T12T14_by_SNR). The
  normalized soft-masks (baseline + ingress + flat_bot = 1 by
  construction) do not drift to 0.5 at low SNR — high-SNR T12/T14
  median 0.637, only 16% within [0.40, 0.60]. Design goal met.
- **V9 direction (priority):** multi-template Taylor-gate bank — N
  parallel gates each pre-tuned to a specific dip morphology
  (symmetric U, asymmetric ingress, inverted secondary, box/grazing,
  Gaussian spot). Per-sample output: which gate fired strongest.
  This gives per-sample morphological selectivity without needing
  closed-form fits or per-sample Parameters.
- **Alternative V9 paths:**
  (a) Wire `DynamicGeometryLoss` (saved in src/models/geometry_loss.py)
      with per-sample B from the closed-form fit as training signal.
  (b) Non-linear classifier head (MLP) so the per-sample feature
      non-linearities get modeled.
  (c) Multi-head attention over the 4 CNN channels + 5 shape
      features — lets the model route per-sample.
- **V8/V8.5 artefacts:**
  - src/models/taylor_layer_v8.py — TaylorGateLayerV8 + custom
    TaylorGateV8Function with hand-coded backward, all three gradients
    (A, B, t0) verified by torch.autograd.gradcheck in double precision.
  - src/models/taylor_cnn_v8.py — TaylorCNNv8, shape-feature fusion.
  - src/models/taylor_cnn_v8.pt, taylor_cnn_v85.pt — trained weights.
  - src/models/geometry_loss.py — DynamicGeometryLoss (unused, for V9).
  - scripts/run_v8.py, scripts/gradcheck_v8.py, scripts/v8_figures.py,
    scripts/v8_ss_validation.py.
  - data/ss_flag_cache.csv — cached NASA Archive koi_fpflag_ss +
    koi_model_snr for all 9564 KOIs.
  - notebooks/figures/v8_B_histogram.png, v8_AUCnorm_histogram.png,
    v8_B_vs_AUCnorm.png, v8_K03745_gate.png, B_vs_SS_flag.png,
    v8_T12T14_by_SNR.png.

### V7 Null Result — Archive koi_duration has pipeline bias (CRITICAL)
- **Soft Kepler loss doesn't help on this dataset.** Lambda sweep in both
  symmetric and asymmetric modes showed best result at lambda=0. At lambda=0.1
  the Kepler term is only 2.6% of BCE magnitude (too weak); at lambda=10+
  the model collapses to "always FP." No useful middle ground found.
- **Root cause:** 84% of FPs in our dataset (209/250) are flagged as EBs by
  `koi_fpflag_ss`, BUT their archive `koi_duration` has median obs/pred
  ratio = 0.999 — nearly identical to confirmed planets (0.957). The Kepler
  TCE pipeline fits a Mandel-Agol PLANET model to every candidate, which
  constrains fitted duration to planet-consistent ranges even for EBs. Archive
  `koi_duration` for EBs is a "best planet fit" — not the true eclipse duration.
- **Direct trapezoidal fitting doesn't rescue it.** Fit on our 200-bin folded
  curves succeeded for all 10 tested TCEs (5 planets + 5 EBs), recovered
  sensible depths and T14. But EB primary eclipses genuinely *look* planet-
  like when fit as single events, so fitted violations don't separate classes
  (planet max 0.253, EB min 0.016 — reversed from expectation).
- **Only signal that worked:** hard Kepler gate at threshold 1.0 on the safe
  side catches K01091.01 (viol 1.413, 78-hour "transit"). Gives
  +1.7% precision without blocking any planets. Marginal but free.
  Available in `scripts/hard_kepler_gate_sweep.py`.
- **V8 direction — shape-based discrimination (priority path):**
  Taylor gate morphological parameters can distinguish V-shape (grazing
  EBs) from U-shape (real planet transits with flat bottoms):
  - V-shape: short/zero flat-bottom duration T23, long ingress T12.
    Ratio T12 / T14 close to 0.5. EB signature.
  - U-shape: long flat-bottom T23, short T12. Ratio T12 / T14 < 0.25.
    Planet signature.
  Implementation: extend the Taylor gate to learn T12 and T14 as
  separate parameters, feed the ratio as an explicit feature to the
  classifier head. Already have trapezoid-fit tooling from V7 in
  `scripts/trapezoid_fit_feasibility.py` to bootstrap ground-truth
  ratios for the training set.
- **Alternative V8 paths if shape-based falls short:**
  (a) Full Mandel-Agol joint fit with stellar density prior
  (b) Centroid-motion features (pixel-level, needs MAST target-pixel files)
  (c) Stronger weight on V6's existing odd/even-DEPTH channel
- **V7.5 safe gate (implemented):** `src/models/taylor_cnn_v75.pt`
  bundles V6 Config C weights with `kepler_gate_threshold = 1.0`. At
  inference: if violation > 1.0, override prediction to FP. +1.7%
  precision, zero planets blocked. See `docs/paper/v7_findings.md`.
- **Artefacts of V7/V7.5:** `src/models/kepler_loss.py`,
  `scripts/run_v7.py`, `scripts/sweep_kepler_lambda.py`,
  `scripts/hard_kepler_gate_sweep.py`, `scripts/trapezoid_fit_feasibility.py`,
  `scripts/build_v75_and_figures.py`, `notebooks/figures/*.png`.
  Kepler physics utilities are sound; they just don't land on this data.

---

## Code Conventions
- Python 3.11+, PEP 8, type hints, Google docstrings
- snake_case variables/functions, PascalCase classes
- Notebooks for exploration only — production code in src/

## Safety Rules
- Never commit .env, data/raw/, API keys, or kepler_tce.pt
- Never edit nasaml/ venv
- Never use MathWorks work resources or IP
- Always activate nasaml venv before running anything

## PRODUCTION MODEL PROTECTION

NEVER overwrite files in `src/models/production/`.
These are the published paper models.

Production models (READ-ONLY):
  v6b_recall947.pt   → V6 Config B (94.7% recall)
  v10_f1861.pt       → V10 λ=0.1 (F1=0.861, best single)
  v10_log_mdwarf.pt  → V10 log R* normalization
  v75_safe_gate.pt   → V7.5 safe Kepler gate
  v5_baseline.pt     → V5 secondary eclipse baseline

`AdaptivePINNClassifier` loads from this directory by default.

Experiment naming convention — write NEW models to `src/models/`
(never `src/models/production/`) with a descriptive version suffix:
  taylor_cnn_v10_[description]_[date].pt
  Examples:
    taylor_cnn_v10_1114_noweight.pt
    taylor_cnn_v10_tess_native.pt
    taylor_cnn_v10_wider_cnn.pt

Training scripts should carry the protection banner at the top:

    # ═══════════════════════════════════════════
    # PRODUCTION MODEL PROTECTION
    # NEVER save to src/models/production/
    # Save experiments to src/models/ only
    # with descriptive version suffix
    # ═══════════════════════════════════════════

---

## Active Projects

### 1. Light Curve Classifier — Taylor-CNN PINN

#### Current Status: V5 Complete ✅

| Metric | V4 | **V5** | AstroNet (2018) |
|---|---|---|---|
| Accuracy | 75.0% | **81.2%** | 96% |
| Precision | 66.7% | **72.7%** | — |
| Recall | 100.0% | **100.0%** | — |
| F1 | 0.800 | **0.842** | — |
| Training data | 70 TCEs | 70 TCEs | 15,000 TCEs |
| Parameters | 798 | **856** | ~30,000 |
| Learned A | 0.013 | 0.015 | — |

100% recall retained. V5 caught 5 of 8 test FPs including the 2 clearest EBs
(K03719.01 s_depth=-0.046 → prob=0.00, K01178.01 s_depth=-0.008 → prob=0.06).
3 surviving FPs have flat secondary channels — non-EB failure modes, see V6.

#### Architecture (V5)
```
phase          → Taylor Gate: min(0, -A*(1 - x²/2)) ──┐
primary_flux   ───────────────────────────────────────┼→ stack (B,3,200) → BatchNorm
secondary_flux ───────────────────────────────────────┘   → Conv1d(3→8) → Conv1d(8→16)
                                                          → AvgPool → Linear(16→1) → Sigmoid
```

#### Next Session: V6 — Scale-up + fix pretrain

**V6 starter prompt for Claude Code:**
```
V5 is complete — 81.2% accuracy, 72.7% precision, 100% recall on 16 real
Kepler TCEs. 3 surviving FPs are non-EB (flat secondary channel). V6 goals:
1) Scale dataset to 500+ TCEs from Kepler DR25 so the test set has a
   representative FP mix. Apply data augmentation (depth ±20%, phase
   jitter, noise, time reversal) during training.
2) Fix synthetic pretraining: reset BatchNorm running stats between
   pretrain and finetune, OR match synthetic depth distribution to real
   Kepler (median 1300 ppm). Validate pretrain → finetune beats from-
   scratch on the larger test set.
3) Optional stretch: add centroid-offset channel to address remaining
   non-EB FPs.
Target: precision > 80% AND recall = 100% on the 500+-TCE test set.
```

**Other V6 improvements (priority order):**
1. Scale to 500+ TCEs + data augmentation (highest impact on test-set
   composition; makes precision target reachable)
2. Fix synthetic pretraining BN/depth mismatch
3. Higher-order Taylor: `min(0, -A*(1 - x²/2 + x⁴/24))` for flat-bottom
   dips (hot Jupiters, grazing transits)
4. Centroid-offset channel for non-EB FPs
5. BLS baseline comparison for paper

#### Two-Channel Architecture (V3 — after V5)
```
Raw light curve
  ├── Channel 1: Taylor Gate → physics model
  └── Channel 2: Residual = raw - physics → 1D CNN → detects TTVs, spots
                                          → Linear Fusion → Transit / No Transit
```

#### ECG Extension (after transit work)
- RLC physical model: R-peak = inductive (sin terms), T-wave = capacitive (alternating)
- Inductive: A*(x - x³/6) | Capacitive: B*(1 - x + x²/2) | Asymmetry: C*x²
- Datasets: PTB-XL + MIT-BIH | Baseline: Ribeiro 2020 (Nature Comms)
- Paper: cross-domain generalization of physics-informed Taylor layer

#### Key Papers
- AstroNet: arxiv.org/abs/1712.05044
- ExoMiner: arxiv.org/abs/2111.10009
- Ribeiro ECG: arxiv.org/abs/1904.01949

#### Research Questions
1. Does secondary eclipse view eliminate EB false positives?
2. Does Taylor term count act as a physically meaningful hyperparameter?
3. Does two-channel residual detect TTVs and spot crossings?
4. Does architecture generalize Kepler → TESS without retraining?
5. Does same layer work for ECG QRS classification?

---

### 2. Technical Documentation Assistant (Next)
- Goal: RAG pipeline over NASA technical documentation
- Stack: Sentence Transformers + vector DB + LLM
- Status: Planned — Phase 4

---

## Learning Milestones
- [x] 3Blue1Brown — Linear Algebra
- [x] 3Blue1Brown — Calculus
- [x] ML environment set up (GPU verified)
- [x] 3Blue1Brown — Neural Networks videos 1-3
- [x] Taylor-CNN PINN V4 — 100% recall on real Kepler data ✅
- [x] Taylor-CNN PINN V5 — secondary eclipse view, F1 0.842 ✅
- [x] Taylor-CNN V6 — 500 TCEs + odd/even channel, F1 0.815 ✅
- [x] V7 Kepler soft loss — null result, diagnosed + documented ✅
- [x] V7.5 safe Kepler gate — +1.7% precision, zero planet risk ✅
- [x] V8 + V8.5 shape discrimination — null result, F1 0.795/0.786 ⚠
- [x] V9 DynamicGeometryLoss — null result, best F1 0.765 at λ=0.5 ⚠
- [x] V10 multi-template gate bank + inverted loss — **F1 0.861** ✅
- [x] V10.5 threshold sweep — no threshold hits both targets
- [x] V6b + V10 AND ensemble — **F1 0.872** (session best) ✅
- [x] V10 TESS zero-shot — 100% recall, 80% precision ✅
- [x] V10 M-dwarf R*² norm — rescues K00254 but trashes precision ⚠
- [x] V10 diversity loss — A1 still collapses, F1 down ⚠
- [x] V5 + V10 ensemble — V5 drags V10 down, no gain ⚠
- [x] V10_5b clipped R* — F1 0.805, only K00912 rescued ⚠
- [x] V10_5b log R* — F1 0.854, rescues K00912 + K00183 ✅
- [x] Triple OR @ 0.4 — **100% recall (38/38)** ✅
- [x] Paper draft v1 — `docs/paper/paper_draft.md` ✅
- [x] V10 + 1114 TCEs scale-up — NULL, precision collapsed (Exp 4) ⚠
- [x] V10 TESS-native — degenerate collapse (Exp 4b) ⚠
- [x] V10 cross-mission AND ensemble — NULL, equals best single (Exp 4c) ⚠
- [x] V10 + 1114 TCEs, no pos_weight — precision +18pp, recall halves ⚠
- [ ] 3Blue1Brown — Neural Networks video 4
- [ ] 3Blue1Brown — Transformers (chapters 5-7)
- [ ] Vizuara — Foundations for ML
- [ ] VanderPlas — Astronomy ML book
- [ ] Light Curve Classifier V8 (shape-based discrimination)
- [ ] Quantic MS AI Engineering (June 2025)

---

## NASA Force Target
- nasa.gov/careers/nasaforce — signed up for updates
- Target: AI/ML Engineer or Data/IT roles
- Portfolio: light curve classifier + doc assistant

---

## Claude Code Usage Notes
- Explain concepts not just code — learning is the goal
- Line-by-line comments on non-obvious code
- Always explain WHY, not just WHAT
- Build from scratch over boilerplate — builds intuition
- Walk through math before coding any new layer
- Claude.ai → concepts, strategy, architecture
- Claude Code → building, debugging, iterating

---

## Version Roadmap

```
V4   ✅  Taylor gate + 1D CNN (COMPLETE — 100% recall)
V5   ✅  + Secondary eclipse view (COMPLETE — F1 0.842, +6.2% acc)
V6   ✅  + Scale to 500 TCEs + odd/even channel (COMPLETE — F1 0.815)
V7   ⚠   Soft Kepler loss — NULL RESULT. Archive koi_duration biased by
         pipeline Mandel-Agol fit (see docs/paper/v7_findings.md)
V7.5 ✅  Safe hard Kepler gate at thr=1.0 — precision 75.0% → 76.7%,
         zero planet risk. Catches K01091.01 (78h "transit", viol=1.413).
         (src/models/taylor_cnn_v75.pt)
V8   ⚠  Learnable shape parameter B added to Taylor gate
         (y = min(0, -A·(1 - x²/2 + B·x⁴/24))), plus learnable
         phase offset t0. Gradcheck PASS on all three gradients.
         F1 0.795 — slightly below V6 Config C. B drifted to +0.58
         (intermediate between U and V), acknowledging a mixed
         population. (src/models/taylor_cnn_v8.pt)
V8.5 ⚠  V8 + 5-feature shape fusion (B, AUC_raw, AUC/A, T12/T14,
         flat_bottom) extracted per-sample from primary_flux on a
         fixed [-1, 1] phase grid with normalized soft masks.
         F1 0.786. Features discriminate classes in aggregate
         (AUC_norm 5x separation) but the linear classifier head
         didn't capture it; gate's A collapsed to zero.
         SS-flag validation: per-sample B does NOT recover
         koi_fpflag_ss (separation gap 0.012) — primary-eclipse
         shape alone is insufficient. (src/models/taylor_cnn_v85.pt)
V9   ⚠  Shape features moved from CNN input into the loss via
         DynamicGeometryLoss (SNR-weighted, pivot 12). Same 4-
         channel CNN as V6. A post-step clamped to min=0.001.
         Lambda sweep: best F1 0.765 at lambda=0.5 (worse than V6).
         Penalty direction anti-aligned with class-separation on
         this data (planet T12/T14 > FP T12/T14). Confirms that
         a single global B cannot encode per-sample morphology.
         (src/models/taylor_cnn_v9.pt)
```

### V8 PSNN Architecture
```
Raw light curve
      ↓
Layer 1: Taylor Gate     → LOCAL physics (dip morphology)
      ↓
Layer 2: Kepler Gate     → GLOBAL physics (orbital mechanics)
         Learnable: stellar density ρ*, impact parameter b,
                    limb darkening u
         Hard gate: suppresses physically impossible orbits
                    BEFORE CNN sees them
      ↓
Layer 3: 1D CNN          → LEARNED features (residual patterns)
      ↓
Sigmoid                  → Transit / No Transit
```

### Key research question for V7 vs V8:
Does enforcing Kepler's Laws as a hard gate (V8 layer) outperform
enforcing them as a soft loss penalty (V7)? This comparison is the
paper's central experiment.

### V8 Session Starter Prompt
```
V7 is complete with Kepler loss function. Now implement V8 PSNN:
Add a KeplerGateLayer after the Taylor gate. The layer receives
(period, duration, depth) and computes a consistency score using
Kepler's Third Law. Learnable parameters: stellar density rho_star,
impact parameter b. Output attenuates the signal proportionally to
orbital consistency. Compare V7 vs V8 on same test set.
```

### V9: Multi-Template Gate Bank
```
Instead of one Taylor gate, N parallel gates each tuned to
a different physical dip morphology:

Gate 1: min(0, -A1*(1 - x²/2))    → planet transit (symmetric)
Gate 2: min(0, -A2*(x - x³/6))    → EB primary (asymmetric)
Gate 3: max(0,  A3*(1 - x²/2))    → secondary eclipse (inverted)
Gate 4: min(0, -A4*box(x))         → grazing transit (flat bottom)
Gate 5: min(0, -A5*gaussian(x))    → spot crossing (narrow)

CNN input: (B, N, 200) — stack of N gate outputs
CNN just reads which gates fired → trivial classification
Replaces ExoMiner's 6 input views with 1 input + N physics gates
```

### Full version roadmap
```
V4   ✅  Single Taylor gate (COMPLETE — 100% recall, 798 params)
V5   ✅  + Secondary eclipse view (COMPLETE — F1 0.842, 856 params)
V6   ✅  + 500 TCEs + odd/even channel (COMPLETE — F1 0.815)
V7   ⚠   Soft Kepler loss — NULL RESULT
V7.5 ✅  Safe hard Kepler gate at thr=1.0 — zero planet risk
V8   ⚠   Learnable B curvature + t0 offset — F1 0.795, gradcheck PASS
V8.5 ⚠   V8 + 5 shape features (primary_flux, normalized masks) — F1 0.786
V9   ⚠   DynamicGeometryLoss (SNR-weighted shape + AUC penalty) —
         NULL RESULT, best F1 0.765 at lambda=0.5. Root-cause
         unified with V7: Kepler pipeline Mandel-Agol bias
         contaminates every primary-fold shape metric.
V10  ✅  Multi-template gate bank (5 fixed morphologies, each with
         learnable amplitude) + InvertedGeometryLoss (sign corrected
         from V9). Shape features in loss only, not input. lambda=0.1
         hits **F1 0.861 / prec 82.9% / rec 89.5%** — FIRST GAIN
         over V6 baseline. 1150 params (236 more than V6).
         (src/models/taylor_cnn_v10.pt)
```
