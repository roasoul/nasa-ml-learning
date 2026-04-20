# CLAUDE.md — NASA ML Learning Project

## Project Context
This is a personal ML/AI learning project. All code here is independent of my
employer (MathWorks). I retain full IP ownership of everything in this repo.

**Goal:** Build ML/AI skills targeting a NASA Force AI/ML or Data role.
**Current phase:** Light Curve Classifier V5 complete → V6 scale-up
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
- [ ] 3Blue1Brown — Neural Networks video 4
- [ ] 3Blue1Brown — Transformers (chapters 5-7)
- [ ] Vizuara — Foundations for ML
- [ ] VanderPlas — Astronomy ML book
- [ ] Light Curve Classifier V6 (scale to 500+ TCEs)
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
V4  ✅  Taylor gate + 1D CNN (COMPLETE — 100% recall)
V5  ✅  + Secondary eclipse view (COMPLETE — F1 0.842, +6.2% acc)
V6  ⬜  + Scale to 500+ TCEs + data augmentation + fix BN pretrain
V7  ⬜  + Kepler loss function (global physics as SOFT penalty)
        L_total = L_class + λ·L_Kepler + λ·L_sparsity
V8  ⬜  Physics Stack Neural Network (PSNN)
        Taylor Gate (local) → Kepler Gate (global) → CNN
        Compare PINN (V7) vs PSNN (V8) — does hard gate beat soft?
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
V4  ✅  Single Taylor gate (COMPLETE — 100% recall, 798 params)
V5  ✅  + Secondary eclipse view (COMPLETE — F1 0.842, 856 params)
V6  ⬜  + 500+ TCEs + augmentation + BN pretrain fix
V7  ⬜  + Kepler LOSS (soft penalty)
V8  ⬜  + Kepler GATE (hard gate) = PSNN
V9  ⬜  + Multi-template gate bank (N parallel Taylor gates)
        Compare V4→V9 ablation study = the paper
```
