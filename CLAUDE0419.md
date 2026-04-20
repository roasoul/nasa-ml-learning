# CLAUDE.md — NASA ML Learning Project

## Project Context
This is a personal ML/AI learning project. All code here is independent of my
employer (MathWorks). I retain full IP ownership of everything in this repo.

**Goal:** Build ML/AI skills targeting a NASA Force AI/ML or Data role.
**Current phase:** 3Blue1Brown Neural Networks → Transformers + Vizuara → Projects
**Degree:** MS AI Engineering at Quantic (starting June 2025)

---

## Project Structure
```
C:\Users\skapa\Projects\ml-learning\nasa-ml-learning\
├── CLAUDE.md
├── requirements.txt
├── nasaml\                   # Virtual environment — never edit
├── notebooks\                # Jupyter exploration notebooks
│   ├── 00_installation_test.ipynb
│   ├── 01_taylor_gate_verification.ipynb
│   ├── 02_taylor_cnn_training.ipynb
│   ├── 03_real_kepler_data.ipynb
│   └── 04_real_data_training.ipynb
├── src\                      # Reusable modules and scripts
│   ├── data\                 # Data loading and preprocessing
│   │   ├── synthetic.py      # Synthetic light curve generator
│   │   ├── kepler.py         # Kepler download + preprocessing pipeline
│   │   └── build_dataset.py  # Batch TCE download → cached .pt file
│   ├── models\               # Model definitions
│   │   ├── taylor_layer.py   # Custom Taylor gate (cos-based)
│   │   ├── taylor_cnn.py     # Two-channel PINN classifier
│   │   └── taylor_cnn_kepler.pt  # Trained weights (real Kepler data)
│   └── utils\                # Helper functions
├── projects\
│   ├── light_curve\          # Kepler/TESS exoplanet transit classifier
│   └── doc_assistant\        # Technical documentation RAG tool
└── data\                     # Raw and processed datasets (not committed)
    └── kepler_tce.pt         # Cached preprocessed TCE dataset (100 TCEs)
```

---

## Environment Setup (Windows)
```bash
# Navigate to project
cd C:\Users\skapa\Projects\ml-learning\nasa-ml-learning

# Activate virtual environment
nasaml\Scripts\activate

# Verify GPU
python -c "import torch; print(torch.cuda.is_available())"

# Launch Jupyter
jupyter notebook notebooks\

# Run tests
pytest tests\

# Install new packages — ALWAYS use PyPI explicitly
# (machine has a rogue Artifactory config that intercepts pip)
pip install <package> --index-url https://pypi.org/simple/

# PyTorch specifically — RTX 5060 is Blackwell (sm_120), needs nightly cu128
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Save environment
pip freeze > requirements.txt
```

---

## Hardware
- **GPU:** NVIDIA GeForce RTX 5060 Laptop GPU (Blackwell architecture, sm_120)
- **VRAM:** 8GB
- **CUDA:** 12.8
- **Driver:** 591.74
- **PyTorch:** 2.12.0.dev (nightly) — required for sm_120 Blackwell support
- **Note:** Standard stable PyTorch only supports up to sm_90 — always use nightly cu128

---

## Installed Libraries (all verified working)
- **torch 2.12.0 / torchvision / torchaudio** — PyTorch with CUDA 12.8 (nightly)
- **numpy 2.4.3** — array operations
- **pandas 2.3.3** — data manipulation
- **matplotlib 3.10.8** — visualization
- **scikit-learn 1.8.0** — classical ML
- **jupyter / notebook** — exploration
- **lightkurve 2.6.0** — NASA Kepler/TESS light curve data
- **astropy 7.2.0** — astronomical data handling
- **transformers / datasets** — HuggingFace (for later)
- **sentence-transformers** — embeddings (for later)

---

## Known Issues & Fixes
- **Artifactory pip error:** Machine has a rogue corporate pip config. Always use
  `--index-url https://pypi.org/simple/` flag when installing packages.
- **Blackwell GPU (sm_120):** Stable PyTorch doesn't support RTX 5060. Use nightly
  cu128 build. Switch to stable channel once official Blackwell support ships.
- **lightkurve warning:** `oktopus not installed` warning on import is harmless —
  only needed for advanced PRF fitting, not required for transit classification.
- **lightkurve preprocessing pipeline:** Always follow this sequence before passing
  data to any model:
  1. `lc.flatten(window_length=401)` — remove stellar variability
  2. `.normalize()` — center flux at 1.0
  3. `.fold(period, epoch_time)` — fold on known or candidate period
  4. Scale phase to [-π, π] via `(folded.time.value / period) * 2π`
  5. Subtract 1.0 from flux — so baseline = 0 and transit dip is negative
  6. Phase-bin to 200 points using median per bin
  Skipping any step will cause the Taylor approximation to diverge or produce
  meaningless gradients.
- **remove_outliers destroys transits:** NEVER use `lc.remove_outliers(sigma=5)` —
  it clips transit dips (which are 5–50 sigma deep) along with real outliers.
  Always use `remove_outliers(sigma_lower=float('inf'), sigma_upper=5)` to clip
  only upward outliers (cosmic rays) and preserve downward dips (transits).
- **Taylor gate phase alignment (sin→cos fix):** The original sin(x) gate
  `min(0, -A*(x - x³/6))` placed its dip at phase π/2, but fold() centers
  transits at phase 0. This 90-degree misalignment was fixed by switching to
  cos(x): `min(0, -A*(1 - x²/2))`, which has its dip at phase 0. Any future
  Taylor layer work must use cos(x) as the base function, not sin(x).
- **BatchNorm required before Conv1d on flux data:** Transit flux values are
  ~0.01 (1%) but Conv1d weights initialize at ~0.1. Without BatchNorm, gradients
  are too small and the model stalls at 50% accuracy. Always add
  `nn.BatchNorm1d(n_channels)` before the first Conv1d layer.

---

## Code Conventions
- **Language:** Python 3.11+
- **Style:** PEP 8, 4-space indentation
- **Typing:** Use type hints on all function signatures
- **Docstrings:** Google style — every function and class gets one
- **Naming:** `snake_case` for variables/functions, `PascalCase` for classes
- **Notebooks:** Use for exploration only — production code goes in `src\`

---

## Safety Rules
- **Never commit** `.env`, `data\raw\`, or any file containing API keys
- **Never edit** anything inside `nasaml\`
- **Never use** work resources, work data, or MathWorks IP in this repo
- **Always activate** nasaml venv before installing or running anything
- **Always use** `--index-url` flags when installing (see Known Issues above)

---

## Active Projects

### 1. Light Curve Classifier (Current Focus)
- **Goal:** Classify Kepler/TESS exoplanet transit signals using a
  Physics-Informed Neural Network (PINN)
- **Data:** Kepler TCE (Threshold Crossing Events) via `lightkurve`
- **Benchmark comparisons:**
  - AstroNet (Shallue & Vanderburg 2018) — primary ML baseline
  - BLS (Box Least Squares) — standard non-ML baseline used by Kepler team itself
- **Status:** V4 built and trained on real Kepler TCEs (Sessions 1–4 complete)

#### Architecture: Two-Channel PINN (V4 — implemented)

```
phase (B, 200)    flux (B, 200)
      │                 │
      ▼                 │
  TaylorGate            │
  min(0, -A·cos(x))     │
  → gate_out (B, 200)   │
      │                 │
      ▼                 ▼
  stack → (B, 2, 200)  [ch0=gate, ch1=flux]
      │
  BatchNorm1d(2)
  Conv1d(2→8, k=7) + ReLU
  Conv1d(8→16, k=5) + ReLU
  AdaptiveAvgPool1d → (B, 16)
      │
  Linear(16, 1) → sigmoid → P(transit)
```

**Total parameters:** 798
**Saved weights:** `src/models/taylor_cnn_kepler.pt`

#### Custom Taylor Layer — Key Details
- **Forward:** `output = min(0, -A * (1 - x²/2))` where x is scaled phase
- **Backward:** `grad_x = upstream_grad * A * x * mask`
              `grad_A = upstream_grad * (-(1 - x²/2)) * mask`
- **Mask:** 1 where output < 0 (dip region), 0 elsewhere — prevents dead gradients
- **Learnable parameters:** A (amplitude/depth)
- **Dip shape:** centered at x=0, zero-crossings at x = ±√2 ≈ ±1.41 rad
- **Why cos(x):** cos(x) ≈ 1 - x²/2 has its maximum at x=0, so -A·cos(x) has
  its dip at x=0 — exactly where fold() centers the transit. The original sin(x)
  version placed the dip at x=π/2, causing a 90-degree misalignment.
- **Why two channels:** The CNN receives both the gate model (physics) and the raw
  flux as a 2-channel input. Its first-layer filters learn to compare them —
  implicitly computing the residual plus any other useful comparison.

#### Test Results (Session 4 — Real Kepler Data)

| Metric | Taylor-CNN | AstroNet (2018) |
|--------|-----------|-----------------|
| Accuracy | 75.0% | 96% |
| Precision | 66.7% | — |
| Recall | 100.0% | — |
| F1 | 0.800 | — |
| Training data | 70 TCEs | 15,000 TCEs |
| Parameters | 798 | ~30,000 |
| Input views | 1 (local) | 3 (global+local+secondary) |

100% recall means no confirmed planet was missed. The 4 false positives
predicted as transit are likely eclipsing binaries with real dip-shaped signals
that require secondary eclipse data to distinguish from planets.

#### Planned ECG Extension
- Same Taylor layer architecture generalizes to ECG QRS complex classification
- QRS complex morphologically equivalent to transit dip (half-wave rectifier shape)
- Dataset: PTB-XL (21,837 12-lead ECGs, free on PhysioNet)
- Baseline: Ribeiro et al. 2020 (Nature Communications) — 1D ResNet on 2M ECGs
- Goal: Cross-domain paper demonstrating architecture generalization

#### Key Papers (all free on arxiv.org)
- AstroNet — arxiv.org/abs/1712.05044
- ExoMiner — arxiv.org/abs/2111.10009
- Astronet-Triage-v2 — arxiv.org/abs/2301.01371
- ExoMiner++ 2.0 — arxiv.org/abs/2601.14877
- ExoSpikeNet — arxiv.org/abs/2406.07927
- Ribeiro ECG ResNet — arxiv.org/abs/1904.01949

#### Research Questions
1. Does the Taylor layer improve precision over AstroNet and BLS?
2. Does Taylor term count (2 vs 3 vs 4) act as a physically meaningful hyperparameter?
3. Does the two-channel residual approach detect TTVs and spot crossings?
4. Does the architecture generalize from Kepler → TESS without retraining?
5. Does the same layer work for ECG QRS classification (cross-domain)?

---

### 2. Technical Documentation Assistant (Next)
- **Goal:** RAG pipeline over NASA technical documentation
- **Stack:** Sentence Transformers + vector DB + LLM
- **Status:** Planned — Phase 4 (embeddings phase)

---

## Learning Milestones
- [x] 3Blue1Brown — Essence of Linear Algebra
- [x] 3Blue1Brown — Essence of Calculus
- [x] ML environment set up and verified (GPU confirmed working)
- [ ] 3Blue1Brown — Neural Networks (4 videos)        <- YOU ARE HERE
- [ ] 3Blue1Brown — Transformers (chapters 5-7)        <- in parallel with Vizuara
- [ ] Vizuara — Foundations for ML                     <- in parallel with Transformers
- [ ] VanderPlas — Statistics, Data Mining & ML in Astronomy (after above)
- [x] Light Curve Classifier V4 — built and trained on real Kepler TCEs
- [ ] Sentence Transformers + embeddings
- [ ] Quantic MS AI Engineering (June 2025)

---

## NASA Force Target
- **URL:** nasa.gov/careers/nasaforce
- Signed up for email updates — role windows are short (first window was 4 days)
- Target roles: AI/ML Engineer, Data/IT (first opening was aerospace only)
- Portfolio goal: light curve classifier + doc assistant as flagship projects
- Books: VanderPlas astronomy ML book (owned, to read after Vizuara)

---

## Claude Code Usage Notes
- Always explain concepts, not just write code — learning is the primary goal
- Add line-by-line comments on any non-obvious code
- Always explain WHY something works, not just WHAT it does
- Prefer building from scratch over copying boilerplate — builds intuition
- When implementing the custom Taylor layer, walk through the math before coding
- Use Claude.ai (chat) for concepts, strategy, architecture decisions
- Use Claude Code (terminal) for building, debugging, iterating, and running code

---

## Next Session Prompt — V4 Improvements

The V4 Taylor-CNN is built and trained. Current accuracy is 75% on 100 real
Kepler TCEs (vs AstroNet 96%). The next session should focus on closing this
gap. Copy and paste into Claude Code:

```
The V4 Taylor-CNN (cos gate) is trained on 100 real Kepler TCEs with 75%
accuracy, 100% recall, 66.7% precision, F1=0.800. Weights are saved at
src/models/taylor_cnn_kepler.pt. The main weakness is 4 false positives
(likely eclipsing binaries) predicted as transit.

Next steps to improve accuracy toward AstroNet's 96%:
1. Scale up training data — download 500+ TCEs using build_dataset.py
2. Add a secondary eclipse view as a third CNN input channel (this is
   how AstroNet distinguishes EBs from planets)
3. Try higher-order Taylor terms (1 - x²/2 + x⁴/24) as a hyperparameter
4. Add data augmentation (noise injection, phase jitter)
5. Evaluate on the Autovetter Planet Candidate Catalog for direct
   AstroNet comparison

Start with #1 (more data) and #2 (secondary eclipse channel) as these
will have the biggest impact on precision.
```

### Completed Sessions
1. **Session 1:** Taylor gate verified on synthetic data (baseline=0, gradients pass)
2. **Session 2:** Two-channel CNN connected, 94.5% val accuracy on synthetic data
3. **Session 3:** Real Kepler pipeline built, remove_outliers bug found and fixed
4. **Session 4:** sin→cos gate fix (90° misalignment), trained on 100 real TCEs,
   75% accuracy / 100% recall / F1=0.800
