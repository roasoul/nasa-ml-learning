# V8 / V8.5 / V9 Findings — Shape-Based Discrimination

Status: **null result** across three approaches. Shape features separate
classes 5× in aggregate, but no single-global-B formulation converts
that into F1 gain.

## Summary table

| Version | Acc   | Prec  | Recall | F1     | Params |
|---|---|---|---|---|---|
| V6 C    | 80.3% | 76.7% | 86.8% | **0.815** | 914 |
| V7.5    | 80.3% | 76.7% | 86.8% | **0.815** | 914 |
| V8      | 77.6% | 73.3% | 86.8% | 0.795 | 916 |
| V8.5    | 76.3% | 71.7% | 86.8% | 0.786 | 921 |
| V9 (λ=0.1) | 73.7% | 69.6% | 84.2% | 0.762 | 916 |
| V9 (λ=0.5) | 75.0% | 72.1% | 81.6% | 0.765 | 916 |
| V9 (λ=1.0) | 50.0% |  0.0% |  0.0% | 0.000 | 916 |

Seed=42 stratified 70/15/15 split, 76-TCE test set — identical across
all rows so comparisons are direct.

---

## V8 — Learnable curvature parameter B

Extended the Taylor gate with a 4th-order Taylor term controlled by a
learnable curvature parameter and a learnable phase offset:

    y = min(0, -A · (1 − x²/2 + B · x⁴/24)),   x = clamp(phase − t0, −π, π)

### Analytic gradients (verified by torch.autograd.gradcheck)

    ∂y/∂A = −(1 − x²/2 + B · x⁴/24) · mask
    ∂y/∂B = −A · (x⁴/24) · mask
    ∂y/∂x = A · (x − B · x³/6) · mask

Three gradcheck variants all PASS in float64 at atol=1e-5, rtol=1e-4
(Function-level A & B, Module-level A & B & t0, baseline zero-gradient
sanity). The hand-coded backward is correct.

### V8 result: F1 0.795

A converged to 0.026, B drifted from init 1.0 down to +0.58, t0 stayed
near zero. No F1 gain over V6 Config C. The single global B learns a
*population-averaged* curvature — somewhere between the U-shape planets
and V-shape EBs, which is useful to neither class.

---

## V8.5 — Five shape features fused into the classifier

    shape = [B_global, AUC_raw, AUC_norm = AUC/A, T12_T14, flat_bottom]

Normalized soft masks operating on `depth_norm = depth/A` with k=50:

    dip_all  = sigmoid(k · (depth_norm − 0.1))
    flat_bot = sigmoid(k · (depth_norm − 0.8))
    ingress  = dip_all − flat_bot
    total    = (1 − dip_all) + ingress + flat_bot + ε   # =1 by construction
    ingress  = ingress / total
    flat_bot = flat_bot / total

Fixed `[−1, 1]` phase grid via `F.interpolate` makes AUC
instrument-agnostic. Concatenate `(batch, 16)` CNN features with
`(batch, 5)` shape features; classifier = Linear(21, 1).

### V8.5-v1 bug — features were batch-constant

Initial extraction pulled features from `gate_output`, which is
byte-identical for every TCE because the fold pipeline outputs the
same `−π..π` 200-bin phase grid across all 500 samples. All shape
features came out identical per row:

    shape_features[0] == shape_features[1] == ... == shape_features[N-1]

The five "features" were effectively five learned biases. Caught before
proceeding to analysis.

### V8.5-v2 fix — features from primary_flux

Moved extraction onto `primary_flux` (per-sample). Kept `B` as a
shared global prior (same value per row by design, still useful as
"population template context"). Aggregate feature separation on the
test set:

| Feature     | Planet median | FP median  | Ratio |
|-------------|--------------:|-----------:|:-----:|
| AUC_norm    |        0.070  |     0.390  |  ~5.6× |
| flat_bottom |        0.010  |     0.050  |   ~5× |
| T12/T14     |        0.688  |     0.598  |   weak |

### V8.5 result: F1 0.786

Features are real and separating, but the Linear(21, 1) fusion head
did not convert that into F1 gain. Gate's A *collapsed to −0.0013* —
the classifier found a pathway through the five shape features and
zeroed out the gate channel.

### SS-flag validation v1 (closed-form least-squares): NULL

Per-sample B via closed-form `(A, AB)` lstsq on dip-only samples:

- SS=0 (not-EB): n=42, median B = +1.156
- SS=1 (EB-flag): n=34, median B = +1.168
- **Separation gap = 0.012 → NULL**

### Mask-overlap check: PASSED

High-SNR T12/T14 median 0.637; only 16 % of high-SNR samples fall in
[0.4, 0.6]. Normalized soft masks do not drift to 0.5 at low SNR
(design goal met).

---

## V9 — DynamicGeometryLoss (shape in the loss, not the input)

V8.5 proved that fusing features into the classifier input lets BCE
bypass them. V9 moves the shape signal into the loss:

    L = BCE + 0.01 · |B|.mean()
        + λ(SNR) · planet_prob · t12_t14
        + λ(SNR) · planet_prob · (1 − AUC_norm)

    λ(SNR) = λ_min + (λ_max − λ_min) · sigmoid(SNR − 12)

A post-step `A.clamp(min=0.001)` was added so the gate cannot die.
CNN input is unchanged from V6 — clean baseline.

### λ sweep result: NULL across the board

| λ_max | Acc   | Prec  | Rec   | F1    | A_final | B_final |
|---|---|---|---|---|---|---|
| 0.1   | 73.7% | 69.6% | 84.2% | 0.762 | 0.0010 (floor) | +0.57 |
| 0.5   | 75.0% | 72.1% | 81.6% | 0.765 | 0.0010 (floor) | +0.82 |
| 1.0   | 50.0% |  0.0% |  0.0% | 0.000 | 0.0085         | +1.00 |

Best V9 (F1 0.765 at λ=0.5) is **worse than V6 Config C (0.815)** by
0.05 F1. λ=1.0 is catastrophic — the geometry term overwhelms BCE and
the classifier predicts "not-planet" for everything.

### Root cause: the penalty direction is anti-aligned with class
### separation *on this dataset*

The loss was designed under the hypothesis

> planets are U-shape (low T12/T14, high AUC_norm);
> EBs are V-shape (high T12/T14, low AUC_norm).

The actual numbers on the 76-TCE test set (V9 winner λ=0.5):

| Feature     | Planet median | FP median  | Direction in L |
|-------------|--------------:|-----------:|----------------|
| T12/T14     |        0.673  |     0.604  | planets HIGHER — penalty punishes planets |
| AUC_norm    |        0.090  |     0.506  | planets LOWER — `(1 − AUC_norm)` punishes planets |

Both terms push in the wrong direction. Why?

- Normalized soft masks use `depth_norm = depth/A`, so deep FP
  eclipses (`depth_norm > 0.8`) route into the `flat_bot` bucket,
  not `ingress`. This *lowers* their T12/T14. Planets with modest
  depth keep most of the signal in the ingress mask.
- `AUC_norm = AUC/A` is small for narrow planet transits (~0.07) and
  large for wide EB eclipses (0.5–8). The `(1 − AUC_norm)` term gives
  a large positive penalty multiplied into planet predictions and
  a negative "reward" for deep-dip FPs — exactly the opposite of the
  intended pressure.

Fixing the sign (use `prob · (1 − t12_t14)` and `prob · AUC_norm`)
would likely help, but that's a new experiment, and the broader lesson
supersedes it.

### Gate collapse despite the clamp

A hit the 0.001 floor within ~25 epochs for λ ∈ {0.1, 0.5}. The clamp
kept A technically alive but `gate_out ≈ 0` everywhere. At λ=1.0 the
gate barely moved because training collapsed too fast for any parameter
drift to develop.

### SS-flag validation v2 (bounded scipy curve_fit on dip-only points): NULL

Bounded `curve_fit` with `A ∈ [0, 0.1]`, `B ∈ [−2, 2]`, run on all 500
TCEs, fitting `taylor_poly(x, A, B) = −A · (1 − x²/2 + B · x⁴/24)` to
the dip-region points only (dropping the non-smooth `min(0, ·)` clamp
that had pinned B at the lower bound in v1):

- SS=0 (not-EB): n=285, median B = +1.212
- SS=1 (EB-flag): n=215, median B = +1.177
- **Separation gap = 0.034 → NULL**

Two independent per-sample B estimators (closed-form lstsq in V8.5
and bounded scipy fit in V9) both converge on the same answer:
**primary-eclipse shape cannot recover koi_fpflag_ss.**

---

## Root cause unifying V7, V8, V8.5, V9: Kepler pipeline bias

The V7 duration-mystery finding applies one level deeper. The Kepler
TCE pipeline fits a Mandel-Agol *planet* model to every candidate —
including EBs. That means:

- Archive `koi_duration` (V7 input) is a "best planet fit" T14, not
  the true eclipse duration — EBs look planet-consistent in duration.
- Archive folded flux (V8/V8.5/V9 input) inherits the same bias. The
  folded light curve is the data the model would see, but the per-sample
  morphology you can extract from it is constrained by how the pipeline
  detrended, phase-folded, and clipped the signal — all tuned for a
  planet hypothesis.

The signal the classifier has available is therefore *primary-eclipse
morphology through a planet-biased lens.* The SS flag lives in:

- **Secondary eclipse depth** (not in the primary fold that V8–V9
  fit against).
- **Centroid motion** (pixel-level data; would require re-downloading
  target-pixel files from MAST).
- **Odd/even transit depth** differences (already a channel in V6,
  which is why V6 works — it gets a little of the EB signal from
  there — but not strongly enough to push past 0.815 F1).

Every attempt to extract an EB signature from primary-fold shape alone
— Kepler duration violation (V7), learnable curvature B (V8), fused
shape features (V8.5), shape penalty loss (V9), per-sample B fit (SS
validation v1 and v2) — converges on the same null. The signal the
pipeline removed, the model cannot recover.

---

## Implication for V10: multi-template gate bank

A single global B learns a population-averaged template — useful to
neither planets nor EBs. A per-sample morphology signal requires
per-sample routing, not a global prior.

V10 direction: **N parallel Taylor gates**, each tuned to a distinct
dip morphology:

    Gate 1: y = min(0, −A₁ · (1 − x²/2))           # symmetric planet U
    Gate 2: y = min(0, −A₂ · (x − x³/6))           # asymmetric ingress
    Gate 3: y = max(0,  A₃ · (1 − x²/2))           # inverted secondary
    Gate 4: y = min(0, −A₄ · box(x))               # grazing flat-box
    Gate 5: y = min(0, −A₅ · gaussian(x))          # spot crossing

Each gate produces its own `(batch, 200)` output. The CNN consumes
the stack — `(batch, N, 200)` — and learns which gate fired for each
sample. This replaces the "fit one B to the population" bottleneck
with "let each sample pick its own gate from a learnable bank."

Whether that also pushes past the Kepler-pipeline-bias ceiling is an
open question — but V9 foreclosed the single-template approach, so
V10 is the next structural step worth testing.

Alternative V10 paths if the gate bank also fails:

1. Reinstate the geometry loss with the sign corrected empirically
   from the V9 class-separation data.
2. Non-linear classifier head (MLP) — V8.5 used Linear(21, 1); the
   bypass of shape features suggests the head wasn't expressive
   enough to use them.
3. Re-download target-pixel files to build a centroid-motion channel
   (genuine new physics, not derivable from the folded light curve).

---

## Artifacts

| File                                         | Purpose |
|---|---|
| `src/models/taylor_layer_v8.py`              | TaylorGateLayerV8 + custom Function (A, B, t0) |
| `src/models/taylor_cnn_v8.py`                | V8 / V8.5 classifier (4-channel + optional shape fusion) |
| `src/models/taylor_cnn_v9.py`                | V9 classifier (shape features in loss only) |
| `src/models/geometry_loss.py`                | DynamicGeometryLoss (SNR-weighted shape + AUC penalty) |
| `src/models/taylor_cnn_v8.pt`                | V8 trained weights |
| `src/models/taylor_cnn_v85.pt`               | V8.5 trained weights |
| `src/models/taylor_cnn_v9.pt`                | V9 winner (λ=0.5) weights |
| `scripts/gradcheck_v8.py`                    | Analytic-backward verification for (A, B, t0) |
| `scripts/run_v8.py`                          | V8 / V8.5 training |
| `scripts/v8_figures.py`                      | V8.5 shape-feature plots (a–d) |
| `scripts/v8_ss_validation.py`                | SS-flag validation v1 (closed-form) |
| `scripts/run_v9.py`                          | V9 λ-sweep training |
| `scripts/v9_figures.py`                      | V9 metrics-vs-λ, A-trace, T12/T14 distribution |
| `scripts/v9_ss_validation.py`                | SS-flag validation v2 (bounded scipy curve_fit) |
| `data/ss_flag_cache.csv`                     | Cached NASA Archive SS + SNR for 9564 KOIs |
| `data/v8_training.log`, `data/v9_training.log` | Full training output |
| `notebooks/figures/v8_*.png`                 | V8.5 diagnostic figures |
| `notebooks/figures/v9_*.png`                 | V9 λ-sweep diagnostic figures |
| `notebooks/figures/B_vs_SS_flag{,_v2}.png`   | SS-flag validation figures |
