# Taylor-CNN: Physics-Informed Transit Classification

A physics-informed 1D CNN for exoplanet transit classification on phase-folded
light curves. The model uses a bank of five fixed-morphology Taylor gates with
learnable amplitudes. Achieves F1=0.872 with 1150 parameters and 100% TESS
recall zero-shot.

Preprint: https://doi.org/10.5281/zenodo.19696691

- **Best single model:** V10 λ=0.1 — F1 0.861 (1150 params)
- **Best ensemble:** V6b + V10 AND — F1 0.872
- **TESS zero-shot:** 100% recall, 80% precision
- **Paper draft:** `docs/paper/paper_draft.md`

## DR25 Batch Scoring (3 batches, ~2700 TCEs)

Applied V10+V6b AND ensemble to Kepler DR25 catalog.

| Batch | TCEs | F1    | False alarms | Missed confirmed |
|-------|------|-------|--------------|-----------------|
| 1     | 850  | 0.775 | 110          | 0               |
| 2     | 850  | 0.735 | 288          | 0               |
| 3     | 1000 | 0.725 | 349          | 17              |

Key findings:
- F1 drops from 0.872 (curated test) to 0.725-0.775
  (broader distribution) — expected generalisation gap
- AND ensemble misses confirmed planets where V10
  is confident but V6b disagrees:
  K01797.01: p_v10=0.832, p_v6b=0.151 → missed
  K01911.01: p_v10=0.811, p_v6b=0.476 → missed
- 0 confirmed planets found in DR25 FP archive
- Motivates weighted ensemble for V10.2

## Citation

If you use this code or paper in your research, please cite:

```bibtex
@article{kapali2026taylorcnn,
  author    = {Kapali, Srikanth},
  title     = {Taylor-CNN: Physics-Informed Transit
               Classification with a Multi-Template Gate Bank},
  year      = {2026},
  doi       = {10.5281/zenodo.19696691},
  url       = {https://doi.org/10.5281/zenodo.19696691},
  note      = {Zenodo preprint}
}
```
