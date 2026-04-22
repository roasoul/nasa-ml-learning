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

## Citation

If you use this code or paper in your research, please cite:

```bibtex
@article{kapali2026taylorcnn,
  author    = {Kapali, Srikanth},
  title     = {Taylor-CNN: Physics-Informed Transit
               Classification with a Multi-Template Gate Bank},
  year      = {2026},
  doi       = {10.5281/zenodo.15696691},
  url       = {https://doi.org/10.5281/zenodo.15696691},
  note      = {Zenodo preprint}
}
```
