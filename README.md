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

## Confirmed Planet Recoveries from DR25 False Positive Archive

Planets recovered by V10+V6b AND ensemble that were
marked FALSE POSITIVE in Kepler DR25 but subsequently
confirmed as genuine exoplanets by independent methods.

| KOI | Name | p_v10 | p_v6b | Depth (ppm) | Period (d) | Radius (R⊕) | NASA Confirmation |
|-----|------|-------|-------|-------------|------------|-------------|-------------------|
| K01227.01 | Kepler-1227 b | 0.9963 | 0.8420 | 20,739 | TBD | TBD | [NASA Archive](https://exoplanetarchive.ipac.caltech.edu/overview/Kepler-1227) |
| K01829.01 | Kepler-1829 b | 0.9200 | 0.5936 | 2,941 | 5.19 | 2.48 | [NASA Archive](https://exoplanetarchive.ipac.caltech.edu/overview/Kepler-1829) |
| K00946.01 | Kepler-946 b  | 0.8805 | 0.7412 | 1,273 | 11.79 | 1.92 | [NASA Archive](https://exoplanetarchive.ipac.caltech.edu/overview/Kepler-946) |
| K03289.01 | Kepler-1455 b | 0.8805 | 0.7394 | 1,047 | 49.30 | 2.16 | [NASA Archive](https://exoplanetarchive.ipac.caltech.edu/overview/Kepler-1455) |

> **Notes:**
> - All four marked `FALSE POSITIVE` in Kepler DR25 cumulative KOI catalog
> - Both V10 and V6b independently agreed → AND ensemble prediction = PLANET
> - Kepler-1455 b equilibrium temperature ~295K — within or near habitable zone
> - Recovery rate: 4/4 top candidates verified (full DR25 run in progress)
> - Root cause: Mandel-Agol pipeline bias documented in V7 null result
>   causes genuine planets to inherit EB-like duration signatures → miscategorized as FP

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
