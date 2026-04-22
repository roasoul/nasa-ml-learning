# Production Model Registry
## READ-ONLY — Never overwrite these files

| File                  | F1    | Prec  | Recall | Notes              |
|-----------------------|-------|-------|--------|--------------------|
| v5_baseline.pt        | 0.842 | 72.7% | 100%   | 16-TCE easy set    |
| v6b_recall947.pt      | 0.818 | 72.0% | 94.7%  | high recall anchor |
| v75_safe_gate.pt      | 0.815 | 76.7% | 86.8%  | safe Kepler gate   |
| v10_f1861.pt          | 0.861 | 82.9% | 89.5%  | best single model  |
| v10_log_mdwarf.pt     | 0.854 | TBD   | TBD    | M-dwarf aware      |

## Ensemble configurations (paper results)
| Ensemble              | F1    | Prec  | Recall | Use case          |
|-----------------------|-------|-------|--------|-------------------|
| v6b+v10 AND           | 0.872 | 85.0% | 89.5%  | balanced (best)   |
| v6b+v10+log OR(0.4)   | 0.800 | 66.7% | 100%   | discovery mode    |
| v6b+v10 weighted      | 0.854 | 79.5% | 92.1%  | high recall       |
| TESS zero-shot (v10)  | —     | 80.0% | 100%   | cross-instrument  |

## Rules
- NEVER save experiments to this directory
- All experiments → src/models/ with version suffix
- Load production models READ-ONLY in all scripts
- AdaptivePINNClassifier loads from here
