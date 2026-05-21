# ═══════════════════════════════════════════
# PRODUCTION MODEL PROTECTION
# NEVER save to src/models/production/
# Save experiments to src/models/ only
# with descriptive version suffix
# ═══════════════════════════════════════════

import numpy as np
import pandas as pd

df = pd.read_csv('data/batch3_candidates_verified.csv')

missed  = df[(df['truth']==1) & (df['ensemble_pred']==0)]
falseal = df[(df['truth']==0) & (df['ensemble_pred']==1)]

print(f"Missed confirmed planets : {len(missed)}")
print(f"False alarms             : {len(falseal)}")
print()

weights = [
    (0.5, 0.5),   # equal weight
    (0.6, 0.4),   # slight V10 preference
    (0.7, 0.3),   # V10 dominant (recommended)
    (0.8, 0.2),   # strong V10 preference
    (1.0, 0.0),   # V10 only
]

thresholds = [0.4, 0.45, 0.5, 0.55, 0.6]

print("Weighted ensemble sweep on 17 missed planets")
print("="*70)
print(f"{'w_v10':>6} {'w_v6b':>6} {'thr':>5} "
      f"{'recovered':>10} {'new_FA':>8} {'net':>6}")
print("-"*70)

results = []
for w10, w6b in weights:
    for thr in thresholds:
        missed_score = (w10 * missed['p_v10'] +
                       w6b  * missed['p_v6b'])
        recovered = (missed_score > thr).sum()

        fa_score = (w10 * falseal['p_v10'] +
                   w6b  * falseal['p_v6b'])
        new_fa = (fa_score > thr).sum()

        # Net gain: recovered planets minus additional false alarms vs baseline
        net = recovered - (new_fa - len(falseal))

        print(f"{w10:>6.1f} {w6b:>6.1f} {thr:>5.2f} "
              f"{recovered:>8d}/17 "
              f"{new_fa:>6d}/349 "
              f"{net:>+6d}")

        results.append({
            'w_v10': w10, 'w_v6b': w6b,
            'threshold': thr,
            'recovered': recovered,
            'false_alarms': new_fa,
        })

results_df = pd.DataFrame(results)
print("\nTop 3 configurations by recovered/17:")
top = results_df.sort_values(
    ['recovered', 'false_alarms'],
    ascending=[False, True]
).head(3)
print(top.to_string())

results_df.to_csv('data/weighted_ensemble_sweep.csv', index=False)
print("\nSaved: data/weighted_ensemble_sweep.csv")
