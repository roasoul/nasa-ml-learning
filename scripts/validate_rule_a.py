# Rule A precision filter validation across batches 1, 2, and 3.
# Rule A: reject if stellar_r > 2.0 Rsun
# Goal: confirm ~15-20% FA reduction with zero confirmed planet loss.

import pandas as pd

RULE_A_THRESHOLD = 2.0

def rule_a(df_):
    return df_['stellar_r'] > RULE_A_THRESHOLD

batches = [
    ('Batch 1', 'data/kepler_dr25_1k_scores.csv'),
    ('Batch 2', 'data/kepler_dr25_2k_scores.csv'),
    ('Batch 3', 'data/batch3_candidates_verified.csv'),
]

print("Rule A validation: reject if stellar_r > {:.1f} Rsun".format(RULE_A_THRESHOLD))
print("=" * 70)

summary_rows = []

for label, path in batches:
    df = pd.read_csv(path)

    confirmed = df[df['truth'] == 1]
    fp_all    = df[df['truth'] == 0]
    falseal   = df[(df['truth'] == 0) & (df['ensemble_pred'] == 1)]

    # Rule A applied to false alarms
    fa_rejected  = rule_a(falseal).sum()
    fa_after     = len(falseal) - fa_rejected
    fa_pct       = fa_rejected / len(falseal) * 100 if len(falseal) else 0.0

    # Rule A applied to confirmed planets
    conf_rejected = rule_a(confirmed).sum()

    print(f"\n{label}  ({path})")
    print(f"  TCEs total         : {len(df)}")
    print(f"  Confirmed planets  : {len(confirmed)}")
    print(f"  False alarms       : {len(falseal)}")
    print(f"  stellar_r range    : {df['stellar_r'].min():.3f} - {df['stellar_r'].max():.3f} Rsun")

    # Stellar radius distribution for FAs
    fa_giants = falseal[rule_a(falseal)]
    print(f"\n  False alarms with stellar_r > {RULE_A_THRESHOLD}:")
    print(f"    count  : {len(fa_giants)}")
    if len(fa_giants):
        print(f"    stellar_r range: {fa_giants['stellar_r'].min():.2f} - {fa_giants['stellar_r'].max():.2f}")
        # Breakdown by stellar_r bin
        bins = [2, 3, 5, 10, 1000]
        for lo, hi in zip(bins, bins[1:]):
            n = ((fa_giants['stellar_r'] >= lo) & (fa_giants['stellar_r'] < hi)).sum()
            print(f"    {lo:4.0f} - {hi:4.0f} Rsun : {n}")

    # Confirmed planets with stellar_r > threshold
    conf_giants = confirmed[rule_a(confirmed)]
    print(f"\n  Confirmed planets with stellar_r > {RULE_A_THRESHOLD}: {len(conf_giants)}")
    if len(conf_giants):
        print("  *** RECALL HIT — these confirmed planets would be lost ***")
        print(conf_giants[['koi_id', 'depth_ppm', 'stellar_r', 'p_v10', 'p_v6b', 'ensemble_pred']].to_string())

    print(f"\n  Rule A result:")
    print(f"    FA before     : {len(falseal)}")
    print(f"    FA after      : {fa_after}")
    print(f"    FA rejected   : {fa_rejected}  ({fa_pct:.1f}%)")
    print(f"    Planets lost  : {conf_rejected}")

    summary_rows.append({
        'batch': label,
        'tce_total': len(df),
        'confirmed': len(confirmed),
        'fa_before': len(falseal),
        'fa_after': fa_after,
        'fa_rejected': fa_rejected,
        'fa_pct': round(fa_pct, 1),
        'planets_lost': conf_rejected,
    })

# Summary table
print()
print("=" * 70)
print("CROSS-BATCH SUMMARY")
print("=" * 70)
print(f"{'Batch':<10} {'FAs before':>10} {'FAs after':>10} {'FA rejected':>12} {'Planets lost':>13}")
print("-" * 60)
for r in summary_rows:
    print(f"{r['batch']:<10} {r['fa_before']:>10} {r['fa_after']:>10} "
          f"{r['fa_rejected']:>8} ({r['fa_pct']:>4.1f}%) {r['planets_lost']:>11}")

total_fa_before  = sum(r['fa_before']   for r in summary_rows)
total_fa_after   = sum(r['fa_after']    for r in summary_rows)
total_fa_rej     = sum(r['fa_rejected'] for r in summary_rows)
total_conf_lost  = sum(r['planets_lost'] for r in summary_rows)
overall_pct      = total_fa_rej / total_fa_before * 100 if total_fa_before else 0.0

print("-" * 60)
print(f"{'TOTAL':<10} {total_fa_before:>10} {total_fa_after:>10} "
      f"{total_fa_rej:>8} ({overall_pct:>4.1f}%) {total_conf_lost:>11}")

print()
print("VERDICT")
print("-" * 60)
if total_conf_lost == 0 and 15.0 <= overall_pct <= 25.0:
    verdict = "IMPLEMENT in V10.2 -- robust ~{:.1f}% FA cut, zero planet loss".format(overall_pct)
elif total_conf_lost == 0 and overall_pct > 5.0:
    verdict = "IMPLEMENT in V10.2 -- {:.1f}% FA cut, zero planet loss".format(overall_pct)
elif total_conf_lost == 0:
    verdict = "MARGINAL -- only {:.1f}% FA cut, but zero planet loss. Include as free filter.".format(overall_pct)
else:
    verdict = "REJECT -- loses {} confirmed planet(s). Raise threshold.".format(total_conf_lost)
print(f"  {verdict}")

# Save
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv('data/rule_a_validation.csv', index=False)
print("\nSaved: data/rule_a_validation.csv")
