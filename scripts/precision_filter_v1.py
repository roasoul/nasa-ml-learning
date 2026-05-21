# ===========================================
# PRODUCTION MODEL PROTECTION
# NEVER save to src/models/production/
# Save experiments to src/models/ only
# with descriptive version suffix
# ===========================================

import numpy as np
import pandas as pd

df = pd.read_csv('data/batch3_candidates_verified.csv')

confirmed = df[df['truth'] == 1]
missed    = df[(df['truth'] == 1) & (df['ensemble_pred'] == 0)]
falseal   = df[(df['truth'] == 0) & (df['ensemble_pred'] == 1)]

print("=" * 60)
print("DATASET SUMMARY")
print("=" * 60)
print(f"Total rows in CSV     : {len(df)}")
print(f"Confirmed (truth=1)   : {len(confirmed)}")
print(f"  -> missed by ensemble: {len(missed)}")
print(f"False alarms          : {len(falseal)}")
print()

# --- STEP 1: EXPLORE THE DISTRIBUTIONS -----------------------

def dist_summary(series, label):
    print(f"  {label}:")
    print(f"    min={series.min():.3f}  p25={series.quantile(0.25):.3f}  "
          f"median={series.median():.3f}  p75={series.quantile(0.75):.3f}  "
          f"p90={series.quantile(0.90):.3f}  p95={series.quantile(0.95):.3f}  "
          f"max={series.max():.3f}")

print("=" * 60)
print("STEP 1: DISTRIBUTIONS")
print("=" * 60)

print("\n--- FALSE ALARMS (n={}) ---".format(len(falseal)))
dist_summary(falseal['stellar_r'],              'stellar_r (Rsun)')
dist_summary(falseal['depth_ppm'],              'depth_ppm')
dist_summary(falseal['depth_ppm'] / falseal['stellar_r']**2,
                                                'depth / stellar_r^2')

print("\n--- CONFIRMED PLANETS (n={}) ---".format(len(confirmed)))
dist_summary(confirmed['stellar_r'],            'stellar_r (Rsun)')
dist_summary(confirmed['depth_ppm'],            'depth_ppm')
dist_summary(confirmed['depth_ppm'] / confirmed['stellar_r']**2,
                                                'depth / stellar_r^2')

print("\n--- MISSED CONFIRMED PLANETS (n={}) ---".format(len(missed)))
dist_summary(missed['stellar_r'],               'stellar_r (Rsun)')
dist_summary(missed['depth_ppm'],               'depth_ppm')
dist_summary(missed['depth_ppm'] / missed['stellar_r']**2,
                                                'depth / stellar_r^2')

# Flag outlier stellar radii in false alarms
fa_giants = falseal[falseal['stellar_r'] > 2.0]
print(f"\nFAs with stellar_r > 2.0: {len(fa_giants)}")
if len(fa_giants):
    print(fa_giants[['koi_id','depth_ppm','stellar_r','p_v10','p_v6b']].to_string())

fa_big_depth = falseal[falseal['depth_ppm'] > 10000]
print(f"\nFAs with depth_ppm > 10000: {len(fa_big_depth)}")
if len(fa_big_depth):
    print(fa_big_depth[['koi_id','depth_ppm','stellar_r','p_v10','p_v6b']].to_string())

# --- STEP 2: DESIGN THE FILTER -------------------------------

def apply_rule(subset, rule_fn):
    mask = rule_fn(subset)
    return mask.sum(), mask

def rule_A(df_):
    return df_['stellar_r'] > 2.0

def rule_B(df_):
    return (df_['depth_ppm'] > 5000) & (df_['stellar_r'] > 1.5)

def rule_C(df_):
    return (df_['stellar_r'] > 2.0) | \
           ((df_['depth_ppm'] > 10000) & (df_['stellar_r'] > 1.0))

print()
print("=" * 60)
print("STEP 2: FILTER RULES")
print("=" * 60)
print(f"\n{'Rule':<6} {'FAs rejected':>14} {'Confirmed lost':>16} {'Missed lost':>13}")
print("-" * 55)

rows = []
for name, fn in [('A', rule_A), ('B', rule_B), ('C', rule_C)]:
    fa_rej,  fa_mask  = apply_rule(falseal,   fn)
    con_rej, con_mask = apply_rule(confirmed, fn)
    mis_rej, mis_mask = apply_rule(missed,    fn)
    print(f"  {name}     {fa_rej:>6}/{len(falseal)}    "
          f"{con_rej:>6}/{len(confirmed)}    "
          f"{mis_rej:>6}/{len(missed)}")
    rows.append({'rule': name,
                 'fa_rejected': fa_rej, 'fa_total': len(falseal),
                 'confirmed_lost': con_rej, 'confirmed_total': len(confirmed),
                 'missed_lost': mis_rej, 'missed_total': len(missed)})

# --- STEP 3: COMBINE WITH WEIGHTED VOTE ----------------------

print()
print("=" * 60)
print("STEP 3: COMBINED PIPELINE  (weighted vote 0.7/0.3 thr=0.40 + Rule A)")
print("=" * 60)

w10, w6b, thr = 0.7, 0.3, 0.40

# Weighted vote applied to missed planets
missed_score  = w10 * missed['p_v10']  + w6b * missed['p_v6b']
missed_vote   = missed_score > thr

# Weighted vote applied to false alarms
fa_score      = w10 * falseal['p_v10'] + w6b * falseal['p_v6b']
fa_vote       = fa_score > thr

# Precision filter (Rule A) applied on top
fa_kept_after_vote = falseal[fa_vote]
fa_pass_filter = ~rule_A(fa_kept_after_vote)
fa_final = fa_kept_after_vote[fa_pass_filter]

# Missed planets recovered by weighted vote, not blocked by filter
missed_recovered = missed[missed_vote & ~rule_A(missed)]
missed_blocked   = missed[missed_vote & rule_A(missed)]

print(f"\nWeighted vote (0.7/0.3, thr=0.40):")
print(f"  Missed planets promoted  : {missed_vote.sum()}/17")
print(f"  FAs promoted             : {fa_vote.sum()}/349")

print(f"\nAfter Rule A precision filter:")
print(f"  Missed planets recovered : {len(missed_recovered)}/17")
print(f"  FAs remaining            : {len(fa_final)}/{fa_vote.sum()} promoted -> "
      f"{len(fa_final)}/349 total")
if len(missed_blocked) > 0:
    print(f"\n  WARNING: {len(missed_blocked)} recovered planet(s) blocked by Rule A:")
    print(missed_blocked[['koi_id','depth_ppm','stellar_r','p_v10','p_v6b']].to_string())

# Compare baseline AND ensemble vs combined pipeline
print(f"\n{'':30} {'AND baseline':>14} {'Weighted+RuleA':>16}")
print("-" * 62)
print(f"{'Missed planets recovered':30} {'0':>14} {len(missed_recovered):>16}")
print(f"{'False alarms':30} {'349':>14} {len(fa_final):>16}")

# --- STEP 4: HONEST ASSESSMENT -------------------------------

print()
print("=" * 60)
print("STEP 4: HONEST ASSESSMENT")
print("=" * 60)

# Check Rule A impact on the full confirmed set
con_blocked_A = confirmed[rule_A(confirmed)]
recall_impact = len(con_blocked_A) / len(confirmed) if len(confirmed) else 0
print(f"\nRule A blocks {len(con_blocked_A)}/{len(confirmed)} confirmed planets "
      f"({recall_impact*100:.1f}% recall hit)")

if len(con_blocked_A) > 0:
    print("\n  Confirmed planets blocked by Rule A (stellar_r > 2.0):")
    print(con_blocked_A[['koi_id','depth_ppm','stellar_r','p_v10','p_v6b']].to_string())

fa_cut_A   = rule_A(falseal).sum()
prec_gain  = fa_cut_A / len(falseal) * 100

print(f"\nRule A summary:")
print(f"  FAs removed    : {fa_cut_A}/{len(falseal)}  ({prec_gain:.1f}% FA reduction)")
print(f"  Confirmed lost : {len(con_blocked_A)}/{len(confirmed)}  ({recall_impact*100:.1f}% recall hit)")
print(f"  Missed lost    : {rule_A(missed).sum()}/{len(missed)}")

print()
if recall_impact == 0 and fa_cut_A > 0:
    verdict = "IMPLEMENT -- zero recall cost, reduces FAs"
elif recall_impact < 0.02 and fa_cut_A > 5:
    verdict = "WORTH CONSIDERING -- small recall cost, meaningful FA reduction"
elif fa_cut_A == 0:
    verdict = "SKIP -- no FA reduction at this threshold"
else:
    verdict = "REJECT -- recall hit too large for FA gain"
print(f"Verdict: {verdict}")

# --- SAVE RESULTS ---------------------------------------------

results_df = pd.DataFrame(rows)
results_df.to_csv('data/precision_filter_results.csv', index=False)
print(f"\nSaved: data/precision_filter_results.csv")
