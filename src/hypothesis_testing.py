import pandas as pd
import numpy as np
from scipy import stats

matches = pd.read_csv('../data/processed/matches_clean.csv')
deliveries = pd.read_csv('../data/processed/deliveries_clean.csv')

print("=" * 60)
print("HYPOTHESIS TEST 1: Does winning the toss give a real advantage?")
print("=" * 60)

# H0 (null hypothesis): Toss winner wins the match 50% of the time (no advantage)
# H1 (alternative): Toss winner wins more/less than 50% of the time

valid = matches[matches['winner'] != 'No Result'].dropna(subset=['winner', 'toss_winner'])
toss_win_match_win = (valid['toss_winner'] == valid['winner']).astype(int)

n = len(toss_win_match_win)
successes = toss_win_match_win.sum()
observed_rate = successes / n

# One-sample proportion z-test against 50%
from statsmodels.stats.proportion import proportions_ztest
count = successes
nobs = n
z_stat, p_value = proportions_ztest(count, nobs, value=0.5)

print(f"Sample size: {n} matches")
print(f"Observed win rate after winning toss: {observed_rate*100:.2f}%")
print(f"Z-statistic: {z_stat:.4f}")
print(f"P-value: {p_value:.4f}")

alpha = 0.05
if p_value < alpha:
    print(f"CONCLUSION: p < {alpha} → statistically significant. Toss winning DOES have a real effect on match outcome.")
else:
    print(f"CONCLUSION: p >= {alpha} → NOT statistically significant. The observed edge could just be random chance.")

print("\n" + "=" * 60)
print("HYPOTHESIS TEST 2: Does batting first vs chasing affect win rate?")
print("=" * 60)

# H0: bat-first teams and chasing teams win at the same rate
# H1: they differ

bat_first_wins = valid[valid['toss_decision'] == 'bat']
chase_wins = valid[valid['toss_decision'] == 'field']

bat_first_win_rate = (bat_first_wins['toss_winner'] == bat_first_wins['winner']).mean()
chase_win_rate = (chase_wins['toss_winner'] == chase_wins['winner']).mean()

print(f"Bat-first win rate (after winning toss): {bat_first_win_rate*100:.2f}% (n={len(bat_first_wins)})")
print(f"Chase win rate (after winning toss): {chase_win_rate*100:.2f}% (n={len(chase_wins)})")

# Two-proportion z-test
count = np.array([
    (bat_first_wins['toss_winner'] == bat_first_wins['winner']).sum(),
    (chase_wins['toss_winner'] == chase_wins['winner']).sum()
])
nobs = np.array([len(bat_first_wins), len(chase_wins)])

z_stat2, p_value2 = proportions_ztest(count, nobs)

print(f"Z-statistic: {z_stat2:.4f}")
print(f"P-value: {p_value2:.4f}")

if p_value2 < alpha:
    print(f"CONCLUSION: p < {alpha} → statistically significant difference between bat-first and chase strategies.")
else:
    print(f"CONCLUSION: p >= {alpha} → no statistically significant difference between the two strategies.")

print("\n" + "=" * 60)
print("HYPOTHESIS TEST 3: Do top batters have a HIGHER strike rate in death overs vs powerplay?")
print("=" * 60)

# H0: strike rate in death overs == strike rate in powerplay (no difference)
# H1: strike rate differs between phases

deliveries['phase'] = pd.cut(deliveries['over'], bins=[0, 6, 15, 20], labels=['Powerplay', 'Middle', 'Death'])

powerplay_runs = deliveries[deliveries['phase'] == 'Powerplay']['batsman_runs']
death_runs = deliveries[deliveries['phase'] == 'Death']['batsman_runs']

# Independent samples t-test
t_stat, p_value3 = stats.ttest_ind(death_runs.dropna(), powerplay_runs.dropna(), equal_var=False)

print(f"Mean runs per ball — Powerplay: {powerplay_runs.mean():.3f}")
print(f"Mean runs per ball — Death overs: {death_runs.mean():.3f}")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value3:.6f}")

if p_value3 < alpha:
    print(f"CONCLUSION: p < {alpha} → statistically significant difference in scoring rate between phases.")
else:
    print(f"CONCLUSION: p >= {alpha} → no statistically significant difference.")

# ============================================
# SAVE RESULTS SUMMARY
# ============================================
summary = pd.DataFrame([
    {'test': 'Toss winner win-rate vs 50%', 'p_value': p_value, 'significant': p_value < alpha},
    {'test': 'Bat-first vs Chase win-rate', 'p_value': p_value2, 'significant': p_value2 < alpha},
    {'test': 'Powerplay vs Death scoring rate', 'p_value': p_value3, 'significant': p_value3 < alpha}
])
summary.to_csv('../reports/hypothesis_test_results.csv', index=False)
print("\nSaved results to reports/hypothesis_test_results.csv")