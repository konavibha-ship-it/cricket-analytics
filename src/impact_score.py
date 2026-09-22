import pandas as pd
import numpy as np

deliveries = pd.read_csv('../data/processed/deliveries_clean.csv')
matches = pd.read_csv('../data/processed/matches_clean.csv')

# ============================================
# BATTING IMPACT
# ============================================
batting = deliveries.groupby('batsman').agg(
    total_runs=('batsman_runs', 'sum'),
    balls_faced=('batsman_runs', 'count'),
    innings=('match_id', 'nunique'),
    fours=('batsman_runs', lambda x: (x == 4).sum()),
    sixes=('batsman_runs', lambda x: (x == 6).sum())
).reset_index()

batting['strike_rate'] = (batting['total_runs'] / batting['balls_faced']) * 100
batting['avg_runs_per_innings'] = batting['total_runs'] / batting['innings']

# Only consider players with meaningful sample size
batting = batting[batting['balls_faced'] >= 100]

# Normalize each component 0-1 (min-max scaling)
def normalize(col):
    return (col - col.min()) / (col.max() - col.min())

batting['sr_norm'] = normalize(batting['strike_rate'])
batting['avg_norm'] = normalize(batting['avg_runs_per_innings'])
batting['boundary_rate'] = (batting['fours'] + batting['sixes']) / batting['balls_faced']
batting['boundary_norm'] = normalize(batting['boundary_rate'])

# Custom weighted Batting Impact Score (your own formula — explainable in interview)
batting['batting_impact'] = (
    batting['sr_norm'] * 0.35 +
    batting['avg_norm'] * 0.40 +
    batting['boundary_norm'] * 0.25
) * 100

# ============================================
# BOWLING IMPACT
# ============================================
bowling = deliveries.groupby('bowler').agg(
    balls_bowled=('total_runs', 'count'),
    runs_conceded=('total_runs', 'sum'),
    innings=('match_id', 'nunique')
).reset_index()

wickets = deliveries[
    deliveries['dismissal_kind'].notnull() &
    ~deliveries['dismissal_kind'].isin(['run out', 'retired hurt', 'obstructing the field'])
].groupby('bowler').size().reset_index(name='wickets')

bowling = bowling.merge(wickets, on='bowler', how='left')
bowling['wickets'] = bowling['wickets'].fillna(0)

bowling = bowling[bowling['balls_bowled'] >= 60]  # meaningful sample

bowling['economy'] = (bowling['runs_conceded'] / bowling['balls_bowled']) * 6
bowling['wickets_per_innings'] = bowling['wickets'] / bowling['innings']

# Lower economy = better, so invert before normalizing
bowling['economy_inv'] = 1 / bowling['economy']
bowling['economy_norm'] = normalize(bowling['economy_inv'])
bowling['wickets_norm'] = normalize(bowling['wickets_per_innings'])

bowling['bowling_impact'] = (
    bowling['economy_norm'] * 0.45 +
    bowling['wickets_norm'] * 0.55
) * 100

# ============================================
# SAVE RESULTS
# ============================================
batting_result = batting[['batsman', 'total_runs', 'strike_rate', 'avg_runs_per_innings', 'batting_impact']]
batting_result = batting_result.sort_values('batting_impact', ascending=False)

bowling_result = bowling[['bowler', 'wickets', 'economy', 'wickets_per_innings', 'bowling_impact']]
bowling_result = bowling_result.sort_values('bowling_impact', ascending=False)

batting_result.to_csv('../data/processed/batting_impact.csv', index=False)
bowling_result.to_csv('../data/processed/bowling_impact.csv', index=False)

print("=== TOP 10 BATTING IMPACT ===")
print(batting_result.head(10).to_string(index=False))

print("\n=== TOP 10 BOWLING IMPACT ===")
print(bowling_result.head(10).to_string(index=False))