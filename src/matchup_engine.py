import pandas as pd
import numpy as np

deliveries = pd.read_csv('../data/processed/deliveries_clean.csv')

# ============================================
# HEAD-TO-HEAD: BATTER vs BOWLER
# ============================================
matchups = deliveries.groupby(['batsman', 'bowler']).agg(
    balls_faced=('batsman_runs', 'count'),
    runs_scored=('batsman_runs', 'sum'),
    dismissals=('dismissal_kind', lambda x: x.notnull().sum())
).reset_index()

# Only keep matchups with a meaningful sample (avoids noisy 1-ball "matchups")
matchups = matchups[matchups['balls_faced'] >= 6]

matchups['strike_rate'] = (matchups['runs_scored'] / matchups['balls_faced']) * 100
matchups['average'] = matchups.apply(
    lambda row: row['runs_scored'] / row['dismissals'] if row['dismissals'] > 0 else row['runs_scored'],
    axis=1
)

matchups = matchups.sort_values('balls_faced', ascending=False)
matchups.to_csv('../data/processed/matchups.csv', index=False)
print(deliveries['batsman'].unique()[:20])
print(deliveries['bowler'].unique()[:20])

print(f"Total unique matchups (6+ balls): {len(matchups)}")
print("\n=== SAMPLE: Most-faced matchups ===")
print(matchups.head(10).to_string(index=False))


# ============================================
# LOOKUP FUNCTION — check a specific matchup
# ============================================
def get_matchup(batsman_name, bowler_name):
    result = matchups[
        (matchups['batsman'].str.lower() == batsman_name.lower()) &
        (matchups['bowler'].str.lower() == bowler_name.lower())
    ]
    if result.empty:
        print(f"No matchup data found for {batsman_name} vs {bowler_name} (or fewer than 6 balls faced)")
        return None
    else:
        print(f"\n{batsman_name} vs {bowler_name}:")
        print(result.to_string(index=False))
        return result


# ============================================
# TRY IT — replace these names with real players from your dataset
# ============================================
# Uncomment and edit the line below with two real player names from your data:
get_matchup("V Kohli", "RA Jadeja")