import pandas as pd
import numpy as np

from data_loader import load_deliveries
deliveries = load_deliveries()

# ============================================
# BUILD BASELINE EXPECTATIONS PER (format, phase) SITUATION
# ============================================
deliveries['phase'] = pd.cut(deliveries['over'], bins=[0, 6, 15, 20, 50], labels=['Powerplay', 'Middle', 'Death', 'Extended'])

# Only use formats with a consistent over-based phase structure
situational = deliveries[deliveries['format'].isin(['IPL', 'T20I', 'ODI'])]

baseline = situational.groupby(['format', 'phase'], observed=True).agg(
    avg_runs_per_ball=('batsman_runs', 'mean'),
    wicket_rate=('dismissal_kind', lambda x: x.notnull().mean())
).reset_index()

print("=== Baseline Expected Rates by Format & Phase ===")
print(baseline)

baseline_lookup = baseline.set_index(['format', 'phase'])


def calculate_xruns_xwickets(player_name, role='batsman'):
    """
    Compares a player's actual runs/wickets against what would be 'expected'
    given the situations (format + phase) they played in — like xG vs actual goals.
    """
    if role == 'batsman':
        player_data = situational[situational['batsman'] == player_name].copy()
    else:
        player_data = situational[situational['bowler'] == player_name].copy()

    if player_data.empty:
        return None

    def get_expected_runs(row):
        try:
            return baseline_lookup.loc[(row['format'], row['phase']), 'avg_runs_per_ball']
        except KeyError:
            return np.nan

    def get_expected_wicket(row):
        try:
            return baseline_lookup.loc[(row['format'], row['phase']), 'wicket_rate']
        except KeyError:
            return np.nan

    player_data['expected_runs'] = player_data.apply(get_expected_runs, axis=1)
    player_data['expected_wicket'] = player_data.apply(get_expected_wicket, axis=1)

    if role == 'batsman':
        actual_runs = player_data['batsman_runs'].sum()
        expected_runs = player_data['expected_runs'].sum()
        actual_dismissals = player_data['dismissal_kind'].notnull().sum()
        expected_dismissals = player_data['expected_wicket'].sum()

        return {
            'player': player_name,
            'balls_faced': len(player_data),
            'actual_runs': int(actual_runs),
            'expected_runs (xRuns)': round(expected_runs, 1),
            'runs_above_expected': round(actual_runs - expected_runs, 1),
            'actual_dismissals': int(actual_dismissals),
            'expected_dismissals (xDismissals)': round(expected_dismissals, 1),
            'dismissals_vs_expected': round(actual_dismissals - expected_dismissals, 1)
        }
    else:
        actual_runs_conceded = player_data['total_runs'].sum()
        expected_runs_conceded = player_data['expected_runs'].sum()
        actual_wickets = player_data['dismissal_kind'].notnull().sum()
        expected_wickets = player_data['expected_wicket'].sum()

        return {
            'player': player_name,
            'balls_bowled': len(player_data),
            'actual_runs_conceded': int(actual_runs_conceded),
            'expected_runs_conceded': round(expected_runs_conceded, 1),
            'runs_saved_vs_expected': round(expected_runs_conceded - actual_runs_conceded, 1),
            'actual_wickets': int(actual_wickets),
            'expected_wickets (xWickets)': round(expected_wickets, 1),
            'wickets_above_expected': round(actual_wickets - expected_wickets, 1)
        }


if __name__ == "__main__":
    print("\n=== Batting xRuns Example ===")
    print(calculate_xruns_xwickets("V Kohli", role='batsman'))

    print("\n=== Bowling xWickets Example ===")
    print(calculate_xruns_xwickets("JJ Bumrah", role='bowler'))