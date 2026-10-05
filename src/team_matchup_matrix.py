import pandas as pd
import numpy as np

deliveries = pd.read_parquet('../data/processed/deliveries_clean.parquet')
matches = pd.read_parquet('../data/processed/matches_clean.parquet')


def build_team_matchup_matrix(team_a, team_b, format_filter='IPL', min_balls=6):
    """
    Builds a full batter x bowler strike-rate matrix between two teams'
    historical squads (anyone who has ever played for either team in this format).
    """
    format_matches = matches[matches['format'] == format_filter]
    format_deliveries = deliveries[deliveries['match_id'].isin(format_matches['match_id'])]

    # Find anyone who has batted for team_a
    team_a_batters = format_deliveries[format_deliveries['batting_team'] == team_a]['batsman'].unique()

    # Find anyone who has bowled for team_b (bowling_team isn't directly in the schema,
    # so infer via: bowler appears in deliveries where batting_team == team_b's opponent)
    team_b_matches = format_matches[(format_matches['team1'] == team_b) | (format_matches['team2'] == team_b)]
    team_b_deliveries = format_deliveries[format_deliveries['match_id'].isin(team_b_matches['match_id'])]
    team_b_bowlers = team_b_deliveries[team_b_deliveries['batting_team'] != team_b]['bowler'].unique()

    matchups = format_deliveries[
        (format_deliveries['batsman'].isin(team_a_batters)) &
        (format_deliveries['bowler'].isin(team_b_bowlers))
    ]

    matrix_data = matchups.groupby(['batsman', 'bowler']).agg(
        runs=('batsman_runs', 'sum'),
        balls=('batsman_runs', 'count'),
        dismissals=('dismissal_kind', lambda x: x.notnull().sum())
    ).reset_index()

    matrix_data = matrix_data[matrix_data['balls'] >= min_balls]
    matrix_data['strike_rate'] = (matrix_data['runs'] / matrix_data['balls'] * 100).round(1)

    if matrix_data.empty:
        return None, None

    # Pivot into a matrix: rows = batters, columns = bowlers, values = strike rate
    pivot = matrix_data.pivot(index='batsman', columns='bowler', values='strike_rate')

    return pivot, matrix_data


def find_key_matchups(matrix_data, top_n=5):
    """Surfaces the most extreme matchups — biggest advantage/disadvantage pairs."""
    if matrix_data is None or matrix_data.empty:
        return None, None

    favorable_for_batter = matrix_data.sort_values('strike_rate', ascending=False).head(top_n)
    favorable_for_bowler = matrix_data.sort_values('strike_rate', ascending=True).head(top_n)

    return favorable_for_batter, favorable_for_bowler


if __name__ == "__main__":
    team_a = "Mumbai Indians"
    team_b = "Chennai Super Kings"

    pivot, matrix_data = build_team_matchup_matrix(team_a, team_b, format_filter='IPL')

    if pivot is not None:
        print(f"=== {team_a} batters vs {team_b} bowlers — Strike Rate Matrix ===")
        print(pivot.round(1))

        favorable_batter, favorable_bowler = find_key_matchups(matrix_data)

        print(f"\n=== Best matchups FOR {team_a} batters (highest SR) ===")
        print(favorable_batter[['batsman', 'bowler', 'balls', 'strike_rate']].to_string(index=False))

        print(f"\n=== Best matchups FOR {team_b} bowlers (lowest SR against) ===")
        print(favorable_bowler[['batsman', 'bowler', 'balls', 'strike_rate']].to_string(index=False))

        matrix_data.to_csv('../data/processed/team_matchup_sample.csv', index=False)
    else:
        print("No matchup data found for this pairing with sufficient sample size.")