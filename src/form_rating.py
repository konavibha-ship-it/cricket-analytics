import pandas as pd
import numpy as np

deliveries = pd.read_parquet('../data/processed/deliveries_clean.parquet')
matches = pd.read_parquet('../data/processed/matches_clean.parquet')

# Attach match date to each delivery so we can order innings chronologically
match_dates = matches[['match_id', 'date']]
deliveries = deliveries.merge(match_dates, on='match_id', how='left')
deliveries['date'] = pd.to_datetime(deliveries['date'])


def calculate_form_rating(player_name, role='batsman', recent_n=10, min_career_innings=10):
    """
    Builds a recency-weighted rating: recent performances count more than older ones.
    Returns career average, recent-N average, and a blended 'form rating'.
    """
    if role == 'batsman':
        player_data = deliveries[deliveries['batsman'] == player_name]
        metric_col = 'batsman_runs'
    else:
        player_data = deliveries[deliveries['bowler'] == player_name]
        metric_col = 'total_runs'

    if player_data.empty:
        return None

    # Group by match to get per-innings totals, ordered by date
    per_innings = player_data.groupby(['match_id', 'date']).agg(
        runs=(metric_col, 'sum'),
        balls=(metric_col, 'count'),
        wickets=('dismissal_kind', lambda x: x.notnull().sum()) if role == 'bowler' else ('dismissal_kind', 'size')
    ).reset_index().sort_values('date')

    if len(per_innings) < min_career_innings:
        return None

    if role == 'batsman':
        per_innings['metric'] = (per_innings['runs'] / per_innings['balls'] * 100)  # strike rate per innings
    else:
        per_innings['metric'] = (per_innings['runs'] / per_innings['balls'] * 6)  # economy per innings

    career_avg = per_innings['metric'].mean()
    recent_form = per_innings['metric'].tail(recent_n)
    recent_avg = recent_form.mean()

    # Exponential recency weighting: most recent innings weighted highest
    weights = np.exp(np.linspace(-1, 0, len(recent_form)))
    weighted_recent = np.average(recent_form, weights=weights)

    if role == 'batsman':
        # Higher recent SR relative to career = trending up
        form_trend = "🔥 In Form" if weighted_recent > career_avg * 1.1 else \
                     "❄️ Out of Form" if weighted_recent < career_avg * 0.85 else "➡️ Steady"
    else:
        # Lower recent economy relative to career = trending up (better)
        form_trend = "🔥 In Form" if weighted_recent < career_avg * 0.9 else \
                     "❄️ Out of Form" if weighted_recent > career_avg * 1.15 else "➡️ Steady"

    return {
        'player': player_name,
        'role': role,
        'career_avg_metric': round(career_avg, 1),
        'recent_avg_metric': round(recent_avg, 1),
        'weighted_recent_metric': round(weighted_recent, 1),
        'form_trend': form_trend,
        'innings_considered': len(recent_form),
        'total_career_innings': len(per_innings)
    }


if __name__ == "__main__":
    result = calculate_form_rating("V Kohli", role='batsman')
    print(result)

    result2 = calculate_form_rating("JJ Bumrah", role='bowler')
    print(result2)