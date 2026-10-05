import pandas as pd
import numpy as np

from data_loader import load_matches, load_deliveries

matches = load_matches()
deliveries = load_deliveries()


def get_par_score(venue, format_filter='IPL', overs_completed=None):
    """
    Returns the historical par score at this venue — either the full first-innings
    par, or the par AT a specific point (overs_completed) if given, using historical
    first-innings trajectories.
    """
    venue_matches = matches[(matches['venue'] == venue) & (matches['format'] == format_filter)]
    match_ids = venue_matches['match_id'].unique()

    first_innings = deliveries[
        (deliveries['match_id'].isin(match_ids)) & (deliveries['inning'] == 1)
    ]

    if first_innings.empty:
        return None

    if overs_completed is None:
        # Full innings par score
        totals = first_innings.groupby('match_id')['total_runs'].sum()
        return {
            'venue': venue,
            'sample_size': len(totals),
            'par_score': round(totals.mean(), 1),
            'par_score_median': round(totals.median(), 1)
        }
    else:
        # Score at a specific over-mark, across historical matches
        ball_cutoff = overs_completed * 6
        cumulative_scores = []
        for match_id, group in first_innings.groupby('match_id'):
            group_sorted = group.sort_values(['over', 'ball'])
            balls_in_range = group_sorted.head(ball_cutoff)
            if len(balls_in_range) >= ball_cutoff * 0.9:  # allow slight tolerance for extras
                cumulative_scores.append(balls_in_range['total_runs'].sum())

        if not cumulative_scores:
            return None

        return {
            'venue': venue,
            'overs_completed': overs_completed,
            'sample_size': len(cumulative_scores),
            'par_score_at_this_stage': round(np.mean(cumulative_scores), 1)
        }


def milestone_pace_tracker(current_runs, current_balls, format_filter='IPL', milestone=50, tolerance=3):
    """
    Given a batter's current innings state, finds historical innings that had
    reached a similar (runs, balls) point DURING the innings, then checks what
    percentage of those innings went on to reach the milestone by the end.
    """
    format_deliveries = deliveries[deliveries['format'] == format_filter].copy()
    format_deliveries = format_deliveries.sort_values(['match_id', 'inning', 'batsman', 'over', 'ball'])

    matches_at_checkpoint = []
    final_totals = {}

    for (match_id, inning, batsman), group in format_deliveries.groupby(['match_id', 'inning', 'batsman']):
        group = group.sort_values(['over', 'ball'])
        cumulative_runs = group['batsman_runs'].cumsum()
        cumulative_balls = np.arange(1, len(group) + 1)

        final_total = cumulative_runs.iloc[-1] if len(cumulative_runs) > 0 else 0
        key = (match_id, inning, batsman)
        final_totals[key] = final_total

        # Find the ball index where this innings was near our checkpoint
        mask = (
            (cumulative_balls >= current_balls) & (cumulative_balls <= current_balls + tolerance) &
            (cumulative_runs >= current_runs - tolerance) & (cumulative_runs <= current_runs + tolerance)
        )

        if mask.any():
            matches_at_checkpoint.append(key)

    if not matches_at_checkpoint:
        return None

    reached_milestone = sum(1 for key in matches_at_checkpoint if final_totals[key] >= milestone)
    pace_pct = (reached_milestone / len(matches_at_checkpoint)) * 100

    return {
        'current_state': f"{current_runs} off {current_balls} balls",
        'milestone': milestone,
        'comparable_historical_innings': len(matches_at_checkpoint),
        'pct_that_reached_milestone': round(pace_pct, 1)
    }


if __name__ == "__main__":
    sample_venue = matches[matches['format'] == 'IPL']['venue'].value_counts().index[0]

    print("=== Full Par Score ===")
    print(get_par_score(sample_venue, format_filter='IPL'))

    print("\n=== Par Score at 10 overs ===")
    print(get_par_score(sample_venue, format_filter='IPL', overs_completed=10))

    print("\n=== Milestone Pace: batter at 35 off 25 balls, chasing a 50 ===")
    print(milestone_pace_tracker(current_runs=35, current_balls=25, format_filter='IPL', milestone=50))

    print("\n=== Milestone Pace: batter at 70 off 45 balls, chasing a 100 ===")
    print(milestone_pace_tracker(current_runs=70, current_balls=45, format_filter='IPL', milestone=100))