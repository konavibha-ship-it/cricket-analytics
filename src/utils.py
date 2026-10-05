"""
Shared calculation helpers used across multiple analysis scripts.
Centralizing these avoids subtle inconsistencies — e.g. different scripts
using slightly different dismissal-type exclusion lists for "real" wickets.
"""
import pandas as pd

# Dismissal types that count as a bowler's wicket (excludes run outs, which
# aren't credited to the bowler)
BOWLER_WICKET_TYPES = {
    'caught', 'bowled', 'lbw', 'stumped', 'caught and bowled',
    'hit wicket', 'obstructing the field', 'hit the ball twice', 'handled the ball'
}


def is_bowler_wicket(dismissal_series):
    """Boolean mask: True where the dismissal counts as the bowler's wicket."""
    return dismissal_series.isin(BOWLER_WICKET_TYPES)


def strike_rate(runs, balls):
    """Standard batting strike rate, safe against division by zero."""
    return (runs / balls * 100) if balls > 0 else 0.0


def economy_rate(runs, balls):
    """Standard bowling economy rate, safe against division by zero."""
    return (runs / balls * 6) if balls > 0 else 0.0


def batting_average(runs, dismissals):
    """Standard batting average. An undismissed player's average is
    conventionally their total runs (not infinity, not zero)."""
    return (runs / dismissals) if dismissals > 0 else runs


def min_sample_filter(df, count_col, min_count):
    """Common pattern: drop rows below a minimum sample size before ranking/comparing."""
    return df[df[count_col] >= min_count]