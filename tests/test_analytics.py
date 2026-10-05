import sys
import os
import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from utils import strike_rate, economy_rate, batting_average, is_bowler_wicket


# ============================================
# UNIT TESTS — pure functions, hand-calculable, no data loading needed
# ============================================
def test_strike_rate_basic():
    assert strike_rate(50, 30) == pytest.approx(166.67, abs=0.01)

def test_strike_rate_zero_balls():
    assert strike_rate(0, 0) == 0.0

def test_economy_rate_basic():
    assert economy_rate(24, 18) == pytest.approx(8.0)

def test_economy_rate_zero_balls():
    assert economy_rate(0, 0) == 0.0

def test_batting_average_normal():
    # 100 runs across 4 dismissals = average 25
    assert batting_average(100, 4) == 25.0

def test_batting_average_never_out():
    # Convention: an undismissed batter's "average" is their total runs
    assert batting_average(45, 0) == 45

def test_is_bowler_wicket_excludes_run_out():
    dismissals = pd.Series(['caught', 'run out', 'bowled', None, 'stumped'])
    result = is_bowler_wicket(dismissals)
    expected = pd.Series([True, False, True, False, True])
    pd.testing.assert_series_equal(result, expected)


# ============================================
# INTEGRATION TESTS — using the real dataset via data_loader
# ============================================
from data_loader import load_format, load_matches

def test_load_format_ipl_only():
    matches, deliveries = load_format('IPL')
    assert (matches['format'] == 'IPL').all()
    assert (deliveries['format'] == 'IPL').all()

def test_load_format_rejects_invalid():
    with pytest.raises(ValueError):
        load_format('NotARealFormat')

def test_load_format_match_delivery_consistency():
    """Every match_id in the filtered deliveries must exist in the filtered matches."""
    matches, deliveries = load_format('ODI')
    match_ids_in_matches = set(matches['match_id'])
    match_ids_in_deliveries = set(deliveries['match_id'])
    assert match_ids_in_deliveries.issubset(match_ids_in_matches)


# ============================================
# REGRESSION TEST — locks in the milestone-tracker bug fix
# ============================================
def test_milestone_tracker_uses_checkpoint_not_final_total():
    """
    Guards against the exact bug found earlier: the first version of
    milestone_pace_tracker compared a batter's FINAL total against the
    milestone, rather than their total AT THE CHECKPOINT — which made
    reaching a higher milestone than the checkpoint mathematically
    impossible. A batter at 35/25 should have a realistic (non-zero,
    non-trivial) chance of reaching 50, not exactly 0%.
    """
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
    from par_score_engine import milestone_pace_tracker

    result = milestone_pace_tracker(current_runs=35, current_balls=25, format_filter='IPL', milestone=50)

    assert result is not None
    assert result['pct_that_reached_milestone'] > 10, (
        "If this is 0%, the checkpoint-vs-final-total bug has regressed."
    )