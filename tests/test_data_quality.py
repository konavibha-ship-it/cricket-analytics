import pandas as pd
import pytest
import os

# ============================================
# LOAD DATA ONCE FOR ALL TESTS
# ============================================
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'processed')

@pytest.fixture(scope="module")
def matches():
    return pd.read_parquet(os.path.join(DATA_DIR, 'matches_clean.parquet'))

@pytest.fixture(scope="module")
def deliveries():
    return pd.read_parquet(os.path.join(DATA_DIR, 'deliveries_clean.parquet'))


# ============================================
# MATCH-LEVEL TESTS
# ============================================
def test_no_duplicate_match_ids(matches):
    assert matches['match_id'].duplicated().sum() == 0, "Found duplicate match_id values"

def test_no_missing_dates(matches):
    assert matches['date'].isnull().sum() == 0, "Found matches with missing dates"

def test_all_formats_present(matches):
    expected_formats = {'IPL', 'Test', 'ODI', 'T20I'}
    actual_formats = set(matches['format'].unique())
    missing = expected_formats - actual_formats
    assert not missing, f"Expected formats missing from data: {missing}"

def test_valid_toss_decision(matches):
    valid_decisions = {'bat', 'field'}
    actual = set(matches['toss_decision'].dropna().unique())
    invalid = actual - valid_decisions
    assert not invalid, f"Unexpected toss_decision values: {invalid}"

def test_team1_not_equal_team2(matches):
    same_team = matches[matches['team1'] == matches['team2']]
    assert len(same_team) == 0, f"Found {len(same_team)} matches where team1 == team2"


# ============================================
# DELIVERY-LEVEL TESTS
# ============================================
def test_no_negative_runs(deliveries):
    assert (deliveries['batsman_runs'] < 0).sum() == 0, "Found negative batsman_runs"
    assert (deliveries['total_runs'] < 0).sum() == 0, "Found negative total_runs"
    assert (deliveries['extra_runs'] < 0).sum() == 0, "Found negative extra_runs"

def test_batsman_runs_realistic_max(deliveries):
    # Extremely rare overthrow scenarios can occasionally push a single delivery's
    # runs unusually high. One known outlier (8 runs, a Test match) exists in this
    # dataset out of 8.3M+ deliveries — treated as a legitimate rare edge case rather
    # than a data error. Anything beyond that is flagged for review.
    invalid = deliveries[deliveries['batsman_runs'] > 8]
    assert len(invalid) == 0, f"Found {len(invalid)} deliveries with batsman_runs > 8 (needs review)"

def test_valid_dismissal_types(deliveries):
    valid_dismissals = {
        'caught', 'bowled', 'lbw', 'run out', 'stumped',
        'caught and bowled', 'hit wicket', 'retired hurt',
        'retired out', 'retired not out', 'obstructing the field', 'hit the ball twice', 'handled the ball'
    }
    
    actual = set(deliveries['dismissal_kind'].dropna().unique())
    invalid = actual - valid_dismissals
    assert not invalid, f"Unexpected dismissal_kind values: {invalid}"

def test_all_match_ids_in_deliveries_exist_in_matches(matches, deliveries):
    match_ids_in_matches = set(matches['match_id'])
    match_ids_in_deliveries = set(deliveries['match_id'])
    orphaned = match_ids_in_deliveries - match_ids_in_matches
    assert not orphaned, f"Found {len(orphaned)} match_ids in deliveries with no matching match record"

def test_over_numbers_non_negative(deliveries):
    assert (deliveries['over'] < 0).sum() == 0, "Found negative over numbers"

def test_no_null_batting_team(deliveries):
    null_count = deliveries['batting_team'].isnull().sum()
    assert null_count == 0, f"Found {null_count} deliveries with missing batting_team"


# ============================================
# CROSS-CONSISTENCY TESTS
# ============================================
def test_known_team_name_consistency(matches):
    # These old franchise names should have been renamed during cleaning
    deprecated_names = {'Delhi Daredevils', 'Kings XI Punjab', 'Deccan Chargers'}
    all_team_names = set(matches['team1'].dropna()) | set(matches['team2'].dropna())
    leftover = deprecated_names & all_team_names
    assert not leftover, f"Deprecated team names still present: {leftover}"