"""
Single, cached entry point for loading the cleaned dataset.
Every script in src/ and the dashboard should import from here instead of
calling pd.read_parquet directly — this avoids loading the 8.3M-row dataset
repeatedly and is the one place format/column fixes need to happen.
"""
import os
import pandas as pd
import functools

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'processed')

MATCHES_PATH = os.path.join(DATA_DIR, 'matches_clean.parquet')
DELIVERIES_PATH = os.path.join(DATA_DIR, 'deliveries_clean.parquet')

VALID_FORMATS = {'IPL', 'Test', 'ODI', 'T20I', 'BBL', 'County Championship', 'The Hundred',
                  'CPL', 'CSA T20 Challenge', 'SA20', 'Syed Mushtaq Ali Trophy', 'Super Smash',
                  "Women's BBL", "Women's CPL", 'WPL'}

PHASE_BINS = [0, 6, 15, 20]
PHASE_LABELS = ['Powerplay', 'Middle', 'Death']


@functools.lru_cache(maxsize=1)
def load_matches():
    """Full matches table, cached — only read from disk once per process."""
    df = pd.read_parquet(MATCHES_PATH)
    return df


@functools.lru_cache(maxsize=1)
def load_deliveries():
    """Full deliveries table, cached — only read from disk once per process."""
    df = pd.read_parquet(DELIVERIES_PATH)
    return df


def load_format(format_name, include_deliveries=True):
    """
    Returns (matches, deliveries) filtered to a single format.
    Raises a clear error for an unrecognised format instead of silently
    returning the full multi-format dataset (the bug that broke the T20
    win-probability model earlier).
    """
    if format_name not in VALID_FORMATS:
        raise ValueError(f"Unknown format '{format_name}'. Valid options: {sorted(VALID_FORMATS)}")

    matches = load_matches()
    matches = matches[matches['format'] == format_name]

    if not include_deliveries:
        return matches, None

    deliveries = load_deliveries()
    deliveries = deliveries[deliveries['match_id'].isin(matches['match_id'])]
    return matches, deliveries


def add_phase_column(deliveries):
    """Adds a standard Powerplay/Middle/Death 'phase' column. Shared logic —
    previously duplicated with slightly different bins across 4+ files."""
    deliveries = deliveries.copy()
    deliveries['phase'] = pd.cut(deliveries['over'], bins=PHASE_BINS, labels=PHASE_LABELS)
    return deliveries


if __name__ == "__main__":
    m = load_matches()
    d = load_deliveries()
    print(f"Full dataset: {len(m):,} matches, {len(d):,} deliveries")

    ipl_m, ipl_d = load_format('IPL')
    print(f"IPL only: {len(ipl_m):,} matches, {len(ipl_d):,} deliveries")

    try:
        load_format('NotARealFormat')
    except ValueError as e:
        print(f"\nCorrectly rejected bad format: {e}")