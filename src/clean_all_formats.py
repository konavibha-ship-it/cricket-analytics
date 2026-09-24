import pandas as pd
import numpy as np
import os

matches = pd.read_csv('../data/raw/matches_all_formats.csv')
deliveries = pd.read_csv('../data/raw/deliveries_all_formats.csv')

print("Before cleaning:", matches.shape, deliveries.shape)

# ============================================
# FIX INCONSISTENT TEAM NAMES (franchise + some country renames)
# ============================================
team_map = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Kings XI Punjab': 'Punjab Kings',
    'Deccan Chargers': 'Sunrisers Hyderabad',
}

for col in ['team1', 'team2', 'winner', 'toss_winner']:
    matches[col] = matches[col].replace(team_map)

deliveries['batting_team'] = deliveries['batting_team'].replace(team_map)

# ============================================
# HANDLE MISSING VALUES
# ============================================
matches['winner'] = matches['winner'].fillna('No Result')
matches['city'] = matches['city'].fillna('Unknown')

# ============================================
# CONVERT DATE
# ============================================
matches['date'] = pd.to_datetime(matches['date'], errors='coerce')
matches['year'] = matches['date'].dt.year

# ============================================
# DROP DUPLICATES
# ============================================
matches = matches.drop_duplicates()
deliveries = deliveries.drop_duplicates()

# ============================================
# DROP ROWS WITH NO VALID MATCH DATE (a few malformed entries are common at this scale)
# ============================================
before = len(matches)
matches = matches.dropna(subset=['date'])
print(f"Dropped {before - len(matches)} matches with invalid/missing dates")

# Keep only deliveries belonging to matches that survived cleaning
valid_ids = matches['match_id'].unique()
deliveries = deliveries[deliveries['match_id'].isin(valid_ids)]

print("\nAfter cleaning:", matches.shape, deliveries.shape)
print("\nDate range:", matches['date'].min(), "to", matches['date'].max())
print("\nMatches per format:")
print(matches['format'].value_counts())

# ============================================
# SAVE
# ============================================
os.makedirs('../data/processed', exist_ok=True)
matches.to_csv('../data/processed/matches_clean.csv', index=False)
deliveries.to_csv('../data/processed/deliveries_clean.csv', index=False)

print("\nSaved cleaned combined dataset to data/processed/")
print("(This OVERWRITES your previous IPL-only matches_clean.csv / deliveries_clean.csv)")