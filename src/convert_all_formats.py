import json
import os
import pandas as pd

# ============================================
# CONFIG — each format's folder and a short code
# ============================================
FORMAT_FOLDERS = {
    'IPL': '../data/raw/json',
    'Test': '../data/raw/json_tests',
    'ODI': '../data/raw/json_odis',
    'T20I': '../data/raw/json_t20is',
    'BBL': '../data/raw/json_bbl',
    'County Championship': '../data/raw/json_county',
    'The Hundred': '../data/raw/json_hundred',
    'CPL': '../data/raw/json_cpl',
    'CSA T20 Challenge': '../data/raw/json_csa',
    'SA20': '../data/raw/json_sa20',
    'Syed Mushtaq Ali Trophy': '../data/raw/json_smat',
    'Super Smash': '../data/raw/json_supersmash',
    "Women's BBL": '../data/raw/json_wbbl',
    "Women's CPL": '../data/raw/json_wcpl',
    'WPL': '../data/raw/json_wpl'
}

match_rows = []
delivery_rows = []
match_id_counter = 0

for format_name, folder_path in FORMAT_FOLDERS.items():
    if not os.path.exists(folder_path):
        print(f"⚠️  Skipping {format_name} — folder not found: {folder_path}")
        continue

    files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    print(f"Processing {format_name}: {len(files)} matches found...")

    for filename in files:
        match_id_counter += 1
        filepath = os.path.join(folder_path, filename)

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"  Skipped {filename} (couldn't parse): {e}")
            continue

        info = data.get('info', {})

        # ---- MATCH LEVEL INFO ----
        teams = info.get('teams', [None, None])
        outcome = info.get('outcome', {})
        winner = outcome.get('winner', 'No Result')
        toss = info.get('toss', {})

        # Determine overs limit for this match (varies by format)
        overs_info = info.get('overs', None)
        if format_name == 'Test':
            overs_limit = None  # no fixed limit
        elif overs_info:
            overs_limit = overs_info
        elif format_name == 'ODI':
            overs_limit = 50
        elif format_name in ['T20I', 'IPL']:
            overs_limit = 20
        else:
            overs_limit = None

        match_rows.append({
            'match_id': match_id_counter,
            'format': format_name,
            'date': info.get('dates', [None])[0],
            'city': info.get('city', None),
            'venue': info.get('venue', None),
            'team1': teams[0] if len(teams) > 0 else None,
            'team2': teams[1] if len(teams) > 1 else None,
            'toss_winner': toss.get('winner'),
            'toss_decision': toss.get('decision'),
            'winner': winner,
            'season': info.get('season', None),
            'overs_limit': overs_limit
        })

        # ---- BALL BY BALL INFO ----
        innings_list = data.get('innings', [])

        for inning_idx, inning in enumerate(innings_list):
            batting_team = inning.get('team')
            overs = inning.get('overs', [])

            for over_data in overs:
                over_num = over_data.get('over')
                balls = over_data.get('deliveries', [])

                for i, ball in enumerate(balls):
                    batter = ball.get('batter', ball.get('batsman'))
                    bowler = ball.get('bowler')
                    runs = ball.get('runs', {})
                    wickets = ball.get('wickets', [])

                    dismissal_kind = None
                    player_out = None
                    if wickets:
                        dismissal_kind = wickets[0].get('kind')
                        player_out = wickets[0].get('player_out')

                    delivery_rows.append({
                        'match_id': match_id_counter,
                        'format': format_name,
                        'inning': inning_idx + 1,
                        'batting_team': batting_team,
                        'over': over_num,
                        'ball': i + 1,
                        'batsman': batter,
                        'bowler': bowler,
                        'batsman_runs': runs.get('batter', 0),
                        'extra_runs': runs.get('extras', 0),
                        'total_runs': runs.get('total', 0),
                        'dismissal_kind': dismissal_kind,
                        'player_dismissed': player_out
                    })

    print(f"  Done with {format_name}.\n")

# ============================================
# SAVE COMBINED DATASETS
# ============================================
matches_df = pd.DataFrame(match_rows)
deliveries_df = pd.DataFrame(delivery_rows)

os.makedirs('../data/raw', exist_ok=True)
matches_df.to_csv('../data/raw/matches_all_formats.csv', index=False)
deliveries_df.to_csv('../data/raw/deliveries_all_formats.csv', index=False)

print("=" * 60)
print("COMBINED DATASET SUMMARY")
print("=" * 60)
print(f"Total matches: {matches_df.shape[0]}")
print(f"Total deliveries: {deliveries_df.shape[0]}")
print("\nMatches per format:")
print(matches_df['format'].value_counts())
print("\nSaved to:")
print("  data/raw/matches_all_formats.csv")
print("  data/raw/deliveries_all_formats.csv")