import json
import os
import pandas as pd

json_folder = '../data/raw/json'
match_rows = []
delivery_rows = []

match_id = 0

for filename in os.listdir(json_folder):
    if not filename.endswith('.json'):
        continue

    match_id += 1
    filepath = os.path.join(json_folder, filename)

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    info = data.get('info', {})

    # ---- MATCH LEVEL INFO ----
    teams = info.get('teams', [None, None])
    outcome = info.get('outcome', {})
    winner = outcome.get('winner', 'No Result')

    toss = info.get('toss', {})

    match_rows.append({
        'match_id': match_id,
        'date': info.get('dates', [None])[0],
        'city': info.get('city', None),
        'venue': info.get('venue', None),
        'team1': teams[0] if len(teams) > 0 else None,
        'team2': teams[1] if len(teams) > 1 else None,
        'toss_winner': toss.get('winner'),
        'toss_decision': toss.get('decision'),
        'winner': winner,
        'season': info.get('season', None)
    })

    # ---- BALL BY BALL INFO ----
    innings_list = data.get('innings', [])

    for inning in innings_list:
        batting_team = inning.get('team')
        overs = inning.get('overs', [])

        for over_data in overs:
            over_num = over_data.get('over')
            deliveries = over_data.get('deliveries', [])

            for i, ball in enumerate(deliveries):
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
                    'match_id': match_id,
                    'inning': innings_list.index(inning) + 1,
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

    if match_id % 100 == 0:
        print(f"Processed {match_id} matches...")

matches_df = pd.DataFrame(match_rows)
deliveries_df = pd.DataFrame(delivery_rows)

matches_df.to_csv('../data/raw/matches.csv', index=False)
deliveries_df.to_csv('../data/raw/deliveries.csv', index=False)

print("Done!")
print("Matches:", matches_df.shape)
print("Deliveries:", deliveries_df.shape)