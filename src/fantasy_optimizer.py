import pandas as pd
import numpy as np
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, PULP_CBC_CMD

deliveries = pd.read_csv('../data/processed/deliveries_clean.csv')

# ============================================
# BUILD A SIMPLE FANTASY POINTS SYSTEM
# (based loosely on common fantasy scoring rules)
# ============================================
batting_pts = deliveries.groupby('batsman').agg(
    runs=('batsman_runs', 'sum'),
    balls=('batsman_runs', 'count'),
    fours=('batsman_runs', lambda x: (x == 4).sum()),
    sixes=('batsman_runs', lambda x: (x == 6).sum()),
    innings=('match_id', 'nunique')
).reset_index().rename(columns={'batsman': 'player'})

batting_pts['batting_points'] = (
    batting_pts['runs'] * 1 +
    batting_pts['fours'] * 1 +
    batting_pts['sixes'] * 2
)

bowling_pts = deliveries[
    deliveries['dismissal_kind'].notnull() &
    ~deliveries['dismissal_kind'].isin(['run out', 'retired hurt', 'obstructing the field'])
].groupby('bowler').size().reset_index(name='wickets').rename(columns={'bowler': 'player'})

bowling_pts['bowling_points'] = bowling_pts['wickets'] * 25

# ============================================
# COMBINE INTO ONE PLAYER POOL
# ============================================
all_players = pd.merge(batting_pts[['player', 'runs', 'innings', 'batting_points']],
                         bowling_pts[['player', 'wickets', 'bowling_points']],
                         on='player', how='outer').fillna(0)

all_players['total_points'] = all_players['batting_points'] + all_players['bowling_points']

# Filter to players with meaningful involvement (avoids single-match outliers)
all_players = all_players[all_players['innings'] >= 10]

# ============================================
# ASSIGN A "CREDIT COST" per player (simulate fantasy game constraints)
# Higher points = higher cost, normalized to a realistic 6-11 credit range
# ============================================
min_pts, max_pts = all_players['total_points'].min(), all_players['total_points'].max()
all_players['credits'] = 6 + (all_players['total_points'] - min_pts) / (max_pts - min_pts) * 5
all_players['credits'] = all_players['credits'].round(1)

# ============================================
# CLASSIFY ROLE (simple heuristic based on points source)
# ============================================
def classify_role(row):
    if row['bowling_points'] > row['batting_points'] * 1.5:
        return 'Bowler'
    elif row['bowling_points'] > 0 and row['batting_points'] > 0:
        return 'All-rounder'
    else:
        return 'Batter'

all_players['role'] = all_players.apply(classify_role, axis=1)

print(f"Player pool size: {len(all_players)}")
print(all_players['role'].value_counts())

# ============================================
# OPTIMIZATION: pick best 11 players under constraints
# ============================================
TOTAL_CREDITS = 100
SQUAD_SIZE = 11
MIN_BATTERS = 3
MIN_BOWLERS = 3
MIN_ALLROUNDERS = 1

prob = LpProblem("Fantasy_XI_Optimizer", LpMaximize)

player_vars = {
    row['player']: LpVariable(f"player_{i}", cat=LpBinary)
    for i, row in all_players.reset_index().iterrows()
}

players_list = all_players.reset_index(drop=True)

# Objective: maximize total fantasy points
prob += lpSum(
    players_list.loc[i, 'total_points'] * player_vars[players_list.loc[i, 'player']]
    for i in players_list.index
)

# Constraint: exactly 11 players
prob += lpSum(player_vars.values()) == SQUAD_SIZE

# Constraint: total credits <= budget
prob += lpSum(
    players_list.loc[i, 'credits'] * player_vars[players_list.loc[i, 'player']]
    for i in players_list.index
) <= TOTAL_CREDITS

# Constraint: minimum batters
prob += lpSum(
    player_vars[players_list.loc[i, 'player']]
    for i in players_list.index if players_list.loc[i, 'role'] == 'Batter'
) >= MIN_BATTERS

# Constraint: minimum bowlers
prob += lpSum(
    player_vars[players_list.loc[i, 'player']]
    for i in players_list.index if players_list.loc[i, 'role'] == 'Bowler'
) >= MIN_BOWLERS

# Constraint: minimum all-rounders
prob += lpSum(
    player_vars[players_list.loc[i, 'player']]
    for i in players_list.index if players_list.loc[i, 'role'] == 'All-rounder'
) >= MIN_ALLROUNDERS

# ============================================
# SOLVE
# ============================================
prob.solve(PULP_CBC_CMD(msg=0))

selected = [p for p in player_vars if player_vars[p].value() == 1]
result = all_players[all_players['player'].isin(selected)].sort_values('total_points', ascending=False)

total_credits_used = result['credits'].sum()
total_points = result['total_points'].sum()

print("\n=== OPTIMAL FANTASY XI ===")
print(result[['player', 'role', 'total_points', 'credits']].to_string(index=False))
print(f"\nTotal Credits Used: {total_credits_used:.1f} / {TOTAL_CREDITS}")
print(f"Total Projected Points: {total_points:.0f}")

result.to_csv('../data/processed/fantasy_xi.csv', index=False)
print("\nSaved to data/processed/fantasy_xi.csv")