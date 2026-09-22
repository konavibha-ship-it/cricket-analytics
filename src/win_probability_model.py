import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# ---- LOAD DATA ----
matches = pd.read_csv('../data/processed/matches_clean.csv')
deliveries = pd.read_csv('../data/processed/deliveries_clean.csv')

# Only use matches with a clear winner
matches = matches[matches['winner'] != 'No Result'].dropna(subset=['winner'])
valid_match_ids = matches['match_id'].unique()
deliveries = deliveries[deliveries['match_id'].isin(valid_match_ids)]

# ---- BUILD BALL-BY-BALL MATCH STATE ----
# For each ball, we need: current score, wickets fallen, balls bowled, 
# innings (1st or 2nd), target (if chasing), and whether the batting team eventually won

rows = []

for match_id, group in deliveries.groupby('match_id'):
    match_info = matches[matches['match_id'] == match_id]
    if match_info.empty:
        continue
    winner = match_info['winner'].values[0]

    for inning_num in group['inning'].unique():
        inning_data = group[group['inning'] == inning_num].sort_values(['over', 'ball'])
        if inning_data.empty:
            continue

        batting_team = inning_data['batting_team'].iloc[0]
        cumulative_runs = 0
        cumulative_wickets = 0
        total_balls = 0

        # First innings final score becomes the target for 2nd innings
        first_innings_total = None
        if inning_num == 2:
            first_inn = group[group['inning'] == 1]
            first_innings_total = first_inn['total_runs'].sum()

        for _, ball in inning_data.iterrows():
            cumulative_runs += ball['total_runs']
            if pd.notnull(ball['dismissal_kind']):
                cumulative_wickets += 1
            total_balls += 1

            overs_completed = total_balls / 6
            balls_remaining = max(120 - total_balls, 0)  # assuming T20 (20 overs = 120 balls)

            target = first_innings_total + 1 if first_innings_total is not None else None
            runs_needed = (target - cumulative_runs) if target is not None else None
            required_run_rate = (runs_needed / (balls_remaining / 6)) if (target is not None and balls_remaining > 0) else None

            rows.append({
                'match_id': match_id,
                'inning': inning_num,
                'batting_team': batting_team,
                'current_score': cumulative_runs,
                'wickets_fallen': cumulative_wickets,
                'balls_bowled': total_balls,
                'balls_remaining': balls_remaining,
                'target': target,
                'runs_needed': runs_needed,
                'required_run_rate': required_run_rate,
                'current_run_rate': cumulative_runs / overs_completed if overs_completed > 0 else 0,
                'won': 1 if batting_team == winner else 0
            })

state_df = pd.DataFrame(rows)
state_df = state_df.fillna(0)  # 1st innings has no target/required_run_rate

print("Ball-by-ball states built:", state_df.shape)

# ---- TRAIN MODEL ----
features = ['inning', 'current_score', 'wickets_fallen', 'balls_bowled', 
            'balls_remaining', 'target', 'runs_needed', 'required_run_rate', 'current_run_rate']

X = state_df[features]
y = state_df['won']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, preds))
print("ROC-AUC:", roc_auc_score(y_test, probs))
print(classification_report(y_test, preds))

# ---- SAVE MODEL for use in dashboard ----
os.makedirs('../data/processed', exist_ok=True)
joblib.dump(model, '../data/processed/win_prob_model.pkl')
state_df.to_csv('../data/processed/match_states.csv', index=False)

print("Model saved to data/processed/win_prob_model.pkl")