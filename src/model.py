import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

matches = pd.read_csv('../data/processed/matches_clean.csv')
matches = matches.dropna(subset=['winner'])
matches = matches[matches['winner'] != 'No Result']

le_team = LabelEncoder()
le_toss = LabelEncoder()
le_venue = LabelEncoder()

# Fit team encoder on ALL team names that appear anywhere (team1, team2, toss_winner, winner)
all_teams = pd.concat([matches['team1'], matches['team2'], matches['toss_winner'], matches['winner']]).unique()
le_team.fit(all_teams)

matches['team1_enc'] = le_team.transform(matches['team1'])
matches['team2_enc'] = le_team.transform(matches['team2'])
matches['toss_winner_enc'] = le_team.transform(matches['toss_winner'])
matches['toss_decision_enc'] = le_toss.fit_transform(matches['toss_decision'])
matches['venue_enc'] = le_venue.fit_transform(matches['venue'].astype(str))
matches['winner_enc'] = le_team.transform(matches['winner'])

features = ['team1_enc', 'team2_enc', 'toss_winner_enc', 'toss_decision_enc', 'venue_enc']
X = matches[features]
y = matches['winner_enc']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds, zero_division=0))