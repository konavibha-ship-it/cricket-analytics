"""Retrain the IPL win-probability model: honest evaluation + small file.

Run from the project root:  python retrain_small_model.py
Saves to data/processed/win_prob_model_small.pkl (the old model is untouched).
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

DATA = Path("data/processed")
OUT = DATA / "win_prob_model_small.pkl"

FEATURES = ["inning", "current_score", "wickets_fallen", "balls_bowled",
            "balls_remaining", "target", "runs_needed",
            "required_run_rate", "current_run_rate"]

df = pd.read_parquet(DATA / "match_states.parquet")

missing = [c for c in FEATURES + ["won", "match_id"] if c not in df.columns]
if missing:
    raise SystemExit(f"match_states.parquet is missing columns: {missing}\n"
                     f"Columns found: {list(df.columns)}")

print("Rows:", len(df), "| Matches:", df["match_id"].nunique())

X, y, groups = df[FEATURES], df["won"], df["match_id"]

# Split by MATCH so no match appears in both train and test
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=50,   # keeps trees small
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]
print("Accuracy (match-level split):", round(accuracy_score(y_test, probs > 0.5), 4))
print("ROC-AUC  (match-level split):", round(roc_auc_score(y_test, probs), 4))

joblib.dump(model, OUT, compress=3)
print(f"Saved {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")
