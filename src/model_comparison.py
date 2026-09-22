import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import joblib
import os

os.makedirs('../reports', exist_ok=True)

# ============================================
# LOAD DATA (from win_probability_model.py output)
# ============================================
state_df = pd.read_csv('../data/processed/match_states.csv')

features = ['inning', 'current_score', 'wickets_fallen', 'balls_bowled',
            'balls_remaining', 'target', 'runs_needed', 'required_run_rate', 'current_run_rate']

X = state_df[features]
y = state_df['won']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression needs scaled features; tree models don't
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = []

# ============================================
# MODEL 1: Logistic Regression (simple baseline)
# ============================================
print("Training Logistic Regression...")
start = time.time()
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)
lr_time = time.time() - start

lr_preds = log_reg.predict(X_test_scaled)
lr_probs = log_reg.predict_proba(X_test_scaled)[:, 1]

results.append({
    'model': 'Logistic Regression',
    'accuracy': accuracy_score(y_test, lr_preds),
    'roc_auc': roc_auc_score(y_test, lr_probs),
    'f1_score': f1_score(y_test, lr_preds),
    'train_time_sec': lr_time
})

# ============================================
# MODEL 2: Random Forest (default, untuned baseline)
# ============================================
print("Training baseline Random Forest...")
start = time.time()
rf_baseline = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_baseline.fit(X_train, y_train)
rf_base_time = time.time() - start

rf_base_preds = rf_baseline.predict(X_test)
rf_base_probs = rf_baseline.predict_proba(X_test)[:, 1]

results.append({
    'model': 'Random Forest (default)',
    'accuracy': accuracy_score(y_test, rf_base_preds),
    'roc_auc': roc_auc_score(y_test, rf_base_probs),
    'f1_score': f1_score(y_test, rf_base_preds),
    'train_time_sec': rf_base_time
})

# ============================================
# MODEL 3: Random Forest — TUNED via GridSearchCV
# ============================================
print("Tuning Random Forest with GridSearchCV (this takes a few minutes)...")
param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5]
}

start = time.time()
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)
rf_tuned_time = time.time() - start

print(f"Best parameters found: {grid_search.best_params_}")

rf_tuned = grid_search.best_estimator_
rf_tuned_preds = rf_tuned.predict(X_test)
rf_tuned_probs = rf_tuned.predict_proba(X_test)[:, 1]

results.append({
    'model': 'Random Forest (tuned)',
    'accuracy': accuracy_score(y_test, rf_tuned_preds),
    'roc_auc': roc_auc_score(y_test, rf_tuned_probs),
    'f1_score': f1_score(y_test, rf_tuned_preds),
    'train_time_sec': rf_tuned_time
})

# ============================================
# MODEL 4: Gradient Boosting
# ============================================
print("Training Gradient Boosting...")
start = time.time()
gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)
gb_time = time.time() - start

gb_preds = gb.predict(X_test)
gb_probs = gb.predict_proba(X_test)[:, 1]

results.append({
    'model': 'Gradient Boosting',
    'accuracy': accuracy_score(y_test, gb_preds),
    'roc_auc': roc_auc_score(y_test, gb_probs),
    'f1_score': f1_score(y_test, gb_preds),
    'train_time_sec': gb_time
})

# ============================================
# COMPARE ALL MODELS
# ============================================
results_df = pd.DataFrame(results).sort_values('roc_auc', ascending=False)
print("\n=== MODEL COMPARISON ===")
print(results_df.to_string(index=False))

results_df.to_csv('../reports/model_comparison.csv', index=False)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].barh(results_df['model'], results_df['roc_auc'], color='steelblue')
axes[0].set_xlabel('ROC-AUC')
axes[0].set_title('Model Comparison — ROC-AUC')
axes[0].invert_yaxis()

axes[1].barh(results_df['model'], results_df['train_time_sec'], color='darkorange')
axes[1].set_xlabel('Training Time (seconds)')
axes[1].set_title('Model Comparison — Training Time')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('../reports/model_comparison_chart.png')
plt.show()

# ============================================
# SAVE THE BEST MODEL (overwrites your original win_prob_model.pkl)
# ============================================
best_model_name = results_df.iloc[0]['model']
print(f"\nBest model: {best_model_name}")

if best_model_name == 'Random Forest (tuned)':
    joblib.dump(rf_tuned, '../data/processed/win_prob_model.pkl')
    print("Saved tuned Random Forest as the new production model.")
elif best_model_name == 'Gradient Boosting':
    joblib.dump(gb, '../data/processed/win_prob_model.pkl')
    print("Saved Gradient Boosting as the new production model.")
else:
    print("Best model wasn't a tree model requiring no scaling — keeping existing model.pkl unless you want to swap it manually.")