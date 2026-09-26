import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, accuracy_score, roc_auc_score
from sklearn.frozen import FrozenEstimator
import joblib
import os

os.makedirs('../reports', exist_ok=True)

# ---- LOAD THE EXISTING TRAINED MODEL AND DATA ----
original_model = joblib.load('../data/processed/win_prob_model.pkl')
state_df = pd.read_csv('../data/processed/match_states.csv')

features = ['inning', 'current_score', 'wickets_fallen', 'balls_bowled',
            'balls_remaining', 'target', 'runs_needed', 'required_run_rate', 'current_run_rate']

X = state_df[features]
y = state_df['won']

# Split fresh — calibration needs its own held-out set separate from what trained the original model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Applying isotonic calibration...")

# 'prefit' tells it the base model is already trained — it only fits the calibration mapping
calibrated_model = CalibratedClassifierCV(FrozenEstimator(original_model), method='sigmoid')
calibrated_model.fit(X_train, y_train)

# ---- COMPARE BEFORE vs AFTER ----
raw_probs = original_model.predict_proba(X_test)[:, 1]
calibrated_probs = calibrated_model.predict_proba(X_test)[:, 1]

raw_brier = brier_score_loss(y_test, raw_probs)
calibrated_brier = brier_score_loss(y_test, calibrated_probs)

raw_auc = roc_auc_score(y_test, raw_probs)
calibrated_auc = roc_auc_score(y_test, calibrated_probs)

print(f"\nBEFORE calibration — Brier: {raw_brier:.4f}, ROC-AUC: {raw_auc:.4f}")
print(f"AFTER calibration  — Brier: {calibrated_brier:.4f}, ROC-AUC: {calibrated_auc:.4f}")
print(f"\nBrier score improvement: {(raw_brier - calibrated_brier):.4f} "
      f"({((raw_brier - calibrated_brier) / raw_brier * 100):.1f}% better)")

# ---- CALIBRATION CURVE COMPARISON CHART ----
frac_pos_raw, mean_pred_raw = calibration_curve(y_test, raw_probs, n_bins=10)
frac_pos_cal, mean_pred_cal = calibration_curve(y_test, calibrated_probs, n_bins=10)

plt.figure(figsize=(8, 8))
plt.plot(mean_pred_raw, frac_pos_raw, marker='o', linewidth=2, label='Before calibration', color='#888888')
plt.plot(mean_pred_cal, frac_pos_cal, marker='o', linewidth=2, label='After calibration', color='#ff4b4b')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
plt.xlabel('Mean Predicted Win Probability')
plt.ylabel('Actual Fraction of Wins')
plt.title('Calibration Improvement — T20 Model (Before vs After Platt Scaling)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../reports/calibration_before_after.png')
plt.close()

print("\nSaved comparison chart to reports/calibration_before_after.png")

# ---- SAVE THE CALIBRATED MODEL (replaces the one the dashboard uses) ----
joblib.dump(calibrated_model, '../data/processed/win_prob_model.pkl')

size_mb = os.path.getsize('../data/processed/win_prob_model.pkl') / (1024*1024)
print(f"\nCalibrated model saved (replaces original). File size: {size_mb:.1f} MB")