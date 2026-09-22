import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, classification_report
)
import joblib
import os

os.makedirs('../reports', exist_ok=True)

# ============================================
# LOAD THE SAVED MODEL + DATA (from win_probability_model.py)
# ============================================
model = joblib.load('../data/processed/win_prob_model.pkl')
state_df = pd.read_csv('../data/processed/match_states.csv')

features = ['inning', 'current_score', 'wickets_fallen', 'balls_bowled',
            'balls_remaining', 'target', 'runs_needed', 'required_run_rate', 'current_run_rate']

X = state_df[features]
y = state_df['won']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

# ============================================
# 1. FEATURE IMPORTANCE
# ============================================
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
importances.plot(kind='barh', color='teal')
plt.title('Feature Importance — Win Probability Model')
plt.xlabel('Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('../reports/feature_importance.png')
plt.show()

print("=== Feature Importance ===")
print(importances)

# ============================================
# 2. CONFUSION MATRIX
# ============================================
cm = confusion_matrix(y_test, preds)

plt.figure(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Lost', 'Won'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix — Win Probability Model')
plt.tight_layout()
plt.savefig('../reports/confusion_matrix.png')
plt.show()

# ============================================
# 3. ROC CURVE
# ============================================
fpr, tpr, thresholds = roc_curve(y_test, probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — Win Probability Model')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('../reports/roc_curve.png')
plt.show()

print(f"\nROC-AUC Score: {roc_auc:.4f}")

# ============================================
# 4. WIN PROBABILITY OVER THE COURSE OF A SINGLE MATCH (storytelling chart)
# ============================================
# Pick one match to visualize as an example
sample_match_id = state_df['match_id'].iloc[0]
sample_match = state_df[
    (state_df['match_id'] == sample_match_id) & (state_df['inning'] == 2)
].sort_values('balls_bowled')

if not sample_match.empty:
    sample_probs = model.predict_proba(sample_match[features])[:, 1]

    plt.figure(figsize=(12, 5))
    plt.plot(sample_match['balls_bowled'], sample_probs * 100, color='crimson', linewidth=2)
    plt.axhline(50, color='gray', linestyle='--', alpha=0.5)
    plt.title(f'Live Win Probability — Match {sample_match_id} (2nd Innings)')
    plt.xlabel('Balls Bowled')
    plt.ylabel('Win Probability (%)')
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig('../reports/sample_win_probability_curve.png')
    plt.show()

    print(f"\nSaved a sample win-probability curve for match_id {sample_match_id}")

print("\nAll evaluation charts saved to reports/")