import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import os

os.makedirs('../reports', exist_ok=True)

# ============================================
# LOAD MODEL + DATA
# ============================================
model = joblib.load('../data/processed/win_prob_model.pkl')
state_df = pd.read_csv('../data/processed/match_states.csv')

features = ['inning', 'current_score', 'wickets_fallen', 'balls_bowled',
            'balls_remaining', 'target', 'runs_needed', 'required_run_rate', 'current_run_rate']

X = state_df[features]

# SHAP on the full dataset can be slow — use a sample for speed
sample_size = min(2000, len(X))
X_sample = X.sample(sample_size, random_state=42)

print(f"Computing SHAP values on a sample of {sample_size} rows (this takes a minute)...")

# ============================================
# COMPUTE SHAP VALUES
# ============================================
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# Handle different SHAP output shapes across versions
if isinstance(shap_values, list):
    # Older SHAP: list of arrays, one per class
    shap_values_win = shap_values[1]
elif shap_values.ndim == 3:
    # Newer SHAP: shape (samples, features, classes)
    shap_values_win = shap_values[:, :, 1]
else:
    # Already 2D (samples, features)
    shap_values_win = shap_values

print(f"SHAP values shape: {shap_values_win.shape}")  # sanity check — should be (2000, 9)

# ============================================
# 1. SUMMARY PLOT — overall feature impact across all predictions
# ============================================
plt.figure()
shap.summary_plot(shap_values_win, X_sample, show=False)
plt.tight_layout()
plt.savefig('../reports/shap_summary_plot.png', bbox_inches='tight')
plt.close()
print("Saved: shap_summary_plot.png")

# ============================================
# 2. BAR PLOT — mean absolute SHAP value per feature (cleaner version of importance)
# ============================================
plt.figure()
shap.summary_plot(shap_values_win, X_sample, plot_type='bar', show=False)
plt.tight_layout()
plt.savefig('../reports/shap_bar_plot.png', bbox_inches='tight')
plt.close()
print("Saved: shap_bar_plot.png")

# ============================================
# 3. EXPLAIN ONE SPECIFIC PREDICTION (the real power of SHAP)
# ============================================
# Pick one example row to explain in detail
example_idx = 0
example_row = X_sample.iloc[[example_idx]]
example_shap = shap_values_win[example_idx]

predicted_prob = model.predict_proba(example_row)[0][1]

print(f"\n=== Explaining one specific prediction ===")
print(f"Match state: {example_row.to_dict('records')[0]}")
print(f"Predicted win probability: {predicted_prob*100:.1f}%")
print(f"\nHow each feature pushed this prediction:")

feature_contributions = pd.DataFrame({
    'feature': features,
    'value': example_row.values[0],
    'shap_contribution': example_shap
}).sort_values('shap_contribution', key=abs, ascending=False)

print(feature_contributions.to_string(index=False))

# Save a waterfall-style plot for this one prediction
plt.figure()
shap.plots.waterfall(
    shap.Explanation(
        values=example_shap,
        base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) and len(np.atleast_1d(explainer.expected_value)) > 1 else explainer.expected_value,
        data=example_row.values[0],
        feature_names=features
    ),
    show=False
)
plt.tight_layout()
plt.savefig('../reports/shap_single_prediction_waterfall.png', bbox_inches='tight')
plt.close()
print("\nSaved: shap_single_prediction_waterfall.png")

print("\nAll SHAP outputs saved to reports/")