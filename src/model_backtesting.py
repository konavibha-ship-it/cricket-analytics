import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import joblib
import os

os.makedirs('../reports', exist_ok=True)

def backtest_model(model_path, state_csv_path, model_name):
    print("=" * 60)
    print(f"BACKTESTING: {model_name}")
    print("=" * 60)

    model = joblib.load(model_path)
    state_df = pd.read_csv(state_csv_path)

    features = ['inning', 'current_score', 'wickets_fallen', 'balls_bowled',
                'balls_remaining', 'target', 'runs_needed', 'required_run_rate', 'current_run_rate']

    X = state_df[features]
    y_true = state_df['won']

    probs = model.predict_proba(X)[:, 1]

    # ---- CALIBRATION CURVE ----
    # Splits predictions into 10 bins (0-10%, 10-20%, ... 90-100%) and checks:
    # within each bin, did the actual win rate match the predicted probability?
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, probs, n_bins=10)

    plt.figure(figsize=(8, 8))
    plt.plot(mean_predicted_value, fraction_of_positives, marker='o', linewidth=2,
              label=f'{model_name} (actual)', color='#ff4b4b')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
    plt.xlabel('Mean Predicted Win Probability')
    plt.ylabel('Actual Fraction of Wins')
    plt.title(f'Calibration Curve — {model_name}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    safe_name = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    plt.savefig(f'../reports/calibration_{safe_name}.png')
    plt.close()

    # ---- BRIER SCORE (lower = better calibrated; 0 = perfect, 0.25 = random guessing) ----
    brier = brier_score_loss(y_true, probs)

    print(f"\nBrier Score: {brier:.4f}  (0 = perfect, 0.25 = no better than guessing 50/50)")
    print("\nCalibration by bucket:")
    calibration_table = pd.DataFrame({
        'predicted_probability': mean_predicted_value.round(3),
        'actual_win_rate': fraction_of_positives.round(3),
        'difference': (mean_predicted_value - fraction_of_positives).round(3)
    })
    print(calibration_table.to_string(index=False))

    # ---- CHECKPOINT VALIDATION: at specific match stages, does higher predicted prob → more wins? ----
    print("\n--- Checkpoint Validation: Mid-innings snapshot ---")
    # Take one row per match, roughly at the midpoint of the chase
    mid_snapshots = state_df[state_df['inning'] == 2].groupby('match_id').apply(
        lambda g: g.iloc[len(g) // 2] if len(g) > 0 else None
    ).dropna()

    if not mid_snapshots.empty:
        mid_probs = model.predict_proba(mid_snapshots[features])[:, 1]
        predicted_favorite_won = ((mid_probs > 0.5) == mid_snapshots['won']).mean()
        print(f"At the mid-point of the chase, the team the model favored (>50%) "
              f"actually won {predicted_favorite_won*100:.1f}% of the time.")

    return brier, calibration_table


# ============================================
# RUN FOR BOTH MODELS
# ============================================
if __name__ == "__main__":
    t20_brier, t20_table = backtest_model(
        '../data/processed/win_prob_model.pkl',
        '../data/processed/match_states.csv',
        'T20 (IPL) Model'
    )

    print("\n\n")

    odi_brier, odi_table = backtest_model(
        '../data/processed/win_prob_model_odi.pkl',
        '../data/processed/match_states_odi.csv',
        'ODI Model'
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"T20 Model Brier Score: {t20_brier:.4f}")
    print(f"ODI Model Brier Score: {odi_brier:.4f}")
    print("\nCalibration charts saved to reports/")