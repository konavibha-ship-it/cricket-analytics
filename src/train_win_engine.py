"""Train the win probability & momentum engine (T20 and ODI chases).

Run from the project root:
    python src/train_win_engine.py

Creates in data/processed/:
    win_engine_params.json    - fitted DLS-style resource curves
    win_engine_t20.pkl        - chase model for 20-over matches
    win_engine_odi.pkl        - chase model for 50-over matches
    win_engine_metrics.csv    - honest test results (matches never seen in training)

Evaluation is by MATCH: no match appears in both training and test data.
A baseline model (basic features only) is trained as well, so you can see
whether the new features really help.
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from win_engine import (
    BASE_FEATURES, DATA, ENGINE_FEATURES, FAMILIES, build_chase_states,
    fit_resource_params, first_innings_states, prepare_deliveries,
    resource_pct, state_from_inputs,
)

COLUMNS = [
    "match_id", "format", "inning", "batting_team", "over", "ball", "batsman",
    "bowler", "batsman_runs", "total_runs", "dismissal_kind", "player_dismissed",
]

print("Loading data...")
deliveries = pd.read_parquet(DATA / "deliveries_clean.parquet", columns=COLUMNS)
matches = pd.read_parquet(DATA / "matches_clean.parquet")

# Diagnostic: how often does the 'ball' number go above 6 within an over?
ball_max = deliveries.groupby(["match_id", "inning", "over"], observed=True)["ball"].max()
print(f"Overs where ball number exceeds 6: {(ball_max > 6).mean() * 100:.1f}%")


def make_model():
    return HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.08, max_depth=6,
        min_samples_leaf=200, l2_regularization=1.0, random_state=42,
    )


def evaluate(p, y):
    return {
        "accuracy": accuracy_score(y, p > 0.5),
        "roc_auc": roc_auc_score(y, p),
        "log_loss": log_loss(y, p),
        "brier": brier_score_loss(y, p),
    }


def split_test_ids(match_ids, seed=42, frac=0.2):
    ids = np.array(sorted(set(match_ids)))
    rng = np.random.RandomState(seed)
    return set(rng.choice(ids, size=int(len(ids) * frac), replace=False))


def show_resource_table(params, limit):
    full = limit / 6
    overs_left = [full, full * 0.75, full * 0.5, full * 0.25, full * 0.1]
    wk = [0, 2, 4, 6, 8]
    table = pd.DataFrame(
        {f"{w} wkts": resource_pct(overs_left, [w] * len(overs_left), params, limit) for w in wk},
        index=[round(x, 1) for x in overs_left],
    )
    table.index.name = "overs left"
    print("Resources remaining (%):")
    print(table.round(1).to_string())


all_params, metric_rows = {}, []

for family in ["T20", "ODI"]:
    t0 = time.time()
    limit = FAMILIES[family]["limit"]
    print()
    print("=" * 60)
    print(f"{family} chases")
    print("=" * 60)

    d, m = prepare_deliveries(deliveries, matches, family)
    print(f"Full-length matches with a result: {len(m):,}")
    test_ids = split_test_ids(m["match_id"])

    # Resource curves are fitted on TRAINING matches only
    train_first = first_innings_states(d[~d["match_id"].isin(test_ids)])
    params = fit_resource_params(train_first, limit)
    all_params[family] = {"limit": limit, **params}
    show_resource_table(params, limit)

    c = build_chase_states(d, m, params, family)
    is_test = c["match_id"].isin(test_ids)
    train, test = c[~is_test].reset_index(drop=True), c[is_test].reset_index(drop=True)
    print(f"Training rows: {len(train):,} from {train['match_id'].nunique():,} matches")
    print(f"Test rows:     {len(test):,} from {test['match_id'].nunique():,} matches")

    base = make_model().fit(train[BASE_FEATURES], train["won"])
    enh = make_model().fit(train[ENGINE_FEATURES], train["won"])
    p_base = base.predict_proba(test[BASE_FEATURES])[:, 1]
    p_enh = enh.predict_proba(test[ENGINE_FEATURES])[:, 1]

    for name, p in [("baseline", p_base), ("engine", p_enh)]:
        row = {"family": family, "scope": "all test matches", "model": name,
               "test_matches": test["match_id"].nunique(), "test_rows": len(test)}
        row.update(evaluate(p, test["won"]))
        metric_rows.append(row)

    if family == "T20":
        ipl = (test["format"] == "IPL").to_numpy()
        if ipl.sum() > 0 and test.loc[ipl, "won"].nunique() == 2:
            for name, p in [("baseline", p_base), ("engine", p_enh)]:
                row = {"family": family, "scope": "IPL only", "model": name,
                       "test_matches": test.loc[ipl, "match_id"].nunique(), "test_rows": int(ipl.sum())}
                row.update(evaluate(p[ipl], test.loc[ipl, "won"]))
                metric_rows.append(row)

    # AUC by how far the chase has progressed
    prog = (test["balls_bowled"] / limit).to_numpy()
    print()
    print("ROC-AUC by chase progress (test matches):")
    print(f"{'progress':<10}{'baseline':>10}{'engine':>10}")
    for lo, hi, name in [(0, .25, "0-25%"), (.25, .5, "25-50%"), (.5, .75, "50-75%"), (.75, 1.01, "75-100%")]:
        mask = (prog >= lo) & (prog < hi)
        y = test.loc[mask, "won"]
        if y.nunique() == 2:
            print(f"{name:<10}{roc_auc_score(y, p_base[mask]):>10.3f}{roc_auc_score(y, p_enh[mask]):>10.3f}")

    # Sanity check with a typed-in situation
    if family == "T20":
        ex = state_from_inputs(80, 3, 60, 170, family, params)
        label = "T20 example: 80/3 after 10 overs, target 170"
    else:
        ex = state_from_inputs(150, 3, 150, 280, family, params)
        label = "ODI example: 150/3 after 25 overs, target 280"
    prob = enh.predict_proba(pd.DataFrame([ex])[ENGINE_FEATURES])[0][1]
    print()
    print(f"{label}")
    print(f"   chasing team win probability: {prob * 100:.1f}%")
    print(f"   resources left: {ex['resources_pct']:.1f}%  pressure index: {ex['pressure_index']:.2f}  "
          f"vs DLS-style par: {ex['dls_par_diff']:+.1f} runs")

    out = DATA / f"win_engine_{family.lower()}.pkl"
    joblib.dump({"model": enh, "features": ENGINE_FEATURES, "family": family, "limit": limit}, out, compress=3)
    print(f"Saved {out.name} ({out.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f}s")

with open(DATA / "win_engine_params.json", "w") as f:
    json.dump(all_params, f, indent=2)

metrics = pd.DataFrame(metric_rows)
metrics.round(4).to_csv(DATA / "win_engine_metrics.csv", index=False)

print()
print("=" * 60)
print("Test results (matches the models never saw in training)")
print("=" * 60)
print(metrics[["family", "scope", "model", "test_matches", "accuracy", "roc_auc", "log_loss", "brier"]]
      .round(4).to_string(index=False))
