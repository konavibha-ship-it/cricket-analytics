"""Match phase analysis: powerplay / middle overs / death overs.

Run from the project root:
    python src/phase_analysis.py

Creates small CSV files in data/processed/:
    phase_summary.csv   - format x phase totals and rates
    phase_by_year.csv   - run rate per phase per year
    phase_team.csv      - team batting and bowling by phase
    phase_batters.csv   - batter strike rate etc. by phase
    phase_bowlers.csv   - bowler economy etc. by phase

Phases
    20-over formats: Powerplay = overs 1-6, Middle = 7-15, Death = 16-20
    ODI:             Powerplay = overs 1-10, Middle = 11-40, Death = 41-50
The Hundred, Tests and first-class matches are left out (different structure).
Super overs (innings 3 and 4) are left out.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"

T20_FORMATS = [
    "IPL", "T20I", "BBL", "Women's BBL", "CPL", "Women's CPL", "SA20",
    "CSA T20 Challenge", "Super Smash", "Syed Mushtaq Ali Trophy", "WPL",
]
FORMATS = T20_FORMATS + ["ODI"]
BOWLER_WICKETS = {"bowled", "caught", "lbw", "stumped", "caught and bowled", "hit wicket"}
MIN_BALLS = 60  # drop players with fewer balls in a phase

print("Loading data...")
d = pd.read_parquet(DATA / "deliveries_clean.parquet")
matches = pd.read_parquet(DATA / "matches_clean.parquet")
d["match_id"] = d["match_id"].astype(str)
matches["match_id"] = matches["match_id"].astype(str)

d = d[d["format"].isin(FORMATS) & d["inning"].isin([1, 2])].copy()
# These columns are stored as pandas "categoricals" with different category lists,
# which cannot be compared directly. Plain text columns avoid that (NaN is kept).
for col in ["batsman", "bowler", "player_dismissed", "dismissal_kind", "batting_team", "format"]:
    d[col] = d[col].astype(object)

print(f"Using {len(d):,} deliveries from {d['match_id'].nunique():,} matches")

# ---- Phase of each ball ----
is_odi = d["format"] == "ODI"
over = d["over"]
d["phase"] = np.select(
    [~is_odi & (over <= 5), ~is_odi & (over <= 14), is_odi & (over <= 9), is_odi & (over <= 39)],
    ["Powerplay", "Middle", "Powerplay", "Middle"],
    default="Death",
)

# ---- Match context: year and bowling team ----
ctx = matches[["match_id", "year", "team1", "team2"]]
d = d.merge(ctx, on="match_id", how="left")
d["bowling_team"] = np.where(d["batting_team"] == d["team1"], d["team2"], d["team1"])

# ---- Ball flags ----
d["is_boundary"] = d["batsman_runs"].isin([4, 6])
d["is_dot"] = d["total_runs"] == 0
d["is_wicket"] = d["player_dismissed"].notna()
d["out_striker"] = d["player_dismissed"] == d["batsman"]
d["bowler_wicket"] = d["dismissal_kind"].str.lower().isin(BOWLER_WICKETS)

# ---- 1. Summary by format and phase ----
print("Building summary...")
g = d.groupby(["format", "phase"]).agg(
    runs=("total_runs", "sum"),
    balls=("total_runs", "size"),
    wickets=("is_wicket", "sum"),
    boundaries=("is_boundary", "sum"),
    dots=("is_dot", "sum"),
).reset_index()
reached = (
    d.drop_duplicates(["match_id", "inning", "phase"])
    .groupby(["format", "phase"]).size().rename("innings").reset_index()
)
summary = g.merge(reached, on=["format", "phase"])
summary["run_rate"] = summary["runs"] / summary["balls"] * 6
summary["boundary_pct"] = summary["boundaries"] / summary["balls"] * 100
summary["dot_pct"] = summary["dots"] / summary["balls"] * 100
summary["runs_per_innings"] = summary["runs"] / summary["innings"]
summary["wickets_per_innings"] = summary["wickets"] / summary["innings"]
summary.round(2).to_csv(DATA / "phase_summary.csv", index=False)

# ---- 2. Run rate by year ----
print("Building yearly trend...")
y = d.dropna(subset=["year"]).groupby(["format", "year", "phase"]).agg(
    runs=("total_runs", "sum"), balls=("total_runs", "size")
).reset_index()
y["year"] = y["year"].astype(int)
y["run_rate"] = y["runs"] / y["balls"] * 6
y.round(2).to_csv(DATA / "phase_by_year.csv", index=False)

# ---- 3. Teams ----
print("Building team table...")
bat_t = d.groupby(["format", "batting_team", "phase"]).agg(
    runs=("total_runs", "sum"), balls=("total_runs", "size"), wickets=("is_wicket", "sum")
).reset_index().rename(columns={"batting_team": "team"})
bat_t["side"] = "Batting"
bowl_t = d.groupby(["format", "bowling_team", "phase"]).agg(
    runs=("total_runs", "sum"), balls=("total_runs", "size"), wickets=("is_wicket", "sum")
).reset_index().rename(columns={"bowling_team": "team"})
bowl_t["side"] = "Bowling"
team = pd.concat([bat_t, bowl_t], ignore_index=True)
team = team[team["balls"] >= 120].copy()
team["run_rate"] = team["runs"] / team["balls"] * 6
team.round(2).to_csv(DATA / "phase_team.csv", index=False)

# ---- 4. Batters ----
print("Building batter table...")
bt = d.groupby(["format", "batsman", "phase"]).agg(
    runs=("batsman_runs", "sum"),
    balls=("batsman_runs", "size"),
    boundaries=("is_boundary", "sum"),
    outs=("out_striker", "sum"),
).reset_index().rename(columns={"batsman": "player"})
bt = bt[bt["balls"] >= MIN_BALLS]
bt.to_csv(DATA / "phase_batters.csv", index=False)

# ---- 5. Bowlers ----
print("Building bowler table...")
bw = d.groupby(["format", "bowler", "phase"]).agg(
    balls=("total_runs", "size"),
    runs=("total_runs", "sum"),
    wickets=("bowler_wicket", "sum"),
    dots=("is_dot", "sum"),
).reset_index().rename(columns={"bowler": "player"})
bw = bw[bw["balls"] >= MIN_BALLS]
bw.to_csv(DATA / "phase_bowlers.csv", index=False)

# ---- Report ----
print()
print("Files written to", DATA)
for name, df in [("phase_summary", summary), ("phase_by_year", y), ("phase_team", team),
                 ("phase_batters", bt), ("phase_bowlers", bw)]:
    print(f"  {name}.csv: {len(df):,} rows")

print()
print("IPL phase summary:")
ipl = summary[summary["format"] == "IPL"].set_index("phase").reindex(["Powerplay", "Middle", "Death"])
print(ipl[["run_rate", "boundary_pct", "dot_pct", "wickets_per_innings"]].round(2).to_string())
