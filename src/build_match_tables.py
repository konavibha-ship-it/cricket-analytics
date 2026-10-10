"""Build per-player, per-match tables used by several modules.

Run from the project root:
    python src/build_match_tables.py

Creates (in data/processed/):
    batting_innings.parquet  - one row per batter per innings
    bowling_innings.parquet  - one row per bowler per innings

Known approximations (the source data has no wide/no-ball/bye flags):
  * balls faced / bowled count every delivery, so wides are included
  * runs conceded include byes and leg byes
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"

KEYS = ["match_id", "inning"]
BOWLER_WICKETS = {"bowled", "caught", "lbw", "stumped", "caught and bowled", "hit wicket"}

print("Loading data...")
d = pd.read_parquet(DATA / "deliveries_clean.parquet")
matches = pd.read_parquet(DATA / "matches_clean.parquet")

d["match_id"] = d["match_id"].astype(str)
matches["match_id"] = matches["match_id"].astype(str)

# Keep deliveries in true order
d = d.sort_values(KEYS + ["over", "ball"], kind="mergesort").reset_index(drop=True)
d["row_id"] = np.arange(len(d))
d["is_four"] = d["batsman_runs"] == 4
d["is_six"] = d["batsman_runs"] == 6
d["is_dot"] = d["total_runs"] == 0
d["bowler_wicket"] = d["dismissal_kind"].str.lower().isin(BOWLER_WICKETS)

# One row per innings: who batted, in which format
inning_info = (
    d.groupby(KEYS)
    .agg(batting_team=("batting_team", "first"), format=("format", "first"))
    .reset_index()
)

match_info = matches[["match_id", "date", "season", "year", "venue", "team1", "team2", "winner"]]


def add_match_context(df: pd.DataFrame, team_col: str) -> pd.DataFrame:
    """Attach match details and the opposition team."""
    df = df.merge(match_info, on="match_id", how="left")
    df["opposition"] = np.where(df[team_col] == df["team1"], df["team2"], df["team1"])
    return df.drop(columns=["team1", "team2"])


# ---------------- Batting ----------------
print("Building batting table...")
bat = (
    d.groupby(KEYS + ["batsman"], sort=False)
    .agg(
        runs=("batsman_runs", "sum"),
        balls=("batsman_runs", "size"),
        fours=("is_four", "sum"),
        sixes=("is_six", "sum"),
        first_row=("row_id", "min"),
    )
    .reset_index()
)

# A dismissal belongs to the dismissed player, even if they were the non-striker
out = (
    d[d["player_dismissed"].notna()][KEYS + ["player_dismissed", "dismissal_kind"]]
    .drop_duplicates(KEYS + ["player_dismissed"])
    .rename(columns={"player_dismissed": "batsman"})
)
bat = bat.merge(out, on=KEYS + ["batsman"], how="outer")
for col in ["runs", "balls", "fours", "sixes"]:
    bat[col] = bat[col].fillna(0).astype(int)

bat = bat.merge(inning_info, on=KEYS, how="left")
bat["dismissed"] = bat["dismissal_kind"].notna()
bat["strike_rate"] = np.where(bat["balls"] > 0, bat["runs"] / bat["balls"] * 100, np.nan).round(2)
bat["bat_order"] = bat.groupby(KEYS)["first_row"].rank(method="first")
bat = bat.drop(columns=["first_row"])
bat = add_match_context(bat, "batting_team")

# ---------------- Bowling ----------------
print("Building bowling table...")
bowl = (
    d.groupby(KEYS + ["bowler"], sort=False)
    .agg(
        balls=("total_runs", "size"),
        runs=("total_runs", "sum"),
        wickets=("bowler_wicket", "sum"),
        dots=("is_dot", "sum"),
    )
    .reset_index()
)
bowl["wickets"] = bowl["wickets"].astype(int)
bowl["economy"] = (bowl["runs"] / (bowl["balls"] / 6)).round(2)
bowl = bowl.merge(inning_info, on=KEYS, how="left")
bowl = add_match_context(bowl, "batting_team")
# The bowling side is the opposition of the batting side
bowl = bowl.rename(columns={"batting_team": "facing_team", "opposition": "bowling_team"})

# ---------------- Save + sanity checks ----------------
bat.to_parquet(DATA / "batting_innings.parquet", index=False)
bowl.to_parquet(DATA / "bowling_innings.parquet", index=False)

print()
print(f"batting_innings.parquet : {len(bat):,} rows, {bat['match_id'].nunique():,} matches")
print(f"bowling_innings.parquet : {len(bowl):,} rows, {bowl['match_id'].nunique():,} matches")
print("Matches with no details in matches_clean:", int(bat["date"].isna().groupby(bat["match_id"]).all().sum()))

print()
print("Check 1 - total runs match the source data:")
print("  from batting table :", int(bat["runs"].sum()))
print("  from ball-by-ball  :", int(d["batsman_runs"].sum()))

print()
print("Check 2 - V Kohli total runs (all formats), compare with your batting_impact.csv (9346):")
print("  ", int(bat.loc[bat["batsman"] == "V Kohli", "runs"].sum()))

print()
print("Top 5 IPL run scorers:")
ipl = bat[bat["format"] == "IPL"].groupby("batsman")["runs"].sum().sort_values(ascending=False).head(5)
print(ipl.to_string())
