"""Batch 2: ball-by-ball win probability replay (T20 / IPL model)."""
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"

router = APIRouter()

model = joblib.load(DATA / "win_prob_model_small.pkl")
FEATURES = list(getattr(model, "feature_names_in_", [
    "inning", "current_score", "wickets_fallen", "balls_bowled",
    "balls_remaining", "target", "runs_needed",
    "required_run_rate", "current_run_rate",
]))

matches = pd.read_parquet(DATA / "matches_clean.parquet")
matches = matches[matches["format"] == "IPL"].copy()
matches["match_id"] = matches["match_id"].astype(str)

states = pd.read_parquet(DATA / "match_states.parquet")
chase = states[states["inning"] == 2].copy()
chase["match_id"] = chase["match_id"].astype(str)
chase = chase.sort_values(["match_id", "balls_bowled"])

replayable = matches[matches["match_id"].isin(chase["match_id"].unique())].copy()
replayable = replayable.sort_values("date", ascending=False)


def _s(v):
    return None if pd.isna(v) else str(v)


def info(r):
    return {
        "match_id": r["match_id"],
        "date": str(r["date"])[:10],
        "team1": _s(r["team1"]),
        "team2": _s(r["team2"]),
        "winner": _s(r["winner"]),
        "venue": _s(r["venue"]),
        "season": _s(r["season"]),
    }


@router.get("/win-probability/matches")
def replay_matches(
    search: Optional[str] = Query(None, description="Team or venue name"),
    limit: int = Query(30, ge=1, le=100),
):
    df = replayable
    if search:
        mask = (
            df["team1"].str.contains(search, case=False, na=False, regex=False)
            | df["team2"].str.contains(search, case=False, na=False, regex=False)
            | df["venue"].str.contains(search, case=False, na=False, regex=False)
        )
        df = df[mask]
    if df.empty:
        raise HTTPException(404, "No matches found")
    return [info(r) for _, r in df.head(limit).iterrows()]


@router.get("/win-probability/match")
def replay_match(match_id: str = Query(...)):
    d = chase[chase["match_id"] == match_id]
    if d.empty:
        raise HTTPException(404, "Match not found")

    m = replayable[replayable["match_id"] == match_id]
    out = info(m.iloc[0]) if not m.empty else {"match_id": match_id, "winner": None}

    probs = model.predict_proba(d[FEATURES])[:, 1] * 100
    chasing_team = str(d["batting_team"].iloc[0])
    out["chasing_team"] = chasing_team
    out["target"] = int(d["target"].iloc[0])
    out["chasing_team_won"] = out.get("winner") == chasing_team

    points = [
        {"balls": int(b), "p": round(float(p), 1)}
        for b, p in zip(d["balls_bowled"], probs)
    ]
    return {"info": out, "points": points}
