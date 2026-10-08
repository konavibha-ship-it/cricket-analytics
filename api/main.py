import json
from pathlib import Path
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"

app = FastAPI(title="Cricket Analytics API")

# ---------- Load data once at startup ----------
model = joblib.load(DATA / "win_prob_model_small.pkl")
batters = pd.read_csv(DATA / "batting_impact.csv", encoding="utf-8")
bowlers = pd.read_csv(DATA / "bowling_impact.csv", encoding="utf-8")
venues = pd.read_csv(DATA / "venue_intelligence.csv", encoding="utf-8")

FEATURES = list(getattr(model, "feature_names_in_", [
    "inning", "current_score", "wickets_fallen", "balls_bowled",
    "balls_remaining", "target", "runs_needed",
    "required_run_rate", "current_run_rate",
]))


def to_records(df: pd.DataFrame):
    """DataFrame -> JSON-safe list of dicts (NaN becomes null, floats rounded)."""
    return json.loads(df.round(2).to_json(orient="records"))


# ---------- Health ----------
@app.get("/health")
def health():
    return {"status": "ok"}


# ---------- Players ----------
@app.get("/players/top-batters")
def top_batters(
    limit: int = Query(10, ge=1, le=100),
    min_runs: int = Query(0, ge=0, description="Hide players with fewer runs"),
    sort_by: Literal["batting_impact", "total_runs", "strike_rate",
                     "avg_runs_per_innings"] = "batting_impact",
):
    df = batters[batters["total_runs"] >= min_runs]
    df = df.sort_values(sort_by, ascending=False).head(limit)
    return to_records(df)


@app.get("/players/top-bowlers")
def top_bowlers(
    limit: int = Query(10, ge=1, le=100),
    min_wickets: int = Query(0, ge=0, description="Hide bowlers with fewer wickets"),
    sort_by: Literal["bowling_impact", "wickets", "economy",
                     "wickets_per_innings"] = "bowling_impact",
):
    df = bowlers[bowlers["wickets"] >= min_wickets]
    # lower economy is better; everything else higher is better
    df = df.sort_values(sort_by, ascending=(sort_by == "economy")).head(limit)
    return to_records(df)


@app.get("/players/search")
def search_player(name: str = Query(..., min_length=2)):
    bat = batters[batters["batsman"].str.contains(name, case=False, na=False)]
    bowl = bowlers[bowlers["bowler"].str.contains(name, case=False, na=False)]
    if bat.empty and bowl.empty:
        raise HTTPException(404, f"No player found matching '{name}'")
    return {"batting": to_records(bat.head(10)), "bowling": to_records(bowl.head(10))}


# ---------- Venues ----------
@app.get("/venues")
def list_venues(
    search: Optional[str] = Query(None, description="Part of a venue name"),
    limit: int = Query(20, ge=1, le=200),
):
    df = venues
    if search:
        df = df[df["venue"].str.contains(search, case=False, na=False)]
    if df.empty:
        raise HTTPException(404, "No venue found")
    return to_records(df.head(limit))


# ---------- Live win probability ----------
class MatchState(BaseModel):
    inning: int = Field(..., ge=1, le=2, description="1 or 2")
    current_score: int = Field(..., ge=0)
    wickets_fallen: int = Field(..., ge=0, le=10)
    balls_bowled: int = Field(..., ge=0, le=120)
    target: Optional[int] = Field(None, description="Required in 2nd innings")


@app.post("/predict/win-probability")
def predict_win_probability(s: MatchState):
    if s.inning == 2 and s.target is None:
        raise HTTPException(400, "target is required for the 2nd innings")

    balls_remaining = max(120 - s.balls_bowled, 0)
    overs = s.balls_bowled / 6
    crr = s.current_score / overs if overs > 0 else 0

    target = s.target if s.inning == 2 else 0
    runs_needed = (target - s.current_score) if s.inning == 2 else 0
    rrr = runs_needed / (balls_remaining / 6) if s.inning == 2 and balls_remaining > 0 else 0

    row = {
        "inning": s.inning,
        "current_score": s.current_score,
        "wickets_fallen": s.wickets_fallen,
        "balls_bowled": s.balls_bowled,
        "balls_remaining": balls_remaining,
        "target": target,
        "runs_needed": runs_needed,
        "required_run_rate": rrr,
        "current_run_rate": crr,
    }
    X = pd.DataFrame([row])[FEATURES]
    p = float(model.predict_proba(X)[0][1])

    return {
        "batting_team_win_probability": round(p, 4),
        "bowling_team_win_probability": round(1 - p, 4),
    }
