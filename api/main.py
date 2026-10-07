from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "data" / "processed" / "win_prob_model.pkl"

app = FastAPI(title="Cricket Analytics API")
model = joblib.load(MODEL_PATH)

# Same feature order the model was trained with
FEATURES = list(getattr(model, "feature_names_in_", [
    "inning", "current_score", "wickets_fallen", "balls_bowled",
    "balls_remaining", "target", "runs_needed",
    "required_run_rate", "current_run_rate",
]))


class MatchState(BaseModel):
    inning: int = Field(..., ge=1, le=2, description="1 or 2")
    current_score: int = Field(..., ge=0)
    wickets_fallen: int = Field(..., ge=0, le=10)
    balls_bowled: int = Field(..., ge=0, le=120)
    target: Optional[int] = Field(None, description="Required in 2nd innings")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/win-probability")
def predict_win_probability(s: MatchState):
    if s.inning == 2 and s.target is None:
        raise HTTPException(400, "target is required for the 2nd innings")

    balls_remaining = max(120 - s.balls_bowled, 0)
    overs = s.balls_bowled / 6
    crr = s.current_score / overs if overs > 0 else 0

    # 1st innings: target-related features are 0 (same as training code)
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
