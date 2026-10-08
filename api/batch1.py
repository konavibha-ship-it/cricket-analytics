"""Batch 1 endpoints: overview, matchups, archetypes."""
import json
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"

router = APIRouter()

matches = pd.read_parquet(DATA / "matches_clean.parquet")
matchups = pd.read_csv(DATA / "matchups.csv")
clusters = pd.read_csv(DATA / "player_clusters.csv")

Fmt = Literal["IPL", "T20I", "ODI"]


def to_records(df: pd.DataFrame):
    df = df.replace([np.inf, -np.inf], np.nan).round(2)
    return json.loads(df.to_json(orient="records"))


def fmt_matches(fmt: str) -> pd.DataFrame:
    d = matches[matches["format"] == fmt]
    if d.empty:
        raise HTTPException(404, f"No matches found for format {fmt}")
    return d


# ---------- Overview ----------
@router.get("/overview")
def overview(format: Fmt = "IPL"):
    d = fmt_matches(format)
    teams = pd.concat([d["team1"], d["team2"]]).nunique()
    return {
        "matches": int(len(d)),
        "teams": int(teams),
        "seasons": int(d["year"].nunique()),
    }


@router.get("/overview/team-wins")
def team_wins(format: Fmt = "IPL", limit: int = Query(10, ge=1, le=50)):
    d = fmt_matches(format)
    w = d["winner"].dropna()
    w = w[~w.isin(["No Result", "no result", "tie", "Tie"])]
    counts = w.value_counts().head(limit)
    return [{"team": t, "wins": int(n)} for t, n in counts.items()]


@router.get("/overview/matches-per-season")
def matches_per_season(format: Fmt = "IPL"):
    d = fmt_matches(format).dropna(subset=["year"])
    c = d.groupby("year").size().sort_index()
    return [{"year": int(y), "matches": int(n)} for y, n in c.items()]


# ---------- Matchups ----------
@router.get("/matchups/batters")
def matchup_batters(
    search: str = Query("", description="Part of a batter's name"),
    limit: int = Query(30, ge=1, le=100),
):
    g = matchups.groupby("batsman")["balls_faced"].sum().sort_values(ascending=False)
    if search:
        g = g[g.index.str.contains(search, case=False, na=False, regex=False)]
    return [{"batsman": k, "balls_faced": int(v)} for k, v in g.head(limit).items()]


@router.get("/matchups/by-batter")
def matchups_by_batter(
    name: str = Query(..., min_length=1),
    bowler: Optional[str] = Query(None, description="Filter by part of a bowler's name"),
    min_balls: int = Query(1, ge=1),
    limit: int = Query(30, ge=1, le=200),
):
    df = matchups[matchups["batsman"] == name]
    if bowler:
        df = df[df["bowler"].str.contains(bowler, case=False, na=False, regex=False)]
    df = df[df["balls_faced"] >= min_balls]
    df = df.sort_values("balls_faced", ascending=False).head(limit)
    if df.empty:
        raise HTTPException(404, "No matchups found")
    return to_records(df)


@router.get("/matchups/most-faced")
def most_faced(limit: int = Query(30, ge=1, le=100)):
    df = matchups.sort_values("balls_faced", ascending=False).head(limit)
    return to_records(df)


# ---------- Archetypes ----------
@router.get("/archetypes")
def archetypes():
    cols = ["total_runs", "strike_rate", "powerplay_sr", "middle_sr",
            "death_sr", "boundary_pct"]
    out = []
    for cid, g in clusters.groupby("cluster"):
        item = {"cluster": int(cid), "players": int(len(g))}
        for c in cols:
            item[c] = round(float(g[c].mean()), 1)
        top = g.sort_values("total_runs", ascending=False)["batsman"].head(5)
        item["top_players"] = top.tolist()
        out.append(item)
    return sorted(out, key=lambda x: x["cluster"])
