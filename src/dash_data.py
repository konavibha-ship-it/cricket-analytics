"""Shared, cached data loader for dashboard tabs.

The ball-by-ball file is big, so it is loaded once and shared by every tab
that imports get_data().
"""
import pandas as pd
import streamlit as st

from win_engine import DATA

COLS = [
    "match_id", "format", "inning", "batting_team", "over", "ball", "batsman",
    "bowler", "batsman_runs", "total_runs", "dismissal_kind", "player_dismissed",
]


@st.cache_resource(show_spinner="Loading ball-by-ball data (first time only)...")
def get_data():
    d = pd.read_parquet(DATA / "deliveries_clean.parquet", columns=COLS)
    d["match_id"] = d["match_id"].astype(str).astype("category")
    m = pd.read_parquet(DATA / "matches_clean.parquet")
    m["match_id"] = m["match_id"].astype(str)
    return d, m
