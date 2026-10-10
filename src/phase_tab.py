"""Streamlit tab for match phase analysis.

Used by dashboard/app.py:
    from phase_tab import render_phase_tab
    render_phase_tab(DATA_DIR)

Needs the files created by src/phase_analysis.py.
"""
import os

import pandas as pd
import plotly.express as px
import streamlit as st

PHASES = ["Powerplay", "Middle", "Death"]
NAMES = ["phase_summary", "phase_by_year", "phase_team", "phase_batters", "phase_bowlers"]


@st.cache_data(show_spinner=False)
def _load(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def render_phase_tab(data_dir):
    data = {n: _load(os.path.join(data_dir, n + ".csv")) for n in NAMES}
    if any(v is None for v in data.values()):
        st.info("Run `python src/phase_analysis.py` first to create the phase files.")
        return
    summary, by_year, team, batters, bowlers = (data[n] for n in NAMES)

    st.subheader("⏱️ Match Phase Analysis")
    formats = sorted(summary["format"].unique(), key=lambda f: (f != "IPL", f))
    fmt = st.selectbox("Format", formats, key="phase_format")

    if fmt == "ODI":
        st.caption("Powerplay = overs 1–10 · Middle = overs 11–40 · Death = overs 41–50")
    else:
        st.caption("Powerplay = overs 1–6 · Middle = overs 7–15 · Death = overs 16–20")

    # ---- Phase summary ----
    s = summary[summary["format"] == fmt].set_index("phase").reindex(PHASES)
    cols = st.columns(3)
    for col, ph in zip(cols, PHASES):
        r = s.loc[ph]
        col.markdown(f"**{ph}**")
        col.metric("Run rate", f"{r['run_rate']:.2f}")
        col.metric("Boundary %", f"{r['boundary_pct']:.1f}%")
        col.metric("Dot ball %", f"{r['dot_pct']:.1f}%")
        col.metric("Wickets / innings", f"{r['wickets_per_innings']:.2f}")

    # ---- Trend by year ----
    st.markdown("#### Run rate by year")
    y = by_year[by_year["format"] == fmt]
    fig = px.line(
        y, x="year", y="run_rate", color="phase", markers=True,
        category_orders={"phase": PHASES}, template="plotly_dark",
        labels={"run_rate": "Runs per over", "year": "Year", "phase": "Phase"},
    )
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

    # ---- Teams ----
    st.markdown("#### Teams by phase")
    c1, c2, c3 = st.columns(3)
    side = c1.radio("Side", ["Batting", "Bowling"], horizontal=True, key="phase_team_side")
    phase = c2.selectbox("Phase", PHASES, key="phase_team_phase")
    min_balls = c3.slider("Minimum balls", 120, 2000, 300, 60, key="phase_team_min")
    t = team[
        (team["format"] == fmt) & (team["side"] == side)
        & (team["phase"] == phase) & (team["balls"] >= min_balls)
    ]
    t = t.sort_values("run_rate", ascending=(side == "Bowling")).head(15)
    if t.empty:
        st.info("No teams meet the minimum number of balls.")
    else:
        fig = px.bar(
            t, x="run_rate", y="team", orientation="h", template="plotly_dark",
            labels={"run_rate": "Runs per over", "team": ""},
        )
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Batting: higher is better. Bowling: lower is better (runs conceded per over)."
        )
        st.dataframe(
            t[["team", "balls", "runs", "wickets", "run_rate"]].round(2),
            use_container_width=True, hide_index=True,
        )

    # ---- Players ----
    st.markdown("#### Players by phase")
    role = st.radio("Players", ["Batters", "Bowlers"], horizontal=True, key="phase_player_role")
    c1, c2 = st.columns(2)
    p_phase = c1.selectbox("Phase", PHASES, key="phase_player_phase")
    p_min = c2.slider("Minimum balls", 60, 1000, 120, 30, key="phase_player_min")

    if role == "Batters":
        p = batters[
            (batters["format"] == fmt) & (batters["phase"] == p_phase) & (batters["balls"] >= p_min)
        ].copy()
        p["strike_rate"] = p["runs"] / p["balls"] * 100
        p["boundary_pct"] = p["boundaries"] / p["balls"] * 100
        p["runs_per_out"] = p["runs"] / p["outs"].where(p["outs"] > 0)
        show = p.sort_values("strike_rate", ascending=False).head(20)[
            ["player", "balls", "runs", "strike_rate", "boundary_pct", "runs_per_out"]
        ]
    else:
        p = bowlers[
            (bowlers["format"] == fmt) & (bowlers["phase"] == p_phase) & (bowlers["balls"] >= p_min)
        ].copy()
        p["economy"] = p["runs"] / p["balls"] * 6
        p["dot_pct"] = p["dots"] / p["balls"] * 100
        show = p.sort_values("economy").head(20)[
            ["player", "balls", "runs", "wickets", "economy", "dot_pct"]
        ]

    if show.empty:
        st.info("No players meet the minimum number of balls.")
    else:
        st.dataframe(show.round(2), use_container_width=True, hide_index=True)
        st.caption(
            "Wides count as balls and byes count as runs (the source data does not mark them), "
            "so strike rates and economy rates are slightly approximate."
        )
