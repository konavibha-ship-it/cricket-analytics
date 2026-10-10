"""Streamlit tab: Win Probability & Momentum Engine.

Used by dashboard/app.py:
    from win_engine_tab import render_win_engine_tab
    render_win_engine_tab()

Needs the files created by src/train_win_engine.py.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dash_data import get_data as _data
from win_engine import (
    DATA, FAMILIES, build_chase_states, load_engine, load_params,
    prepare_deliveries, state_from_inputs,
)

GREEN, RED = "#2ecc71", "#e74c3c"


# ------------------------------------------------------------------ loaders
def _files_ready():
    need = ["win_engine_params.json", "win_engine_t20.pkl", "win_engine_odi.pkl"]
    return all((DATA / n).exists() for n in need)


@st.cache_resource(show_spinner="Loading engine...")
def _engine(family):
    return load_engine(family)


@st.cache_resource(show_spinner=False)
def _params():
    return load_params()


def _eligible(m, family):
    cfg = FAMILIES[family]
    x = m.copy()
    x["overs_limit"] = pd.to_numeric(x["overs_limit"], errors="coerce")
    x = x[x["format"].isin(cfg["formats"]) & (x["overs_limit"] == cfg["overs_limit"])]
    x = x.dropna(subset=["winner"])
    x = x[~x["winner"].astype(str).str.lower().isin({"no result", "tie", "draw"})]
    return x.sort_values("date", ascending=False)


def _parse_overs(text, max_overs):
    try:
        o, _, b = text.strip().partition(".")
        overs, balls = int(o or 0), int(b or 0)
    except ValueError:
        return None
    if overs < 0 or balls < 0 or balls > 5:
        return None
    total = overs * 6 + balls
    return total if total <= max_overs * 6 else None


def _prob(engine, row):
    X = pd.DataFrame([row])[engine["features"]]
    return float(engine["model"].predict_proba(X)[0][1] * 100)


# ------------------------------------------------------------------ main
def render_win_engine_tab():
    st.subheader("⚡ Win Probability & Momentum Engine")
    if not _files_ready():
        st.info("Run `python src/train_win_engine.py` first to build the engine.")
        return

    params = _params()
    c1, c2 = st.columns(2)
    family = c1.radio("Match length", ["T20", "ODI"], horizontal=True, key="we_family")
    mode = c2.radio("Mode", ["Replay a match", "Enter a live score"], horizontal=True, key="we_mode")
    engine = _engine(family)
    st.caption(
        "Win probability of the team that is CHASING, from a model trained on full-length "
        f"{'T20' if family == 'T20' else 'ODI'} chases. Team strength is not included."
    )

    if mode == "Replay a match":
        _replay(family, engine, params[family])
    else:
        _live(family, engine, params[family])
    _about()


# ------------------------------------------------------------------ replay
def _replay(family, engine, p):
    d_all, m_all = _data()
    elig = _eligible(m_all, family)
    if elig.empty:
        st.info("No matches available.")
        return

    formats = sorted(elig["format"].astype(str).unique(), key=lambda f: (f not in ("IPL", "ODI"), f))
    c1, c2 = st.columns(2)
    fmt = c1.selectbox("Competition", formats, key=f"we_fmt_{family}")
    q = c2.text_input("Search team or venue (optional)", key=f"we_q_{family}").strip()

    x = elig[elig["format"].astype(str) == fmt]
    if q:
        mask = (
            x["team1"].astype(str).str.contains(q, case=False, regex=False)
            | x["team2"].astype(str).str.contains(q, case=False, regex=False)
            | x["venue"].astype(str).str.contains(q, case=False, regex=False)
        )
        x = x[mask]
    x = x.head(300)
    if x.empty:
        st.info("No matches found.")
        return

    labels = {r.match_id: f"{str(r.date)[:10]} · {r.team1} vs {r.team2} · {r.venue}"
              for r in x.itertuples()}
    mid = st.selectbox("Match", list(labels), format_func=lambda i: labels[i], key=f"we_match_{family}")

    d_one = d_all[d_all["match_id"] == mid]
    m_one = m_all[m_all["match_id"] == mid]
    try:
        d_prep, m_prep = prepare_deliveries(d_one, m_one, family)
        c = build_chase_states(d_prep, m_prep, p, family)
    except ValueError as e:
        st.warning(str(e))
        return
    if c.empty:
        st.warning("This match has no usable second-innings data.")
        return

    c["win_prob"] = engine["model"].predict_proba(c[engine["features"]])[:, 1] * 100
    c["x"] = np.arange(1, len(c) + 1)
    chasing = str(c["batting_team"].iloc[0])
    target = int(c["target"].iloc[0])
    won = bool(c["won"].iloc[0])

    k = st.columns(5)
    k[0].metric("Chasing team", chasing)
    k[1].metric("Target", target)
    k[2].metric("Result", "Chase succeeded" if won else "Chase failed")
    k[3].metric("Win % at the start", f"{c['win_prob'].iloc[0]:.0f}%")
    k[4].metric("Lowest / highest", f"{c['win_prob'].min():.0f}% / {c['win_prob'].max():.0f}%")

    # ---- Win probability ----
    hover = (
        c["over"].astype(int).astype(str) + "." + c["ball"].astype(int).astype(str)
        + "  " + c["current_score"].astype(int).astype(str)
        + "/" + c["wickets_fallen"].astype(int).astype(str)
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=c["x"], y=c["win_prob"], mode="lines", line=dict(color=GREEN, width=3),
        text=hover, hovertemplate="%{text}<br>Win %{y:.1f}%<extra></extra>",
    ))
    w = c[c["is_wicket"] == 1]
    fig.add_trace(go.Scatter(
        x=w["x"], y=w["win_prob"], mode="markers",
        marker=dict(color=RED, size=10, symbol="x"),
        text=w["player_dismissed"].astype(str) + " out",
        hovertemplate="%{text}<br>Win %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=50, line_dash="dash", line_color="gray")
    fig.update_layout(
        template="plotly_dark", height=420, showlegend=False,
        yaxis_title=f"{chasing} win probability (%)", yaxis_range=[0, 100],
        xaxis_title="Delivery number of the chase (red crosses = wickets)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- Momentum ----
    st.markdown("#### Momentum")
    base = c["win_prob"].shift(12).fillna(c["win_prob"].iloc[0])
    mom = c["win_prob"] - base
    figm = go.Figure(go.Bar(x=c["x"], y=mom, marker_color=np.where(mom >= 0, GREEN, RED).tolist()))
    figm.update_layout(
        template="plotly_dark", height=280, showlegend=False,
        yaxis_title="Change in win % over last 2 overs", xaxis_title="Delivery number of the chase",
    )
    st.plotly_chart(figm, use_container_width=True)
    st.caption("Green = the last two overs moved the chase towards the chasing team. Red = away from it.")

    # ---- Pressure and par ----
    a, b = st.columns(2)
    f1 = go.Figure(go.Scatter(x=c["x"], y=c["pressure_index"], mode="lines", line=dict(color="#f5a623", width=3)))
    f1.add_hline(y=1, line_dash="dash", line_color="gray")
    f1.update_layout(template="plotly_dark", height=320, title="Pressure index (above 1 = hard)",
                     xaxis_title="Delivery number", showlegend=False)
    a.plotly_chart(f1, use_container_width=True)

    f2 = go.Figure(go.Scatter(x=c["x"], y=c["dls_par_diff"], mode="lines", line=dict(color="#4aa3ff", width=3)))
    f2.add_hline(y=0, line_dash="dash", line_color="gray")
    f2.update_layout(template="plotly_dark", height=320, title="Runs ahead (+) or behind (-) DLS-style par",
                     xaxis_title="Delivery number", showlegend=False)
    b.plotly_chart(f2, use_container_width=True)

    # ---- Biggest swings ----
    st.markdown("#### Biggest swings")
    c["swing"] = c["win_prob"].diff()
    c["before"] = c["win_prob"].shift(1)
    top = c.loc[c["swing"].abs().nlargest(8).index].sort_values("x").copy()

    def event(r):
        if r["is_wicket"] == 1:
            return f"Wicket: {r['player_dismissed']}"
        if r["batsman_runs"] == 6:
            return "SIX"
        if r["batsman_runs"] == 4:
            return "FOUR"
        return f"{int(r['total_runs'])} run(s)"

    top["Event"] = top.apply(event, axis=1)
    show = pd.DataFrame({
        "Ball": top["over"].astype(int).astype(str) + "." + top["ball"].astype(int).astype(str),
        "Batter": top["batsman"], "Bowler": top["bowler"], "Event": top["Event"],
        "Win % before": top["before"].round(1), "Win % after": top["win_prob"].round(1),
        "Swing (pts)": top["swing"].round(1),
    })
    st.dataframe(show, use_container_width=True, hide_index=True)


# ------------------------------------------------------------------ live score
def _live(family, engine, p):
    limit = FAMILIES[family]["limit"]
    d_target, d_score, d_over = (170, 80, "10.0") if family == "T20" else (280, 150, "25.0")

    c1, c2, c3, c4 = st.columns(4)
    target = c1.number_input("Target", 1, 600, d_target, key=f"we_t_{family}")
    score = c2.number_input("Current score", 0, 600, d_score, key=f"we_s_{family}")
    wk = c3.number_input("Wickets fallen", 0, 9, 3, key=f"we_w_{family}")
    overs_txt = c4.text_input("Overs bowled (e.g. 10.3)", d_over, key=f"we_o_{family}")

    balls = _parse_overs(overs_txt, limit // 6)
    if balls is None:
        st.error(f"Enter overs like 10.3 (10 overs and 3 balls), up to {limit // 6} overs.")
        return
    if score >= target:
        st.success("The target has been reached.")
        return
    if balls >= limit:
        st.warning("All overs have been bowled and the target was not reached.")
        return

    row = state_from_inputs(int(score), int(wk), balls, int(target), family, p)
    prob = _prob(engine, row)

    k = st.columns(5)
    k[0].metric("Chasing team win %", f"{prob:.1f}%")
    k[1].metric("Runs needed", f"{int(row['runs_needed'])} off {row['balls_remaining']}")
    k[2].metric("Required run rate", f"{row['required_run_rate']:.2f}")
    k[3].metric("Resources left", f"{row['resources_pct']:.1f}%")
    k[4].metric("Pressure index", f"{row['pressure_index']:.2f}")

    pi = row["pressure_index"]
    mood = "comfortable" if pi < 0.8 else "balanced" if pi <= 1.2 else "under pressure"
    diff = row["dls_par_diff"]
    side = "ahead of" if diff >= 0 else "behind"
    st.write(
        f"The chasing team is **{mood}** (rule of thumb: below 0.8 comfortable, above 1.2 under pressure) "
        f"and **{abs(diff):.1f} runs {side}** the DLS-style par score."
    )

    st.markdown("#### What if the next over goes…")
    per_over = score / max(balls, 1) * 6
    rows = []
    for runs in [0, 4, 8, 12, 16, 20]:
        r = {"Next over scores": f"{runs} runs"}
        for w, name in [(0, "No wicket"), (1, "1 wicket")]:
            s2 = state_from_inputs(
                int(score) + runs, int(wk) + w, min(balls + 6, limit), int(target), family, p,
                recent_runs=runs + per_over, recent_wkts=w + wk / max(balls, 1) * 6,
            )
            r[name] = f"{_prob(engine, s2):.0f}%"
        rows.append(r)
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("Chasing team win probability after the next over. Recent-form inputs are estimated.")


# ------------------------------------------------------------------ about
def _about():
    with st.expander("How this works"):
        st.markdown(
            "- **Win probability:** a model trained on past full-length chases, using score, wickets, "
            "balls left, required run rate and recent runs and wickets.\n"
            "- **Resources left (DLS-style):** the share of scoring potential still in hand, from a curve "
            "fitted to this project's data using the same shape as the original Duckworth-Lewis method. "
            "It is not the official DLS table.\n"
            "- **DLS-style par:** the score the chasing team would need by now to be level on resources.\n"
            "- **Pressure index:** runs still needed divided by what an average team scores from the same "
            "situation. Above 1 means better-than-average batting is needed.\n"
            "- **Momentum:** how much the win probability moved over the last 12 deliveries.\n"
            "- **Limits:** no team or player strength, and the T20 model is trained on all T20 cricket "
            "together, so IPL chases may be judged slightly harsher than they should be."
        )
        path = DATA / "win_engine_metrics.csv"
        if path.exists():
            st.markdown("**Test results (matches the models never saw in training):**")
            st.dataframe(pd.read_csv(path), hide_index=True, use_container_width=True)
