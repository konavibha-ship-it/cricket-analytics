"""Streamlit tab: Player Performance Tracking.

Used by dashboard/app.py:
    from player_tracking_tab import render_player_tracking_tab
    render_player_tracking_tab()

Needs the files created by src/build_match_tables.py
(batting_innings.parquet and bowling_innings.parquet).

The watchlist is saved in data/processed/watchlist.json on this computer.
"""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from win_engine import DATA

GREEN, GOLD, BLUE, RED, ORANGE = "#2ecc71", "#f5a623", "#4aa3ff", "#e74c3c", "#ff7f50"
WATCH_FILE = DATA / "watchlist.json"
TEXT_COLS = ["batsman", "bowler", "batting_team", "bowling_team", "facing_team",
             "opposition", "venue", "dismissal_kind", "season"]


# ------------------------------------------------------------------ data
@st.cache_resource(show_spinner="Loading player tables...")
def _tables():
    bat = pd.read_parquet(DATA / "batting_innings.parquet")
    bowl = pd.read_parquet(DATA / "bowling_innings.parquet")
    for df in (bat, bowl):
        for col in TEXT_COLS:
            if col in df.columns:
                df[col] = df[col].astype(object)
        df["format"] = df["format"].astype(str)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    bat = bat.sort_values(["date", "match_id", "inning"], kind="mergesort").reset_index(drop=True)
    bowl = bowl.sort_values(["date", "match_id", "inning"], kind="mergesort").reset_index(drop=True)
    return bat, bowl


def _f(v, nd=1):
    return "–" if v is None or pd.isna(v) else f"{v:.{nd}f}"


def _load_watch():
    try:
        return json.loads(WATCH_FILE.read_text())
    except Exception:  # noqa: BLE001
        return []


def _save_watch(items):
    WATCH_FILE.write_text(json.dumps(items, indent=2))


# ------------------------------------------------------------------ stats
def _bat_stats(x):
    inn = len(x)
    outs = int(x["dismissed"].sum())
    runs = int(x["runs"].sum())
    balls = int(x["balls"].sum())
    hs, star = 0, ""
    if inn:
        i = x["runs"].idxmax()
        hs = int(x.loc[i, "runs"])
        star = "" if x.loc[i, "dismissed"] else "*"
    return {
        "Innings": inn, "Runs": runs,
        "Average": runs / outs if outs else np.nan,
        "SR": runs / balls * 100 if balls else np.nan,
        "RPI": runs / inn if inn else np.nan,
        "HS": f"{hs}{star}",
        "50s": int(((x["runs"] >= 50) & (x["runs"] < 100)).sum()),
        "100s": int((x["runs"] >= 100).sum()),
        "4s": int(x["fours"].sum()), "6s": int(x["sixes"].sum()),
        "Boundary %": (4 * x["fours"].sum() + 6 * x["sixes"].sum()) / runs * 100 if runs else np.nan,
    }


def _bowl_stats(x):
    inn = len(x)
    balls, runs, wk = int(x["balls"].sum()), int(x["runs"].sum()), int(x["wickets"].sum())
    best = "–"
    if inn:
        b = x.sort_values(["wickets", "runs"], ascending=[False, True]).iloc[0]
        best = f"{int(b['wickets'])}/{int(b['runs'])}"
    return {
        "Innings": inn, "Wickets": wk,
        "Average": runs / wk if wk else np.nan,
        "Economy": runs / (balls / 6) if balls else np.nan,
        "Strike rate": balls / wk if wk else np.nan,
        "WPI": wk / inn if inn else np.nan,
        "Best": best,
        "3W+": int((x["wickets"] >= 3).sum()), "5W+": int((x["wickets"] >= 5).sum()),
    }


def _bat_form(x, n):
    if len(x) < 3:
        return None
    c, r = _bat_stats(x), _bat_stats(x.tail(n))
    if pd.isna(c["SR"]) or pd.isna(r["SR"]) or not c["RPI"] or not c["SR"]:
        return None
    return 0.5 * (r["RPI"] / c["RPI"]) + 0.5 * (r["SR"] / c["SR"])


def _bowl_form(x, n):
    if len(x) < 3:
        return None
    c, r = _bowl_stats(x), _bowl_stats(x.tail(n))
    if not c["WPI"] or pd.isna(c["Economy"]) or pd.isna(r["Economy"]) or not r["Economy"]:
        return None
    return 0.5 * (r["WPI"] / c["WPI"]) + 0.5 * (c["Economy"] / r["Economy"])


def _label(idx):
    if idx is None:
        return "not enough data"
    if idx >= 1.2:
        return "🔥 In form"
    if idx >= 0.85:
        return "➖ Steady"
    return "❄️ Out of form"


# ------------------------------------------------------------------ charts
def _bat_charts(x, n, show_n):
    x = x.copy()
    x["roll_runs"] = x["runs"].rolling(n, min_periods=1).sum()
    x["roll_balls"] = x["balls"].rolling(n, min_periods=1).sum()
    x["roll_outs"] = x["dismissed"].astype(int).rolling(n, min_periods=1).sum()
    x["roll_avg"] = x["roll_runs"] / x["roll_outs"].where(x["roll_outs"] > 0)
    x["roll_sr"] = x["roll_runs"] / x["roll_balls"].where(x["roll_balls"] > 0) * 100
    v = x.tail(show_n).reset_index(drop=True)
    xs = np.arange(1, len(v) + 1)

    colors = np.where(v["runs"] >= 100, GREEN, np.where(v["runs"] >= 50, GOLD, BLUE)).tolist()
    text = [f"{int(r)}{'' if d else '*'}" for r, d in zip(v["runs"], v["dismissed"])]
    hover = (v["date"].dt.strftime("%Y-%m-%d").fillna("") + " vs " + v["opposition"].astype(str)
             + "<br>" + v["balls"].astype(int).astype(str) + " balls · " + v["venue"].astype(str))

    fig = go.Figure()
    fig.add_trace(go.Bar(x=xs, y=v["runs"], marker_color=colors, text=text, textposition="outside",
                         customdata=hover, hovertemplate="%{y} runs<br>%{customdata}<extra></extra>",
                         name="Runs"))
    fig.add_trace(go.Scatter(x=xs, y=v["roll_avg"], mode="lines", line=dict(color=ORANGE, width=3),
                             name=f"Average of last {n}"))
    fig.update_layout(template="plotly_dark", height=380, xaxis_title="Innings (oldest to newest)",
                      yaxis_title="Runs", title="Runs per innings (* = not out, gold = 50+, green = 100+)",
                      legend=dict(orientation="h", y=-0.25))

    fig2 = go.Figure(go.Scatter(x=xs, y=v["roll_sr"], mode="lines", line=dict(color=GREEN, width=3)))
    fig2.update_layout(template="plotly_dark", height=300, xaxis_title="Innings (oldest to newest)",
                       yaxis_title="Strike rate", title=f"Rolling strike rate (last {n} innings)")
    return fig, fig2


def _bowl_charts(x, n, show_n):
    x = x.copy()
    x["roll_runs"] = x["runs"].rolling(n, min_periods=1).sum()
    x["roll_balls"] = x["balls"].rolling(n, min_periods=1).sum()
    x["roll_wpi"] = x["wickets"].rolling(n, min_periods=1).mean()
    x["roll_econ"] = x["roll_runs"] / x["roll_balls"].where(x["roll_balls"] > 0) * 6
    v = x.tail(show_n).reset_index(drop=True)
    xs = np.arange(1, len(v) + 1)

    colors = np.where(v["wickets"] >= 5, GREEN, np.where(v["wickets"] >= 3, GOLD, BLUE)).tolist()
    hover = (v["date"].dt.strftime("%Y-%m-%d").fillna("") + " vs " + v["facing_team"].astype(str)
             + "<br>" + v["runs"].astype(int).astype(str) + " runs from " + v["balls"].astype(int).astype(str)
             + " balls")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=xs, y=v["wickets"], marker_color=colors, customdata=hover,
                         hovertemplate="%{y} wickets<br>%{customdata}<extra></extra>", name="Wickets"))
    fig.add_trace(go.Scatter(x=xs, y=v["roll_wpi"], mode="lines", line=dict(color=ORANGE, width=3),
                             name=f"Wickets per innings, last {n}"))
    fig.update_layout(template="plotly_dark", height=380, xaxis_title="Innings (oldest to newest)",
                      yaxis_title="Wickets", title="Wickets per innings (gold = 3+, green = 5+)",
                      legend=dict(orientation="h", y=-0.25))

    fig2 = go.Figure(go.Scatter(x=xs, y=v["roll_econ"], mode="lines", line=dict(color=RED, width=3)))
    fig2.update_layout(template="plotly_dark", height=300, xaxis_title="Innings (oldest to newest)",
                       yaxis_title="Economy", title=f"Rolling economy (last {n} innings, lower is better)")
    return fig, fig2


# ------------------------------------------------------------------ breakdowns
def _bat_table(x, by, min_inn=1, top=15, sort="Innings"):
    g = x.dropna(subset=[by]).groupby(by).agg(
        Innings=("runs", "size"), Runs=("runs", "sum"), Balls=("balls", "sum"),
        Outs=("dismissed", "sum"), Fifties=("runs", lambda s: int((s >= 50).sum())),
    )
    g = g[g["Innings"] >= min_inn]
    g["Avg"] = (g["Runs"] / g["Outs"].where(g["Outs"] > 0)).round(1)
    g["SR"] = (g["Runs"] / g["Balls"].where(g["Balls"] > 0) * 100).round(1)
    g = g.sort_values(sort, ascending=False).head(top)
    return g[["Innings", "Runs", "Avg", "SR", "Fifties"]].rename(columns={"Fifties": "50+"}).reset_index()


def _bowl_table(x, by, min_inn=1, top=15, sort="Innings"):
    g = x.dropna(subset=[by]).groupby(by).agg(
        Innings=("wickets", "size"), Wickets=("wickets", "sum"),
        Runs=("runs", "sum"), Balls=("balls", "sum"),
    )
    g = g[g["Innings"] >= min_inn]
    g["Econ"] = (g["Runs"] / g["Balls"].where(g["Balls"] > 0) * 6).round(2)
    g["Avg"] = (g["Runs"] / g["Wickets"].where(g["Wickets"] > 0)).round(1)
    g = g.sort_values(sort, ascending=False).head(top)
    return g[["Innings", "Wickets", "Econ", "Avg"]].reset_index()


# ------------------------------------------------------------------ watchlist
def _watch_row(item, bat, bowl, n):
    role, player, fmt = item["role"], item["player"], item["format"]
    df, col = (bat, "batsman") if role == "Batter" else (bowl, "bowler")
    x = df[df[col] == player]
    if fmt != "All formats":
        x = x[x["format"] == fmt]
    if x.empty:
        return None
    r = x.tail(n)
    if role == "Batter":
        s = _bat_stats(r)
        recent = f"{s['RPI']:.1f} runs/inn · SR {_f(s['SR'], 0)}"
        idx = _bat_form(x, n)
    else:
        s = _bowl_stats(r)
        recent = f"{s['WPI']:.2f} wkts/inn · econ {_f(s['Economy'], 2)}"
        idx = _bowl_form(x, n)
    return {"Player": player, "Role": role, "Competition": fmt, "Innings": len(x),
            f"Last {n}": recent, "Form index": round(idx, 2) if idx is not None else None,
            "Form": _label(idx)}


def _watchlist_section(bat, bowl, n, role, player, fmt):
    st.markdown("---")
    st.markdown("### ⭐ Watchlist")
    items = _load_watch()

    new = {"role": role, "player": player, "format": fmt}
    if st.button(f"➕ Add {player} ({fmt}) to the watchlist", key="pt_add"):
        if new not in items:
            items.append(new)
            _save_watch(items)
            st.success("Added.")
        else:
            st.info("Already on the watchlist.")

    rows = [r for r in (_watch_row(i, bat, bowl, n) for i in items) if r]
    if not rows:
        st.caption("Nothing on the watchlist yet. Pick a player above and add them.")
        return
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    names = [f"{i['player']} · {i['role']} · {i['format']}" for i in items]
    remove = st.multiselect("Remove from watchlist", names, key="pt_remove")
    if remove and st.button("Remove selected", key="pt_remove_btn"):
        keep = [i for i, nm in zip(items, names) if nm not in remove]
        _save_watch(keep)
        st.rerun()


# ------------------------------------------------------------------ main
def render_player_tracking_tab():
    st.subheader("📈 Player Performance Tracking")
    if not (DATA / "batting_innings.parquet").exists() or not (DATA / "bowling_innings.parquet").exists():
        st.info("Run `python src/build_match_tables.py` first to create the player tables.")
        return
    bat, bowl = _tables()

    c1, c2, c3, c4 = st.columns([1, 1, 2, 1])
    role = c1.radio("Role", ["Batter", "Bowler"], horizontal=True, key="pt_role")
    fmts = sorted(set(bat["format"]) | set(bowl["format"]), key=lambda f: (f != "IPL", f))
    fmt = c2.selectbox("Competition", ["All formats"] + fmts, key="pt_fmt")
    min_inn = c4.number_input("Min innings", 3, 200, 20, key="pt_min")

    df, col = (bat, "batsman") if role == "Batter" else (bowl, "bowler")
    pool = df if fmt == "All formats" else df[df["format"] == fmt]
    counts = pool[col].value_counts()
    names = list(counts[counts >= min_inn].index)
    if not names:
        st.info("No players meet the minimum number of innings. Lower 'Min innings'.")
        return
    player = c3.selectbox("Player (type to search)", names, key=f"pt_player_{role}_{fmt}")

    s1, s2 = st.columns(2)
    n = s1.slider("Recent form window (innings)", 5, 25, 10, key="pt_n")
    show_n = s2.slider("Innings shown in charts", 20, 100, 40, key="pt_show")

    x = pool[pool[col] == player].reset_index(drop=True)
    if x.empty:
        st.info("No data for this player.")
        return

    if role == "Batter":
        _render_batter(x, n, show_n)
    else:
        _render_bowler(x, n, show_n)

    _watchlist_section(bat, bowl, n, role, player, fmt)
    st.caption(
        "Form index = half the recent output compared with career (runs per innings, or wickets per innings) "
        "plus half the recent rate (strike rate, or economy). Above 1.2 = in form, below 0.85 = out of form. "
        "It is a simple rule of thumb and is noisy over few innings. Balls faced include wides and bowlers' "
        "runs include byes, so rates are slightly approximate."
    )


def _render_batter(x, n, show_n):
    c = _bat_stats(x)
    k = st.columns(4)
    k[0].metric("Innings", c["Innings"])
    k[1].metric("Runs", f"{c['Runs']:,}")
    k[2].metric("Average", _f(c["Average"]))
    k[3].metric("Strike rate", _f(c["SR"]))
    k = st.columns(4)
    k[0].metric("Highest score", c["HS"])
    k[1].metric("100s / 50s", f"{c['100s']} / {c['50s']}")
    k[2].metric("Fours / sixes", f"{c['4s']} / {c['6s']}")
    k[3].metric("Runs from boundaries", f"{_f(c['Boundary %'], 0)}%")

    last = x.tail(n)
    r = _bat_stats(last)
    idx = _bat_form(x, n)
    st.markdown(f"#### Recent form (last {len(last)} innings)")
    k = st.columns(4)
    k[0].metric("Runs per innings", _f(r["RPI"]), f"{r['RPI'] - c['RPI']:+.1f} vs career")
    k[1].metric("Strike rate", _f(r["SR"]), f"{r['SR'] - c['SR']:+.1f} vs career"
                if not pd.isna(r["SR"]) and not pd.isna(c["SR"]) else None)
    k[2].metric("50s / 100s", f"{int(((last['runs'] >= 50) & (last['runs'] < 100)).sum())} / {int((last['runs'] >= 100).sum())}")
    k[3].metric("Form index", _f(idx, 2), _label(idx), delta_color="off")

    fig, fig2 = _bat_charts(x, n, show_n)
    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Breakdowns")
    t1, t2, t3, t4 = st.tabs(["By year", "By opponent", "By venue", "By batting position"])
    with t1:
        y = x.copy()
        y["Year"] = y["year"].astype("Int64")
        st.dataframe(_bat_table(y, "Year", top=40, sort="Year"), hide_index=True, use_container_width=True)
    with t2:
        st.dataframe(_bat_table(x, "opposition", top=12), hide_index=True, use_container_width=True)
        st.caption("The 12 opponents faced most often.")
    with t3:
        st.dataframe(_bat_table(x, "venue", min_inn=3, top=12, sort="Runs"), hide_index=True,
                     use_container_width=True)
        st.caption("Venues with 3 or more innings, ordered by runs.")
    with t4:
        p = x.dropna(subset=["bat_order"]).copy()
        p["Position"] = pd.cut(p["bat_order"], bins=[0, 2, 3, 5, 7, 12],
                               labels=["Opener (1-2)", "No. 3", "No. 4-5", "No. 6-7", "No. 8-11"]).astype(object)
        st.dataframe(_bat_table(p, "Position", top=10), hide_index=True, use_container_width=True)
        st.caption("Position is estimated from the order in which batters first faced a ball.")


def _render_bowler(x, n, show_n):
    c = _bowl_stats(x)
    k = st.columns(4)
    k[0].metric("Innings bowled", c["Innings"])
    k[1].metric("Wickets", f"{c['Wickets']:,}")
    k[2].metric("Average", _f(c["Average"]))
    k[3].metric("Economy", _f(c["Economy"], 2))
    k = st.columns(4)
    k[0].metric("Strike rate (balls per wicket)", _f(c["Strike rate"]))
    k[1].metric("Best figures", c["Best"])
    k[2].metric("3-wicket hauls", c["3W+"])
    k[3].metric("5-wicket hauls", c["5W+"])

    last = x.tail(n)
    r = _bowl_stats(last)
    idx = _bowl_form(x, n)
    st.markdown(f"#### Recent form (last {len(last)} innings)")
    k = st.columns(4)
    k[0].metric("Wickets per innings", _f(r["WPI"], 2), f"{r['WPI'] - c['WPI']:+.2f} vs career")
    k[1].metric("Economy", _f(r["Economy"], 2),
                f"{r['Economy'] - c['Economy']:+.2f} vs career" if not pd.isna(r["Economy"]) else None,
                delta_color="inverse")
    k[2].metric("Wickets", int(last["wickets"].sum()))
    k[3].metric("Form index", _f(idx, 2), _label(idx), delta_color="off")

    fig, fig2 = _bowl_charts(x, n, show_n)
    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Breakdowns")
    t1, t2, t3 = st.tabs(["By year", "By opponent", "By venue"])
    with t1:
        y = x.copy()
        y["Year"] = y["year"].astype("Int64")
        st.dataframe(_bowl_table(y, "Year", top=40, sort="Year"), hide_index=True, use_container_width=True)
    with t2:
        st.dataframe(_bowl_table(x, "facing_team", top=12), hide_index=True, use_container_width=True)
        st.caption("The 12 teams bowled against most often.")
    with t3:
        st.dataframe(_bowl_table(x, "venue", min_inn=3, top=12, sort="Wickets"), hide_index=True,
                     use_container_width=True)
        st.caption("Venues with 3 or more innings, ordered by wickets.")
