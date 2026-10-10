"""Streamlit tab: Match Scorecard Analysis.

Used by dashboard/app.py:
    from scorecard_tab import render_scorecard_tab
    render_scorecard_tab()

What the data cannot tell us (so the cards are slightly approximate):
  * fielders are not recorded - catches show only the bowler
  * wides are not marked - balls faced include wides
  * byes and leg byes are counted in the bowler's runs
  * overs are estimated from the over and ball numbers
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dash_data import get_data
from win_engine import T20_FORMATS

BOWLER_WICKETS = {"bowled", "caught", "lbw", "stumped", "caught and bowled", "hit wicket"}
TEXT_COLS = ["batsman", "bowler", "player_dismissed", "dismissal_kind", "batting_team", "format"]
COLORS = ["#2ecc71", "#4aa3ff", "#f5a623", "#e74c3c"]
RED_BALL = ("Test", "County Championship")


# ------------------------------------------------------------------ helpers
def _ord(n):
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _overs(balls):
    return f"{balls // 6}.{balls % 6}"


def _legal_balls(di):
    per_over = di.groupby("over").size()
    return int(np.minimum(per_over, 6).sum())


def _how_out(kind, bowler):
    k = str(kind).lower()
    if k == "bowled":
        return f"b {bowler}"
    if k == "lbw":
        return f"lbw b {bowler}"
    if k == "caught":
        return f"caught b {bowler}"
    if k == "caught and bowled":
        return f"c & b {bowler}"
    if k == "stumped":
        return f"stumped b {bowler}"
    if k == "hit wicket":
        return f"hit wicket b {bowler}"
    return k


@st.cache_resource(show_spinner=False)
def _innings_totals():
    d, m = get_data()
    t = d.groupby(["match_id", "inning"], observed=True).agg(
        total=("total_runs", "sum"), fmt=("format", "first")
    ).reset_index()
    t["match_id"] = t["match_id"].astype(str)
    t["fmt"] = t["fmt"].astype(str)
    return t.merge(m[["match_id", "venue"]], on="match_id", how="left")


# ------------------------------------------------------------------ cards
def _batting_card(di):
    dis = {}
    for r in di[di["player_dismissed"].notna()].itertuples():
        dis.setdefault(r.player_dismissed, (r.dismissal_kind, r.bowler))

    d2 = di.assign(is4=di["batsman_runs"] == 4, is6=di["batsman_runs"] == 6)
    g = d2.groupby("batsman", sort=False).agg(
        R=("batsman_runs", "sum"), B=("batsman_runs", "size"),
        F=("is4", "sum"), S=("is6", "sum"),
    ).reset_index()

    rows = []
    for r in g.itertuples():
        out = _how_out(*dis[r.batsman]) if r.batsman in dis else "not out"
        rows.append({
            "Batter": r.batsman, "How out": out, "R": int(r.R), "B": int(r.B),
            "4s": int(r.F), "6s": int(r.S),
            "SR": round(r.R / r.B * 100, 1) if r.B else None,
        })
    seen = set(g["batsman"])
    for p, (k, b) in dis.items():
        if p not in seen:  # dismissed without facing a ball (e.g. run out at the non-striker's end)
            rows.append({"Batter": p, "How out": _how_out(k, b), "R": 0, "B": 0,
                         "4s": 0, "6s": 0, "SR": None})
    return pd.DataFrame(rows)


def _bowling_card(di):
    bw = di.assign(is_bw=di["dismissal_kind"].astype(str).str.lower().isin(BOWLER_WICKETS))
    rows = []
    for bowler, g in bw.groupby("bowler", sort=False):
        per_over = g.groupby("over").agg(cnt=("total_runs", "size"), runs=("total_runs", "sum"))
        balls = int(np.minimum(per_over["cnt"], 6).sum())
        maidens = int(((per_over["runs"] == 0) & (per_over["cnt"] >= 6)).sum())
        runs = int(g["total_runs"].sum())
        rows.append({
            "Bowler": bowler, "O": _overs(balls), "M": maidens, "R": runs,
            "W": int(g["is_bw"].sum()),
            "Econ": round(runs / (balls / 6), 2) if balls else None,
            "0s": int((g["total_runs"] == 0).sum()),
        })
    return pd.DataFrame(rows)


def _fall_of_wickets(di):
    di = di.reset_index(drop=True)
    cum = di["total_runs"].cumsum()
    out, n = [], 0
    for i in di.index[di["player_dismissed"].notna()]:
        n += 1
        r = di.loc[i]
        out.append(f"{n}-{int(cum[i])} ({r['player_dismissed']}, {int(r['over'])}.{int(r['ball'])})")
    return out


def _partnerships(di):
    di = di.reset_index(drop=True)
    wk_idx = list(di.index[di["player_dismissed"].notna()])
    bounds, start = [], 0
    for n, i in enumerate(wk_idx, 1):
        bounds.append((start, i, n))
        start = i + 1
    if start < len(di):
        bounds.append((start, len(di) - 1, None))

    rows = []
    for s, e, n in bounds:
        seg = di.iloc[s:e + 1]
        names = list(dict.fromkeys(seg["batsman"]))
        rows.append({
            "Partnership": f"{_ord(n)} wicket" if n else "Unbroken",
            "Batters": " & ".join(str(x) for x in names[:3]),
            "Runs (incl. extras)": int(seg["total_runs"].sum()),
            "Deliveries": len(seg),
        })
    return pd.DataFrame(rows)


def _innings_view(di, key):
    team = str(di["batting_team"].iloc[0])
    runs = int(di["total_runs"].sum())
    wk = int(di["player_dismissed"].notna().sum())
    balls = _legal_balls(di)
    extras = int((di["total_runs"] - di["batsman_runs"]).sum())
    rr = runs / max(balls, 1) * 6

    st.markdown(f"### {team}: {runs}/{wk} ({_overs(balls)} overs)")

    bat_runs = int(di["batsman_runs"].sum())
    bound_runs = int(di.loc[di["batsman_runs"].isin([4, 6]), "batsman_runs"].sum())
    k = st.columns(4)
    k[0].metric("Run rate", f"{rr:.2f}")
    k[1].metric("Runs from boundaries", f"{bound_runs / max(bat_runs, 1) * 100:.0f}%")
    k[2].metric("Dot balls", f"{(di['total_runs'] == 0).mean() * 100:.0f}%")
    k[3].metric("Extras", extras)

    st.markdown("**Batting**")
    st.dataframe(_batting_card(di), hide_index=True, use_container_width=True)
    st.caption(f"Extras {extras} · Total {runs}/{wk} ({_overs(balls)} overs)")

    st.markdown("**Bowling**")
    st.dataframe(_bowling_card(di), hide_index=True, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Fall of wickets**")
        fow = _fall_of_wickets(di)
        st.write(" · ".join(fow) if fow else "No wickets fell.")
    with c2:
        st.markdown("**Partnerships**")
        pt = _partnerships(di)
        st.dataframe(pt, hide_index=True, use_container_width=True)
        if not pt.empty:
            best = pt.loc[pt["Runs (incl. extras)"].idxmax()]
            st.caption(f"Biggest: {best['Batters']} added {best['Runs (incl. extras)']}")


# ------------------------------------------------------------------ charts
def _charts(d):
    worm = go.Figure()
    overs = go.Figure()
    for k, inn in enumerate(sorted(d["inning"].unique())):
        di = d[d["inning"] == inn]
        team = str(di["batting_team"].iloc[0])
        color = COLORS[k % len(COLORS)]

        x = di["over"].to_numpy() + np.minimum(di["ball"].to_numpy(), 6) / 6
        y = di["total_runs"].cumsum().to_numpy()
        worm.add_trace(go.Scatter(x=x, y=y, mode="lines", name=f"{team} (inn {inn})",
                                  line=dict(color=color, width=3)))
        w = di["player_dismissed"].notna().to_numpy()
        worm.add_trace(go.Scatter(
            x=x[w], y=y[w], mode="markers", showlegend=False,
            marker=dict(color=color, size=10, symbol="x"),
            text=di.loc[w, "player_dismissed"].astype(str) + " out",
            hovertemplate="%{text}<extra></extra>",
        ))

        po = di.groupby("over")["total_runs"].sum()
        overs.add_trace(go.Bar(x=po.index + 1, y=po.values, name=f"{team} (inn {inn})",
                               marker_color=color))

    worm.update_layout(template="plotly_dark", height=400, xaxis_title="Overs",
                       yaxis_title="Runs", title="Worm chart (x = wicket)")
    overs.update_layout(template="plotly_dark", height=340, barmode="group",
                        xaxis_title="Over", yaxis_title="Runs in the over",
                        title="Runs per over")
    return worm, overs


def _phase_table(d, fmt):
    if fmt == "ODI":
        cuts = (10, 40)
    elif fmt in T20_FORMATS:
        cuts = (6, 15)
    else:
        return None
    rows = []
    for inn in sorted(d["inning"].unique()):
        if inn > 2:
            continue
        di = d[d["inning"] == inn]
        over = di["over"].to_numpy()
        phase = np.where(over < cuts[0], "Powerplay", np.where(over < cuts[1], "Middle", "Death"))
        for name in ["Powerplay", "Middle", "Death"]:
            seg = di[phase == name]
            if seg.empty:
                continue
            ov = seg["over"].nunique()
            runs = int(seg["total_runs"].sum())
            rows.append({
                "Innings": int(inn), "Team": str(seg["batting_team"].iloc[0]), "Phase": name,
                "Overs": ov, "Runs": runs, "Wickets": int(seg["player_dismissed"].notna().sum()),
                "Run rate": round(runs / ov, 2),
            })
    return pd.DataFrame(rows)


def _context(d, fmt, venue):
    if fmt in RED_BALL:
        return None
    totals = _innings_totals()
    lines = []
    for inn in sorted(d["inning"].unique()):
        if inn > 2:
            continue
        di = d[d["inning"] == inn]
        total = int(di["total_runs"].sum())
        team = str(di["batting_team"].iloc[0])
        same = totals[(totals["fmt"] == fmt) & (totals["inning"] == inn)]
        if len(same) < 20:
            continue
        pct = (same["total"] < total).mean() * 100
        line = (f"**{team}** ({total}): higher than {pct:.0f}% of innings {inn} totals in {fmt} "
                f"(average {same['total'].mean():.0f}")
        at_venue = same[same["venue"] == venue]
        if len(at_venue) >= 5:
            line += f"; at this venue {at_venue['total'].mean():.0f} from {len(at_venue)} matches"
        lines.append(line + ")")
    return lines


# ------------------------------------------------------------------ main
def render_scorecard_tab():
    st.subheader("📋 Match Scorecard Analysis")
    d_all, m_all = get_data()

    formats = sorted(m_all["format"].astype(str).unique(), key=lambda f: (f != "IPL", f))
    c1, c2 = st.columns(2)
    fmt = c1.selectbox("Competition", formats, key="sc_fmt")
    q = c2.text_input("Search team or venue (optional)", key="sc_q").strip()

    x = m_all[m_all["format"].astype(str) == fmt].sort_values("date", ascending=False)
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
    mid = st.selectbox("Match", list(labels), format_func=lambda i: labels[i], key="sc_match")

    row = m_all[m_all["match_id"] == mid].iloc[0]
    d = d_all[d_all["match_id"] == mid].copy()
    if d.empty:
        st.warning("There is no ball-by-ball data for this match.")
        return
    for col in TEXT_COLS:
        d[col] = d[col].astype(object)
    d = d.sort_values(["inning", "over", "ball"], kind="mergesort").reset_index(drop=True)

    winner = row["winner"]
    st.markdown(f"**{row['team1']} vs {row['team2']}** · {row['venue']} · {str(row['date'])[:10]}")
    st.write(
        f"Toss: {row['toss_winner']} chose to {row['toss_decision']}. "
        f"Result: **{winner if pd.notna(winner) else 'no result'}**"
        f"{' won' if pd.notna(winner) and str(winner).lower() not in ('tie', 'no result', 'draw') else ''}."
    )

    innings = sorted(d["inning"].unique())
    labels_tabs = [
        f"Innings {i}: {d.loc[d['inning'] == i, 'batting_team'].iloc[0]}" for i in innings
    ]
    for tab, inn in zip(st.tabs(labels_tabs), innings):
        with tab:
            _innings_view(d[d["inning"] == inn], key=f"sc_{mid}_{inn}")

    st.markdown("#### Match charts")
    worm, overs = _charts(d)
    st.plotly_chart(worm, use_container_width=True)
    st.plotly_chart(overs, use_container_width=True)

    phases = _phase_table(d, str(row["format"]))
    if phases is not None and not phases.empty:
        st.markdown("#### Phase breakdown")
        st.dataframe(phases, hide_index=True, use_container_width=True)
        st.caption("T20: powerplay = overs 1-6, middle = 7-15, death = 16-20. "
                   "ODI: 1-10, 11-40, 41-50.")

    ctx = _context(d, str(row["format"]), row["venue"])
    if ctx:
        st.markdown("#### How the scores compare")
        for line in ctx:
            st.markdown("- " + line)

    st.caption(
        "Approximate where the data is thin: fielders are not recorded, balls faced include wides, "
        "bowlers' runs include byes and leg byes, and overs are estimated."
    )
