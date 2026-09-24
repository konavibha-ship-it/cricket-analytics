import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from strength_weakness import batsman_report, bowler_report
from pitch_visuals import over_by_over_batting, draw_dismissal_field_diagram, generate_bowling_plan

st.set_page_config(
    page_title="Cricket Analytics Dashboard",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM STYLING
# ============================================
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div[data-testid="stMetric"] {
        background-color: #1a1c24;
        border: 1px solid #2d2f3b;
        border-radius: 12px;
        padding: 15px 10px;
    }
    div[data-testid="stMetricLabel"] { color: #9ca3af; }
    div[data-testid="stMetricValue"] { color: #ff4b4b; font-weight: 700; }
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1c24;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# LOAD DATA
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'processed')

def safe_load_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None

def safe_load_model(path):
    return joblib.load(path) if os.path.exists(path) else None

matches = pd.read_parquet(os.path.join(DATA_DIR, 'matches_clean.parquet'))
deliveries = pd.read_parquet(os.path.join(DATA_DIR, 'deliveries_clean.parquet'))
batting_impact = safe_load_csv(os.path.join(DATA_DIR, 'batting_impact.csv'))
bowling_impact = safe_load_csv(os.path.join(DATA_DIR, 'bowling_impact.csv'))
matchups = safe_load_csv(os.path.join(DATA_DIR, 'matchups.csv'))
state_df = safe_load_csv(os.path.join(DATA_DIR, 'match_states.csv'))
player_clusters = safe_load_csv(os.path.join(DATA_DIR, 'player_clusters.csv'))
win_model = safe_load_model(os.path.join(DATA_DIR, 'win_prob_model.pkl'))

PLOTLY_TEMPLATE = "plotly_dark"
ACCENT = "#ff4b4b"

# ============================================
# SIDEBAR — FORMAT + SEASON FILTERS
# ============================================
with st.sidebar:
    st.title("🏏 Cricket Analytics")
    st.caption("International, IPL & domestic T20 leagues — 2001–2026")
    st.divider()

    available_formats = sorted(matches['format'].dropna().unique())
    format_filter = st.multiselect(
        "Filter by Competition",
        options=available_formats,
        default=['IPL']
    )

    season_filter = st.multiselect(
        "Filter by Year",
        options=sorted(matches['year'].dropna().unique()),
        default=[]
    )
    st.divider()
    st.caption("Built with Python, scikit-learn, SHAP, PuLP & Streamlit")

matches_view = matches.copy()
if format_filter:
    matches_view = matches_view[matches_view['format'].isin(format_filter)]
if season_filter:
    matches_view = matches_view[matches_view['year'].isin(season_filter)]

deliveries_view = deliveries[deliveries['match_id'].isin(matches_view['match_id'])]

# ============================================
# HEADER
# ============================================
st.title("🏏 Cricket Performance Analytics")
st.caption("Ball-by-ball data → SQL → ML → live win probability")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊  Overview", "🌟  Player Impact", "🆚  Matchups", "📈  Win Probability",
    "🧩  Player Archetypes", "🔍  Scouting Report", "🎯  Bowling Plan"
])

# ---------------- TAB 1: OVERVIEW ----------------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Matches", f"{len(matches_view):,}")
    c2.metric("Teams", matches_view['team1'].nunique())
    c3.metric("Seasons", matches_view['year'].nunique())
    c4.metric("Balls Analyzed", f"{len(deliveries_view):,}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Team Wins")
        team_wins = matches_view['winner'].value_counts().head(10).reset_index()
        team_wins.columns = ['team', 'wins']
        fig = px.bar(team_wins, x='wins', y='team', orientation='h',
                     template=PLOTLY_TEMPLATE, color='wins', color_continuous_scale='Reds')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Matches Per Season")
        season_counts = matches_view['year'].value_counts().sort_index().reset_index()
        season_counts.columns = ['year', 'matches']
        fig = px.line(season_counts, x='year', y='matches', markers=True, template=PLOTLY_TEMPLATE)
        fig.update_traces(line_color=ACCENT, line_width=3, marker_size=8)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Top Run Scorers")
        top_batsmen = deliveries_view.groupby('batsman')['batsman_runs'].sum().sort_values(ascending=False).head(10).reset_index()
        fig = px.bar(top_batsmen, x='batsman_runs', y='batsman', orientation='h',
                     template=PLOTLY_TEMPLATE, color='batsman_runs', color_continuous_scale='Oranges')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Top Wicket Takers")
        wkts = deliveries_view[deliveries_view['dismissal_kind'].notnull()]
        top_bowlers = wkts.groupby('bowler').size().sort_values(ascending=False).head(10).reset_index(name='wickets')
        fig = px.bar(top_bowlers, x='wickets', y='bowler', orientation='h',
                     template=PLOTLY_TEMPLATE, color='wickets', color_continuous_scale='Blues')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

# ---------------- TAB 2: PLAYER IMPACT ----------------
with tab2:
    st.caption("⚠️ Impact scores below are computed from the full historical dataset and are not affected by the sidebar filter.")
    if batting_impact is not None and bowling_impact is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🏏 Top 10 Batting Impact")
            fig = px.bar(batting_impact.head(10), x='batting_impact', y='batsman', orientation='h',
                         template=PLOTLY_TEMPLATE, color='batting_impact', color_continuous_scale='Sunset')
            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False, height=450)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("🎯 Top 10 Bowling Impact")
            fig = px.bar(bowling_impact.head(10), x='bowling_impact', y='bowler', orientation='h',
                         template=PLOTLY_TEMPLATE, color='bowling_impact', color_continuous_scale='Teal')
            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False, height=450)
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        c1, c2 = st.columns(2)
        c1.dataframe(batting_impact.head(25), use_container_width=True, height=400)
        c2.dataframe(bowling_impact.head(25), use_container_width=True, height=400)
    else:
        st.info("Run src/impact_score.py to generate this data.")

# ---------------- TAB 3: MATCHUPS ----------------
with tab3:
    st.caption("⚠️ Matchup data is computed from the full historical dataset and is not affected by the sidebar filter.")
    if matchups is not None:
        st.subheader("Batter vs Bowler Head-to-Head")
        col1, col2 = st.columns(2)
        selected_batter = col1.selectbox("Batter", sorted(matchups['batsman'].unique()))
        selected_bowler = col2.selectbox("Bowler", sorted(matchups['bowler'].unique()))

        result = matchups[(matchups['batsman'] == selected_batter) & (matchups['bowler'] == selected_bowler)]

        if not result.empty:
            r = result.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Balls Faced", int(r['balls_faced']))
            c2.metric("Runs Scored", int(r['runs_scored']))
            c3.metric("Strike Rate", f"{r['strike_rate']:.1f}")
            c4.metric("Dismissals", int(r['dismissals']))

            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=r['strike_rate'],
                title={'text': "Strike Rate in this Matchup"},
                gauge={'axis': {'range': [0, 200]}, 'bar': {'color': ACCENT}}
            ))
            fig.update_layout(template=PLOTLY_TEMPLATE, height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No matchup data (fewer than 6 balls faced)")

        st.divider()
        st.subheader("Most-Faced Matchups Overall")
        st.dataframe(matchups.sort_values('balls_faced', ascending=False).head(15), use_container_width=True)
    else:
        st.info("Run src/matchup_engine.py to generate this data.")

# ---------------- TAB 4: WIN PROBABILITY ----------------
with tab4:
    st.caption("⚠️ Live win probability is trained on T20-format data (IPL). Best used with the IPL filter selected.")
    if state_df is not None and win_model is not None:
        st.subheader("Live Win Probability — Ball by Ball")
        match_ids = state_df['match_id'].unique()
        selected_match = st.selectbox("Select a Match", match_ids)

        features = ['inning', 'current_score', 'wickets_fallen', 'balls_bowled',
                    'balls_remaining', 'target', 'runs_needed', 'required_run_rate', 'current_run_rate']

        match_data = state_df[
            (state_df['match_id'] == selected_match) & (state_df['inning'] == 2)
        ].sort_values('balls_bowled')

        if not match_data.empty:
            probs = win_model.predict_proba(match_data[features])[:, 1] * 100

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=match_data['balls_bowled'], y=probs, mode='lines',
                line=dict(color=ACCENT, width=3), fill='tozeroy',
                fillcolor='rgba(255,75,75,0.1)', name='Win Probability'
            ))
            fig.add_hline(y=50, line_dash="dash", line_color="gray")
            fig.update_layout(template=PLOTLY_TEMPLATE, height=450,
                               xaxis_title="Balls Bowled", yaxis_title="Win Probability (%)",
                               yaxis_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)

            current_prob = probs[-1] if len(probs) else 50
            st.metric("Current Win Probability", f"{current_prob:.1f}%")
        else:
            st.warning("No 2nd innings data for this match.")
    else:
        st.info("Run src/win_probability_model.py to generate this data.")

# ---------------- TAB 5: PLAYER ARCHETYPES ----------------
with tab5:
    st.caption("⚠️ Clustering is computed from the full historical dataset and is not affected by the sidebar filter.")
    if player_clusters is not None:
        st.subheader("Batter Archetypes (K-Means Clustering)")
        fig = px.scatter(
            player_clusters, x='strike_rate', y='death_sr', color='cluster',
            size='total_runs', hover_name='batsman', template=PLOTLY_TEMPLATE,
            color_continuous_scale='Viridis',
            labels={'strike_rate': 'Overall Strike Rate', 'death_sr': 'Death Overs Strike Rate'}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(player_clusters, use_container_width=True, height=400)
    else:
        st.info("Run src/player_clustering.py to generate this data.")

# ---------------- TAB 6: SCOUTING REPORT ----------------
with tab6:
    st.subheader("🔍 Player Scouting Report — Strengths & Weaknesses")
    st.caption("Filtered by the competitions selected in the sidebar.")

    report_type = st.radio("Analyze as:", ["Batsman", "Bowler"], horizontal=True)

    if report_type == "Batsman":
        player_list = sorted(deliveries_view['batsman'].unique())
        selected_player = st.selectbox("Select Batter", player_list, key="scout_batter")
        report = batsman_report(selected_player, verbose=False)

        if report:
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Runs", report['overall_runs'])
            c2.metric("Balls Faced", report['overall_balls'])
            c3.metric("Strike Rate", report['overall_sr'])

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**By Match Phase**")
                st.dataframe(report['phase'], use_container_width=True)
                if not report['phase'].empty:
                    best = report['phase']['strike_rate'].idxmax()
                    worst = report['phase']['strike_rate'].idxmin()
                    st.success(f"✅ Strength: {best} (SR {report['phase'].loc[best, 'strike_rate']})")
                    st.warning(f"⚠️ Weakness: {worst} (SR {report['phase'].loc[worst, 'strike_rate']})")
            with col2:
                st.markdown("**Dismissal Patterns**")
                if report['dismissals'] is not None and not report['dismissals'].empty:
                    fig = px.pie(values=report['dismissals'].values, names=report['dismissals'].index,
                                 template="plotly_dark", hole=0.4)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No dismissal data available.")

            st.markdown("**Top 5 Venues**")
            st.dataframe(report['venue'].head(5), use_container_width=True)
            st.markdown("**Vs Opposition Teams**")
            st.dataframe(report['opponent'], use_container_width=True)
        else:
            st.warning("No data found for this player.")
    else:
        player_list = sorted(deliveries_view['bowler'].unique())
        selected_player = st.selectbox("Select Bowler", player_list, key="scout_bowler")
        report = bowler_report(selected_player, verbose=False)

        if report:
            c1, c2 = st.columns(2)
            c1.metric("Total Wickets", report['overall_wickets'])
            c2.metric("Economy", report['overall_economy'])

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**By Match Phase**")
                st.dataframe(report['phase'], use_container_width=True)
                if not report['phase'].empty:
                    best = report['phase']['economy'].idxmin()
                    worst = report['phase']['economy'].idxmax()
                    st.success(f"✅ Strength: {best} (Econ {report['phase'].loc[best, 'economy']})")
                    st.warning(f"⚠️ Weakness: {worst} (Econ {report['phase'].loc[worst, 'economy']})")
            with col2:
                st.markdown("**Top 5 Venues**")
                st.dataframe(report['venue'].head(5), use_container_width=True)

            st.markdown("**Vs Opposition Teams**")
            st.dataframe(report['opponent'], use_container_width=True)
        else:
            st.warning("No data found for this player.")

# ---------------- TAB 7: BOWLING PLAN ----------------
with tab7:
    st.subheader("🎯 Bowling Plan & Field Visualization")
    st.caption("Built from real phase/venue/dismissal data — field zones are illustrative, not shot-tracking data. Filtered by sidebar selection.")

    player_list = sorted(deliveries_view['batsman'].unique())
    selected_player = st.selectbox("Select Batter to Plan Against", player_list, key="bowling_plan_player")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Over-by-Over Strike Rate**")
        over_data = over_by_over_batting(selected_player).reset_index()
        fig = px.bar(over_data, x='over', y='strike_rate', template="plotly_dark",
                     color='strike_rate', color_continuous_scale='Reds')
        fig.update_layout(height=350, xaxis_title="Over Number", yaxis_title="Strike Rate")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Dismissal Pattern Map**")
        fig = draw_dismissal_field_diagram(selected_player)
        if fig:
            st.pyplot(fig)
        else:
            st.info("Not enough dismissal data for this player.")

    st.divider()
    st.markdown(generate_bowling_plan(selected_player))