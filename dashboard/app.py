import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

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

matches = pd.read_csv(os.path.join(DATA_DIR, 'matches_clean.csv'))
deliveries = pd.read_csv(os.path.join(DATA_DIR, 'deliveries_clean.csv'))
batting_impact = safe_load_csv(os.path.join(DATA_DIR, 'batting_impact.csv'))
bowling_impact = safe_load_csv(os.path.join(DATA_DIR, 'bowling_impact.csv'))
matchups = safe_load_csv(os.path.join(DATA_DIR, 'matchups.csv'))
state_df = safe_load_csv(os.path.join(DATA_DIR, 'match_states.csv'))
player_clusters = safe_load_csv(os.path.join(DATA_DIR, 'player_clusters.csv'))
win_model = safe_load_model(os.path.join(DATA_DIR, 'win_prob_model.pkl'))

PLOTLY_TEMPLATE = "plotly_dark"
ACCENT = "#ff4b4b"

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.title("🏏 Cricket Analytics")
    st.caption("IPL ball-by-ball data analysis")
    st.divider()
    season_filter = st.multiselect(
        "Filter by Season",
        options=sorted(matches['year'].dropna().unique()),
        default=[]
    )
    st.divider()
    st.caption("Built with Python, scikit-learn, SHAP, PuLP & Streamlit")

matches_view = matches if not season_filter else matches[matches['year'].isin(season_filter)]

# ============================================
# HEADER
# ============================================
st.title("🏏 Cricket Performance Analytics")
st.caption("Ball-by-ball IPL data → SQL → ML → live win probability")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  Overview", "🌟  Player Impact", "🆚  Matchups", "📈  Win Probability", "🧩  Player Archetypes"
])

# ---------------- TAB 1: OVERVIEW ----------------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Matches", f"{len(matches_view):,}")
    c2.metric("Teams", matches_view['team1'].nunique())
    c3.metric("Seasons", matches_view['year'].nunique())
    c4.metric("Balls Analyzed", f"{len(deliveries):,}")

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
        fig = px.line(season_counts, x='year', y='matches', markers=True,
                       template=PLOTLY_TEMPLATE)
        fig.update_traces(line_color=ACCENT, line_width=3, marker_size=8)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Top Run Scorers")
        top_batsmen = deliveries.groupby('batsman')['batsman_runs'].sum().sort_values(ascending=False).head(10).reset_index()
        fig = px.bar(top_batsmen, x='batsman_runs', y='batsman', orientation='h',
                     template=PLOTLY_TEMPLATE, color='batsman_runs', color_continuous_scale='Oranges')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Top Wicket Takers")
        wkts = deliveries[deliveries['dismissal_kind'].notnull()]
        top_bowlers = wkts.groupby('bowler').size().sort_values(ascending=False).head(10).reset_index(name='wickets')
        fig = px.bar(top_bowlers, x='wickets', y='bowler', orientation='h',
                     template=PLOTLY_TEMPLATE, color='wickets', color_continuous_scale='Blues')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

# ---------------- TAB 2: PLAYER IMPACT ----------------
with tab2:
    if batting_impact is not None and bowling_impact is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🏏 Top 10 Batting Impact")
            top_bat = batting_impact.head(10)
            fig = px.bar(top_bat, x='batting_impact', y='batsman', orientation='h',
                         template=PLOTLY_TEMPLATE, color='batting_impact', color_continuous_scale='Sunset')
            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False, height=450)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("🎯 Top 10 Bowling Impact")
            top_bowl = bowling_impact.head(10)
            fig = px.bar(top_bowl, x='bowling_impact', y='bowler', orientation='h',
                         template=PLOTLY_TEMPLATE, color='bowling_impact', color_continuous_scale='Teal')
            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False, height=450)
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Full Leaderboards")
        c1, c2 = st.columns(2)
        c1.dataframe(batting_impact.head(25), use_container_width=True, height=400)
        c2.dataframe(bowling_impact.head(25), use_container_width=True, height=400)
    else:
        st.info("Run src/impact_score.py to generate this data.")

# ---------------- TAB 3: MATCHUPS ----------------
with tab3:
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
                mode="gauge+number",
                value=r['strike_rate'],
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
                x=match_data['balls_bowled'], y=probs,
                mode='lines', line=dict(color=ACCENT, width=3),
                fill='tozeroy', fillcolor='rgba(255,75,75,0.1)',
                name='Win Probability'
            ))
            fig.add_hline(y=50, line_dash="dash", line_color="gray")
            fig.update_layout(
                template=PLOTLY_TEMPLATE, height=450,
                xaxis_title="Balls Bowled", yaxis_title="Win Probability (%)",
                yaxis_range=[0, 100]
            )
            st.plotly_chart(fig, use_container_width=True)

            current_prob = probs[-1] if len(probs) else 50
            st.metric("Current Win Probability", f"{current_prob:.1f}%")
        else:
            st.warning("No 2nd innings data for this match.")
    else:
        st.info("Run src/win_probability_model.py to generate this data.")

# ---------------- TAB 5: PLAYER ARCHETYPES ----------------
with tab5:
    if player_clusters is not None:
        st.subheader("Batter Archetypes (K-Means Clustering)")
        fig = px.scatter(
            player_clusters, x='strike_rate', y='death_sr',
            color='cluster', size='total_runs', hover_name='batsman',
            template=PLOTLY_TEMPLATE, color_continuous_scale='Viridis',
            labels={'strike_rate': 'Overall Strike Rate', 'death_sr': 'Death Overs Strike Rate'}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(player_clusters, use_container_width=True, height=400)
    else:
        st.info("Run src/player_clustering.py to generate this data.")