import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

st.set_page_config(page_title="Cricket Analytics Dashboard", layout="wide")

# ============================================
# LOAD DATA
# ============================================
# Get the folder this script lives in, then go up one level to the project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'processed')

matches = pd.read_csv(os.path.join(DATA_DIR, 'matches_clean.csv'))
deliveries = pd.read_csv(os.path.join(DATA_DIR, 'deliveries_clean.csv'))

# Advanced files (loaded safely — app still works if these haven't been generated)
def safe_load_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None

def safe_load_model(path):
    return joblib.load(path) if os.path.exists(path) else None

batting_impact = safe_load_csv(os.path.join(DATA_DIR, 'batting_impact.csv'))
bowling_impact = safe_load_csv(os.path.join(DATA_DIR, 'bowling_impact.csv'))
matchups = safe_load_csv(os.path.join(DATA_DIR, 'matchups.csv'))
state_df = safe_load_csv(os.path.join(DATA_DIR, 'match_states.csv'))
win_model = safe_load_model(os.path.join(DATA_DIR, 'win_prob_model.pkl'))

st.title("🏏 Cricket Performance Analytics Dashboard")

# ============================================
# TABS
# ============================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview", "🌟 Player Impact", "🆚 Matchups", "📈 Live Win Probability"
])

# ---------------- TAB 1: OVERVIEW ----------------
with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Matches", len(matches))
    col2.metric("Total Teams", matches['team1'].nunique())
    col3.metric("Total Seasons", matches['year'].nunique())

    st.subheader("Team Wins")
    st.bar_chart(matches['winner'].value_counts())

    st.subheader("Top Run Scorers")
    top_batsmen = deliveries.groupby('batsman')['batsman_runs'].sum().sort_values(ascending=False).head(10)
    st.bar_chart(top_batsmen)

    st.subheader("Top Wicket Takers")
    wickets = deliveries[deliveries['dismissal_kind'].notnull()]
    top_bowlers = wickets.groupby('bowler').size().sort_values(ascending=False).head(10)
    st.bar_chart(top_bowlers)

# ---------------- TAB 2: PLAYER IMPACT ----------------
with tab2:
    st.subheader("Batting Impact Score (custom weighted metric)")
    if batting_impact is not None:
        st.dataframe(batting_impact.head(20), use_container_width=True)
        st.bar_chart(batting_impact.set_index('batsman')['batting_impact'].head(10))
    else:
        st.info("Run src/impact_score.py first to generate this data.")

    st.subheader("Bowling Impact Score (custom weighted metric)")
    if bowling_impact is not None:
        st.dataframe(bowling_impact.head(20), use_container_width=True)
        st.bar_chart(bowling_impact.set_index('bowler')['bowling_impact'].head(10))
    else:
        st.info("Run src/impact_score.py first to generate this data.")

# ---------------- TAB 3: MATCHUPS ----------------
with tab3:
    st.subheader("Batter vs Bowler Head-to-Head")
    if matchups is not None:
        batter_list = sorted(matchups['batsman'].unique())
        bowler_list = sorted(matchups['bowler'].unique())

        col1, col2 = st.columns(2)
        selected_batter = col1.selectbox("Select Batter", batter_list)
        selected_bowler = col2.selectbox("Select Bowler", bowler_list)

        result = matchups[
            (matchups['batsman'] == selected_batter) &
            (matchups['bowler'] == selected_bowler)
        ]

        if not result.empty:
            r = result.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Balls Faced", int(r['balls_faced']))
            c2.metric("Runs Scored", int(r['runs_scored']))
            c3.metric("Strike Rate", f"{r['strike_rate']:.1f}")
            c4.metric("Dismissals", int(r['dismissals']))
        else:
            st.warning("No matchup data for this pair (fewer than 6 balls faced)")

        st.subheader("Most-Faced Matchups Overall")
        st.dataframe(matchups.sort_values('balls_faced', ascending=False).head(15), use_container_width=True)
    else:
        st.info("Run src/matchup_engine.py first to generate this data.")

# ---------------- TAB 4: LIVE WIN PROBABILITY ----------------
with tab4:
    st.subheader("Live Win Probability — Example Match")
    if state_df is not None and win_model is not None:
        match_ids = state_df['match_id'].unique()
        selected_match = st.selectbox("Select a Match ID", match_ids)

        features = ['inning', 'current_score', 'wickets_fallen', 'balls_bowled',
                    'balls_remaining', 'target', 'runs_needed', 'required_run_rate', 'current_run_rate']

        match_data = state_df[
            (state_df['match_id'] == selected_match) & (state_df['inning'] == 2)
        ].sort_values('balls_bowled')

        if not match_data.empty:
            probs = win_model.predict_proba(match_data[features])[:, 1] * 100

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(match_data['balls_bowled'], probs, color='crimson', linewidth=2)
            ax.axhline(50, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel('Balls Bowled')
            ax.set_ylabel('Win Probability (%)')
            ax.set_title(f'Win Probability — Match {selected_match} (2nd Innings, Batting Team)')
            ax.set_ylim(0, 100)
            st.pyplot(fig)
        else:
            st.warning("This match has no 2nd innings data (e.g. no result / abandoned).")
    else:
        st.info("Run src/win_probability_model.py first to generate this data.")