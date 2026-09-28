import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

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

def safe_load_parquet(path):
    return pd.read_parquet(path) if os.path.exists(path) else None

matches = pd.read_parquet(os.path.join(DATA_DIR, 'matches_clean.parquet'))
deliveries = pd.read_parquet(os.path.join(DATA_DIR, 'deliveries_clean.parquet'))
batting_impact = safe_load_csv(os.path.join(DATA_DIR, 'batting_impact.csv'))
bowling_impact = safe_load_csv(os.path.join(DATA_DIR, 'bowling_impact.csv'))
matchups = safe_load_csv(os.path.join(DATA_DIR, 'matchups.csv'))
state_df = safe_load_parquet(os.path.join(DATA_DIR, 'match_states.parquet'))
state_df_odi = safe_load_parquet(os.path.join(DATA_DIR, 'match_states_odi.parquet'))
win_model_odi = safe_load_model(os.path.join(DATA_DIR, 'win_prob_model_odi.pkl'))

player_clusters = safe_load_csv(os.path.join(DATA_DIR, 'player_clusters.csv'))
win_model = safe_load_model(os.path.join(DATA_DIR, 'win_prob_model.pkl'))
# ============================================
# CACHED WRAPPERS FOR EXPENSIVE COMPUTATIONS
# ============================================
@st.cache_data(show_spinner="Loading venue intelligence...")
def cached_venue_comparison(format_filter):
    return compare_all_venues(min_matches=15, format_filter=format_filter)

@st.cache_data(show_spinner="Computing form rating...")
def cached_form_rating(player, role):
    return calculate_form_rating(player, role=role)

@st.cache_data(show_spinner="Computing expected metrics...")
def cached_xmetrics(player, role):
    return calculate_xruns_xwickets(player, role=role)

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

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📊  Overview", "🌟  Player Impact", "🆚  Matchups", "📈  Win Probability",
    "🧩  Player Archetypes", "🔍  Scouting Report", "🎯  Bowling Plan",
    "📈  Advanced Analytics", "🧠  Match Intelligence"
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
    st.subheader("Live Win Probability — Ball by Ball")

    model_format = st.radio("Model:", ["T20 (IPL)", "ODI"], horizontal=True)

    if model_format == "T20 (IPL)":
        active_state_df = state_df
        active_model = win_model
        model_note = "Trained on IPL data — 72.5% accuracy, 0.82 ROC-AUC"
    else:
        active_state_df = state_df_odi
        active_model = win_model_odi
        model_note = "Trained on ODI data — 74.6% accuracy, 0.84 ROC-AUC"

    st.caption(model_note)

    if active_state_df is not None and active_model is not None:
        match_ids = active_state_df['match_id'].unique()
        selected_match = st.selectbox("Select a Match", match_ids, key=f"match_select_{model_format}")

        features = ['inning', 'current_score', 'wickets_fallen', 'balls_bowled',
                    'balls_remaining', 'target', 'runs_needed', 'required_run_rate', 'current_run_rate']

        match_data = active_state_df[
            (active_state_df['match_id'] == selected_match) & (active_state_df['inning'] == 2)
        ].sort_values('balls_bowled')

        if not match_data.empty:
            probs = active_model.predict_proba(match_data[features])[:, 1] * 100

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
            st.divider()
            st.subheader("🔮 What-If Predictor — Enter Any Live Scenario")
            st.caption("Type in a match situation and get an instant prediction from the model.")

        else:
            st.warning("No 2nd innings data for this match.")

            st.divider()
    st.subheader("🔮 What-If Predictor — Enter Any Live Scenario")
    st.caption("Type in a match situation and get an instant prediction from the model.")

    wc1, wc2, wc3 = st.columns(3)
    with wc1:
        wi_score = st.number_input("Current Score", min_value=0, max_value=500, value=100, key="wi_score")
        wi_wickets = st.number_input("Wickets Fallen", min_value=0, max_value=10, value=3, key="wi_wickets")
    with wc2:
        max_balls = 120 if model_format == "T20 (IPL)" else 300
        wi_balls_bowled = st.number_input("Balls Bowled", min_value=1, max_value=max_balls, value=60, key="wi_balls")
        wi_target = st.number_input("Target (0 if 1st innings)", min_value=0, max_value=500, value=180, key="wi_target")
    with wc3:
        wi_inning = st.selectbox("Innings", [1, 2], index=1, key="wi_inning")

    if st.button("Predict Win Probability", key=f"predict_btn_{model_format}"):
        balls_remaining = max(max_balls - wi_balls_bowled, 0)
        overs_completed = wi_balls_bowled / 6
        current_run_rate = wi_score / overs_completed if overs_completed > 0 else 0

        if wi_inning == 2 and wi_target > 0:
            runs_needed = wi_target - wi_score
            required_run_rate = (runs_needed / (balls_remaining / 6)) if balls_remaining > 0 else 0
        else:
            runs_needed = 0
            required_run_rate = 0
            wi_target = 0

        input_row = pd.DataFrame([{
            'inning': wi_inning,
            'current_score': wi_score,
            'wickets_fallen': wi_wickets,
            'balls_bowled': wi_balls_bowled,
            'balls_remaining': balls_remaining,
            'target': wi_target,
            'runs_needed': runs_needed,
            'required_run_rate': required_run_rate,
            'current_run_rate': current_run_rate
        }])

        prediction = active_model.predict_proba(input_row[features])[0][1] * 100

        st.metric("Predicted Win Probability", f"{prediction:.1f}%")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            title={'text': "Win Probability"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': ACCENT},
                'steps': [
                    {'range': [0, 50], 'color': '#2d2f3b'},
                    {'range': [50, 100], 'color': '#3a3d4d'}
                ],
                'threshold': {'line': {'color': "white", 'width': 3}, 'thickness': 0.8, 'value': 50}
            }
        ))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=300)
        st.plotly_chart(fig, use_container_width=True)    
    else:
        st.info(f"Model data not found for {model_format}. Run the corresponding training script first.")
        

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
    from strength_weakness import batsman_report, bowler_report
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
    from pitch_visuals import over_by_over_batting, draw_dismissal_field_diagram, generate_bowling_plan
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
    # ---------------- TAB 8: ADVANCED ANALYTICS ----------------
with tab8:
    from form_rating import calculate_form_rating
    from expected_metrics import calculate_xruns_xwickets
    st.subheader("📈 Player Form & Expected Performance")
    st.caption("Form rating uses recency-weighted recent innings. xRuns/xWickets compare actual output against situational expectations (like xG in football).")

    adv_role = st.radio("Role:", ["Batsman", "Bowler"], horizontal=True, key="adv_role")
    role_key = 'batsman' if adv_role == "Batsman" else 'bowler'

    player_pool = sorted(deliveries['batsman'].unique()) if role_key == 'batsman' else sorted(deliveries['bowler'].unique())
    adv_player = st.selectbox("Select Player", player_pool, key="adv_player")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🔥 Form Rating**")
        form = cached_form_rating(adv_player, role_key)
        if form:
            st.metric("Form Trend", form['form_trend'])
            c1, c2 = st.columns(2)
            c1.metric("Career Avg", form['career_avg_metric'])
            c2.metric("Recent (weighted)", form['weighted_recent_metric'])
            st.caption(f"Based on last {form['innings_considered']} of {form['total_career_innings']} career innings")
        else:
            st.info("Not enough career innings for a reliable form rating.")

    with col2:
        st.markdown("**⚡ Expected vs Actual (xRuns / xWickets)**")
        xmetrics = cached_xmetrics(adv_player, role_key)
        if xmetrics:
            if role_key == 'batsman':
                st.metric("Runs Above Expected", xmetrics['runs_above_expected'])
                st.metric("Dismissals vs Expected", xmetrics['dismissals_vs_expected'],
                           help="Negative means fewer dismissals than expected — a good sign")
            else:
                st.metric("Wickets Above Expected", xmetrics['wickets_above_expected'])
                st.metric("Runs Saved vs Expected", xmetrics['runs_saved_vs_expected'])
        else:
            st.info("Not enough situational data for this player.")

# ---------------- TAB 9: MATCH INTELLIGENCE ----------------
with tab9:
    from venue_intelligence import venue_profile, compare_all_venues
    from par_score_engine import get_par_score
    from captaincy_auditor import audit_toss_decision, audit_bowling_change_timing
    from team_matchup_matrix import build_team_matchup_matrix, find_key_matchups
    st.subheader("🧠 Venue Intelligence, Par Scores & Captaincy Audit")

    mi_format = st.selectbox("Format", ['IPL', 'T20I', 'ODI'], key="mi_format")

    st.markdown("**🏟️ Venue Rankings**")
    venue_table = cached_venue_comparison(mi_format)
    st.dataframe(venue_table, use_container_width=True, height=300)

    st.divider()

    mi_venue = st.selectbox("Select a Venue for Detailed Audit", venue_table['venue'].tolist(), key="mi_venue")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📊 Par Score**")
        par = get_par_score(mi_venue, format_filter=mi_format)
        if par:
            st.metric("Full Innings Par Score", par['par_score'])
            st.caption(f"Based on {par['sample_size']} historical matches")

    with col2:
        st.markdown("**🎯 Toss Decision Audit**")
        audit = audit_toss_decision(mi_venue, format_filter=mi_format)
        if audit and audit['data_backed_recommendation']:
            st.metric("Data-Backed Recommendation", audit['data_backed_recommendation'])
            st.caption(f"Bat first: {audit['bat_first_win_rate_pct']}% win rate ({audit['bat_first_sample']} matches) | "
                       f"Chase: {audit['chase_win_rate_pct']}% win rate ({audit['chase_sample']} matches)")

    st.divider()
    st.markdown("**🆚 Team-vs-Team Batter/Bowler Matchup Matrix**")
    all_teams = sorted(matches[matches['format'] == mi_format]['team1'].dropna().unique())

    tc1, tc2 = st.columns(2)
    team_a = tc1.selectbox("Batting Team", all_teams, key="mi_team_a")
    team_b = tc2.selectbox("Bowling Team", all_teams, key="mi_team_b", index=1 if len(all_teams) > 1 else 0)

    if team_a != team_b:
        pivot, matrix_data = build_team_matchup_matrix(team_a, team_b, format_filter=mi_format, min_balls=10)
        if pivot is not None:
            favorable_batter, favorable_bowler = find_key_matchups(matrix_data)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"*Best matchups for {team_a} batters*")
                st.dataframe(favorable_batter[['batsman', 'bowler', 'balls', 'strike_rate']], use_container_width=True)
            with c2:
                st.markdown(f"*Best matchups for {team_b} bowlers*")
                st.dataframe(favorable_bowler[['batsman', 'bowler', 'balls', 'strike_rate']], use_container_width=True)
        else:
            st.info("Not enough head-to-head data for this pairing.")
    else:
        st.warning("Select two different teams.")
        # ---------------- BALL TRAJECTORY POPUP (separate add-on) ----------------
try:
    sys.path.append(os.path.join(BASE_DIR, '..'))
    from ball_trajectory.dashboard_popup import render_trajectory_popup
    render_trajectory_popup()
except Exception as e:
    st.sidebar.caption(f"Ball trajectory add-on unavailable: {e}")
