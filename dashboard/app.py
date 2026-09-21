import streamlit as st
import pandas as pd

st.set_page_config(page_title="Cricket Analytics Dashboard", layout="wide")

matches = pd.read_csv('data/processed/matches_clean.csv')
deliveries = pd.read_csv('data/processed/deliveries_clean.csv')

st.title("🏏 Cricket Performance Analytics Dashboard")

col1, col2, col3 = st.columns(3)
col1.metric("Total Matches", len(matches))
col2.metric("Total Teams", matches['team1'].nunique())
col3.metric("Total Seasons", matches['year'].nunique())

st.subheader("Team Wins")
team_wins = matches['winner'].value_counts()
st.bar_chart(team_wins)

st.subheader("Top Run Scorers")
top_batsmen = deliveries.groupby('batsman')['batsman_runs'].sum().sort_values(ascending=False).head(10)
st.bar_chart(top_batsmen)

st.subheader("Top Wicket Takers")
wickets = deliveries[deliveries['dismissal_kind'].notnull()]
top_bowlers = wickets.groupby('bowler').size().sort_values(ascending=False).head(10)
st.bar_chart(top_bowlers)

st.subheader("Filter by Team")
team_choice = st.selectbox("Select a team", sorted(matches['team1'].dropna().unique()))
team_matches = matches[(matches['team1'] == team_choice) | (matches['team2'] == team_choice)]
st.write(f"{team_choice} played {len(team_matches)} matches")
st.dataframe(team_matches[['date','team1','team2','winner','venue']].sort_values('date', ascending=False))