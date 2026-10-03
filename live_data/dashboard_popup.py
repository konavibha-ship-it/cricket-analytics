"""
Floating "Live Player Lookup" button + popup.
Fully separate from the Cricsheet dataset — talks only to the Cricbuzz API.
"""
import pandas as pd
import streamlit as st

from live_data.cricbuzz_client import (
    search_player, get_player_batting_stats,
    get_player_bowling_stats, get_live_matches, CricbuzzUnavailable
)

_FAB_CSS = """
<style>
.st-key-live_fab {
    position: fixed;
    bottom: 90px;
    right: 24px;
    z-index: 999;
    width: auto !important;
}
.st-key-live_fab button {
    border-radius: 999px;
    padding: 0.6rem 1.1rem;
    background: #1e88e5;
    color: white;
    border: none;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.45);
    font-weight: 600;
}
.st-key-live_fab button:hover { background: #42a5f5; color: white; }
</style>
"""


def _stats_table(stats_json):
    """Cricbuzz returns {'headers': [...], 'values': [{'values': [...]}]} — turn it into a DataFrame."""
    headers = stats_json.get("headers", [])
    rows = [v["values"] for v in stats_json.get("values", [])]
    if not headers or not rows:
        return None
    df = pd.DataFrame(rows, columns=headers)
    return df.set_index(headers[0])


@st.dialog("🏏 Live Player & Match Lookup (Cricbuzz)", width="large")
def _live_dialog():
    st.caption("Live data from the Cricbuzz API — separate from this dashboard's historical dataset.")

    tab_a, tab_b = st.tabs(["Player Search", "Live Matches"])

    with tab_a:
        name = st.text_input("Search player name", value="Kohli", key="live_player_search")
        if name:
            try:
                results = search_player(name)
            except CricbuzzUnavailable as e:
                st.error(f"Cricbuzz API unavailable: {e}")
                results = []

            if results:
                options = {f"{r['name']} ({r.get('teamName', '?')})": r['id'] for r in results}
                choice = st.selectbox("Select player", list(options.keys()), key="live_player_choice")
                player_id = options[choice]

                try:
                    batting = _stats_table(get_player_batting_stats(player_id))
                    bowling = _stats_table(get_player_bowling_stats(player_id))
                except CricbuzzUnavailable as e:
                    st.error(f"Cricbuzz API unavailable: {e}")
                    batting = bowling = None

                if batting is not None:
                    st.markdown("**Batting — by format**")
                    st.dataframe(batting, use_container_width=True)
                if bowling is not None and not bowling.empty:
                    st.markdown("**Bowling — by format**")
                    st.dataframe(bowling, use_container_width=True)
            elif name:
                st.info("No players found for that name.")

    with tab_b:
        try:
            live = get_live_matches()
        except CricbuzzUnavailable as e:
            st.error(f"Cricbuzz API unavailable: {e}")
            live = None

        if live:
            for match_type in live.get("typeMatches", []):
                for series in match_type.get("seriesMatches", []):
                    wrapper = series.get("seriesAdWrapper")
                    if not wrapper:
                        continue
                    for m in wrapper.get("matches", []):
                        info = m.get("matchInfo", {})
                        score = m.get("matchScore", {})
                        t1 = info.get("team1", {}).get("teamSName", "?")
                        t2 = info.get("team2", {}).get("teamSName", "?")
                        st.markdown(f"**{t1} vs {t2}** — {info.get('matchDesc', '')} ({info.get('matchFormat', '')})")
                        st.caption(info.get("status", ""))
                        st.divider()


@st.fragment
def render_live_popup():
    st.markdown(_FAB_CSS, unsafe_allow_html=True)
    with st.container(key="live_fab"):
        if st.button("📡 Live Player Lookup", key="live_open"):
            _live_dialog()