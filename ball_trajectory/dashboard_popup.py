"""
Floating "Ball Trajectory" button + popup for the dashboard.

Completely separate from the cricket dataset: it never reads any data file.
It only uses physics.py and viz3d.py from this folder.
"""
import streamlit as st

_FAB_CSS = """
<style>
.st-key-traj_fab {
    position: fixed;
    bottom: 24px;
    right: 24px;
    z-index: 999;
    width: auto !important;
}
.st-key-traj_fab button {
    border-radius: 999px;
    padding: 0.6rem 1.1rem;
    background: #ff4b4b;
    color: white;
    border: none;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.45);
    font-weight: 600;
}
.st-key-traj_fab button:hover { background: #ff6b6b; color: white; }
</style>
"""


@st.dialog("🏏 Ball Trajectory Simulator (3D)", width="large")
def _trajectory_dialog():
    from ball_trajectory.physics import BallPhysics, deliver, summarise
    from ball_trajectory.viz3d import make_figure

    st.caption("Physics simulation driven by the values you set. It is not tracking data "
               "and is not connected to the match dataset.")

    c1, c2, c3 = st.columns(3)
    speed = c1.slider("Speed (km/h)", 90, 160, 140, key="traj_speed")
    length = c2.slider("Length (m from batter's stumps)", 0.0, 14.0, 7.0, 0.5, key="traj_length")
    line = c3.slider("Line (m sideways, + = off side)", -0.6, 1.0, 0.2, 0.05, key="traj_line")

    c4, c5, c6 = st.columns(3)
    swing = c4.slider("Swing (m/s², + = towards off side)", -2.5, 2.5, 0.0, 0.1, key="traj_swing")
    height = c5.slider("Release height (m)", 1.6, 2.5, 2.2, 0.05, key="traj_height")
    turn = c6.slider("Turn off the pitch (°, + = towards off side)", -8.0, 8.0, 0.0, 0.5, key="traj_turn")

    physics = BallPhysics(swing_accel=swing, turn_deg=turn)
    result = deliver(speed_kmh=speed, length_m=length, line_y=line,
                     release_height=height, physics=physics)

    err = result["solver_error_m"]
    if err is not None and err > 0.1:
        st.warning("That speed and length can't be reached from this release height, "
                   "so this shows the closest delivery.")

    st.plotly_chart(make_figure([result], [f"{speed} km/h"]), width="stretch")

    s = summarise(result)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Length", s["Length"])
    m2.metric("Line", s["Line"])
    m3.metric("Height at stumps (m)", s["Height at stumps (m)"])
    m4.metric("Hits stumps", s["Hits stumps"])
    st.caption(f"Speed at bounce {s['Speed at bounce (km/h)']} km/h · "
               f"at stumps {s['Speed at stumps (km/h)']} km/h · "
               f"flight time {s['Flight time (s)']} s. "
               "Height and width are exaggerated in the 3D view for visibility.")


@st.fragment
def render_trajectory_popup():
    """Draws the floating button. Opening the popup does not rerun the rest of the dashboard."""
    st.markdown(_FAB_CSS, unsafe_allow_html=True)
    with st.container(key="traj_fab"):
        if st.button("🏏 Ball Trajectory 3D", key="traj_open"):
            _trajectory_dialog()