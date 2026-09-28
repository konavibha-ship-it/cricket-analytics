import numpy as np
import plotly.graph_objects as go

try:
    from .physics import PITCH_LENGTH, STUMP_HEIGHT      # when imported by the dashboard
except ImportError:
    from physics import PITCH_LENGTH, STUMP_HEIGHT       # when run as a script (demo.py)

COLOURS = ["#ff4b4b", "#4bafff", "#ffb84b", "#6be675", "#b84bff", "#ffffff"]


def _rect(x0, x1, y0, y1, z, colour, opacity):
    return go.Mesh3d(
        x=[x0, x1, x1, x0], y=[y0, y0, y1, y1], z=[z] * 4,
        i=[0, 0], j=[1, 2], k=[2, 3],
        color=colour, opacity=opacity, hoverinfo="skip", showlegend=False,
    )


def _stumps(x):
    return [
        go.Scatter3d(x=[x, x], y=[y, y], z=[0, STUMP_HEIGHT], mode="lines",
                     line=dict(color="#f5deb3", width=6), hoverinfo="skip", showlegend=False)
        for y in (-0.095, 0.0, 0.095)
    ]


def _crease(x):
    return go.Scatter3d(x=[x, x], y=[-1.32, 1.32], z=[0.003, 0.003], mode="lines",
                        line=dict(color="white", width=3), hoverinfo="skip", showlegend=False)


def make_figure(results, labels=None):
    """3D view of one or more simulated deliveries (list of results from physics.deliver)."""
    if labels is None:
        labels = [f"Delivery {i + 1}" for i in range(len(results))]

    L = results[0]["inputs"]["release_to_stumps"]
    bowler_end = L - PITCH_LENGTH

    fig = go.Figure()
    fig.add_trace(_rect(bowler_end - 2, L + 2, -3.0, 3.0, -0.002, "#2d5a3d", 0.55))   # outfield
    fig.add_trace(_rect(bowler_end, L, -1.525, 1.525, 0.0, "#c8a96e", 0.9))            # pitch strip
    fig.add_trace(_crease(L - 1.22))
    fig.add_trace(_crease(bowler_end + 1.22))
    for tr in _stumps(L) + _stumps(bowler_end):
        fig.add_trace(tr)

    for idx, (res, label) in enumerate(zip(results, labels)):
        colour = COLOURS[idx % len(COLOURS)]
        pts = res["trajectory"]
        fig.add_trace(go.Scatter3d(
            x=pts[:, 1], y=pts[:, 2], z=pts[:, 3], mode="lines",
            line=dict(color=colour, width=6), name=label,
            hovertemplate=f"{label}<br>x=%{{x:.2f}} m<br>y=%{{y:.2f}} m<br>height=%{{z:.2f}} m<extra></extra>",
        ))
        fig.add_trace(go.Scatter3d(
            x=[pts[0, 1]], y=[pts[0, 2]], z=[pts[0, 3]], mode="markers",
            marker=dict(size=5, color=colour, symbol="circle"),
            hoverinfo="skip", showlegend=False))
        if res["bounce"]:
            b = res["bounce"]
            fig.add_trace(go.Scatter3d(
                x=[b["x"]], y=[b["y"]], z=[0.04], mode="markers",
                marker=dict(size=6, color=colour, symbol="diamond"),
                hovertemplate=f"{label} pitches: {res['length_label']}, {res['line_label']}<extra></extra>",
                showlegend=False))
        if res["crossing"]:
            c = res["crossing"]
            fig.add_trace(go.Scatter3d(
                x=[L], y=[c["y"]], z=[c["z"]], mode="markers",
                marker=dict(size=6, color="#6be675" if res["hits_stumps"] else "#ffffff", symbol="x"),
                hovertemplate=f"{label} at stumps: {'hits' if res['hits_stumps'] else 'misses'}<extra></extra>",
                showlegend=False))

    fig.update_layout(
        template="plotly_dark", height=560, margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", y=1.02),
        scene=dict(
            xaxis=dict(title="Along pitch (m)", range=[bowler_end - 2, L + 2]),
            yaxis=dict(title="Sideways (m), + = off side", range=[-2.5, 2.5]),
            zaxis=dict(title="Height (m)", range=[0, 3.0]),
            aspectmode="manual", aspectratio=dict(x=4.0, y=1.0, z=0.7),
            camera=dict(eye=dict(x=-1.9, y=-1.3, z=0.7)),
        ),
    )
    return fig