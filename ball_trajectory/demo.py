import os

from physics import BallPhysics, deliver, summarise
from viz3d import make_figure

deliveries = {
    "Yorker, 145 km/h": dict(speed_kmh=145, length_m=1.0, line_y=0.0),
    "Good length outswinger, 138 km/h": dict(speed_kmh=138, length_m=7.0, line_y=0.25,
                                             physics=BallPhysics(swing_accel=1.2)),
    "Bouncer, 142 km/h": dict(speed_kmh=142, length_m=11.5, line_y=0.1),
}

results = []
for label, kwargs in deliveries.items():
    res = deliver(**kwargs)
    results.append(res)
    print(f"\n{label}")
    for key, value in summarise(res).items():
        print(f"  {key}: {value}")
    print(f"  (solver error: {res['solver_error_m']} m)")

fig = make_figure(results, list(deliveries.keys()))
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_output.html")
fig.write_html(out_path)
print(f"\nSaved 3D view to: {out_path}")