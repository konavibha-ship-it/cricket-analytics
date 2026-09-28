"""
Cricket ball flight simulator (physics model, 3D).

Coordinates (metres)
  x : along the pitch. 0 = release point, increasing toward the batter's stumps
  y : sideways. 0 = middle-stump line, positive = off side (right-hand batter)
  z : height above the ground

IMPORTANT: this is a SIMULATION driven by the parameters you set.
It is NOT learned from real delivery-tracking (Hawk-Eye) data.
"""
import math
from dataclasses import dataclass

import numpy as np

G = 9.81
AIR_DENSITY = 1.2
BALL_MASS = 0.156            # kg
BALL_RADIUS = 0.0359         # m
BALL_AREA = math.pi * BALL_RADIUS ** 2

PITCH_LENGTH = 20.12         # stump to stump, metres
STUMP_HALF_WIDTH = 0.1143
STUMP_HEIGHT = 0.711

KMH_TO_MS = 1.0 / 3.6

# Approximate bands, pace-bowling oriented: (upper limit in metres from batter's stumps, label).
# Sources differ on exact cut-offs, so edit these freely.
LENGTH_BANDS = [
    (2.0, "Yorker"),
    (6.0, "Full"),
    (8.0, "Good length"),
    (10.0, "Short of a length"),
    (float("inf"), "Short"),
]

# (upper limit of sideways position in metres, label)
LINE_BANDS = [
    (-STUMP_HALF_WIDTH, "Down the leg side"),
    (STUMP_HALF_WIDTH, "On the stumps"),
    (0.35, "Off-stump channel"),
    (0.70, "Outside off"),
    (float("inf"), "Wide outside off"),
]


@dataclass
class BallPhysics:
    drag_coeff: float = 0.40          # air drag
    swing_accel: float = 0.0          # sideways m/s^2 while in the air (+ = towards off side)
    restitution: float = 0.55         # share of vertical speed kept after the bounce
    pace_retention: float = 0.78      # share of horizontal speed kept after the bounce
    turn_deg: float = 0.0             # sideways deflection at the bounce (+ = towards off side)
    release_to_stumps: float = 18.5   # metres from release point to the batter's stumps


def classify_length(distance_from_stumps):
    if distance_from_stumps is None:
        return "Full toss"
    for limit, label in LENGTH_BANDS:
        if distance_from_stumps < limit:
            return label
    return LENGTH_BANDS[-1][1]


def classify_line(y):
    for limit, label in LINE_BANDS:
        if y < limit:
            return label
    return LINE_BANDS[-1][1]


def _fly(speed_kmh, release_height, release_y, theta_deg, phi_deg, p, dt, record):
    """Integrate one delivery. theta = angle below horizontal, phi = sideways angle (+ = off side)."""
    v = speed_kmh * KMH_TO_MS
    th = math.radians(theta_deg)
    ph = math.radians(phi_deg)
    vx = v * math.cos(th) * math.cos(ph)
    vy = v * math.cos(th) * math.sin(ph)
    vz = -v * math.sin(th)

    x, y, z = 0.0, release_y, release_height
    t = 0.0
    k = 0.5 * AIR_DENSITY * p.drag_coeff * BALL_AREA / BALL_MASS
    L = p.release_to_stumps

    bounced = False
    bounce = None
    crossing = None
    outcome = "in_flight"
    pts = [(t, x, y, z)] if record else None

    while t < 3.0:
        sp = math.sqrt(vx * vx + vy * vy + vz * vz)
        ax = -k * sp * vx
        ay = -k * sp * vy + (0.0 if bounced else p.swing_accel)
        az = -k * sp * vz - G

        vx += ax * dt
        vy += ay * dt
        vz += az * dt

        px, py, pz = x, y, z
        x += vx * dt
        y += vy * dt
        z += vz * dt
        t += dt

        # ground contact
        if z <= BALL_RADIUS and vz < 0:
            if bounced:
                outcome = "second_bounce"
                break
            frac = (pz - BALL_RADIUS) / (pz - z) if pz != z else 0.0
            frac = min(max(frac, 0.0), 1.0)
            bx = px + (x - px) * frac
            by = py + (y - py) * frac
            bt = t - dt + dt * frac
            bounce = {"x": bx, "y": by, "t": bt, "speed": sp}
            x, y, z = bx, by, BALL_RADIUS
            vz = -vz * p.restitution
            vx *= p.pace_retention
            vy *= p.pace_retention
            if p.turn_deg:
                a = math.radians(p.turn_deg)
                vx, vy = (vx * math.cos(a) - vy * math.sin(a),
                          vx * math.sin(a) + vy * math.cos(a))
            bounced = True
            if record:
                pts.append((bt, bx, by, BALL_RADIUS))

        # reached the batter's stumps plane
        if x >= L:
            frac = (L - px) / (x - px) if x != px else 0.0
            frac = min(max(frac, 0.0), 1.0)
            cy = py + (y - py) * frac
            cz = pz + (z - pz) * frac
            ct = t - dt + dt * frac
            crossing = {"y": cy, "z": cz, "t": ct, "speed": sp}
            if record:
                pts.append((ct, L, cy, cz))
            outcome = "reached_stumps"
            break

        if record:
            pts.append((t, x, y, z))

    hits = False
    if crossing is not None:
        hits = (abs(crossing["y"]) <= STUMP_HALF_WIDTH + BALL_RADIUS and
                0.0 <= crossing["z"] <= STUMP_HEIGHT + BALL_RADIUS)

    return {"bounce": bounce, "crossing": crossing, "outcome": outcome,
            "hits_stumps": hits, "points": pts}


def _bisect(f, lo, hi, iters=28):
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        return lo if abs(flo) < abs(fhi) else hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if flo * fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


def _bounce_xy(speed_kmh, h, y0, th, ph, p, dt):
    r = _fly(speed_kmh, h, y0, th, ph, p, dt, False)
    if r["bounce"] is not None:
        return r["bounce"]["x"], r["bounce"]["y"]
    cy = r["crossing"]["y"] if r["crossing"] else 0.0
    return p.release_to_stumps + 5.0, cy      # full toss: bounce is beyond the stumps


def deliver(speed_kmh=140.0, length_m=7.0, line_y=0.2, release_height=2.2,
            release_y=0.3, physics=None, dt=0.001):
    """
    Simulate a delivery that pitches `length_m` metres in front of the batter's stumps
    at sideways position `line_y`. The release angles are solved numerically.
    """
    p = physics or BallPhysics()
    target_x = p.release_to_stumps - length_m
    solver_dt = 0.002

    th, ph = 6.0, 0.0
    for _ in range(3):
        th = _bisect(lambda a: _bounce_xy(speed_kmh, release_height, release_y, a, ph, p, solver_dt)[0] - target_x,
                     -8.0, 50.0)
        ph = _bisect(lambda a: _bounce_xy(speed_kmh, release_height, release_y, th, a, p, solver_dt)[1] - line_y,
                     -12.0, 12.0)

    r = _fly(speed_kmh, release_height, release_y, th, ph, p, dt, True)
    b, c = r["bounce"], r["crossing"]

    dist = (p.release_to_stumps - b["x"]) if b else None
    return {
        "inputs": {"speed_kmh": speed_kmh, "length_m": length_m, "line_y": line_y,
                   "release_height": release_height, "release_y": release_y,
                   "release_to_stumps": p.release_to_stumps},
        "release_angle_down_deg": th,
        "release_angle_sideways_deg": ph,
        "trajectory": np.array(r["points"]),      # columns: t, x, y, z
        "bounce": b,
        "crossing": c,
        "outcome": r["outcome"],
        "hits_stumps": r["hits_stumps"],
        "length_from_stumps_m": dist,
        "length_label": classify_length(dist),
        "line_label": classify_line(b["y"]) if b else None,
        "solver_error_m": abs(b["x"] - target_x) if b else None,
    }


def summarise(result):
    """Rounded key numbers for printing or showing in a table."""
    b, c = result["bounce"], result["crossing"]
    return {
        "Length": result["length_label"],
        "Line": result["line_label"],
        "Pitches (m from stumps)": None if b is None else round(result["length_from_stumps_m"], 2),
        "Speed at bounce (km/h)": None if b is None else round(b["speed"] * 3.6, 1),
        "Speed at stumps (km/h)": None if c is None else round(c["speed"] * 3.6, 1),
        "Height at stumps (m)": None if c is None else round(c["z"], 2),
        "Hits stumps": "Yes" if result["hits_stumps"] else "No",
        "Flight time (s)": None if c is None else round(c["t"], 2),
    }