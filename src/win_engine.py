"""Win probability & momentum engine for limited-overs chases.

Used by src/train_win_engine.py (training) and by the dashboard tab.

Ideas implemented
  * Resources: the average runs still to come from a state (overs left, wickets
    lost), fitted with the functional form used by the original Duckworth-Lewis
    method:   Z(u, w) = Z0(w) * (1 - exp(-b(w) * u / Z0(w))).
    The parameters are fitted to THIS project's data, so this is "DLS-style".
    It is NOT the official DLS Standard / Professional Edition table.
  * Pressure index: runs still needed / average runs a team scores from the
    same state.  Above 1.0 = the chase needs better-than-average batting.
  * DLS-style par score: target * (share of resources the chasing team has used).
  * Recent form: runs and wickets in the last 12 deliveries (about 2 overs).

Only chases (2nd innings) of full-length matches are modelled:
20 overs for the T20 family, 50 overs for ODI.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"

KEYS = ["match_id", "inning"]

T20_FORMATS = [
    "IPL", "T20I", "BBL", "Women's BBL", "CPL", "Women's CPL", "SA20",
    "CSA T20 Challenge", "Super Smash", "Syed Mushtaq Ali Trophy", "WPL",
]
FAMILIES = {
    "T20": {"formats": T20_FORMATS, "overs_limit": 20, "limit": 120},
    "ODI": {"formats": ["ODI"], "overs_limit": 50, "limit": 300},
}

BASE_FEATURES = [
    "current_score", "wickets_fallen", "balls_bowled", "balls_remaining",
    "target", "runs_needed", "required_run_rate", "current_run_rate",
]
EXTRA_FEATURES = [
    "resources_pct", "pressure_index", "dls_par_diff",
    "recent_runs_12", "recent_wkts_12",
]
ENGINE_FEATURES = BASE_FEATURES + EXTRA_FEATURES

TEXT_COLS = ["batsman", "bowler", "player_dismissed", "dismissal_kind", "batting_team", "format"]


# ---------------------------------------------------------------- resources
def _z(u, z0, b):
    return z0 * (1 - np.exp(-b * u / z0))


def resource_runs(u, w, params):
    """Average runs still to come with u overs left and w wickets lost."""
    z0 = np.asarray(params["z0"], dtype=float)
    b = np.asarray(params["b"], dtype=float)
    w_raw = np.asarray(w).astype(int)
    idx = np.clip(w_raw, 0, 9)
    u = np.maximum(np.asarray(u, dtype=float), 0.0)
    z = _z(u, z0[idx], b[idx])
    return np.where(w_raw >= 10, 0.0, z)


def resource_pct(u, w, params, limit):
    """Share (%) of the full innings' scoring resources that remain."""
    full = float(resource_runs(limit / 6, 0, params))
    return resource_runs(u, w, params) / full * 100


def fit_resource_params(states, limit):
    """Fit Z0(w) and b(w) for w = 0..9 wickets lost.

    states needs the columns: balls_bowled, wickets, remaining_runs
    (one row per ball of full-length first innings).
    """
    from scipy.optimize import curve_fit

    z0_list, b_list = [], []
    for w in range(10):
        s = states[states["wickets"] == w]
        g = s.groupby("balls_bowled")["remaining_runs"].agg(["mean", "size"]).reset_index()
        g = g[g["size"] >= 20]
        popt = None
        if len(g) >= 8:
            u = (limit - g["balls_bowled"].to_numpy(dtype=float)) / 6
            y = g["mean"].to_numpy(dtype=float)
            sigma = 1.0 / np.sqrt(g["size"].to_numpy(dtype=float))
            try:
                popt, _ = curve_fit(
                    _z, u, y, p0=[max(y.max() * 1.3, 50.0), 8.0], sigma=sigma,
                    bounds=([10.0, 0.5], [2000.0, 60.0]), maxfev=20000,
                )
            except Exception as e:  # noqa: BLE001
                print(f"   wickets={w}: curve fit failed ({e})")
        if popt is None:
            if not z0_list:
                raise RuntimeError("Could not fit the zero-wicket resource curve")
            popt = [z0_list[-1] * 0.7, b_list[-1] * 0.7]
            print(f"   wickets={w}: too little data, estimated from {w - 1} wickets")
        z0_list.append(float(popt[0]))
        b_list.append(float(popt[1]))
    return {"z0": z0_list, "b": b_list}


# ---------------------------------------------------------------- data prep
def prepare_deliveries(deliveries, matches, family):
    """Deliveries of full-length matches with a clear winner, plus counters."""
    cfg = FAMILIES[family]
    m = matches.copy()
    m["match_id"] = m["match_id"].astype(str)
    m["overs_limit"] = pd.to_numeric(m["overs_limit"], errors="coerce")
    m = m[m["format"].isin(cfg["formats"]) & (m["overs_limit"] == cfg["overs_limit"])]
    m = m.dropna(subset=["winner"])
    m = m[~m["winner"].astype(str).str.lower().isin({"no result", "tie", "draw"})].copy()
    if m.empty:
        raise ValueError(
            f"No {family} matches found with overs_limit == {cfg['overs_limit']}. "
            "Check the overs_limit column in matches_clean.parquet."
        )
    m["winner"] = m["winner"].astype(object)

    d = deliveries[deliveries["inning"].isin([1, 2])].copy()
    d["match_id"] = d["match_id"].astype(str)
    d = d[d["match_id"].isin(set(m["match_id"]))].copy()
    for col in TEXT_COLS:
        d[col] = d[col].astype(object)

    d = d.sort_values(KEYS + ["over", "ball"], kind="mergesort").reset_index(drop=True)
    # Wides are not marked in the data, so time is taken from over and ball number
    d["balls_bowled"] = np.minimum(d["over"] * 6 + d["ball"].clip(1, 6), cfg["limit"])
    d["is_wicket"] = d["player_dismissed"].notna().astype(int)
    grp = d.groupby(KEYS, sort=False)
    d["cum_runs"] = grp["total_runs"].cumsum()
    d["wickets"] = grp["is_wicket"].cumsum()
    return d, m


def first_innings_states(d):
    """Ball states of first innings with the runs still to come (for the resource fit)."""
    f = d[d["inning"] == 1].copy()
    total = f.groupby("match_id")["total_runs"].sum().rename("inn_total")
    f = f.join(total, on="match_id")
    f["remaining_runs"] = f["inn_total"] - f["cum_runs"]
    return f[["balls_bowled", "wickets", "remaining_runs"]]


def build_chase_states(d, m, params, family):
    """One row per delivery of every chase, with all engine features and the label."""
    limit = FAMILIES[family]["limit"]
    total1 = d[d["inning"] == 1].groupby("match_id")["total_runs"].sum().rename("first_total")
    c = d[d["inning"] == 2].join(total1, on="match_id").dropna(subset=["first_total"])
    c = c.merge(m[["match_id", "winner"]], on="match_id", how="left").reset_index(drop=True)

    c["target"] = c["first_total"] + 1
    c["current_score"] = c["cum_runs"]
    c["wickets_fallen"] = c["wickets"]
    c["balls_remaining"] = (limit - c["balls_bowled"]).clip(lower=0)
    c["runs_needed"] = c["target"] - c["current_score"]
    c["required_run_rate"] = (
        c["runs_needed"].clip(lower=0) / c["balls_remaining"].clip(lower=1) * 6
    ).clip(upper=60)
    c["current_run_rate"] = c["cum_runs"] / (c["balls_bowled"].clip(lower=1) / 6)

    u = c["balls_remaining"] / 6
    z = resource_runs(u, c["wickets"], params)
    full = float(resource_runs(limit / 6, 0, params))
    c["resources_pct"] = z / full * 100
    c["pressure_index"] = (c["runs_needed"].clip(lower=0) / np.maximum(z, 1.0)).clip(upper=10)
    par = (c["target"] - 1) * (100 - c["resources_pct"]) / 100
    c["dls_par_diff"] = c["current_score"] - par

    g = c.groupby("match_id", sort=False)
    c["recent_runs_12"] = c["cum_runs"] - g["cum_runs"].shift(12).fillna(0)
    c["recent_wkts_12"] = c["wickets"] - g["wickets"].shift(12).fillna(0)

    c["won"] = (c["batting_team"] == c["winner"]).astype(int)
    return c


def state_from_inputs(score, wickets, balls_bowled, target, family, params,
                      recent_runs=None, recent_wkts=None):
    """Feature row for a situation typed in by hand (what-if / live entry)."""
    limit = FAMILIES[family]["limit"]
    balls_remaining = max(limit - balls_bowled, 0)
    runs_needed = target - score
    z = float(resource_runs(balls_remaining / 6, wickets, params))
    full = float(resource_runs(limit / 6, 0, params))
    res = z / full * 100
    par = (target - 1) * (100 - res) / 100
    if recent_runs is None:
        recent_runs = score / max(balls_bowled, 1) * 12
    if recent_wkts is None:
        recent_wkts = wickets / max(balls_bowled, 1) * 12
    return {
        "current_score": score,
        "wickets_fallen": wickets,
        "balls_bowled": balls_bowled,
        "balls_remaining": balls_remaining,
        "target": target,
        "runs_needed": runs_needed,
        "required_run_rate": min(max(runs_needed, 0) / max(balls_remaining, 1) * 6, 60),
        "current_run_rate": score / (max(balls_bowled, 1) / 6),
        "resources_pct": res,
        "pressure_index": min(max(runs_needed, 0) / max(z, 1.0), 10),
        "dls_par_diff": score - par,
        "recent_runs_12": recent_runs,
        "recent_wkts_12": recent_wkts,
    }


# ---------------------------------------------------------------- saved files
def load_params():
    with open(DATA / "win_engine_params.json") as f:
        return json.load(f)


def load_engine(family):
    return joblib.load(DATA / f"win_engine_{family.lower()}.pkl")


def predict(engine, states):
    """Chasing team win probability (0-1) for every row of states."""
    return engine["model"].predict_proba(states[engine["features"]])[:, 1]
