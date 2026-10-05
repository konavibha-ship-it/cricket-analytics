"""
Thin client for the Cricbuzz Cricket API (RapidAPI).
Separate from the Cricsheet dataset — this only talks to the live API.
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("RAPIDAPI_KEY")
HOST = "cricbuzz-cricket.p.rapidapi.com"
BASE_URL = f"https://{HOST}"

HEADERS = {
    "X-RapidAPI-Key": API_KEY or "",
    "X-RapidAPI-Host": HOST,
}


class CricbuzzUnavailable(Exception):
    pass


def _get(path, params=None):
    if not API_KEY:
        raise CricbuzzUnavailable("No RAPIDAPI_KEY set in .env")

    resp = requests.get(f"{BASE_URL}{path}", headers=HEADERS, params=params, timeout=10)

    if resp.status_code == 429:
        raise CricbuzzUnavailable("Rate limit / quota exceeded for this plan")
    if resp.status_code != 200:
        raise CricbuzzUnavailable(f"API returned status {resp.status_code}")
    if not resp.text.strip():
        raise CricbuzzUnavailable("API returned an empty response (no data for this request)")

    try:
        return resp.json()
    except ValueError:
        raise CricbuzzUnavailable("API returned a non-JSON response")


def search_player(name):
    """Returns a list of matching players with their Cricbuzz player IDs."""
    data = _get("/stats/v1/player/search", params={"plrN": name})
    return data.get("player", [])


def get_player_info(player_id):
    """Role, batting/bowling style, teams, DOB etc."""
    return _get(f"/stats/v1/player/{player_id}")


def get_player_batting_stats(player_id):
    """Career batting figures, broken out by Test / ODI / T20."""
    return _get(f"/stats/v1/player/{player_id}/batting")


def get_player_bowling_stats(player_id):
    """Career bowling figures, broken out by Test / ODI / T20."""
    return _get(f"/stats/v1/player/{player_id}/bowling")


def get_live_matches():
    """Currently live matches with scores."""
    return _get("/matches/v1/live")