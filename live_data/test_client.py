from cricbuzz_client import search_player, get_player_batting_stats, get_live_matches

print("=== Searching for 'Kohli' ===")
results = search_player("Kohli")
print(results)

if results:
    player_id = results[0]["id"]
    print(f"\n=== Batting stats for player id {player_id} ===")
    print(get_player_batting_stats(player_id))

print("\n=== Live matches ===")
print(get_live_matches())
from cricbuzz_client import get_player_bowling_stats
print("\n=== Bowling stats for Kohli (id 1413) ===")
print(get_player_bowling_stats(1413))