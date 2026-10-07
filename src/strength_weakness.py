import pandas as pd
import numpy as np

# ============================================
# LOAD DATA
# ============================================
from data_loader import load_matches, load_deliveries

deliveries = load_deliveries()
matches = load_matches()

# Join venue and opposition info onto each delivery
match_info = matches[['match_id', 'venue', 'team1', 'team2']]
deliveries = deliveries.merge(match_info, on='match_id', how='left')

deliveries['phase'] = pd.cut(deliveries['over'], bins=[0, 6, 15, 20], labels=['Powerplay', 'Middle', 'Death'])

# Figure out the opposition (bowling team) for each ball, from a BATTER's perspective
deliveries['opponent'] = np.where(
    deliveries['batting_team'] == deliveries['team1'],
    deliveries['team2'],
    deliveries['team1']
)


# ============================================
# BATSMAN STRENGTH/WEAKNESS REPORT
# ============================================
def batsman_report(player_name, min_balls=20, verbose=True):
    player_data = deliveries[deliveries['batsman'] == player_name]
    if player_data.empty:
        if verbose:
            print(f"No data found for {player_name}")
        return None

    overall_runs = player_data['batsman_runs'].sum()
    overall_balls = len(player_data)
    overall_sr = (overall_runs / overall_balls) * 100 if overall_balls else 0

    # ---- By Phase ----
    phase_stats = player_data.groupby('phase', observed=True).agg(
        runs=('batsman_runs', 'sum'), balls=('batsman_runs', 'count')
    )
    phase_stats['strike_rate'] = (phase_stats['runs'] / phase_stats['balls'] * 100).round(1)
    phase_stats = phase_stats[phase_stats['balls'] >= min_balls].sort_values('strike_rate', ascending=False)

    # ---- By Venue ----
    venue_stats = player_data.groupby('venue').agg(
        runs=('batsman_runs', 'sum'), balls=('batsman_runs', 'count')
    )
    venue_stats['strike_rate'] = (venue_stats['runs'] / venue_stats['balls'] * 100).round(1)
    venue_stats = venue_stats[venue_stats['balls'] >= min_balls].sort_values('strike_rate', ascending=False)

    # ---- Dismissal Patterns ----
    dismissals = player_data[player_data['dismissal_kind'].notnull()]
    dismissal_counts = dismissals['dismissal_kind'].value_counts() if not dismissals.empty else pd.Series(dtype=int)

    # ---- Vs Opposition ----
    opp_stats = player_data.groupby('opponent').agg(
        runs=('batsman_runs', 'sum'), balls=('batsman_runs', 'count')
    )
    opp_stats['strike_rate'] = (opp_stats['runs'] / opp_stats['balls'] * 100).round(1)
    opp_stats = opp_stats[opp_stats['balls'] >= min_balls].sort_values('strike_rate', ascending=False)

    if verbose:
        print("=" * 70)
        print(f"BATSMAN REPORT: {player_name}")
        print("=" * 70)
        print(f"\nOverall: {overall_runs} runs off {overall_balls} balls | SR: {overall_sr:.1f}")

        print("\n--- By Match Phase ---")
        print(phase_stats)
        if not phase_stats.empty:
            best_phase = phase_stats['strike_rate'].idxmax()
            worst_phase = phase_stats['strike_rate'].idxmin()
            print(f"\n✅ STRENGTH: Best strike rate in {best_phase} ({phase_stats.loc[best_phase, 'strike_rate']})")
            print(f"⚠️  WEAKNESS: Lowest strike rate in {worst_phase} ({phase_stats.loc[worst_phase, 'strike_rate']})")

        print(f"\n--- By Venue (min {min_balls} balls) ---")
        print(venue_stats.head(5))
        if not venue_stats.empty:
            print(f"\n✅ STRENGTH: Best venue → {venue_stats.index[0]} (SR: {venue_stats.iloc[0]['strike_rate']})")
            print(f"⚠️  WEAKNESS: Worst venue → {venue_stats.index[-1]} (SR: {venue_stats.iloc[-1]['strike_rate']})")

        print("\n--- Dismissal Patterns ---")
        print(dismissal_counts)
        if not dismissal_counts.empty:
            top_dismissal = dismissal_counts.index[0]
            pct = (dismissal_counts.iloc[0] / dismissal_counts.sum()) * 100
            print(f"\n⚠️  WEAKNESS: Most common dismissal → {top_dismissal} ({pct:.0f}% of outs)")

        print(f"\n--- Vs Opposition Teams (min {min_balls} balls) ---")
        print(opp_stats)
        if not opp_stats.empty:
            print(f"\n✅ STRENGTH: Dominates → {opp_stats.index[0]} (SR: {opp_stats.iloc[0]['strike_rate']})")
            print(f"⚠️  WEAKNESS: Struggles vs → {opp_stats.index[-1]} (SR: {opp_stats.iloc[-1]['strike_rate']})")

    return {
        'player': player_name,
        'overall_runs': overall_runs,
        'overall_balls': overall_balls,
        'overall_sr': round(overall_sr, 1),
        'phase': phase_stats,
        'venue': venue_stats,
        'dismissals': dismissal_counts,
        'opponent': opp_stats
    }


# ============================================
# BOWLER STRENGTH/WEAKNESS REPORT
# ============================================
def bowler_report(player_name, min_balls=20, verbose=True):
    player_data = deliveries[deliveries['bowler'] == player_name]
    if player_data.empty:
        if verbose:
            print(f"No data found for {player_name}")
        return None

    overall_runs = player_data['total_runs'].sum()
    overall_balls = len(player_data)
    overall_economy = (overall_runs / overall_balls) * 6 if overall_balls else 0
    wickets = player_data['dismissal_kind'].notnull().sum()

    # ---- By Phase ----
    phase_stats = player_data.groupby('phase', observed=True).agg(
        runs=('total_runs', 'sum'), balls=('total_runs', 'count'),
        wickets=('dismissal_kind', lambda x: x.notnull().sum())
    )
    phase_stats['economy'] = (phase_stats['runs'] / phase_stats['balls'] * 6).round(2)
    phase_stats = phase_stats[phase_stats['balls'] >= min_balls].sort_values('economy')

    # ---- By Venue ----
    venue_stats = player_data.groupby('venue').agg(
        runs=('total_runs', 'sum'), balls=('total_runs', 'count'),
        wickets=('dismissal_kind', lambda x: x.notnull().sum())
    )
    venue_stats['economy'] = (venue_stats['runs'] / venue_stats['balls'] * 6).round(2)
    venue_stats = venue_stats[venue_stats['balls'] >= min_balls].sort_values('economy')

    # ---- Vs Opposition (batting_team = who the bowler actually bowled against) ----
    opp_stats = player_data.groupby('batting_team').agg(
        runs=('total_runs', 'sum'), balls=('total_runs', 'count'),
        wickets=('dismissal_kind', lambda x: x.notnull().sum())
    )
    opp_stats['economy'] = (opp_stats['runs'] / opp_stats['balls'] * 6).round(2)
    opp_stats = opp_stats[opp_stats['balls'] >= min_balls].sort_values('economy')

    if verbose:
        print("=" * 70)
        print(f"BOWLER REPORT: {player_name}")
        print("=" * 70)
        print(f"\nOverall: {wickets} wickets | Economy: {overall_economy:.2f}")

        print("\n--- By Match Phase ---")
        print(phase_stats)
        if not phase_stats.empty:
            best_phase = phase_stats['economy'].idxmin()
            worst_phase = phase_stats['economy'].idxmax()
            print(f"\n✅ STRENGTH: Best economy in {best_phase} ({phase_stats.loc[best_phase, 'economy']})")
            print(f"⚠️  WEAKNESS: Worst economy in {worst_phase} ({phase_stats.loc[worst_phase, 'economy']})")

        print(f"\n--- By Venue (min {min_balls} balls) ---")
        print(venue_stats.head(5))

        print(f"\n--- Vs Opposition Teams (min {min_balls} balls) ---")
        print(opp_stats)
        if not opp_stats.empty:
            print(f"\n✅ STRENGTH: Most effective vs → {opp_stats.index[0]} (Econ: {opp_stats.iloc[0]['economy']})")
            print(f"⚠️  WEAKNESS: Expensive vs → {opp_stats.index[-1]} (Econ: {opp_stats.iloc[-1]['economy']})")

    return {
        'player': player_name,
        'overall_wickets': wickets,
        'overall_economy': round(overall_economy, 2),
        'phase': phase_stats,
        'venue': venue_stats,
        'opponent': opp_stats
    }


# ============================================
# RUN A SAMPLE REPORT — only runs when this file is executed directly,
# NOT when imported by the dashboard
# ============================================
if __name__ == "__main__":
    batsman_report("V Kohli")
    print("\n\n")
    bowler_report("JJ Bumrah")