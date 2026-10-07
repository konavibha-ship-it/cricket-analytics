import pandas as pd
import numpy as np

from data_loader import load_matches, load_deliveries
matches = load_matches()
deliveries = load_deliveries()


def audit_toss_decision(venue, format_filter='IPL'):
    """
    For a given venue, checks whether bat-first or chasing has historically
    been the better toss decision — a data-backed answer to 'what should
    the captain have chosen here?'
    """
    venue_matches = matches[
        (matches['venue'] == venue) & (matches['format'] == format_filter)
    ].dropna(subset=['winner'])
    venue_matches = venue_matches[venue_matches['winner'] != 'No Result']

    bat_first_matches = venue_matches[venue_matches['toss_decision'] == 'bat']
    chase_matches = venue_matches[venue_matches['toss_decision'] == 'field']

    bat_first_win_rate = (bat_first_matches['toss_winner'] == bat_first_matches['winner']).mean() * 100 \
        if len(bat_first_matches) > 0 else None
    chase_win_rate = (chase_matches['toss_winner'] == chase_matches['winner']).mean() * 100 \
        if len(chase_matches) > 0 else None

    recommendation = None
    if bat_first_win_rate is not None and chase_win_rate is not None:
        recommendation = "Bat First" if bat_first_win_rate > chase_win_rate else "Chase"

    return {
        'venue': venue,
        'bat_first_win_rate_pct': round(bat_first_win_rate, 1) if bat_first_win_rate is not None else None,
        'bat_first_sample': len(bat_first_matches),
        'chase_win_rate_pct': round(chase_win_rate, 1) if chase_win_rate is not None else None,
        'chase_sample': len(chase_matches),
        'data_backed_recommendation': recommendation
    }


def audit_bowling_change_timing(bowler_name, format_filter='IPL'):
    """
    Checks whether a bowler performs better/worse when introduced earlier vs
    later in an innings, relative to their own average — flags whether their
    usage pattern matches their actual strength by phase.
    """
    bowler_data = deliveries[
        (deliveries['bowler'] == bowler_name) & (deliveries['format'] == format_filter)
    ].copy()

    if bowler_data.empty:
        return None

    bowler_data['phase'] = pd.cut(bowler_data['over'], bins=[0, 6, 15, 20], labels=['Powerplay', 'Middle', 'Death'])

    phase_stats = bowler_data.groupby('phase', observed=True).agg(
        balls=('total_runs', 'count'),
        runs=('total_runs', 'sum'),
        wickets=('dismissal_kind', lambda x: x.notnull().sum())
    )
    phase_stats['economy'] = (phase_stats['runs'] / phase_stats['balls'] * 6).round(2)
    phase_stats['usage_pct'] = (phase_stats['balls'] / phase_stats['balls'].sum() * 100).round(1)

    if phase_stats.empty:
        return None

    best_phase = phase_stats['economy'].idxmin()
    most_used_phase = phase_stats['usage_pct'].idxmax()

    alignment = "✅ Well-utilized" if best_phase == most_used_phase else \
                f"⚠️ Misaligned — best in {best_phase} but used most in {most_used_phase}"

    return {
        'bowler': bowler_name,
        'phase_breakdown': phase_stats[['economy', 'usage_pct', 'wickets']].to_dict('index'),
        'strongest_phase': best_phase,
        'most_used_phase': most_used_phase,
        'usage_alignment': alignment
    }


if __name__ == "__main__":
    sample_venue = matches[matches['format'] == 'IPL']['venue'].value_counts().index[0]

    print("=== Toss Decision Audit ===")
    print(audit_toss_decision(sample_venue, format_filter='IPL'))

    print("\n=== Bowling Usage Audit: JJ Bumrah ===")
    result = audit_bowling_change_timing("JJ Bumrah", format_filter='IPL')
    print(f"Bowler: {result['bowler']}")
    print(f"Strongest phase: {result['strongest_phase']}")
    print(f"Most used phase: {result['most_used_phase']}")
    print(f"Alignment: {result['usage_alignment']}")
    print("\nPhase breakdown:")
    for phase, stats in result['phase_breakdown'].items():
        print(f"  {phase}: {stats}")