import pandas as pd
import numpy as np

from data_loader import load_matches, load_deliveries
matches = load_matches()
deliveries = load_deliveries()


def venue_profile(venue_name, format_filter=None):
    venue_matches = matches[matches['venue'] == venue_name]
    if format_filter:
        venue_matches = venue_matches[venue_matches['format'] == format_filter]

    if venue_matches.empty:
        return None

    match_ids = venue_matches['match_id'].unique()
    venue_deliveries = deliveries[deliveries['match_id'].isin(match_ids)]

    # First innings scores (the "what's a good total here" question)
    first_innings = venue_deliveries[venue_deliveries['inning'] == 1].groupby('match_id')['total_runs'].sum()
    avg_first_innings_score = first_innings.mean()

    # Chase success rate
    valid_matches = venue_matches[venue_matches['winner'] != 'No Result'].dropna(subset=['winner'])
    chase_wins = valid_matches[valid_matches['toss_decision'] == 'field']
    chase_win_rate = (chase_wins['toss_winner'] == chase_wins['winner']).mean() * 100 if len(chase_wins) > 0 else None

    # Boundary rate (proxy for how "easy" scoring is here)
    boundaries = venue_deliveries[venue_deliveries['batsman_runs'].isin([4, 6])].shape[0]
    boundary_rate = (boundaries / len(venue_deliveries)) * 100 if len(venue_deliveries) > 0 else 0

    # Wicket rate (proxy for pitch difficulty / bowling friendliness)
    wicket_rate = venue_deliveries['dismissal_kind'].notnull().mean() * 100

    # Highest and lowest team totals ever recorded here
    all_innings_totals = venue_deliveries.groupby(['match_id', 'inning'])['total_runs'].sum()
    highest_total = all_innings_totals.max()
    lowest_total = all_innings_totals.min()

    return {
        'venue': venue_name,
        'total_matches': len(venue_matches),
        'avg_first_innings_score': round(avg_first_innings_score, 1) if not pd.isna(avg_first_innings_score) else None,
        'chase_win_rate_pct': round(chase_win_rate, 1) if chase_win_rate is not None else None,
        'boundary_rate_pct': round(boundary_rate, 2),
        'wicket_rate_pct': round(wicket_rate, 2),
        'highest_innings_total': int(highest_total) if not pd.isna(highest_total) else None,
        'lowest_innings_total': int(lowest_total) if not pd.isna(lowest_total) else None,
        'venue_character': classify_venue(boundary_rate, wicket_rate)
    }


def classify_venue(boundary_rate, wicket_rate, all_boundary_rates=None, all_wicket_rates=None):
    """
    Classifies venues RELATIVE to the dataset's own distribution (percentiles),
    rather than fixed thresholds — since T20 cricket overall skews high-scoring,
    absolute cutoffs don't discriminate well between venues.
    """
    if all_boundary_rates is None or all_wicket_rates is None:
        return "⚖️ Balanced"  # fallback if called without dataset context

    boundary_percentile = (all_boundary_rates < boundary_rate).mean()
    wicket_percentile = (all_wicket_rates < wicket_rate).mean()

    if boundary_percentile > 0.75 and wicket_percentile < 0.35:
        return "🏏 Batting Paradise"
    elif wicket_percentile > 0.70 and boundary_percentile < 0.40:
        return "🎯 Bowler-Friendly"
    elif boundary_percentile > 0.65:
        return "⚡ High-Scoring"
    elif wicket_percentile > 0.60:
        return "🛡️ Bowling-Friendly"
    else:
        return "⚖️ Balanced"


def compare_all_venues(min_matches=15, format_filter='IPL'):
    """Ranks all venues with sufficient sample size by scoring character,
    classified relative to each other rather than fixed absolute cutoffs."""
    venues = matches[matches['format'] == format_filter]['venue'].value_counts()
    qualifying_venues = venues[venues >= min_matches].index

    profiles = []
    for v in qualifying_venues:
        profile = venue_profile(v, format_filter=format_filter)
        if profile:
            profiles.append(profile)

    df = pd.DataFrame(profiles)

    # Now reclassify using the full set's own distribution
    all_boundary_rates = df['boundary_rate_pct']
    all_wicket_rates = df['wicket_rate_pct']

    df['venue_character'] = df.apply(
        lambda row: classify_venue(row['boundary_rate_pct'], row['wicket_rate_pct'],
                                     all_boundary_rates, all_wicket_rates),
        axis=1
    )

    return df.sort_values('avg_first_innings_score', ascending=False)


if __name__ == "__main__":
    print("=== Single Venue Example ===")
    sample_venue = matches[matches['format'] == 'IPL']['venue'].value_counts().index[0]
    print(f"Analyzing: {sample_venue}\n")
    print(venue_profile(sample_venue, format_filter='IPL'))

    print("\n\n=== All IPL Venues Ranked ===")
    all_venues = compare_all_venues(min_matches=15, format_filter='IPL')
    print(all_venues.to_string(index=False))

    all_venues.to_csv('../data/processed/venue_intelligence.csv', index=False)
    print("\nSaved to data/processed/venue_intelligence.csv")