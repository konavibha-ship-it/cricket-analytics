import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from data_loader import load_deliveries
deliveries = load_deliveries()

# ============================================
# 1. OVER-BY-OVER SCORING/ECONOMY HEATMAP DATA
# ============================================
def over_by_over_batting(player_name):
    data = deliveries[deliveries['batsman'] == player_name]
    grouped = data.groupby('over').agg(
        runs=('batsman_runs', 'sum'), balls=('batsman_runs', 'count')
    )
    grouped['strike_rate'] = (grouped['runs'] / grouped['balls'] * 100).round(1)
    return grouped.reindex(range(20)).fillna(0)  # ensure all 20 overs shown


def over_by_over_bowling(player_name):
    data = deliveries[deliveries['bowler'] == player_name]
    grouped = data.groupby('over').agg(
        runs=('total_runs', 'sum'), balls=('total_runs', 'count'),
        wickets=('dismissal_kind', lambda x: x.notnull().sum())
    )
    grouped['economy'] = (grouped['runs'] / grouped['balls'] * 6).round(2)
    return grouped.reindex(range(20)).fillna(0)


# ============================================
# 2. DISMISSAL-ZONE FIELD DIAGRAM
# (illustrative — based on real dismissal TYPE counts,
#  NOT real shot-tracking / wagon-wheel data)
# ============================================
def draw_dismissal_field_diagram(player_name):
    data = deliveries[(deliveries['batsman'] == player_name) & (deliveries['dismissal_kind'].notnull())]

    if data.empty:
        return None

    counts = data['dismissal_kind'].value_counts()

    bowled = counts.get('bowled', 0)
    lbw = counts.get('lbw', 0)
    caught = counts.get('caught', 0)
    caught_bowled = counts.get('caught and bowled', 0)
    stumped = counts.get('stumped', 0)
    run_out = counts.get('run out', 0)

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    # Ground boundary
    boundary = plt.Circle((0, 0), 10, color='#2d5a3d', fill=True, alpha=0.6)
    inner_circle = plt.Circle((0, 0), 5.5, color='#3a7050', fill=True, alpha=0.4)
    ax.add_patch(boundary)
    ax.add_patch(inner_circle)

    # Pitch strip
    pitch = patches.Rectangle((-0.5, -2.2), 1, 4.4, color='#d4b483', zorder=3)
    ax.add_patch(pitch)

    # Stumps (batting end) — bowled / lbw / caught-and-bowled
    ax.scatter(0, 2.2, s=max(bowled + lbw + caught_bowled, 1) * 25 + 100,
               color='#ff4b4b', edgecolors='white', zorder=5, label='Bowled/LBW/C&B')
    ax.text(0, 2.2, f"{bowled + lbw + caught_bowled}", ha='center', va='center',
            fontsize=11, fontweight='bold', color='white', zorder=6)

    # Keeper position — stumped
    ax.scatter(0, -3.2, s=max(stumped, 1) * 25 + 100,
               color='#ffb84b', edgecolors='white', zorder=5, label='Stumped')
    ax.text(0, -3.2, f"{stumped}", ha='center', va='center',
            fontsize=11, fontweight='bold', color='white', zorder=6)

    # Caught — distributed across 4 generic outfield zones (illustrative split, not real positions)
    caught_positions = [(6.5, 6.5), (-6.5, 6.5), (6.5, -6.5), (-6.5, -6.5)]
    caught_labels = ['Cover/Off', 'Midwicket/Leg', 'Long-off', 'Long-on']
    per_zone = caught / 4 if caught else 0

    for (x, y), label in zip(caught_positions, caught_labels):
        ax.scatter(x, y, s=per_zone * 20 + 80, color='#4bafff', edgecolors='white', zorder=5)
        ax.text(x, y, f"{per_zone:.0f}", ha='center', va='center',
                fontsize=9, fontweight='bold', color='white', zorder=6)

    # Run out — one generic infield position
    if run_out:
        ax.scatter(4, -1, s=run_out * 25 + 80, color='#b84bff', edgecolors='white', zorder=5)
        ax.text(4, -1, f"{run_out}", ha='center', va='center',
                fontsize=9, fontweight='bold', color='white', zorder=6)

    ax.set_xlim(-11, 11)
    ax.set_ylim(-11, 11)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f"{player_name} — Dismissal Pattern Map\n(by dismissal type, not shot-tracking)",
                 color='white', fontsize=12, pad=15)

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Bowled/LBW/C&B', markerfacecolor='#ff4b4b', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Caught (outfield)', markerfacecolor='#4bafff', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Stumped', markerfacecolor='#ffb84b', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Run Out', markerfacecolor='#b84bff', markersize=10),
    ]
    ax.legend(handles=legend_elements, loc='upper right', facecolor='#1a1c24',
              labelcolor='white', fontsize=8, framealpha=0.8)

    plt.tight_layout()
    return fig


# ============================================
# 3. BOWLING PLAN — text recommendation generated from real weakness data
# ============================================
def generate_bowling_plan(player_name):
    from strength_weakness import batsman_report

    report = batsman_report(player_name, verbose=False)
    if report is None:
        return "No data available for this player."

    lines = [f"### 🎯 Bowling Plan vs {player_name}\n"]

    # Phase weakness
    if not report['phase'].empty:
        worst_phase = report['phase']['strike_rate'].idxmin()
        worst_sr = report['phase'].loc[worst_phase, 'strike_rate']
        best_phase = report['phase']['strike_rate'].idxmax()
        best_sr = report['phase'].loc[best_phase, 'strike_rate']
        lines.append(f"- **Attack in {worst_phase}** — his strike rate drops to {worst_sr} here (vs {best_sr} in {best_phase}).")
        lines.append(f"- **Avoid bowling defensively in {best_phase}** — he scores fastest here ({best_sr} SR), so build pressure with tight lines rather than giving width.")

    # Dismissal pattern
    if report['dismissals'] is not None and not report['dismissals'].empty:
        top_dismissal = report['dismissals'].index[0]
        pct = (report['dismissals'].iloc[0] / report['dismissals'].sum()) * 100
        dismissal_advice = {
            'caught': "target catches in the outfield/boundary riders — encourage aerial shots with full deliveries.",
            'bowled': "attack the stumps with yorkers and full, straight deliveries.",
            'lbw': "bowl straight and full, especially around off-stump, to draw LBW chances.",
            'caught and bowled': "mix pace to induce miscues back at the bowler.",
            'stumped': "use spin and flight to draw him out of the crease.",
            'run out': "tight fielding and quick returns — he's prone to run-out situations."
        }
        advice = dismissal_advice.get(top_dismissal, "study dismissal patterns further.")
        lines.append(f"- **Primary dismissal method: {top_dismissal}** ({pct:.0f}% of outs) — {advice}")

    # Opponent-specific
    if not report['opponent'].empty:
        weakest_opp = report['opponent'].index[-1]
        lines.append(f"- Historically struggles most against **{weakest_opp}** — study their bowling tactics for ideas.")

    return "\n".join(lines)


# ============================================
# TEST
# ============================================
if __name__ == "__main__":
    print(over_by_over_batting("V Kohli"))
    print("\n")
    print(generate_bowling_plan("V Kohli"))

    fig = draw_dismissal_field_diagram("V Kohli")
    if fig:
        fig.savefig('../reports/sample_field_diagram.png', facecolor=fig.get_facecolor())
        print("\nSaved sample field diagram to reports/")