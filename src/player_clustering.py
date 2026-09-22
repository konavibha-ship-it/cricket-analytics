import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

os.makedirs('../reports', exist_ok=True)

deliveries = pd.read_csv('../data/processed/deliveries_clean.csv')
deliveries['phase'] = pd.cut(deliveries['over'], bins=[0, 6, 15, 20], labels=['Powerplay', 'Middle', 'Death'])

# ============================================
# BUILD FEATURE SET PER PLAYER
# ============================================
overall = deliveries.groupby('batsman').agg(
    total_runs=('batsman_runs', 'sum'),
    balls_faced=('batsman_runs', 'count'),
    fours=('batsman_runs', lambda x: (x == 4).sum()),
    sixes=('batsman_runs', lambda x: (x == 6).sum())
).reset_index()
overall = overall[overall['balls_faced'] >= 200]  # meaningful sample only

overall['strike_rate'] = (overall['total_runs'] / overall['balls_faced']) * 100
overall['boundary_pct'] = (overall['fours'] + overall['sixes']) / overall['balls_faced'] * 100
overall['six_pct'] = overall['sixes'] / overall['balls_faced'] * 100

# Phase-wise strike rates (key differentiator between player types)
phase_sr = deliveries.groupby(['batsman', 'phase'], observed=True).agg(
    runs=('batsman_runs', 'sum'),
    balls=('batsman_runs', 'count')
).reset_index()
phase_sr['sr'] = (phase_sr['runs'] / phase_sr['balls']) * 100

phase_pivot = phase_sr.pivot(index='batsman', columns='phase', values='sr').reset_index()
phase_pivot.columns.name = None
phase_pivot = phase_pivot.rename(columns={
    'Powerplay': 'powerplay_sr', 'Middle': 'middle_sr', 'Death': 'death_sr'
})

player_features = overall.merge(phase_pivot, on='batsman', how='left')
player_features = player_features.fillna(player_features[['powerplay_sr', 'middle_sr', 'death_sr']].mean())

# ============================================
# CLUSTER
# ============================================
feature_cols = ['strike_rate', 'boundary_pct', 'six_pct', 'powerplay_sr', 'middle_sr', 'death_sr']
X = player_features[feature_cols]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method to justify choice of k
inertias = []
k_range = range(2, 9)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(list(k_range), inertias, marker='o')
plt.title('Elbow Method — Choosing Optimal Number of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.tight_layout()
plt.savefig('../reports/clustering_elbow.png')
plt.show()

# Based on elbow curve, we use k=4 (adjust after viewing the chart if needed)
k_final = 4
kmeans = KMeans(n_clusters=k_final, random_state=42, n_init=10)
player_features['cluster'] = kmeans.fit_predict(X_scaled)

# ============================================
# LABEL CLUSTERS BASED ON THEIR CHARACTERISTICS
# ============================================
cluster_summary = player_features.groupby('cluster')[feature_cols].mean()
print("=== Cluster Characteristics (average values) ===")
print(cluster_summary)

# ============================================
# VISUALIZE WITH PCA (reduce 6 dimensions to 2 for plotting)
# ============================================
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
player_features['pca1'] = pca_result[:, 0]
player_features['pca2'] = pca_result[:, 1]

plt.figure(figsize=(10, 7))
sns.scatterplot(data=player_features, x='pca1', y='pca2', hue='cluster', palette='Set2', s=80)
for _, row in player_features.sort_values('total_runs', ascending=False).head(15).iterrows():
    plt.text(row['pca1'] + 0.05, row['pca2'], row['batsman'], fontsize=8)
plt.title('Batter Archetypes — Clustered by Playing Style')
plt.tight_layout()
plt.savefig('../reports/player_clusters.png')
plt.show()

# ============================================
# SAVE RESULTS
# ============================================
output = player_features[['batsman', 'total_runs', 'strike_rate', 'powerplay_sr',
                            'middle_sr', 'death_sr', 'boundary_pct', 'cluster']]
output = output.sort_values(['cluster', 'total_runs'], ascending=[True, False])
output.to_csv('../data/processed/player_clusters.csv', index=False)

print("\n=== Sample players per cluster ===")
for c in sorted(output['cluster'].unique()):
    print(f"\nCluster {c}:")
    print(output[output['cluster'] == c].head(5)[['batsman', 'total_runs', 'strike_rate']].to_string(index=False))

print("\nSaved to data/processed/player_clusters.csv")