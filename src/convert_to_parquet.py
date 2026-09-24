import pandas as pd
import os

print("Loading deliveries_clean.csv...")
deliveries = pd.read_csv('../data/processed/deliveries_clean.csv')

print(f"Original shape: {deliveries.shape}")
print(f"Original CSV size: {os.path.getsize('../data/processed/deliveries_clean.csv') / (1024*1024):.1f} MB")

# Convert repetitive text columns to 'category' dtype — this is what makes Parquet compression so effective
categorical_cols = ['format', 'batting_team', 'batsman', 'bowler', 'dismissal_kind', 'player_dismissed']
for col in categorical_cols:
    if col in deliveries.columns:
        deliveries[col] = deliveries[col].astype('category')

# Save as compressed Parquet
output_path = '../data/processed/deliveries_clean.parquet'
deliveries.to_parquet(output_path, engine='pyarrow', compression='snappy', index=False)

new_size = os.path.getsize(output_path) / (1024*1024)
print(f"\nParquet file size: {new_size:.1f} MB")

if new_size < 100:
    print("✅ Under GitHub's 100MB limit — safe to commit!")
else:
    print("⚠️ Still over 100MB — we'll need Option B (smaller deployed dataset) instead.")

# Also convert matches (much smaller, but let's be consistent)
matches = pd.read_csv('../data/processed/matches_clean.csv')
matches.to_parquet('../data/processed/matches_clean.parquet', engine='pyarrow', compression='snappy', index=False)
print(f"\nmatches_clean.parquet size: {os.path.getsize('../data/processed/matches_clean.parquet') / (1024*1024):.2f} MB")