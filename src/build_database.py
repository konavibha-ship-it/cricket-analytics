import pandas as pd
from sqlalchemy import create_engine

matches = pd.read_csv('../data/processed/matches_clean.csv')
deliveries = pd.read_csv('../data/processed/deliveries_clean.csv')

print(f"Loading {matches.shape[0]} matches and {deliveries.shape[0]} deliveries into SQL...")

engine = create_engine('sqlite:///../data/processed/cricket.db')

print("Inserting matches table...")
matches.to_sql('matches', engine, if_exists='replace', index=False, chunksize=1000)

print("Inserting deliveries table (this is the big one — will take a while)...")
deliveries.to_sql('deliveries', engine, if_exists='replace', index=False, chunksize=1000)

print("Database rebuilt: data/processed/cricket.db")

check = pd.read_sql("SELECT format, COUNT(*) as matches FROM matches GROUP BY format ORDER BY matches DESC", engine)
print("\nVerification — matches per format in SQL:")
print(check)