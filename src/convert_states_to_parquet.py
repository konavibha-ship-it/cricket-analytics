import pandas as pd
import os

for name in ['match_states', 'match_states_odi']:
    csv_path = f'../data/processed/{name}.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        parquet_path = f'../data/processed/{name}.parquet'
        df.to_parquet(parquet_path, engine='pyarrow', compression='snappy', index=False)
        print(f"{name}: {os.path.getsize(csv_path)/(1024*1024):.1f} MB -> {os.path.getsize(parquet_path)/(1024*1024):.1f} MB")
    else:
        print(f"{name}.csv not found, skipping")