"""
Combine FIFA ranking CSV files into a single unified rankings.csv
"""

import pandas as pd
from pathlib import Path

# Get all FIFA ranking files
data_dir = Path("data/raw")
ranking_files = sorted(data_dir.glob("fifa_ranking-*.csv"))

print(f"Found {len(ranking_files)} FIFA ranking files:")
for f in ranking_files:
    print(f"  - {f.name}")

# Read and combine all files
dfs = []
for file in ranking_files:
    df = pd.read_csv(file)
    dfs.append(df)
    print(f"  Loaded {len(df)} rows from {file.name}")

# Combine into single dataframe
combined_df = pd.concat(dfs, ignore_index=True)
print(f"\nCombined total: {len(combined_df)} rows")

# Save to unified rankings.csv
output_path = data_dir / "rankings.csv"
combined_df.to_csv(output_path, index=False)
print(f"Saved to {output_path}")

# Delete original files
for file in ranking_files:
    file.unlink()
    print(f"Deleted {file.name}")

print("\nDone! FIFA ranking data unified into rankings.csv")
