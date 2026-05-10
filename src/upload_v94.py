"""
Upload v94 submission to Dacon leaderboard
"""
import pandas as pd
from pathlib import Path

SUBMIT = Path('submissions')
files = sorted(SUBMIT.glob('v94_submission_*.csv'))
if not files:
    print("No v94 submission file found!")
    exit(1)

sub_path = files[-1]
print(f"Uploading: {sub_path}")
print(f"Size: {pd.read_csv(sub_path).shape}")

# Check format
df = pd.read_csv(sub_path)
print(f"Columns: {list(df.columns)}")
print(f"Shape: {df.shape}")
print(f"Nulls:\n{df.isnull().sum()}")
print(f"\nFirst 3 rows:\n{df.head(3)}")

# Save as the canonical name
canonical = SUBMIT / 'v94_submission.csv'
df.to_csv(canonical, index=False)
print(f"\nSaved canonical: {canonical}")
