"""
Upload V97 (Temperature Scaling) to Dacon leaderboard.
Uses requests to call the Dacon submission API.
"""
import pandas as pd
from pathlib import Path
import sys
import os

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data_raw"
SUBMIT = ROOT / "submissions"

# Find latest V97 submission
v97_files = sorted(SUBMIT.glob("submission_v97_temp_*.csv"))
if not v97_files:
    print("ERROR: No V97 submission found")
    sys.exit(1)
sub_path = v97_files[-1]
print(f"Uploading: {sub_path.name}")

# Load and validate
sub = pd.read_csv(sub_path)
sample = pd.read_csv(DATA_RAW / "ch2026_submission_sample.csv")

print(f"Size: {sub.shape}")
print(f"Columns: {list(sub.columns)}")
print(f"Nulls: {sub.isnull().sum().sum()}")
print(f"First 3 rows:\n{sub.head(3)}")

assert sub.shape[0] == sample.shape[0], f"Row count mismatch"
assert sorted(sub.columns) == sorted(sample.columns), "Column mismatch"
print("✅ Format validated")

# Save canonical
canonical = SUBMIT / "v97_submission.csv"
sub.to_csv(canonical, index=False)
print(f"Saved canonical: {canonical}")

# Upload via Dacon API
try:
    import requests
    
    # Get token from env or default
    token = os.getenv("DACON_TOKEN", "54b8510a-9698-423d-8ea66c3597e1")
    
    # Try the codex upload endpoint
    url = "https://dacon.io/competitions/submit/answer/nl/"
    headers = {
        "Authorization": f"Bearer {token}",
    }
    
    with open(sub_path, "rb") as f:
        files = {"file": (sub_path.name, f, "text/csv")}
        data = {"competitionId": 2026}
        
        print(f"\nUploading to {url}...")
        resp = requests.post(url, headers=headers, data=data, files=files, timeout=60)
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.text[:1000]}")
        
        if resp.status_code == 200:
            print("✅ Upload successful!")
        else:
            print("❌ Upload failed")
            
except Exception as e:
    print(f"Upload error: {e}")
    print("\nManual upload required:")
    print(f"  File: {sub_path}")
    print(f"  Competition: https://dacon.io/competitions/official/236690")
