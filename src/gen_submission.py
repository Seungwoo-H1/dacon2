"""
Generate submission CSV from OOF predictions.
Usage: python gen_submission.py [v52|v53|...]
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = ROOT / "data_processed"
SUBMIT_DIR = ROOT / "submissions"
SUBMIT_DIR.mkdir(exist_ok=True)

SAMPLE = pd.read_csv(ROOT / "data_raw" / "ch2026_submission_sample.csv")
TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']

def gen_submission(version):
    # Load OOF
    oof_path = DATA_PROCESSED / f"oof_{version}.csv"
    oof = pd.read_csv(oof_path)
    
    # Load sample
    sample = pd.read_csv(ROOT / "data_raw" / "ch2026_submission_sample.csv")
    
    # Match keys
    oof = oof.merge(sample[['subject_id', 'sleep_date', 'lifelog_date']], 
                    on=['subject_id', 'sleep_date', 'lifelog_date'],
                    how='right')
    
    # Build submission
    sub = sample.copy()
    for t in TARGETS:
        if t in oof.columns:
            sub[t] = oof[t].values
        else:
            print(f"  WARNING: {t} not in OOF!")
    
    # Check NaN
    for t in TARGETS:
        nan_count = sub[t].isna().sum()
        if nan_count > 0:
            print(f"  WARNING: {t} has {nan_count} NaNs, filling with 0.5")
            sub[t] = sub[t].fillna(0.5)
    
    sub_path = SUBMIT_DIR / f"submission_{version}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    sub.to_csv(sub_path, index=False)
    
    print(f"  {version} Meta: avg_cal={json.loads((DATA_PROCESSED / f'{version}_meta.json').read_text()).get('avg_cal_loss', 'N/A')}")
    print(f"  OOF rows: {len(oof)}, Sample rows: {len(sample)}")
    print(f"  ✅ Saved: {sub_path}")
    print(f"  Preview:\n{sub.head(3)}")
    print(f"  Stats:")
    for t in TARGETS:
        print(f"    {t}: min={sub[t].min():.4f} max={sub[t].max():.4f} mean={sub[t].mean():.4f}")
    
    return sub_path

if __name__ == "__main__":
    version = sys.argv[1] if len(sys.argv) > 1 else "v53"
    print(f"Generating submission for {version}...")
    gen_submission(version)
