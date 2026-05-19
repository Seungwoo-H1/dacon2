"""
Generate features_clean_v60.parquet from features.parquet
This is the "V60" step that V61/V257/V127 depend on.
"""
import sys, gc, re, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

ROOT = Path('/home/mwoo423/.openclaw/workspace')
DATA = ROOT / 'data_processed'

TARGETS = {'Q1','Q2','Q3','S1','S2','S3','S4'}
META = {'subject_id','lifelog_date','sleep_date','sleep_date_parsed','date'}

def sanitize(n):
    return re.sub(r'[^a-zA-Z0-9_]','_',n)

def main():
    print("Loading features.parquet...")
    feat = pd.read_parquet(DATA / 'features.parquet')
    print(f"  Shape: {feat.shape}")
    
    # Feature columns (numeric, exclude meta and targets)
    feat_cols = [c for c in feat.columns 
                 if c not in META | TARGETS
                 and np.issubdtype(feat[c].dtype, np.number)]
    print(f"  Base features: {len(feat_cols)}")
    
    # Add z-scores (personalization)
    zscore_cols = []
    for col in feat_cols:
        zc = f'{col}_zscore'
        zscore_cols.append(zc)
        vals = feat[col].fillna(0)
        grp = vals.groupby(feat['subject_id']).agg(mean='mean', std='std').reset_index()
        grp.columns = ['subject_id', f'{col}_subj_mean', f'{col}_subj_std']
        feat = feat.merge(grp, on='subject_id', how='left')
        sm = feat[f'{col}_subj_mean']
        ss = feat[f'{col}_subj_std']
        mask = (ss == 0) | feat[col].isnull()
        feat[f'{col}_zscore'] = np.where(mask, 0.0, (feat[col].fillna(0) - sm) / np.maximum(ss, 1e-8))
        gc.collect()
    
    print(f"  Z-score features: {len(zscore_cols)}")
    print(f"  Total features: {len(feat_cols) + len(zscore_cols)}")
    
    # Save
    out_path = DATA / 'features_clean_v60.parquet'
    feat.to_parquet(out_path, index=False)
    print(f"\n  Saved: {out_path}")
    print(f"  Shape: {feat.shape}")

if __name__ == '__main__':
    main()
