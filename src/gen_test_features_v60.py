"""
gen_test_features_v60.py — Generate test_features_clean_v60.parquet
Build test features from raw parquet files + sample submission.
"""
import sys, gc, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, DATA_PROCESSED, SAMPLE_CSV, PARQUET_FILES

TARGETS = {'Q1','Q2','Q3','S1','S2','S3','S4'}
META_COLS = {'subject_id','lifelog_date','sleep_date','date'}
AGG_FUNCS = ['mean', 'std', 'min', 'max', 'count']

def parse_ambience_array(val):
    """
    Parse m_ambience numpy array of [category, value] pairs.
    val is already a numpy ndarray of ndarray objects, NOT JSON string.
    """
    if val is None:
        return {}
    if not isinstance(val, np.ndarray):
        return {}
    
    result = {}
    for item in val:
        if isinstance(item, (np.ndarray, list)) and len(item) >= 2:
            category = str(item[0])
            try:
                value = float(item[1])
            except:
                value = 0.0
            result[category] = value
    return result

def aggregate_device(df, device_name):
    """Aggregate a single device's data by subject and date."""
    if df.empty:
        return pd.DataFrame()
    
    df = df.copy()
    
    # Handle datetime
    if 'timestamp' in df.columns and 'date' not in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
    
    if 'date' not in df.columns:
        return pd.DataFrame()
    
    # Special handling for mAmbience
    if device_name == 'mAmbience':
        if 'm_ambience' in df.columns:
            records = []
            for _, row in df.iterrows():
                parsed = parse_ambience_array(row['m_ambience'])
                for category, value in parsed.items():
                    records.append({
                        'subject_id': row['subject_id'],
                        'date': row['date'],
                        'ambience_category': category,
                        'ambience_value': value,
                    })
            
            if records:
                amb_df = pd.DataFrame(records)
                agg = amb_df.groupby(['subject_id', 'date', 'ambience_category'])['ambience_value'].sum().reset_index()
                pivoted = agg.pivot_table(
                    index=['subject_id', 'date'],
                    columns='ambience_category',
                    values='ambience_value',
                    fill_value=0
                )
                pivoted.columns = [f'mAmbience_ambience_{col}_sum' for col in pivoted.columns]
                pivoted = pivoted.reset_index()
                return pivoted
        return pd.DataFrame()
    
    # Standard numeric aggregation
    numeric_cols = []
    for col in df.columns:
        if col in ['subject_id', 'date', 'timestamp']:
            continue
        if df[col].dtype in [np.float64, np.int64, float, int, bool, np.bool_]:
            numeric_cols.append(col)
    
    if not numeric_cols:
        return pd.DataFrame()
    
    agg_dfs = []
    for col in numeric_cols:
        col_name = f'{device_name}_{col}'
        for agg_func in AGG_FUNCS:
            agg_col = f'{col_name}_{agg_func}'
            result = df.groupby(['subject_id', 'date'])[col].agg(agg_func).reset_index()
            result.columns = ['subject_id', 'date', agg_col]
            agg_dfs.append(result)
    
    if not agg_dfs:
        return pd.DataFrame()
    
    result = agg_dfs[0]
    for df2 in agg_dfs[1:]:
        result = result.merge(df2, on=['subject_id', 'date'], how='left')
    
    return result

def build_test_features():
    """Build test features from raw parquet files + sample submission."""
    print("Loading sample submission...")
    sample = pd.read_csv(SAMPLE_CSV, parse_dates=['sleep_date', 'lifelog_date'])
    sample['date'] = sample['lifelog_date'].dt.date
    
    print(f"Sample shape: {sample.shape}")
    print(f"Test subjects: {sorted(sample['subject_id'].unique())}")
    
    # Load training features.parquet for z-score statistics
    print("\nLoading training features.parquet for z-score stats...")
    train_feat = pd.read_parquet(DATA_PROCESSED / 'features.parquet')
    
    # Load all raw parquet files
    print("\nLoading raw parquet files...")
    feature_dfs = [sample[['subject_id', 'date', 'lifelog_date']].copy()]
    
    for device_name, fname in PARQUET_FILES.items():
        path = DATA_DIR / fname
        if not path.exists():
            print(f"  Skipping {fname} (not found)")
            continue
        
        print(f"  Loading {fname}...")
        df = pd.read_parquet(path)
        
        # Filter to test subjects only
        df = df[df['subject_id'].isin(sample['subject_id'].unique())].copy()
        
        if df.empty:
            print(f"    Empty for test subjects, skipping")
            continue
        
        print(f"    Shape after filter: {df.shape}")
        
        # Build features for this device
        dev_df = aggregate_device(df, device_name)
        
        if not dev_df.empty:
            feature_dfs.append(dev_df)
            feat_cols_count = len([c for c in dev_df.columns if c not in ['subject_id', 'date', 'lifelog_date', 'sleep_date']])
            print(f"    Generated {feat_cols_count} feature columns")
        
        del df
        gc.collect()
    
    # Merge all features
    print("\nMerging all features...")
    result = feature_dfs[0]
    for df in feature_dfs[1:]:
        if not df.empty:
            result = result.merge(df, on=['subject_id', 'date'], how='left')
    
    result = result.fillna(0)
    
    # Add z-score features (using training data stats)
    print("\nAdding z-score features...")
    base_feat_cols = [c for c in result.columns 
                      if c not in META_COLS | TARGETS
                      and result[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]
    
    target_subjects = set(sample['subject_id'].unique())
    
    for col in base_feat_cols:
        # Get stats from training data for same subjects
        if col in train_feat.columns:
            train_mask = train_feat['subject_id'].isin(target_subjects)
            train_col = train_feat[train_mask][col].fillna(0)
        else:
            continue
        
        if train_col.empty or train_col.nunique() <= 1:
            continue
        
        # Compute global mean and std from training (across all subjects)
        global_mean = train_col.mean()
        global_std = train_col.std()
        if pd.isna(global_std) or global_std == 0:
            global_std = 1e-8
        
        # Apply z-score
        zc = f'{col}_zscore'
        result[zc] = (result[col].fillna(0) - global_mean) / global_std
    
    # Save
    out_path = DATA_PROCESSED / 'test_features_clean_v60.parquet'
    result.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"Shape: {result.shape}")
    
    # Verify
    non_zero = (result != 0).sum().sum()
    total = result.values.size
    print(f"Non-zero values: {non_zero}/{total} ({100*non_zero/total:.1f}%)")
    
    # Check feature columns
    print("\nFeature sample (first 15):")
    feat_cols = [c for c in result.columns if c not in META_COLS | TARGETS and c != 'date'][:15]
    for c in feat_cols:
        if result[c].dtype in [np.float64, np.int64, float, int]:
            print(f"  {c}: min={result[c].min():.4f}, max={result[c].max():.4f}, mean={result[c].mean():.4f}")
    
    return result

if __name__ == '__main__':
    build_test_features()
