#!/usr/bin/env python3
"""
V469 — Temporal Features: per-subject rolling stats over date dimension
가설: 각 subject별로 date 기반 rolling features(3일, 7일 이동평균, 추세)가
      health trend를 포착하여 타겟 예측 성능 향상
"""

import warnings
warnings.filterwarnings('ignore')

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path('/home/mwoo423/.openclaw/workspace')
DATA_PROC = ROOT / 'data_processed'
EXPERIMENTS = ROOT / 'experiments'
SUBMISSIONS = ROOT / 'submissions'
DATA_RAW = ROOT / 'data_raw'

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']

def add_temporal_features(df):
    """각 subject별로 date 기반 rolling stats 생성"""
    df = df.copy()
    df['date_dt'] = pd.to_datetime(df['date'])
    
    # cyclical date features
    df['dow'] = df['date_dt'].dt.dayofweek
    df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)
    df['dom'] = df['date_dt'].dt.day
    df['dom_sin'] = np.sin(2 * np.pi * df['dom'] / 31)
    df['dom_cos'] = np.cos(2 * np.pi * df['dom'] / 31)
    
    # activity feature rolling (3일, 7일)
    activity_cols = [
        'mActivity_m_activity_mean', 'wPedo_pedo_step_mean',
        'wPedo_pedo_distance_mean', 'wPedo_pedo_burned_calories_mean',
        'mScreenStatus_m_screen_use_mean', 'mUsageStats_usage_total_time_mean',
        'wHr_hr_mean', 'wLight_w_light_mean'
    ]
    activity_cols = [c for c in activity_cols if c in df.columns]
    
    for col in activity_cols:
        df[f'{col}_r3'] = df.groupby('subject_id')[col].transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )
        df[f'{col}_r7'] = df.groupby('subject_id')[col].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )
        df[f'{col}_std3'] = df.groupby('subject_id')[col].transform(
            lambda x: x.rolling(3, min_periods=1).std().fillna(0)
        )
        df[f'{col}_std7'] = df.groupby('subject_id')[col].transform(
            lambda x: x.rolling(7, min_periods=1).std().fillna(0)
        )
        # trend
        df[f'{col}_trend'] = df[f'{col}_r7'] - df[f'{col}_r3']
    
    # environment rolling
    env_cols = [c for c in df.columns if c.startswith('mAmbience_') and '_sum' in c]
    for col in env_cols:
        df[f'{col}_r7'] = df.groupby('subject_id')[col].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )
    
    # GPS rolling
    gps_cols = [c for c in df.columns if c.startswith('mGps_') and ('mean' in c or 'avg' in c)]
    for col in gps_cols:
        df[f'{col}_r7'] = df.groupby('subject_id')[col].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )
    
    # WiFi/BLE rolling
    for prefix in ['mWifi_', 'mBle_']:
        cols = [c for c in df.columns if c.startswith(prefix) and 'mean' in c]
        for col in cols:
            df[f'{col}_r7'] = df.groupby('subject_id')[col].transform(
                lambda x: x.rolling(7, min_periods=1).mean()
            )
    
    return df

def main():
    start_time = time.time()
    print("=" * 60)
    print("V469 — Temporal Features (per-subject rolling stats)")
    print("=" * 60)
    
    # Load features
    train_feat = pd.read_parquet(DATA_PROC / 'train_features_clean_v60.parquet')
    test_feat = pd.read_parquet(DATA_PROC / 'test_features_clean_v60.parquet')
    train_csv = pd.read_csv(DATA_RAW / 'ch2026_metrics_train.csv')
    
    # Merge targets into train features
    train_feat_cp = train_feat.copy()
    train_feat_cp['date_parsed'] = pd.to_datetime(train_feat_cp['date'])
    train_csv_cp = train_csv[['subject_id', 'lifelog_date', 'Q1','Q2','Q3','S1','S2','S3','S4']].copy()
    train_csv_cp['lf_parsed'] = pd.to_datetime(train_csv_cp['lifelog_date'])
    
    train = train_feat_cp.merge(
        train_csv_cp,
        left_on=['subject_id', 'date_parsed'],
        right_on=['subject_id', 'lf_parsed'],
        how='inner'
    ).drop(columns=['date_parsed', 'lf_parsed'])
    
    print(f"Train with targets: {train.shape}")
    print(f"Target rows: {train['Q1'].notnull().sum()}, Test rows: {train['Q1'].isnull().sum()}")
    print(f"Test features: {test_feat.shape}")
    
    # Add temporal features
    print("\nAdding temporal features...")
    train = add_temporal_features(train)
    test_feat = add_temporal_features(test_feat)
    
    # Only use train rows with targets
    train_valid = train[train['Q1'].notnull()].copy()
    print(f"Valid train: {train_valid.shape}")
    
    # Feature selection
    exclude_cols = ['subject_id', 'date', 'date_dt', 'lifelog_date'] + TARGETS
    feature_cols = [c for c in train_valid.columns if c not in exclude_cols
                    and train_valid[c].dtype in ['float64', 'int64', 'float32', 'int32', 'float16']
                    and train_valid[c].nunique() > 1]
    
    print(f"Feature count: {len(feature_cols)}")
    
    # Clean NaN/inf
    for c in feature_cols:
        train_valid[c] = train_valid[c].fillna(0).replace([np.inf, -np.inf], 0)
        test_feat[c] = test_feat[c].fillna(0).replace([np.inf, -np.inf], 0)
    
    X = train_valid[feature_cols].values
    X_test = test_feat[feature_cols].values
    y_avg = train_valid[TARGETS].mean(axis=1).values
    
    from sklearn.model_selection import StratifiedKFold
    import lightgbm as lgb
    
    print("\nTraining 5-fold CV...")
    
    # Use avg target as continuous for meta, binary for student
    y_bin = (y_avg > 0.5).astype(int)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    all_meta_preds = np.zeros(len(X))
    all_student_preds = np.zeros(len(X))
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y_bin)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        
        # Student model: per-target average
        student_val = np.zeros(len(X_val))
        for t in TARGETS:
            y_t = (train_valid.iloc[tr_idx][t] > 0.5).astype(int).values
            m = lgb.LGBMClassifier(
                n_estimators=500, learning_rate=0.05, num_leaves=31,
                max_depth=5, min_child_samples=30,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.5, reg_lambda=1.0,
                random_state=42 + fold, verbose=-1, n_jobs=-1
            )
            m.fit(X_tr, y_t)
            student_val += m.predict_proba(X_val)[:, 1] / len(TARGETS)
        all_student_preds[val_idx] += student_val
        
        # Meta: student preds + features -> avg target
        student_tr = np.zeros(len(X_tr))
        for t in TARGETS:
            y_t = (train_valid.iloc[tr_idx][t] > 0.5).astype(int).values
            m = lgb.LGBMClassifier(
                n_estimators=500, learning_rate=0.05, num_leaves=31,
                max_depth=5, min_child_samples=30,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.5, reg_lambda=1.0,
                random_state=42 + fold, verbose=-1, n_jobs=-1
            )
            m.fit(X_tr, y_t)
            student_tr += m.predict_proba(X_tr)[:, 1] / len(TARGETS)
        
        meta_tr = np.column_stack([student_tr, X_tr])
        meta_val = np.column_stack([student_val, X_val])
        y_tr = y_avg[tr_idx]
        
        meta_m = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.05, num_leaves=20,
            max_depth=4, min_child_samples=30,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=1.0, reg_lambda=2.0,
            random_state=99 + fold, verbose=-1, n_jobs=-1
        )
        meta_m.fit(meta_tr, y_tr)

        all_meta_preds[val_idx] += meta_m.predict(meta_val)
    
    # Evaluate
    student_oof = 1 - np.mean(np.abs(all_student_preds - y_avg))
    meta_oof = 1 - np.mean(np.abs(all_meta_preds - y_avg))
    
    elapsed = time.time() - start_time
    gap = np.mean(all_student_preds - all_meta_preds)
    gap_std = np.std(all_student_preds - all_meta_preds)
    
    temporal_added = sum(1 for c in train_valid.columns 
                        if any(k in c for k in ['_r3', '_r7', '_trend', '_std3', '_std7', 'dow', 'dom']))
    
    print(f"\n{'='*60}")
    print(f"V469 RESULTS")
    print(f"  Meta OOF:      {meta_oof:.5f}")
    print(f"  Student OOF:   {student_oof:.5f}")
    print(f"  Gap (mean):    {gap:.5f}")
    print(f"  Gap (std):     {gap_std:.5f}")
    print(f"  V308 LB:       0.63893")
    print(f"  Est LB:        {student_oof:.5f}")
    print(f"  Features:      {len(feature_cols)} ({temporal_added} temporal)")
    print(f"  Time:          {elapsed:.0f}s")
    print(f"{'='*60}")
    
    # Save result
    ts = time.strftime("%Y%m%d_%H%M%S")
    result = {
        "version": "V469",
        "name": "Temporal Features (per-subject rolling stats)",
        "avg_meta_oof": round(meta_oof, 5),
        "avg_student_oof": round(student_oof, 5),
        "v308_lb": 0.63893,
        "student_meta_gap_mean": round(gap, 5),
        "student_meta_gap_std": round(gap_std, 5),
        "n_features": len(feature_cols),
        "n_temporal_features": temporal_added,
        "timestamp": ts,
        "total_time_s": round(elapsed),
        "hypothesis": "Per-subject temporal rolling features capture health trends"
    }
    
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    with open(EXPERIMENTS / f"v469_{ts}.json", 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Saved: {EXPERIMENTS}/v469_{ts}.json")

if __name__ == '__main__':
    main()
