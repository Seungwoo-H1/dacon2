#!/usr/bin/env python3
"""
V471 — Per-Subject Feature Selection
가설: 각 subject별로 predictive한 feature subset이 다름.
      모든 subject에 같은 feature set을 쓰는 대신, subject별 clustering → cluster별 feature ranking.
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
DATA_RAW = ROOT / 'data_raw'

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']

def main():
    start_time = time.time()
    print("=" * 60)
    print("V471 — Per-Subject Feature Selection")
    print("=" * 60)
    
    # Load
    train_feat = pd.read_parquet(DATA_PROC / 'train_features_clean_v60.parquet')
    test_feat = pd.read_parquet(DATA_PROC / 'test_features_clean_v60.parquet')
    train_csv = pd.read_csv(DATA_RAW / 'ch2026_metrics_train.csv')
    
    # Merge targets
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
    
    print(f"Train: {train.shape}")
    
    exclude_cols = ['subject_id', 'date', 'lifelog_date'] + TARGETS
    feature_cols = [c for c in train.columns if c not in exclude_cols
                    and train[c].dtype in ['float64', 'int64', 'float32', 'int32', 'float16']
                    and train[c].nunique() > 1]
    
    print(f"Base features: {len(feature_cols)}")
    
    # Per-subject feature importance analysis
    print("\nAnalyzing per-subject feature importance...")
    subjects = sorted(train['subject_id'].unique())
    
    # Get mean feature importance per subject (using a simple model)
    subject_importance = {}
    for subj in subjects:
        subj_data = train[train['subject_id'] == subj]
        if len(subj_data) < 10:
            continue
        
        X_s = subj_data[feature_cols].fillna(0).values
        y_s = (subj_data[TARGETS].mean(axis=1) > 0.5).astype(int).values
        
        if len(np.unique(y_s)) < 2:
            continue
        
        from sklearn.model_selection import cross_val_score
        import lightgbm as lgb
        
        imp = lgb.LGBMClassifier(
            n_estimators=100, learning_rate=0.1, num_leaves=15,
            max_depth=4, random_state=42, verbose=-1, n_jobs=-1
        )
        imp.fit(X_s, y_s)
        subject_importance[subj] = imp.feature_importances_
        print(f"  {subj}: importance sum = {imp.feature_importances_.sum():.0f}")
    
    # Identify feature subsets that are consistently important across subjects
    all_importances = np.array([v for v in subject_importance.values()])
    mean_imp = all_importances.mean(axis=0)
    std_imp = all_importances.std(axis=0)
    
    # Features with low std → consistently useful across subjects
    # Features with high std → subject-specific importance
    print(f"\nMean importance std: {std_imp.mean():.2f}")
    print(f"Mean importance mean: {mean_imp.mean():.2f}")
    
    # Two strategies:
    # Strategy A: Keep only features with low std (consistent across subjects)
    # Strategy B: Use consensus feature selection (importance > median in >50% subjects)
    
    consensus_threshold = np.median(mean_imp)
    consistent_mask = std_imp < np.median(std_imp)  # top-50% consistent features
    
    # Count subjects where each feature is important (> mean importance)
    above_mean = (all_importances > mean_imp[np.newaxis, :]).astype(float)
    subject_consensus = above_mean.mean(axis=0)  # fraction of subjects with above-mean importance
    
    print(f"\nFeatures above-mean importance in >50% subjects: {(subject_consensus > 0.5).sum()}")
    print(f"Features with std < median: {consistent_mask.sum()}")
    
    # Try different feature selection strategies
    # Fallback to 'all' if empty
    c50 = [feature_cols[i] for i in range(len(feature_cols)) if subject_consensus[i] > 0.5]
    c70 = [feature_cols[i] for i in range(len(feature_cols)) if subject_consensus[i] > 0.7]
    
    strategies = {
        'all': feature_cols,
        'consistent': [feature_cols[i] for i in range(len(feature_cols)) if consistent_mask[i]],
        'consensus_50': c50 if c50 else feature_cols,  # fallback to all
        'consensus_70': c70 if c70 else feature_cols,  # fallback to all
    }
    
    from sklearn.model_selection import StratifiedKFold
    import lightgbm as lgb
    
    print("\n" + "=" * 60)
    print("Testing strategies with 5-fold CV...")
    print("=" * 60)
    
    X_all = train[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    y_avg = train[TARGETS].mean(axis=1).values
    y_bin = (y_avg > 0.5).astype(int)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {}
    
    for name, cols in strategies.items():
        print(f"\n--- {name}: {len(cols)} features ---")
        
        X = train[cols].fillna(0).replace([np.inf, -np.inf], 0).values
        
        all_meta_preds = np.zeros(len(X))
        all_student_preds = np.zeros(len(X))
        
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y_bin)):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr_avg = y_avg[tr_idx]
            
            # Student predictions
            student_tr = np.zeros(len(X_tr))
            student_val = np.zeros(len(X_val))
            
            for t in TARGETS:
                y_t_tr = (train.iloc[tr_idx][t] > 0.5).astype(int).values
                m = lgb.LGBMClassifier(
                    n_estimators=500, learning_rate=0.05, num_leaves=31,
                    max_depth=5, min_child_samples=30,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.5, reg_lambda=1.0,
                    random_state=42 + fold, verbose=-1, n_jobs=-1
                )
                m.fit(X_tr, y_t_tr)
                student_tr += m.predict_proba(X_tr)[:, 1] / len(TARGETS)
                student_val += m.predict_proba(X_val)[:, 1] / len(TARGETS)
            
            all_student_preds[val_idx] += student_val
            
            # Meta predictions
            meta_m = lgb.LGBMRegressor(
                n_estimators=500, learning_rate=0.03, num_leaves=25,
                max_depth=5, min_child_samples=30,
                subsample=0.8, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=3.0,
                random_state=99 + fold, verbose=-1, n_jobs=-1
            )
            meta_m.fit(X_tr, y_tr_avg)
            all_meta_preds[val_idx] += meta_m.predict(X_val)
        
        student_oof = 1 - np.mean(np.abs(all_student_preds - y_avg))
        meta_oof = 1 - np.mean(np.abs(all_meta_preds - y_avg))
        gap = np.mean(all_meta_preds - all_student_preds)
        
        results[name] = {
            'meta': round(meta_oof, 5),
            'student': round(student_oof, 5),
            'gap': round(gap, 5),
            'n_features': len(cols)
        }
        
        print(f"  Meta OOF:      {meta_oof:.5f}")
        print(f"  Student OOF:   {student_oof:.5f}")
        print(f"  Gap:           {gap:.5f}")
        print(f"  Student-Meta:  {student_oof - meta_oof:+.5f}")
    
    # Find best strategy
    best = min(results.items(), key=lambda x: x[1]['student'])
    print(f"\n{'='*60}")
    print(f"BEST: {best[0]} (student OOF: {best[1]['student']})")
    print("=" * 60)
    
    elapsed = time.time() - start_time
    
    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    result = {
        "version": "V471",
        "name": "Per-Subject Feature Selection",
        "strategies": results,
        "best_strategy": best[0],
        "timestamp": ts,
        "total_time_s": round(elapsed)
    }
    
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    with open(EXPERIMENTS / f"v471_{ts}.json", 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Saved: {EXPERIMENTS}/v471_{ts}.json")

if __name__ == '__main__':
    main()
