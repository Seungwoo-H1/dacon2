#!/usr/bin/env python3
"""
V473 — Target Transformation: Binary → Ordinal/Multi-class Reformulation
가설: binary classification의 log-loss 한계 대신:
      1. Binary targets → ordinal regression (0=low, 1=mid, 2=high)
      2. Thresholds로 multi-class: [0, 0.3)=0, [0.3, 0.7)=1, [0.7, 1.0]=2
      Ordinal regression이 binary보다 더 많은 signal을 활용.
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
    print("V473 — Target Transformation: Binary → Ordinal")
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
    
    X_all = train[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    y_avg = train[TARGETS].mean(axis=1).values
    y_bin = (y_avg > 0.5).astype(int)
    
    # Create ordinal targets for each original target
    # Thresholds: low=0 if y<0.3, mid=1 if 0.3<=y<0.7, high=2 if y>=0.7
    def to_ordinal(y):
        if y < 0.3:
            return 0
        elif y < 0.7:
            return 1
        else:
            return 2
    
    y_ordinal = np.array([to_ordinal(v) for v in y_avg])
    
    # Check class distribution
    unique, counts = np.unique(y_ordinal, return_counts=True)
    print(f"\nOrdinal class distribution: {dict(zip(unique, counts))}")
    
    from sklearn.model_selection import StratifiedKFold
    import lightgbm as lgb
    
    print("\n" + "=" * 60)
    print("Testing approaches: ordinal vs binary")
    print("=" * 60)
    
    approaches = {
        'ordinal_lgbm': 'ordinal with LGBM',
        'ordinal_xgb': 'ordinal with XGB',
        'binary_baseline': 'binary baseline (same as V466)',
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {}
    
    for name, desc in approaches.items():
        print(f"\n--- {desc} ---")
        
        all_meta_preds = np.zeros(len(X_all))
        all_student_preds = np.zeros(len(X_all))
        
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_all, y_bin)):
            X_tr, X_val = X_all[tr_idx], X_all[val_idx]
            y_tr_avg = y_avg[tr_idx]
            
            # Student predictions (binary, same for all approaches)
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
            if name == 'ordinal_lgbm':
                # Ordinal regression meta
                y_tr_ord = y_ordinal[tr_idx]
                meta_m = lgb.LGBMClassifier(
                    n_estimators=500, learning_rate=0.03, num_leaves=25,
                    max_depth=5, min_child_samples=30,
                    subsample=0.8, colsample_bytree=0.7,
                    reg_alpha=1.0, reg_lambda=3.0,
                    random_state=99 + fold, verbose=-1, n_jobs=-1,
                    objective='multiclass', num_class=3
                )
                meta_m.fit(X_tr, y_tr_ord)
                # Convert ordinal probs to continuous: weighted average
                probs = meta_m.predict_proba(X_val)  # (N, 3)
                class_weights = np.array([0, 0.5, 1.0])
                all_meta_preds[val_idx] += (probs * class_weights).sum(axis=1)
            elif name == 'ordinal_xgb':
                try:
                    from xgboost import XGBClassifier
                    y_tr_ord = y_ordinal[tr_idx]
                    meta_m = XGBClassifier(
                        n_estimators=500, learning_rate=0.03, max_depth=5,
                        reg_alpha=1.0, reg_lambda=3.0,
                        subsample=0.8, colsample_bytree=0.7,
                        random_state=99 + fold, verbosity=0,
                        objective='multi:softprob', num_class=3
                    )
                    meta_m.fit(X_tr, y_tr_ord)
                    probs = meta_m.predict_proba(X_val)
                    class_weights = np.array([0, 0.5, 1.0])
                    all_meta_preds[val_idx] += (probs * class_weights).sum(axis=1)
                except ImportError:
                    print("  XGB not available, skipping")
                    break
            else:
                # Binary baseline (regression meta, same as V466)
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
        }
        
        print(f"  Meta OOF:      {meta_oof:.5f}")
        print(f"  Student OOF:   {student_oof:.5f}")
        print(f"  Gap:           {gap:.5f}")
    
    # Also test: continuous ordinal target (treating ordinal as regression)
    print(f"\n--- ordinal regression (LGBMRegressor) ---")
    all_meta_preds = np.zeros(len(X_all))
    all_student_preds = np.zeros(len(X_all))
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_all, y_bin)):
        X_tr, X_val = X_all[tr_idx], X_all[val_idx]
        y_tr_avg = y_avg[tr_idx]
        
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
        
        y_tr_ord = y_ordinal[tr_idx]
        meta_m = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.03, num_leaves=25,
            max_depth=5, min_child_samples=30,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=1.0, reg_lambda=3.0,
            random_state=99 + fold, verbose=-1, n_jobs=-1
        )
        meta_m.fit(X_tr, y_tr_ord)
        all_meta_preds[val_idx] += meta_m.predict(X_val)
    
    student_oof = 1 - np.mean(np.abs(all_student_preds - y_avg))
    meta_oof = 1 - np.mean(np.abs(all_meta_preds - y_avg))
    gap = np.mean(all_meta_preds - all_student_preds)
    results['ordinal_reg'] = {
        'meta': round(meta_oof, 5),
        'student': round(student_oof, 5),
        'gap': round(gap, 5),
    }
    print(f"  Meta OOF:      {meta_oof:.5f}")
    print(f"  Student OOF:   {student_oof:.5f}")
    print(f"  Gap:           {gap:.5f}")
    
    # Find best
    best = min(results.items(), key=lambda x: x[1]['student'])
    print(f"\n{'='*60}")
    print(f"BEST: {best[0]} (student OOF: {best[1]['student']})")
    print("=" * 60)
    
    elapsed = time.time() - start_time
    
    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    result = {
        "version": "V473",
        "name": "Target Transformation: Binary → Ordinal",
        "results": {k: v for k, v in results.items()},
        "best_approach": best[0],
        "ordinal_distribution": {str(int(k)): int(v) for k, v in zip(unique, counts)},
        "timestamp": ts,
        "total_time_s": round(elapsed)
    }
    
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    with open(EXPERIMENTS / f"v473_{ts}.json", 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Saved: {EXPERIMENTS}/v473_{ts}.json")

if __name__ == '__main__':
    main()
