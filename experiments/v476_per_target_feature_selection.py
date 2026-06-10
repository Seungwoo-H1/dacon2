#!/usr/bin/env python3
"""
V476 — Per-Target Feature Selection
가설: 모든 타겟에 같은 feature set을 쓰는 대신, 각 타겟별로 independently feature ranking →
      타겟별 최적 feature set을 사용하면 student 성능이 향상될 것.
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
    print("V476 — Per-Target Feature Selection")
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
    
    from sklearn.model_selection import StratifiedKFold
    import lightgbm as lgb
    
    X_all = train[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    y_avg = train[TARGETS].mean(axis=1).values
    y_bin = (y_avg > 0.5).astype(int)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\n" + "=" * 60)
    print("Testing per-target feature selection")
    print("=" * 60)
    
    k_values = [5, 10, 15, 20, 30, 40, 50, 68]  # number of features per target
    results = {}
    
    for k in k_values:
        print(f"\n--- k={k} (per-target top-{k} features) ---")
        
        # For each fold, find top-k features per target independently
        all_meta_preds = np.zeros(len(X_all))
        all_student_preds = np.zeros(len(X_all))
        
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_all, y_bin)):
            X_tr, X_val = X_all[tr_idx], X_all[val_idx]
            y_tr_avg = y_avg[tr_idx]
            
            # Find top-k features per target using training data
            per_target_features = {}
            for t in TARGETS:
                y_t_tr = (train.iloc[tr_idx][t] > 0.5).astype(int).values
                m = lgb.LGBMClassifier(
                    n_estimators=200, learning_rate=0.1, num_leaves=31,
                    max_depth=5, random_state=42 + fold, verbose=-1, n_jobs=-1
                )
                m.fit(X_tr, y_t_tr)
                importances = m.feature_importances_
                top_k_idx = np.argsort(importances)[-k:][::-1]
                per_target_features[t] = top_k_idx
            
            # For student: per-target model uses per-target features
            student_tr = np.zeros(len(X_tr))
            student_val = np.zeros(len(X_val))
            
            for t in TARGETS:
                y_t_tr = (train.iloc[tr_idx][t] > 0.5).astype(int).values
                feat_idx = per_target_features[t]
                
                m = lgb.LGBMClassifier(
                    n_estimators=500, learning_rate=0.05, num_leaves=31,
                    max_depth=5, min_child_samples=30,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.5, reg_lambda=1.0,
                    random_state=42 + fold, verbose=-1, n_jobs=-1
                )
                m.fit(X_tr[:, feat_idx], y_t_tr)
                student_tr += m.predict_proba(X_tr[:, feat_idx])[:, 1] / len(TARGETS)
                student_val += m.predict_proba(X_val[:, feat_idx])[:, 1] / len(TARGETS)
            
            all_student_preds[val_idx] += student_val
            
            # Meta: use union of all per-target features
            all_feat_idx = set()
            for t in TARGETS:
                all_feat_idx.update(per_target_features[t])
            all_feat_idx = sorted(all_feat_idx)
            
            meta_m = lgb.LGBMRegressor(
                n_estimators=500, learning_rate=0.03, num_leaves=25,
                max_depth=5, min_child_samples=30,
                subsample=0.8, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=3.0,
                random_state=99 + fold, verbose=-1, n_jobs=-1
            )
            meta_m.fit(X_tr[:, all_feat_idx], y_tr_avg)
            all_meta_preds[val_idx] += meta_m.predict(X_val[:, all_feat_idx])
        
        student_oof = 1 - np.mean(np.abs(all_student_preds - y_avg))
        meta_oof = 1 - np.mean(np.abs(all_meta_preds - y_avg))
        gap = np.mean(all_meta_preds - all_student_preds)
        
        results[k] = {
            'meta': round(meta_oof, 5),
            'student': round(student_oof, 5),
            'gap': round(gap, 5),
            'improvement': round(0.78570 - student_oof, 5),
        }
        
        print(f"  Meta OOF:      {meta_oof:.5f}")
        print(f"  Student OOF:   {student_oof:.5f}")
        print(f"  Gap:           {gap:.5f}")
        print(f"  Δ vs baseline: {0.78570 - student_oof:+.5f}")
    
    # Baseline for comparison (all features for all targets)
    print(f"\n--- baseline (all features) ---")
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
    results['baseline'] = {
        'meta': round(meta_oof, 5),
        'student': round(student_oof, 5),
        'gap': round(np.mean(all_meta_preds - all_student_preds), 5),
    }
    print(f"  Meta OOF:      {meta_oof:.5f}")
    print(f"  Student OOF:   {student_oof:.5f}")
    
    # Find best
    best = min(results.items(), key=lambda x: x[1]['student'])
    print(f"\n{'='*60}")
    print(f"BEST: k={best[0]} (student OOF: {best[1]['student']})")
    print("=" * 60)
    
    elapsed = time.time() - start_time
    
    ts = time.strftime("%Y%m%d_%H%M%S")
    result = {
        "version": "V476",
        "name": "Per-Target Feature Selection",
        "results": {str(k): v for k, v in results.items()},
        "best_k": best[0],
        "timestamp": ts,
        "total_time_s": round(elapsed)
    }
    
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    with open(EXPERIMENTS / f"v476_{ts}.json", 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Saved: {EXPERIMENTS}/v476_{ts}.json")

if __name__ == '__main__':
    main()
