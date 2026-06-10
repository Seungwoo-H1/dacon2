#!/usr/bin/env python3
"""
V478 — Regularization Sweep
가설: 적절한 strong regularization(Reg_alpha, Reg_lambda, min_child_samples)이 student overfitting을 줄이고
      OOF를 낮출 것. V465의 실패(n_est=5000)를 재현하지 않도록 n_est는 500으로 고정.
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
    print("V478 — Regularization Sweep")
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
    
    from sklearn.model_selection import StratifiedKFold
    import lightgbm as lgb
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Configurations to test
    configs = [
        # (student_lr, student_nl, student_md, student_min, student_sub, student_cbt, meta_lr, meta_nl, meta_md, meta_min, meta_sub, meta_cbt, reg_alpha, reg_lambda)
        ("baseline", 0.05, 31, 5, 30, 0.8, 0.8, 0.03, 25, 5, 30, 0.8, 0.7, 0.5, 1.0),
        ("strong_reg_alpha", 0.05, 31, 5, 30, 0.8, 0.8, 0.03, 25, 5, 30, 0.8, 0.7, 5.0, 10.0),
        ("strong_reg_both", 0.05, 31, 5, 30, 0.8, 0.8, 0.03, 25, 5, 30, 0.8, 0.7, 10.0, 20.0),
        ("deep_tree_high_reg", 0.05, 63, 8, 50, 0.7, 0.7, 0.03, 31, 8, 50, 0.7, 0.6, 5.0, 10.0),
        ("shallow_tree_high_reg", 0.05, 15, 3, 50, 0.9, 0.9, 0.03, 10, 3, 50, 0.9, 0.9, 10.0, 30.0),
        ("low_lr_high_est", 0.01, 31, 5, 30, 0.8, 0.8, 0.005, 25, 5, 30, 0.8, 0.7, 5.0, 10.0),
        ("low_lr_nest1000", 0.02, 31, 5, 30, 0.8, 0.8, 0.01, 25, 5, 30, 0.8, 0.7, 3.0, 5.0),
        ("aggressive_subsample", 0.05, 31, 5, 40, 0.5, 0.5, 0.03, 25, 5, 40, 0.5, 0.5, 3.0, 5.0),
    ]
    
    print(f"\nTesting {len(configs)} configurations...")
    
    results = {}
    
    for name, sl, sn, sm, smin, ss, sc, ml, mn, mm, mmin, ms, mc, ra, rl in configs:
        print(f"\n--- {name} ---")
        
        all_meta_preds = np.zeros(len(X_all))
        all_student_preds = np.zeros(len(X_all))
        
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_all, y_bin)):
            X_tr, X_val = X_all[tr_idx], X_all[val_idx]
            y_tr_avg = y_avg[tr_idx]
            
            # Student
            student_tr = np.zeros(len(X_tr))
            student_val = np.zeros(len(X_val))
            
            for t in TARGETS:
                y_t_tr = (train.iloc[tr_idx][t] > 0.5).astype(int).values
                m = lgb.LGBMClassifier(
                    n_estimators=500, learning_rate=sl, num_leaves=sn,
                    max_depth=sm, min_child_samples=smin,
                    subsample=ss, colsample_bytree=sc,
                    reg_alpha=ra, reg_lambda=rl,
                    random_state=42 + fold, verbose=-1, n_jobs=-1
                )
                m.fit(X_tr, y_t_tr)
                student_tr += m.predict_proba(X_tr)[:, 1] / len(TARGETS)
                student_val += m.predict_proba(X_val)[:, 1] / len(TARGETS)
            
            all_student_preds[val_idx] += student_val
            
            # Meta
            meta_m = lgb.LGBMRegressor(
                n_estimators=500, learning_rate=ml, num_leaves=mn,
                max_depth=mm, min_child_samples=mmin,
                subsample=ms, colsample_bytree=mc,
                reg_alpha=ra, reg_lambda=rl,
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
            'improvement': round(0.78570 - student_oof, 5),
        }
        
        print(f"  Meta OOF:      {meta_oof:.5f}")
        print(f"  Student OOF:   {student_oof:.5f}")
        print(f"  Gap:           {gap:.5f}")
        print(f"  Δ student:     {0.78570 - student_oof:+.5f}")
    
    # Find best
    best = min(results.items(), key=lambda x: x[1]['student'])
    print(f"\n{'='*60}")
    print(f"BEST: {best[0]} (student OOF: {best[1]['student']})")
    print("=" * 60)
    
    elapsed = time.time() - start_time
    
    ts = time.strftime("%Y%m%d_%H%M%S")
    result = {
        "version": "V478",
        "name": "Regularization Sweep",
        "results": {k: v for k, v in results.items()},
        "best_config": best[0],
        "timestamp": ts,
        "total_time_s": int(elapsed)
    }
    
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    with open(EXPERIMENTS / f"v478_{ts}.json", 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Saved: {EXPERIMENTS}/v478_{ts}.json")

if __name__ == '__main__':
    main()
