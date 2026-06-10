#!/usr/bin/env python3
"""
V479 — Ensemble of Diverse Configs
가설: 서로 다른 hyperparameter config의 student predictions을 averaging하면
      overfitting이 감소하고 student OOF가 낮아질 것.
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
    print("V479 — Ensemble of Diverse Configs")
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
    
    # Configs: diverse hyperparameter combinations
    student_configs = [
        # (lr, nl, md, min, sub, cbt, ra, rl)
        ("conservative", 0.03, 15, 3, 50, 0.9, 0.9, 2.0, 5.0),
        ("aggressive", 0.10, 63, 8, 20, 0.6, 0.6, 0.1, 0.1),
        ("balanced", 0.05, 31, 5, 30, 0.8, 0.8, 0.5, 1.0),
        ("deep", 0.05, 127, 10, 20, 0.7, 0.7, 0.1, 0.5),
        ("shallow", 0.08, 10, 2, 80, 0.95, 0.95, 5.0, 10.0),
        ("low_lr", 0.01, 31, 5, 30, 0.8, 0.8, 0.5, 1.0),
        ("high_sub", 0.05, 31, 5, 30, 0.5, 0.5, 0.5, 1.0),
    ]
    
    print(f"\nTesting {len(student_configs)} configs...")
    
    # OOF predictions for each config
    config_oofs = {}
    
    for cname, sl, sn, sm, smin, ss, sc, ra, rl in student_configs:
        print(f"\n--- {cname} ---")
        
        all_student_preds = np.zeros(len(X_all))
        
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_all, y_bin)):
            X_tr, X_val = X_all[tr_idx], X_all[val_idx]
            
            student_fold = np.zeros(len(X_val))
            
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
                student_fold += m.predict_proba(X_val)[:, 1] / len(TARGETS)
            
            all_student_preds[val_idx] += student_fold
        
        student_oof = 1 - np.mean(np.abs(all_student_preds - y_avg))
        config_oofs[cname] = student_oof
        print(f"  Student OOF:   {student_oof:.5f}")
    
    # Ensemble: average all configs
    print(f"\n{'='*60}")
    print("Ensemble: average all configs")
    print("=" * 60)
    
    # Run all configs and average predictions
    all_config_preds = np.zeros((len(X_all), len(student_configs)))
    
    for ci, (cname, sl, sn, sm, smin, ss, sc, ra, rl) in enumerate(student_configs):
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_all, y_bin)):
            X_tr, X_val = X_all[tr_idx], X_all[val_idx]
            student_fold = np.zeros(len(X_val))
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
                student_fold += m.predict_proba(X_val)[:, 1] / len(TARGETS)
            all_config_preds[val_idx, ci] += student_fold
    
    # Average across configs
    ensemble_student = all_config_preds.mean(axis=1)
    ensemble_student_oof = 1 - np.mean(np.abs(ensemble_student - y_avg))
    
    print(f"\nEnsemble Student OOF: {ensemble_student_oof:.5f}")
    print(f"  Δ vs baseline: {0.78570 - ensemble_student_oof:+.5f}")
    print(f"  Best single config: {min(config_oofs.items(), key=lambda x: x[1])[0]} ({min(config_oofs.values()):.5f})")
    
    # Meta predictions (use balanced config for meta)
    all_meta_preds = np.zeros(len(X_all))
    balanced_cfg = ("balanced", 0.05, 31, 5, 30, 0.8, 0.8, 0.5, 1.0)
    _, sl, sn, sm, smin, ss, sc, ra, rl = balanced_cfg
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_all, y_bin)):
        X_tr, X_val = X_all[tr_idx], X_all[val_idx]
        y_tr_avg = y_avg[tr_idx]
        
        meta_m = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.03, num_leaves=25,
            max_depth=5, min_child_samples=30,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=1.0, reg_lambda=3.0,
            random_state=99 + fold, verbose=-1, n_jobs=-1
        )
        meta_m.fit(X_tr, y_tr_avg)
        all_meta_preds[val_idx] += meta_m.predict(X_val)
    
    meta_oof = 1 - np.mean(np.abs(all_meta_preds - y_avg))
    gap = np.mean(all_meta_preds - ensemble_student)
    
    print(f"  Meta OOF:      {meta_oof:.5f}")
    print(f"  Gap:           {gap:.5f}")
    
    elapsed = time.time() - start_time
    
    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    result = {
        "version": "V479",
        "name": "Ensemble of Diverse Configs",
        "config_oofs": config_oofs,
        "ensemble_student_oof": round(ensemble_student_oof, 5),
        "meta_oof": round(meta_oof, 5),
        "gap": round(gap, 5),
        "improvement": round(0.78570 - ensemble_student_oof, 5),
        "n_configs": len(student_configs),
        "timestamp": ts,
        "total_time_s": int(elapsed)
    }
    
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    with open(EXPERIMENTS / f"v479_{ts}.json", 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Saved: {EXPERIMENTS}/v479_{ts}.json")

if __name__ == '__main__':
    main()
