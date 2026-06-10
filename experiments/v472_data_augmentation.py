#!/usr/bin/env python3
"""
V472 — Data Augmentation: Bootstrap Resampling with Noise Injection
가설: bootstrap resampling + noise injection으로 effective sample size를 늘리면
      student overfitting을 줄이고 OOF가 낮아질 것.
      특히 noise level을 다양하게 실험.
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
    print("V472 — Data Augmentation (Bootstrap + Noise Injection)")
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
    
    # Prepare clean data
    X_all = train[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    y_avg = train[TARGETS].mean(axis=1).values
    y_bin = (y_avg > 0.5).astype(int)
    
    # Compute per-feature std for noise injection
    feature_stds = np.std(X_all, axis=0)
    feature_stds = np.where(feature_stds < 1e-10, 1.0, feature_stds)
    
    from sklearn.model_selection import StratifiedKFold
    import lightgbm as lgb
    
    print("\n" + "=" * 60)
    print("Testing noise levels with 5-fold CV...")
    print("=" * 60)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    noise_levels = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]  # 0.0 = no augmentation (baseline)
    results = {}
    
    for noise_level in noise_levels:
        print(f"\n--- noise_level={noise_level:.1f} ---")
        
        all_meta_preds = np.zeros(len(X_all))
        all_student_preds = np.zeros(len(X_all))
        all_train_aug = np.zeros((len(X_all), X_all.shape[1]))  # for tracking
        
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_all, y_bin)):
            X_tr, X_val = X_all[tr_idx], X_all[val_idx]
            y_tr_avg = y_avg[tr_idx]
            
            # ===== AUGMENTATION =====
            if noise_level > 0:
                # Bootstrap resampling with noise injection
                n_extra = len(X_tr)  # double the training set
                noise = np.random.randn(n_extra, X_tr.shape[1]) * feature_stds[np.newaxis, :] * noise_level
                X_aug = X_tr[np.random.choice(len(X_tr), n_extra)] + noise
                X_tr_aug = np.concatenate([X_tr, X_aug], axis=0)
                y_tr_avg_aug = np.concatenate([y_tr_avg, y_tr_avg[np.random.choice(len(y_tr_avg), n_extra)]])
            else:
                X_tr_aug = X_tr.copy()
                y_tr_avg_aug = y_tr_avg.copy()
            
            print(f"  Fold {fold+1}: train size = {X_tr_aug.shape[0]} ({len(X_tr)} + {X_tr_aug.shape[0]-len(X_tr)} augmented)")
            
            # Student predictions (use original X_tr for prediction)
            student_tr = np.zeros(len(X_tr))
            student_val = np.zeros(len(X_val))
            
            # Define y_t_tr first (before augmentation)
            y_t_tr_orig = (train.iloc[tr_idx][TARGETS[0]] > 0.5).astype(int).values
            
            # Augment target labels too
            if noise_level > 0:
                y_t_aug = np.array([y_t_tr_orig[np.random.choice(len(y_t_tr_orig))] for _ in range(n_extra)])
                y_tr_aug = np.concatenate([y_t_tr_orig, y_t_aug])
            else:
                y_tr_aug = y_t_tr_orig.copy()
            
            for t in TARGETS:
                y_t_tr_full = y_tr_aug  # use augmented labels
                m = lgb.LGBMClassifier(
                    n_estimators=500, learning_rate=0.05, num_leaves=31,
                    max_depth=5, min_child_samples=30,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.5, reg_lambda=1.0,
                    random_state=42 + fold, verbose=-1, n_jobs=-1
                )
                m.fit(X_tr_aug, y_t_tr_full)  # train on augmented
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
            if noise_level > 0:
                y_tr_avg_aug = np.concatenate([y_tr_avg, y_tr_avg[np.random.choice(len(y_tr_avg), n_extra)]])
            meta_m.fit(X_tr_aug, y_tr_avg_aug)
            all_meta_preds[val_idx] += meta_m.predict(X_val)
        
        student_oof = 1 - np.mean(np.abs(all_student_preds - y_avg))
        meta_oof = 1 - np.mean(np.abs(all_meta_preds - y_avg))
        gap = np.mean(all_meta_preds - all_student_preds)
        
        results[noise_level] = {
            'meta': round(meta_oof, 5),
            'student': round(student_oof, 5),
            'gap': round(gap, 5),
            'improvement': round(0.78570 - student_oof, 5)  # vs baseline
        }
        
        print(f"  Meta OOF:      {meta_oof:.5f}")
        print(f"  Student OOF:   {student_oof:.5f}")
        print(f"  Gap:           {gap:.5f}")
        print(f"  Δ vs baseline: {0.78570 - student_oof:+.5f}")
    
    # Find best noise level
    best_level = min(results.items(), key=lambda x: x[1]['student'])
    print(f"\n{'='*60}")
    print(f"BEST noise_level: {best_level[0]} (student OOF: {best_level[1]['student']})")
    print("=" * 60)
    
    elapsed = time.time() - start_time
    
    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    result = {
        "version": "V472",
        "name": "Data Augmentation (Bootstrap + Noise Injection)",
        "results": {str(k): v for k, v in results.items()},
        "best_noise_level": best_level[0],
        "timestamp": ts,
        "total_time_s": round(elapsed)
    }
    
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    with open(EXPERIMENTS / f"v472_{ts}.json", 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Saved: {EXPERIMENTS}/v472_{ts}.json")

if __name__ == '__main__':
    main()
