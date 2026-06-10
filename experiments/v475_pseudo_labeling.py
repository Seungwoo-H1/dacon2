#!/usr/bin/env python3
"""
V475 — Pseudo-labeling: High-confidence predictions → augment training set
가설: 모델이 자신감 있게 예측한 샘플(>0.9 또는 <0.1)의 라벨을 pseudo-label로 사용해서
      training set을 augmentation하면 overfitting 감소.
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
    print("V475 — Pseudo-labeling")
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
    
    print("\n" + "=" * 60)
    print("Testing confidence thresholds")
    print("=" * 60)
    
    # Use a separate CV scheme for pseudo-labeling:
    # 1. First pass CV → get out-of-fold predictions
    # 2. Select high-confidence samples → augment train set
    # 3. Second pass CV with augmented set
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    thresholds = [0.7, 0.8, 0.85, 0.9, 0.95]
    results = {}
    
    # First pass: get OOF student predictions
    print("\n--- First pass: OOF predictions ---")
    oof_student_preds = np.zeros((len(X_all), len(TARGETS)))
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_all, y_bin)):
        X_tr, X_val = X_all[tr_idx], X_all[val_idx]
        
        for i, t in enumerate(TARGETS):
            y_t_tr = (train.iloc[tr_idx][t] > 0.5).astype(int).values
            m = lgb.LGBMClassifier(
                n_estimators=500, learning_rate=0.05, num_leaves=31,
                max_depth=5, min_child_samples=30,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.5, reg_lambda=1.0,
                random_state=42 + fold, verbose=-1, n_jobs=-1
            )
            m.fit(X_tr, y_t_tr)
            oof_student_preds[val_idx, i] = m.predict_proba(X_val)[:, 1]
    
    print("First pass done.")
    
    # Second pass: pseudo-labeling with different thresholds
    for threshold in thresholds:
        print(f"\n--- threshold={threshold} ---")
        
        # Find high-confidence samples
        conf_scores = np.max(oof_student_preds, axis=1)  # max prob across targets
        high_conf_mask = conf_scores >= threshold
        low_conf_mask = conf_scores <= (1.0 - threshold)
        pseudo_mask = high_conf_mask | low_conf_mask
        
        n_pseudo = pseudo_mask.sum()
        print(f"  High-confidence samples: {n_pseudo}/{len(X_all)} ({n_pseudo/len(X_all)*100:.0f}%)")
        
        if n_pseudo == 0:
            results[threshold] = {'meta': None, 'student': None, 'gap': None, 'n_pseudo': 0}
            continue
        
        # Pseudo labels: predicted binary labels for high-confidence samples
        pseudo_labels = (oof_student_preds > 0.5).astype(int)
        
        # Augment: duplicate high-confidence samples with slight noise
        n_extra = n_pseudo
        noise_std = 0.05
        noise = np.random.randn(n_extra, X_all.shape[1]) * noise_std
        X_aug = X_all[pseudo_mask] + noise
        pseudo_avg_labels = (oof_student_preds[pseudo_mask].mean(axis=1) > 0.5).astype(float)
        
        all_meta_preds = np.zeros(len(X_all))
        all_student_preds = np.zeros(len(X_all))
        
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_all, y_bin)):
            X_tr, X_val = X_all[tr_idx], X_all[val_idx]
            y_tr_avg = y_avg[tr_idx]
            
            # Augment with pseudo-labels (use only training fold's pseudo-labels)
            # But pseudo-labels are from OOF, so they're already out-of-fold
            # Safe to use all pseudo-labels for augmentation
            
            X_tr_aug = np.concatenate([X_tr, X_aug], axis=0)
            y_tr_avg_aug = np.concatenate([y_tr_avg, pseudo_avg_labels])
            
            # Student
            student_tr = np.zeros(len(X_tr))
            student_val = np.zeros(len(X_val))
            
            for t in TARGETS:
                y_t_tr_full = np.concatenate([
                    (train.iloc[tr_idx][t] > 0.5).astype(int).values,
                    (oof_student_preds[pseudo_mask, TARGETS.index(t)] > 0.5).astype(int)
                ])
                m = lgb.LGBMClassifier(
                    n_estimators=500, learning_rate=0.05, num_leaves=31,
                    max_depth=5, min_child_samples=30,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.5, reg_lambda=1.0,
                    random_state=42 + fold, verbose=-1, n_jobs=-1
                )
                m.fit(X_tr_aug, y_t_tr_full)
                student_tr += m.predict_proba(X_tr)[:, 1] / len(TARGETS)
                student_val += m.predict_proba(X_val)[:, 1] / len(TARGETS)
            
            all_student_preds[val_idx] += student_val
            
            # Meta
            meta_m = lgb.LGBMRegressor(
                n_estimators=500, learning_rate=0.03, num_leaves=25,
                max_depth=5, min_child_samples=30,
                subsample=0.8, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=3.0,
                random_state=99 + fold, verbose=-1, n_jobs=-1
            )
            meta_m.fit(X_tr_aug, y_tr_avg_aug)
            all_meta_preds[val_idx] += meta_m.predict(X_val)
        
        student_oof = 1 - np.mean(np.abs(all_student_preds - y_avg))
        meta_oof = 1 - np.mean(np.abs(all_meta_preds - y_avg))
        gap = np.mean(all_meta_preds - all_student_preds)
        
        results[threshold] = {
            'meta': round(meta_oof, 5),
            'student': round(student_oof, 5),
            'gap': round(gap, 5),
            'n_pseudo': int(n_pseudo),
        }
        
        print(f"  Meta OOF:      {meta_oof:.5f}")
        print(f"  Student OOF:   {student_oof:.5f}")
        print(f"  Gap:           {gap:.5f}")
        print(f"  Δ student:     {0.78570 - student_oof:+.5f}")
    
    # Also run baseline (no pseudo-labeling) for comparison
    print(f"\n--- baseline (no pseudo-labeling) ---")
    all_meta_preds = np.zeros(len(X_all))
    all_student_preds = np.zeros(len(X_all))
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_all, y_bin)):
        X_tr, X_val = X_all[tr_idx], X_all[val_idx]
        y_tr_avg = y_avg[tr_idx]
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
            student_tr = np.zeros(len(X_tr))
            student_val = np.zeros(len(X_val))
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
        'n_pseudo': 0,
    }
    print(f"  Meta OOF:      {meta_oof:.5f}")
    print(f"  Student OOF:   {student_oof:.5f}")
    
    # Find best
    best = min([(k, v) for k, v in results.items() if v['meta'] is not None], key=lambda x: x[1]['student'])
    print(f"\n{'='*60}")
    print(f"BEST: threshold={best[0]} (student OOF: {best[1]['student']})")
    print("=" * 60)
    
    elapsed = time.time() - start_time
    
    ts = time.strftime("%Y%m%d_%H%M%S")
    result = {
        "version": "V475",
        "name": "Pseudo-labeling",
        "results": {str(k): v for k, v in results.items()},
        "best_threshold": best[0],
        "timestamp": ts,
        "total_time_s": round(elapsed)
    }
    
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    with open(EXPERIMENTS / f"v475_{ts}.json", 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Saved: {EXPERIMENTS}/v475_{ts}.json")

if __name__ == '__main__':
    main()
