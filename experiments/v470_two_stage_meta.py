#!/usr/bin/env python3
"""
V470 — Two-Stage Meta Stacking
가설: Stage1에서 target별 student predictions → Stage2에서 student preds + 
      meta features + raw features로 final prediction (deep stacking이 shallow보다 성능 향상)
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
    print("V470 — Two-Stage Meta Stacking")
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
    
    # Feature selection — use same features as V466/V468 (good baseline)
    exclude_cols = ['subject_id', 'date', 'lifelog_date'] + TARGETS
    feature_cols = [c for c in train.columns if c not in exclude_cols
                    and train[c].dtype in ['float64', 'int64', 'float32', 'int32', 'float16']
                    and train[c].nunique() > 1]
    
    print(f"Base features: {len(feature_cols)}")
    
    # Clean
    for c in feature_cols:
        train[c] = train[c].fillna(0).replace([np.inf, -np.inf], 0)
    
    X = train[feature_cols].values
    y_avg = train[TARGETS].mean(axis=1).values
    y_bin = (y_avg > 0.5).astype(int)
    
    from sklearn.model_selection import StratifiedKFold
    import lightgbm as lgb
    
    print("\nTraining Two-Stage Meta with 5-fold CV...")
    print("  Stage 1: Per-target LGBM → student predictions")
    print("  Stage 2: Deep stacking (student preds + features + interactions) → final")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    all_stage2_preds = np.zeros(len(X))
    all_student_preds = np.zeros(len(X))
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y_bin)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr_avg = y_avg[tr_idx]
        y_val_avg = y_avg[val_idx]
        
        # ===== STAGE 1: Student predictions per target =====
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
        
        # ===== STAGE 2: Deep Stacking =====
        # Stage 2 features:
        # 1. Student predictions (1D)
        # 2. Raw features (full set)
        # 3. Per-target predicted probabilities (7D)
        # 4. Interaction: student * raw_features (top-20 by importance)
        
        # Get per-target probs
        target_probs_tr = np.zeros((len(X_tr), len(TARGETS)))
        target_probs_val = np.zeros((len(X_val), len(TARGETS)))
        
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
            target_probs_tr[:, i] = m.predict_proba(X_tr)[:, 1]
            target_probs_val[:, i] = m.predict_proba(X_val)[:, 1]
        
        # Feature importance for top-20 selection (use training subset only)
        imp_model = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.1, num_leaves=31,
            max_depth=5, random_state=42 + fold, verbose=-1, n_jobs=-1
        )
        imp_model.fit(X_tr, y_bin[tr_idx])
        importances = imp_model.feature_importances_
        top_k = min(20, len(feature_cols))
        top_indices = np.argsort(importances)[-top_k:]
        
        # Top features subset
        X_tr_top = X_tr[:, top_indices]
        X_val_top = X_val[:, top_indices]
        
        # Interactions: student * top_features (broadcast: element-wise per sample)
        # student_tr is (N,), X_tr_top is (N, 20) → result is (N, 20)
        student_tr_top = student_tr[:, np.newaxis] * X_tr_top  # (N, 20)
        student_val_top = student_val[:, np.newaxis] * X_val_top  # (N, 20)
        
        # Stage 2 features: all have shape (N, *)
        meta_tr = np.column_stack([
            student_tr[:, np.newaxis],       # 1D → (N, 1)
            X_tr,                            # (N, 68)
            target_probs_tr,                 # (N, 7)
            X_tr_top,                        # (N, 20)
            student_tr_top,                  # (N, 20)
        ])
        meta_val = np.column_stack([
            student_val[:, np.newaxis],
            X_val,
            target_probs_val,
            X_val_top,
            student_val_top,
        ])
        
        print(f"  Fold {fold+1}: Stage 2 features = {meta_tr.shape[1]}")
        
        # Stage 2 model
        stage2 = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.03, num_leaves=25,
            max_depth=5, min_child_samples=30,
            subsample=0.8, colsample_bytree=0.6,
            reg_alpha=2.0, reg_lambda=5.0,
            random_state=99 + fold, verbose=-1, n_jobs=-1
        )
        stage2.fit(meta_tr, y_tr_avg)
        all_stage2_preds[val_idx] += stage2.predict(meta_val)
    
    # Evaluate
    student_oof = 1 - np.mean(np.abs(all_student_preds - y_avg))
    meta_oof = 1 - np.mean(np.abs(all_stage2_preds - y_avg))
    
    elapsed = time.time() - start_time
    gap = np.mean(all_stage2_preds - all_student_preds)
    
    print(f"\n{'='*60}")
    print(f"V470 RESULTS — Two-Stage Meta Stacking")
    print(f"  Meta OOF (stage2): {meta_oof:.5f}")
    print(f"  Student OOF:       {student_oof:.5f}")
    print(f"  Gap (meta-student): {gap:.5f}")
    print(f"  Improvement:       {student_oof - meta_oof:+.5f}")
    print(f"  V308 LB:           0.63893")
    print(f"  Est LB:            {student_oof:.5f}")
    print(f"  Time:              {elapsed:.0f}s")
    print(f"{'='*60}")
    
    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    result = {
        "version": "V470",
        "name": "Two-Stage Meta Stacking",
        "avg_meta_oof": round(meta_oof, 5),
        "avg_student_oof": round(student_oof, 5),
        "v308_lb": 0.63893,
        "student_meta_gap": round(gap, 5),
        "improvement": round(student_oof - meta_oof, 5),
        "n_features": len(feature_cols),
        "timestamp": ts,
        "total_time_s": round(elapsed),
        "hypothesis": "Two-stage deep stacking outperforms single-stage meta"
    }
    
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    with open(EXPERIMENTS / f"v470_{ts}.json", 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Saved: {EXPERIMENTS}/v470_{ts}.json")

if __name__ == '__main__':
    main()
