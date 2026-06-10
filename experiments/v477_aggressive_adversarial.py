#!/usr/bin/env python3
"""
V477 — Aggressive Adversarial Feature Selection
가설: V466의 CV-internal adversarial validation을 더 extreme하게 적용하면
      student OOF를 더 낮출 수 있음.
      - Lower adversarial threshold (0.7 → 0.5)
      - Multiple adversarial models averaging
      - Per-fold adversarial selection + consensus
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
    print("V477 — Aggressive Adversarial Feature Selection")
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
    
    # Create adversarial dataset: train=0, test=1
    n_train = len(X_all)
    y_adversarial = np.array([0]*n_train + [1]*len(X_all))
    X_adversarial = np.vstack([X_all, X_all])  # use same features for train/test
    
    from sklearn.model_selection import StratifiedKFold
    import lightgbm as lgb
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\n" + "=" * 60)
    print("Running aggressive adversarial feature selection")
    print("=" * 60)
    
    # Run adversarial validation per fold to get consistent feature rankings
    print("\nStep 1: CV-internal adversarial validation")
    
    # For each fold, train adversarial model on train portion
    fold_adversarial_importances = []
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_all, np.zeros(len(X_all)))):
        X_tr_adv = X_all[tr_idx]
        X_val_adv = X_all[val_idx]
        y_tr_adversarial = np.array([0]*len(tr_idx) + [1]*len(val_idx))  # fold train=0, fold val=1
        X_combined = np.concatenate([X_tr_adv, X_val_adv])
        
        adv_model = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=31,
            max_depth=5, min_child_samples=20,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.5, reg_lambda=1.0,
            random_state=42 + fold, verbose=-1, n_jobs=-1
        )
        adv_model.fit(X_combined, y_tr_adversarial)
        fold_adversarial_importances.append(adv_model.feature_importances_)
        # AUC on combined train+val
        preds = adv_model.predict_proba(X_combined)[:, 1]
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(y_tr_adversarial, preds)
            print(f"  Fold {fold+1} adv AUC: {auc:.3f}")
        except:
            print(f"  Fold {fold+1} adv AUC: N/A")
    
    # Average adversarial importances across folds
    mean_adv_imp = np.mean(fold_adversarial_importances, axis=0)
    std_adv_imp = np.std(fold_adversarial_importances, axis=0)
    
    # Normalize adversarial importances
    adv_imp_norm = mean_adv_imp / (mean_adv_imp.max() + 1e-10)
    
    print(f"\nAdversarial importance stats:")
    print(f"  Mean: {adv_imp_norm.mean():.3f}")
    print(f"  Std: {adv_imp_norm.std():.3f}")
    print(f"  Max: {adv_imp_norm.max():.3f}")
    print(f"  Features with adv_imp > 0.5: {(adv_imp_norm > 0.5).sum()}")
    print(f"  Features with adv_imp > 0.3: {(adv_imp_norm > 0.3).sum()}")
    print(f"  Features with adv_imp > 0.1: {(adv_imp_norm > 0.1).sum()}")
    
    # Test different adversarial thresholds
    thresholds = [0.0, 0.1, 0.2, 0.3, 0.5]  # 0.0 = no filtering (V466-like baseline)
    
    # Also try: keep only features with adv_imp below percentile
    percentiles = [10, 20, 30, 40, 50]
    
    strategies = {}
    for t in thresholds:
        mask = adv_imp_norm <= t
        strategies[f'adv_thresh_{t}'] = mask
    
    for p in percentiles:
        cutoff = np.percentile(adv_imp_norm, p)
        mask = adv_imp_norm <= cutoff
        strategies[f'adv_pctile_{p}'] = mask
    
    print(f"\nTesting {len(strategies)} strategies...")
    
    results = {}
    
    for name, mask in strategies.items():
        n_kept = mask.sum()
        print(f"\n--- {name}: keep {n_kept}/{len(feature_cols)} features ---")
        
        if n_kept == 0:
            results[name] = None
            continue
        
        X_selected = X_all[:, mask]
        y_avg = train[TARGETS].mean(axis=1).values
        y_bin = (y_avg > 0.5).astype(int)
        
        all_meta_preds = np.zeros(len(X_selected))
        all_student_preds = np.zeros(len(X_selected))
        
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_selected, y_bin)):
            X_tr, X_val = X_selected[tr_idx], X_selected[val_idx]
            y_tr_avg = y_avg[tr_idx]
            
            # Student
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
            
            # Meta
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
            'n_features': n_kept,
            'improvement': round(0.78570 - student_oof, 5),
        }
        
        print(f"  Meta OOF:      {meta_oof:.5f}")
        print(f"  Student OOF:   {student_oof:.5f}")
        print(f"  Gap:           {gap:.5f}")
        print(f"  Δ student:     {0.78570 - student_oof:+.5f}")
    
    # Find best
    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        best = min(valid_results.items(), key=lambda x: x[1]['student'])
        print(f"\n{'='*60}")
        print(f"BEST: {best[0]} (student OOF: {best[1]['student']})")
        print(f"  Features: {best[1]['n_features']}")
        print("=" * 60)
    else:
        best = None
    
    elapsed = time.time() - start_time
    
    ts = time.strftime("%Y%m%d_%H%M%S")
    
    def to_json(obj):
        if isinstance(obj, dict): return {str(k): to_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [to_json(x) for x in obj]
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj
    
    result = {
        "version": "V477",
        "name": "Aggressive Adversarial Feature Selection",
        "results": to_json({k: v for k, v in results.items() if v}),
        "best_strategy": str(best[0]) if best else None,
        "adversarial_stats": {"mean": round(float(adv_imp_norm.mean()),4), "std": round(float(adv_imp_norm.std()),4), "max": round(float(adv_imp_norm.max()),4)},
        "timestamp": ts,
        "total_time_s": int(elapsed)
    }
    
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    with open(EXPERIMENTS / f"v477_{ts}.json", 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Saved: {EXPERIMENTS}/v477_{ts}.json")

if __name__ == '__main__':
    main()
