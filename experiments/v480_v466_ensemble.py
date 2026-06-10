#!/usr/bin/env python3
"""
V480 — V466 Pipeline Reproduction + Ensemble
가설: V466의 CV-internal adversarial + consensus feature selection pipeline을 정확히 재현한 후,
      diverse student configs를 ensemble하면 student OOF를 더 낮출 수 있음.
      V466 student 0.602 → ensemble으로 0.590대 진입 목표.
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
    print("V480 — V466 Pipeline Reproduction + Ensemble")
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
    
    # === STEP 1: V466 Pipeline — CV-internal adversarial ===
    print("\n" + "=" * 60)
    print("Step 1: V466-style CV-internal adversarial validation")
    print("=" * 60)
    
    X_all = train[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    y_avg = train[TARGETS].mean(axis=1).values
    y_bin = (y_avg > 0.5).astype(int)
    
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    import lightgbm as lgb
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # For each fold, train adversarial model on fold train vs fold val
    fold_adv_importances = []
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_all, y_bin)):
        X_tr = X_all[tr_idx]
        X_val = X_all[val_idx]
        y_adv = np.array([0]*len(tr_idx) + [1]*len(val_idx))
        X_both = np.concatenate([X_tr, X_val])
        
        adv_model = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=31,
            max_depth=5, min_child_samples=20,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.5, reg_lambda=1.0,
            random_state=42 + fold, verbose=-1, n_jobs=-1
        )
        adv_model.fit(X_both, y_adv)
        fold_adv_importances.append(adv_model.feature_importances_)
        
        preds = adv_model.predict_proba(X_both)[:, 1]
        auc = roc_auc_score(y_adv, preds)
        print(f"  Fold {fold+1}: adv AUC={auc:.4f}")
    
    # Consensus: average importances, keep features with avg adv_imp below threshold
    mean_adv_imp = np.mean(fold_adv_importances, axis=0)
    adv_imp_norm = mean_adv_imp / (mean_adv_imp.max() + 1e-10)
    
    # Try consensus thresholds (keep features below percentile)
    print(f"\nAdversarial importance distribution:")
    print(f"  Mean: {adv_imp_norm.mean():.3f}, Std: {adv_imp_norm.std():.3f}")
    
    # Reproduce V466 V468 pattern: K=30 consensus features
    k_values = [20, 25, 30, 35, 40]
    
    print(f"\n{'='*60}")
    print("Step 2: Feature reduction K sweep")
    print("=" * 60)
    
    # For each K, get lowest adv_imp features
    all_feature_masks = {}
    for k in k_values:
        sorted_indices = np.argsort(adv_imp_norm)  # ascending: lowest first
        keep_mask = np.zeros(len(feature_cols), dtype=bool)
        keep_mask[sorted_indices[:k]] = True
        all_feature_masks[k] = keep_mask
    
    # === STEP 3: For each K, run student + meta with diverse configs ===
    print(f"\n{'='*60}")
    print("Step 3: Running configs per K")
    print("=" * 60)
    
    # Student configs (from V479)
    student_configs = {
        "aggressive": (0.10, 63, 8, 20, 0.6, 0.6, 0.1, 0.1),
        "balanced": (0.05, 31, 5, 30, 0.8, 0.8, 0.5, 1.0),
        "conservative": (0.03, 15, 3, 50, 0.9, 0.9, 2.0, 5.0),
        "deep": (0.05, 127, 10, 20, 0.7, 0.7, 0.1, 0.5),
        "high_sub": (0.05, 31, 5, 30, 0.5, 0.5, 0.5, 1.0),
    }
    
    results = {}
    
    for k, mask in all_feature_masks.items():
        X_selected = X_all[:, mask]
        n_features = mask.sum()
        print(f"\n--- K={k} ({n_features} features) ---")
        
        # For each student config, get OOF
        config_results = {}
        for cname, (sl, sn, sm, smin, ss, sc, ra, rl) in student_configs.items():
            all_stu_preds = np.zeros(len(X_selected))
            
            for fold, (tr_idx, val_idx) in enumerate(skf.split(X_selected, y_bin)):
                X_tr, X_val = X_selected[tr_idx], X_selected[val_idx]
                stu_fold = np.zeros(len(X_val))
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
                    stu_fold += m.predict_proba(X_val)[:, 1] / len(TARGETS)
                all_stu_preds[val_idx] += stu_fold
            
            stu_oof = 1 - np.mean(np.abs(all_stu_preds - y_avg))
            config_results[cname] = round(stu_oof, 5)
        
        # Also run meta with balanced config
        all_meta_preds = np.zeros(len(X_selected))
        sl_bal, sn_bal, sm_bal, smin_bal, ss_bal, sc_bal, ra_bal, rl_bal = student_configs["balanced"]
        
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_selected, y_bin)):
            X_tr, X_val = X_selected[tr_idx], X_selected[val_idx]
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
        
        # Ensemble: average all student configs
        ens_preds = np.zeros(len(X_selected))
        for cname, (sl, sn, sm, smin, ss, sc, ra, rl) in student_configs.items():
            all_stu_preds = np.zeros(len(X_selected))
            for fold, (tr_idx, val_idx) in enumerate(skf.split(X_selected, y_bin)):
                X_tr, X_val = X_selected[tr_idx], X_selected[val_idx]
                stu_fold = np.zeros(len(X_val))
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
                    stu_fold += m.predict_proba(X_val)[:, 1] / len(TARGETS)
                all_stu_preds[val_idx] += stu_fold
            ens_preds += all_stu_preds / len(student_configs)
        
        ens_student_oof = 1 - np.mean(np.abs(ens_preds - y_avg))
        
        results[k] = {
            'n_features': n_features,
            'configs': config_results,
            'meta_oof': round(meta_oof, 5),
            'ensemble_student_oof': round(ens_student_oof, 5),
        }
        
        print(f"  Meta OOF:       {meta_oof:.5f}")
        print(f"  Ensemble Student OOF: {ens_student_oof:.5f}")
        for cn, cv in config_results.items():
            print(f"    {cn}: {cv}")
    
    # === Find best ===
    best_k = None
    best_student = 1.0
    for k, r in results.items():
        if r['ensemble_student_oof'] < best_student:
            best_student = r['ensemble_student_oof']
            best_k = k
    
    print(f"\n{'='*60}")
    print(f"BEST K: {best_k} (ensemble student OOF: {best_student})")
    print(f"Best config: {min(results[best_k]['configs'].items(), key=lambda x: x[1])}")
    print("=" * 60)
    
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
        "version": "V480",
        "name": "V466 Pipeline Reproduction + Ensemble",
        "results": to_json(results),
        "best_k": int(best_k),
        "timestamp": ts,
        "total_time_s": int(elapsed)
    }
    
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    with open(EXPERIMENTS / f"v480_{ts}.json", 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Saved: {EXPERIMENTS}/v480_{ts}.json")

if __name__ == '__main__':
    main()
