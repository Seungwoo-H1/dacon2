#!/usr/bin/env python3
"""
V482 — V461 Exact Reproduction (47 features removed)
가설: V461의 train vs test adversarial validation으로 정확히 47 features 제거 (keep 21).
      V461 student OOF: 0.607. 이 config로 student + ensemble 테스트.
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
    print("V482 — V461 Exact Reproduction (47 features removed)")
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
    print(f"Test: {test_feat.shape}")
    
    exclude_cols = ['subject_id', 'date', 'lifelog_date'] + TARGETS
    feature_cols = [c for c in train.columns if c not in exclude_cols
                    and train[c].dtype in ['float64', 'int64', 'float32', 'int32', 'float16']
                    and train[c].nunique() > 1]
    
    print(f"Base features: {len(feature_cols)}")
    
    X_train = train[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    y_avg = train[TARGETS].mean(axis=1).values
    y_bin = (y_avg > 0.5).astype(int)
    
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    import lightgbm as lgb
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # === V461: train vs test adversarial ===
    print("\n" + "=" * 60)
    print("V461-style: train vs test adversarial (47 features removed)")
    print("=" * 60)
    
    n_train = len(X_train)
    X_test = test_feat[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    y_adv = np.array([0]*n_train + [1]*len(X_test))
    X_both = np.vstack([X_train, X_test])
    
    # Multi-seed adversarial
    adv_models = []
    for seed in range(15):  # V466은 n_seeds=15
        adv = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=31,
            max_depth=5, min_child_samples=20,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.5, reg_lambda=1.0,
            random_state=42 + seed, verbose=-1, n_jobs=-1
        )
        adv.fit(X_both, y_adv)
        adv_models.append(adv)
    
    mean_adv_imp = np.mean([m.feature_importances_ for m in adv_models], axis=0)
    adv_imp_norm = mean_adv_imp / (mean_adv_imp.max() + 1e-10)
    
    # Sort features by adversarial importance
    sorted_indices = np.argsort(-mean_adv_imp)  # descending: highest first
    
    print(f"\nTop 15 adversarial important features:")
    for i in range(min(15, len(feature_cols))):
        idx = sorted_indices[i]
        print(f"  #{i+1}: {feature_cols[idx]} (adv_imp={mean_adv_imp[idx]:.3f})")
    
    # V461: remove top 47 features → keep bottom 21
    keep_47_removed = np.zeros(len(feature_cols), dtype=bool)
    keep_47_removed[sorted_indices[47:]] = True  # indices 47 onwards = lowest 21
    
    # Also try: keep exactly 21 features (bottom 21 adv importance)
    print(f"\nV461 exact: remove top 47 → keep {keep_47_removed.sum()} features")
    
    # Also test: keep top N lowest adv_imp features
    for n_keep in [15, 20, 21, 25, 30]:
        mask = np.zeros(len(feature_cols), dtype=bool)
        mask[sorted_indices[-n_keep:]] = True  # lowest adv_imp first
        print(f"Keep bottom {n_keep}: {[feature_cols[i] for i in sorted_indices[-n_keep:]]}")
    
    # === Run student + meta with keep 21 (V461 exact) ===
    print(f"\n{'='*60}")
    print("Running with V461 config (21 features)")
    print("=" * 60)
    
    X_21 = X_train[:, keep_47_removed]
    
    # Try different student configs
    student_configs = {
        "aggressive": (0.10, 63, 8, 20, 0.6, 0.6, 0.1, 0.1),
        "balanced": (0.05, 31, 5, 30, 0.8, 0.8, 0.5, 1.0),
        "conservative": (0.03, 15, 3, 50, 0.9, 0.9, 2.0, 5.0),
        "deep": (0.05, 127, 10, 20, 0.7, 0.7, 0.1, 0.5),
        "high_sub": (0.05, 31, 5, 30, 0.5, 0.5, 0.5, 1.0),
        "high_reg": (0.05, 31, 5, 40, 0.7, 0.7, 5.0, 10.0),
        "shallow": (0.08, 10, 2, 80, 0.95, 0.95, 5.0, 10.0),
    }
    
    print(f"\nBase features: 68 → V461 features: {keep_47_removed.sum()}")
    print(f"Selected features: {[feature_cols[i] for i in np.where(keep_47_removed)[0]]}")
    
    results = {}
    
    for n_keep in [15, 20, 21, 25, 30]:
        mask = np.zeros(len(feature_cols), dtype=bool)
        mask[sorted_indices[-n_keep:]] = True
        X_sel = X_train[:, mask]
        
        print(f"\n--- Keep bottom {n_keep} features ---")
        
        config_results = {}
        for cname, (sl, sn, sm, smin, ss, sc, ra, rl) in student_configs.items():
            all_stu = np.zeros(len(X_sel))
            for fold, (tr_idx, val_idx) in enumerate(skf.split(X_sel, y_bin)):
                stu_fold = np.zeros(len(X_train[val_idx]))
                for t in TARGETS:
                    y_t_tr = (train.iloc[tr_idx][t] > 0.5).astype(int).values
                    m = lgb.LGBMClassifier(
                        n_estimators=500, learning_rate=sl, num_leaves=sn,
                        max_depth=sm, min_child_samples=smin,
                        subsample=ss, colsample_bytree=sc,
                        reg_alpha=ra, reg_lambda=rl,
                        random_state=42 + fold, verbose=-1, n_jobs=-1
                    )
                    m.fit(X_sel[tr_idx], y_t_tr)
                    stu_fold += m.predict_proba(X_sel[val_idx])[:, 1] / len(TARGETS)
                all_stu[val_idx] += stu_fold
            
            stu_oof = 1 - np.mean(np.abs(all_stu - y_avg))
            config_results[cname] = round(stu_oof, 5)
        
        # Ensemble
        ens = np.zeros(len(X_sel))
        for cname, (sl, sn, sm, smin, ss, sc, ra, rl) in student_configs.items():
            all_stu = np.zeros(len(X_sel))
            for fold, (tr_idx, val_idx) in enumerate(skf.split(X_sel, y_bin)):
                stu_fold = np.zeros(len(X_train[val_idx]))
                for t in TARGETS:
                    y_t_tr = (train.iloc[tr_idx][t] > 0.5).astype(int).values
                    m = lgb.LGBMClassifier(
                        n_estimators=500, learning_rate=sl, num_leaves=sn,
                        max_depth=sm, min_child_samples=smin,
                        subsample=ss, colsample_bytree=sc,
                        reg_alpha=ra, reg_lambda=rl,
                        random_state=42 + fold, verbose=-1, n_jobs=-1
                    )
                    m.fit(X_sel[tr_idx], y_t_tr)
                    stu_fold += m.predict_proba(X_sel[val_idx])[:, 1] / len(TARGETS)
                all_stu[val_idx] += stu_fold
            ens += all_stu / len(student_configs)
        
        ens_oof = 1 - np.mean(np.abs(ens - y_avg))
        
        # Meta
        all_meta = np.zeros(len(X_sel))
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_sel, y_bin)):
            m_meta = lgb.LGBMRegressor(
                n_estimators=500, learning_rate=0.03, num_leaves=25,
                max_depth=5, min_child_samples=30,
                subsample=0.8, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=3.0,
                random_state=99 + fold, verbose=-1, n_jobs=-1
            )
            m_meta.fit(X_sel[tr_idx], y_avg[tr_idx])
            all_meta[val_idx] += m_meta.predict(X_sel[val_idx])
        
        meta_oof = 1 - np.mean(np.abs(all_meta - y_avg))
        gap = np.mean(all_meta - ens)
        
        best_single = min(config_results.items(), key=lambda x: x[1])
        
        results[f'keep_{n_keep}'] = {
            'n_features': n_keep,
            'configs': config_results,
            'ensemble_student_oof': round(ens_oof, 5),
            'meta_oof': round(meta_oof, 5),
            'gap': round(gap, 5),
            'best_single': {best_single[0]: best_single[1]},
        }
        
        print(f"  Ensemble: {ens_oof:.5f} | Meta: {meta_oof:.5f} | Gap: {gap:.3f}")
        print(f"  Best single: {best_single[0]} ({best_single[1]:.5f})")
        for cn, cv in config_results.items():
            print(f"    {cn}: {cv}")
    
    # Find best
    valid = results
    best = min(valid.items(), key=lambda x: x[1]['ensemble_student_oof'])
    print(f"\n{'='*60}")
    print(f"BEST: {best[0]} (ensemble: {best[1]['ensemble_student_oof']})")
    print(f"  Features: {best[1]['n_features']}")
    print(f"  Meta: {best[1]['meta_oof']}, Gap: {best[1]['gap']}")
    print(f"  V461 target: student 0.607")
    print("=" * 60)
    
    elapsed = time.time() - start_time
    
    ts = time.strftime("%Y%m%d_%H%M%S")
    result = {
        "version": "V482",
        "name": "V461 Exact Reproduction (47 features removed)",
        "results": {str(k): v for k, v in valid.items()},
        "best_strategy": str(best[0]),
        "timestamp": ts,
        "total_time_s": int(elapsed)
    }
    
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    with open(EXPERIMENTS / f"v482_{ts}.json", 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Saved: {EXPERIMENTS}/v482_{ts}.json")

if __name__ == '__main__':
    main()
