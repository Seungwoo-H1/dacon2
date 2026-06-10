#!/usr/bin/env python3
"""
V481 — V461/V466 Hybrid Pipeline
가설: V461(train+test adversarial) + V466(CV-internal consensus)의 hybrid를 student ensemble과 결합.
- V461 방식: train vs test adversarial로 47 features 제거 (V461: student 0.607)
- V466 방식: CV-internal consensus로 features 선택 (V466: student 0.602)
- diverse student configs ensemble 적용
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
    print("V481 — V461/V466 Hybrid Pipeline")
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
    X_test = test_feat[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    y_avg = train[TARGETS].mean(axis=1).values
    y_bin = (y_avg > 0.5).astype(int)
    
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    import lightgbm as lgb
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # === V461-style: train vs test adversarial ===
    print("\n" + "=" * 60)
    print("V461-style: train vs test adversarial")
    print("=" * 60)
    
    n_train = len(X_train)
    y_adv = np.array([0]*n_train + [1]*len(X_test))
    X_both = np.vstack([X_train, X_test])
    
    adv_models = []
    for seed in range(5):
        adv = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=31,
            max_depth=5, min_child_samples=20,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.5, reg_lambda=1.0,
            random_state=42 + seed, verbose=-1, n_jobs=-1
        )
        adv.fit(X_both, y_adv)
        adv_models.append(adv)
        preds = adv.predict_proba(X_both)[:, 1]
        auc = roc_auc_score(y_adv, preds)
        print(f"  Seed {seed}: adv AUC={auc:.4f}")
    
    mean_adv_imp = np.mean([m.feature_importances_ for m in adv_models], axis=0)
    adv_imp_norm = mean_adv_imp / (mean_adv_imp.max() + 1e-10)
    
    # Try different thresholds for V461 style
    print(f"\nV461-style feature removal:")
    for pct in [5, 10, 15, 20, 25, 30]:
        cutoff = np.percentile(adv_imp_norm, pct)
        keep = adv_imp_norm <= cutoff
        print(f"  Top-{pct} percentile (>{cutoff:.3f}): keep {keep.sum()}/{len(feature_cols)}")
    
    # === V466-style: CV-internal adversarial ===
    print(f"\n{'='*60}")
    print("V466-style: CV-internal adversarial")
    print("=" * 60)
    
    fold_adv_importances = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_bin)):
        X_tr = X_train[tr_idx]
        X_val = X_train[val_idx]
        y_adv2 = np.array([0]*len(tr_idx) + [1]*len(val_idx))
        X_both2 = np.concatenate([X_tr, X_val])
        
        adv = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=31,
            max_depth=5, min_child_samples=20,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.5, reg_lambda=1.0,
            random_state=42 + fold, verbose=-1, n_jobs=-1
        )
        adv.fit(X_both2, y_adv2)
        fold_adv_importances.append(adv.feature_importances_)
    
    cv_mean_adv = np.mean(fold_adv_importances, axis=0)
    cv_adv_norm = cv_mean_adv / (cv_mean_adv.max() + 1e-10)
    
    # Hybrid: features that are LOW in BOTH v461 and v466 adversarial importances
    # These are "safe" features that don't discriminate train/test
    hybrid_score = adv_imp_norm + cv_adv_norm
    
    print(f"\nHybrid score stats: mean={hybrid_score.mean():.3f}, std={hybrid_score.std():.3f}")
    
    # Test different thresholds
    strategies = {}
    
    # V461-style thresholds
    for pct in [5, 10, 15, 20, 25, 30]:
        cutoff = np.percentile(adv_imp_norm, pct)
        mask = adv_imp_norm <= cutoff
        strategies[f"v461_p{pct}"] = mask
    
    # V466-style thresholds
    for pct in [10, 20, 30, 40, 50]:
        cutoff = np.percentile(cv_adv_norm, pct)
        mask = cv_adv_norm <= cutoff
        strategies[f"v466_p{pct}"] = mask
    
    # Hybrid thresholds
    for pct in [10, 20, 30, 40, 50]:
        cutoff = np.percentile(hybrid_score, pct)
        mask = hybrid_score <= cutoff
        strategies[f"hybrid_p{pct}"] = mask
    
    print(f"\nTesting {len(strategies)} strategies...")
    
    student_configs = {
        "aggressive": (0.10, 63, 8, 20, 0.6, 0.6, 0.1, 0.1),
        "balanced": (0.05, 31, 5, 30, 0.8, 0.8, 0.5, 1.0),
        "conservative": (0.03, 15, 3, 50, 0.9, 0.9, 2.0, 5.0),
        "deep": (0.05, 127, 10, 20, 0.7, 0.7, 0.1, 0.5),
        "high_sub": (0.05, 31, 5, 30, 0.5, 0.5, 0.5, 1.0),
    }
    
    results = {}
    
    for sname, mask in strategies.items():
        n_kept = mask.sum()
        if n_kept < 5 or n_kept > 70:
            print(f"  SKIP {sname}: {n_kept} features")
            continue
        
        X_sel = X_train[:, mask]
        
        print(f"\n--- {sname}: {n_kept} features ---")
        
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
        
        results[sname] = {
            'n_features': n_kept,
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
    valid = {k: v for k, v in results.items()}
    if valid:
        best = min(valid.items(), key=lambda x: x[1]['ensemble_student_oof'])
        print(f"\n{'='*60}")
        print(f"BEST: {best[0]} (ensemble: {best[1]['ensemble_student_oof']})")
        print(f"  Features: {best[1]['n_features']}")
        print(f"  Meta: {best[1]['meta_oof']}, Gap: {best[1]['gap']}")
        print("=" * 60)
    
    elapsed = time.time() - start_time
    
    ts = time.strftime("%Y%m%d_%H%M%S")
    def to_json(o):
        if isinstance(o, dict): return {str(k): to_json(v) for k,v in o.items()}
        if isinstance(o, (list,tuple)): return [to_json(x) for x in o]
        if isinstance(o,(np.integer,)): return int(o)
        if isinstance(o,(np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o
    result = {
        "version": "V481",
        "name": "V461/V466 Hybrid Pipeline",
        "results": to_json({k:v for k,v in valid.items()}) if valid else {},
        "best_strategy": str(best[0]) if valid else None,
        "timestamp": ts,
        "total_time_s": int(elapsed)
    }
    
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    with open(EXPERIMENTS / f"v481_{ts}.json", 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Saved: {EXPERIMENTS}/v481_{ts}.json")

if __name__ == '__main__':
    main()
