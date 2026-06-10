#!/usr/bin/env python3
"""
V483 — V461 Adversarial Feature Importance Analysis
가설: V461의 train vs test adversarial feature importance를 분석한 후,
      가장 낮은 importance를 가진 features만 제거하면 student가 낮아질 것.
      V461은 47 features 제거 → keep 21. 하지만 이 features가 signal일 수 있음.
      → V461 features 제거 시 student 0.607은 V461의 student config 때문일 수 있음.
      
      V483: train vs test adv importance → features별 OOF impact 분석
      그리고 V461 config (21 features)에서 다양한 student config 테스트
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
    print("V483 — Adversarial Feature Importance Analysis")
    print("=" * 60)
    
    # Load
    train_feat = pd.read_parquet(DATA_PROC / 'train_features_clean_v60.parquet')
    test_feat = pd.read_parquet(DATA_PROC / 'test_features_clean_v60.parquet')
    train_csv = pd.read_csv(DATA_RAW / 'ch2026_metrics_train.csv')
    
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
    
    exclude_cols = ['subject_id', 'date', 'lifelog_date'] + TARGETS
    feature_cols = [c for c in train.columns if c not in exclude_cols
                    and train[c].dtype in ['float64', 'int64', 'float32', 'int32', 'float16']
                    and train[c].nunique() > 1]
    
    X_train = train[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    X_test = test_feat[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    y_avg = train[TARGETS].mean(axis=1).values
    y_bin = (y_avg > 0.5).astype(int)
    
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    import lightgbm as lgb
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # === Train vs test adversarial (5 seeds only) ===
    print("\n" + "=" * 60)
    print("Train vs Test Adversarial (5 seeds)")
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
    
    mean_adv_imp = np.mean([m.feature_importances_ for m in adv_models], axis=0)
    adv_imp_norm = mean_adv_imp / (mean_adv_imp.max() + 1e-10)
    sorted_indices = np.argsort(-mean_adv_imp)
    
    print("\nTop 20 adversarial important features:")
    for i in range(20):
        idx = sorted_indices[i]
        print(f"  #{i+1}: {feature_cols[idx]} (adv={adv_imp_norm[idx]:.3f}, imp={mean_adv_imp[idx]:.1f})")
    
    print("\nBottom 10 adversarial important features:")
    for i in range(-10, 0):
        idx = sorted_indices[i]
        print(f"  #{len(sorted_indices)+i+1}: {feature_cols[idx]} (adv={adv_imp_norm[idx]:.3f}, imp={mean_adv_imp[idx]:.1f})")
    
    # === Feature importance (target-predictive) ===
    print("\n" + "=" * 60)
    print("Target-predictive feature importance (global)")
    print("=" * 60)
    
    global_imp = np.zeros(len(feature_cols))
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_bin)):
        X_tr = X_train[tr_idx]
        for t in TARGETS:
            y_t_tr = (train.iloc[tr_idx][t] > 0.5).astype(int).values
            m = lgb.LGBMClassifier(
                n_estimators=200, learning_rate=0.1, num_leaves=31,
                max_depth=5, random_state=42 + fold, verbose=-1, n_jobs=-1
            )
            m.fit(X_tr, y_t_tr)
            global_imp += m.feature_importances_ / len(TARGETS)
    
    global_imp /= 5  # average over folds
    target_sorted = np.argsort(-global_imp)
    
    print("\nTop 20 target-predictive features:")
    for i in range(20):
        idx = target_sorted[i]
        print(f"  #{i+1}: {feature_cols[idx]} (target_imp={global_imp[idx]:.1f})")
    
    # === Analyze overlap between adversarial and target-predictive ===
    print("\n" + "=" * 60)
    print("Feature overlap analysis")
    print("=" * 60)
    
    # Top 10 adversarial features that are also top-20 target-predictive
    adv_top10 = set(sorted_indices[:10])
    target_top20 = set(target_sorted[:20])
    
    overlap_adv_target = adv_top10 & target_top20
    print(f"Top-10 adv ∩ Top-20 target: {len(overlap_adv_target)} features")
    for f in overlap_adv_target:
        print(f"  - {feature_cols[f]} (adv={adv_imp_norm[f]:.3f}, target={global_imp[f]:.1f})")
    
    # Top 10 adversarial that are LOW in target importance
    low_target = set(target_sorted[30:])  # bottom 38
    adv_signal = adv_top10 - low_target
    print(f"\nAdv-top10 that are NOT in target bottom-38: {len(adv_signal)}")
    for f in adv_signal:
        print(f"  - {feature_cols[f]} (adv={adv_imp_norm[f]:.3f}, target={global_imp[f]:.1f})")
    
    # === Now test: Keep bottom N adversarial features (lowest adv_imp) ===
    # AND remove only features that are HIGH in adv_imp but LOW in target_imp
    print(f"\n{'='*60}")
    print("Feature selection strategies")
    print("=" * 60)
    
    strategies = {}
    
    # Strategy 1: Keep bottom N adv_imp (pure V461 style)
    for n_keep in [21, 30, 40]:
        mask = np.zeros(len(feature_cols), dtype=bool)
        mask[sorted_indices[-n_keep:]] = True
        strategies[f'v461_keep_{n_keep}'] = mask
    
    # Strategy 2: Remove only features with adv_imp > 0.2 AND target_imp in bottom 50%
    for adv_thresh in [0.1, 0.15, 0.2, 0.3]:
        remove = (adv_imp_norm > adv_thresh) & (global_imp < np.median(global_imp))
        mask = ~remove
        if mask.sum() >= 10:
            strategies[f'remove_adv{adv_thresh}_lowtarget'] = mask
    
    # Strategy 3: Combined — low adv_imp OR high target_imp
    for adv_thresh in [0.1, 0.15]:
        for target_thresh_pct in [30, 50, 70]:
            target_thresh = np.percentile(global_imp, target_thresh_pct)
            keep = (adv_imp_norm <= adv_thresh) | (global_imp >= target_thresh)
            if keep.sum() >= 10:
                strategies[f'lowadv{adv_thresh}_or_hightarget{target_thresh_pct}'] = keep
    
    # Strategy 4: Pure target importance — keep top N by target_imp
    for n_keep in [20, 30, 40, 50, 60, 68]:
        mask = np.zeros(len(feature_cols), dtype=bool)
        mask[target_sorted[:n_keep]] = True
        strategies[f'target_keep_{n_keep}'] = mask
    
    print(f"\nTesting {len(strategies)} strategies...")
    
    student_configs = {
        "aggressive": (0.10, 63, 8, 20, 0.6, 0.6, 0.1, 0.1),
        "balanced": (0.05, 31, 5, 30, 0.8, 0.8, 0.5, 1.0),
        "conservative": (0.03, 15, 3, 50, 0.9, 0.9, 2.0, 5.0),
    }
    
    results = {}
    
    for sname, mask in strategies.items():
        n_kept = mask.sum()
        if n_kept < 5 or n_kept > 68:
            continue
        
        X_sel = X_train[:, mask]
        best_single_oof = 1.0
        best_single_name = ""
        
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
            if stu_oof < best_single_oof:
                best_single_oof = stu_oof
                best_single_name = cname
        
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
        
        results[sname] = {
            'n_features': n_kept,
            'best_single': {best_single_name: round(best_single_oof, 5)},
            'meta_oof': round(meta_oof, 5),
        }
        
        print(f"  {sname}: {n_kept}f | best_single={best_single_name}({best_single_oof:.5f}) | meta={meta_oof:.5f}")
    
    # Find best
    valid = {k: v for k, v in results.items()}
    best = min(valid.items(), key=lambda x: x[1]['best_single'][list(x[1]['best_single'].keys())[0]])
    print(f"\n{'='*60}")
    print(f"BEST: {best[0]} (student: {best[1]['best_single']})")
    print("=" * 60)
    
    elapsed = time.time() - start_time
    
    ts = time.strftime("%Y%m%d_%H%M%S")
    result = {
        "version": "V483",
        "name": "Adversarial Feature Importance Analysis",
        "results": {str(k): v for k, v in valid.items()},
        "best_strategy": str(best[0]),
        "timestamp": ts,
        "total_time_s": int(elapsed)
    }
    
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    with open(EXPERIMENTS / f"v483_{ts}.json", 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Saved: {EXPERIMENTS}/v483_{ts}.json")

if __name__ == '__main__':
    main()
