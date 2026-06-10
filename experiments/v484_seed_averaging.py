#!/usr/bin/env python3
"""
V484 — Seed Averaging + Bottom-N Feature Selection
V481에서 이미 train vs test adversarial importance를 확인함:
- AUC 1.000 (모든 seed) → train과 test의 distribution이 완전히 다름
- V461: 47 features removed → keep 21

V484: V481의 train vs test adv ranking을 재계산하되 n_estimators=100으로 빠르게.
그 후 bottom-N features로 seed averaging student/meta 테스트.
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
    print("V484 — Seed Averaging + Bottom-N Adversarial Features")
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

    print(f"Train: {train.shape}")

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
    import lightgbm as lgb

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # === Fast train vs test adversarial (3 seeds, 100 trees) ===
    print("\n" + "=" * 60)
    print("Fast Train vs Test Adversarial (3 seeds x 100 trees)")
    print("=" * 60)

    n_train = len(X_train)
    y_adv = np.array([0]*n_train + [1]*len(X_test))
    X_both = np.vstack([X_train, X_test])

    adv_models = []
    for seed in range(3):
        adv = lgb.LGBMClassifier(
            n_estimators=100, learning_rate=0.05, num_leaves=31,
            max_depth=5, min_child_samples=20,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.5, reg_lambda=1.0,
            random_state=42 + seed, verbose=-1, n_jobs=-1
        )
        adv.fit(X_both, y_adv)
        adv_models.append(adv)

    mean_adv_imp = np.mean([m.feature_importances_ for m in adv_models], axis=0)
    adv_imp_norm = mean_adv_imp / (mean_adv_imp.max() + 1e-10)
    sorted_idx = np.argsort(-mean_adv_imp)

    print("\nTop 20 adversarial important features:")
    for i in range(20):
        idx = sorted_idx[i]
        print(f"  #{i+1}: {feature_cols[idx]} (adv={adv_imp_norm[idx]:.3f})")

    # === Test strategies: seed averaging with different N_keep ===
    print(f"\n{'='*60}")
    print("Seed Averaging (15 seeds) x N_keep strategies")
    print("=" * 60)

    # Also include ALL 68 features as baseline
    n_keep_values = [68, 54, 47, 40, 34, 30, 25, 21, 15]
    results = {}

    for n_keep in n_keep_values:
        if n_keep < 68:
            mask = np.zeros(len(feature_cols), dtype=bool)
            mask[sorted_idx[-n_keep:]] = True
        else:
            mask = np.ones(len(feature_cols), dtype=bool)
        X_sel = X_train[:, mask]
        n_kept = mask.sum()

        print(f"\n--- Keep {n_kept} features (bottom-{n_keep} adv) ---")

        # Strategy 1: Balanced config + 15 seeds
        all_student_preds_arr = np.zeros(len(X_sel))
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_sel, y_bin)):
            X_tr, X_val = X_sel[tr_idx], X_sel[val_idx]
            all_seeds_stu = np.zeros((len(X_val), 15))
            for seed in range(15):
                stu_fold = np.zeros(len(X_val))
                for t in TARGETS:
                    y_t_tr = (train.iloc[tr_idx][t] > 0.5).astype(int).values
                    m = lgb.LGBMClassifier(
                        n_estimators=500, learning_rate=0.05, num_leaves=31,
                        max_depth=5, min_child_samples=30,
                        subsample=0.8, colsample_bytree=0.8,
                        reg_alpha=0.5, reg_lambda=1.0,
                        random_state=42 + seed, verbose=-1, n_jobs=-1
                    )
                    m.fit(X_tr, y_t_tr)
                    stu_fold += m.predict_proba(X_val)[:, 1] / len(TARGETS)
                all_seeds_stu[:, seed] = stu_fold
            all_student_preds_arr[val_idx] = all_seeds_stu.mean(axis=1)

        student_oof = 1 - np.mean(np.abs(all_student_preds_arr - y_avg))

        # Strategy 2: Aggressive config + 15 seeds
        all_aggr_preds_arr = np.zeros(len(X_sel))
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_sel, y_bin)):
            X_tr, X_val = X_sel[tr_idx], X_sel[val_idx]
            all_seeds_aggr = np.zeros((len(X_val), 15))
            for seed in range(15):
                stu_fold = np.zeros(len(X_val))
                for t in TARGETS:
                    y_t_tr = (train.iloc[tr_idx][t] > 0.5).astype(int).values
                    m = lgb.LGBMClassifier(
                        n_estimators=500, learning_rate=0.10, num_leaves=63,
                        max_depth=8, min_child_samples=20,
                        subsample=0.6, colsample_bytree=0.6,
                        reg_alpha=0.1, reg_lambda=0.1,
                        random_state=42 + seed, verbose=-1, n_jobs=-1
                    )
                    m.fit(X_tr, y_t_tr)
                    stu_fold += m.predict_proba(X_val)[:, 1] / len(TARGETS)
                all_seeds_aggr[:, seed] = stu_fold
            all_aggr_preds_arr[val_idx] = all_seeds_aggr.mean(axis=1)

        aggr_student_oof = 1 - np.mean(np.abs(all_aggr_preds_arr - y_avg))

        # Meta (5 seeds)
        all_meta = np.zeros(len(X_sel))
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_sel, y_bin)):
            meta_seeds = np.zeros((len(X_val), 5))
            for seed in range(5):
                m_meta = lgb.LGBMRegressor(
                    n_estimators=500, learning_rate=0.03, num_leaves=25,
                    max_depth=5, min_child_samples=30,
                    subsample=0.8, colsample_bytree=0.7,
                    reg_alpha=1.0, reg_lambda=3.0,
                    random_state=99 + seed, verbose=-1, n_jobs=-1
                )
                m_meta.fit(X_sel[tr_idx], y_avg[tr_idx])
                meta_seeds[:, seed] = m_meta.predict(X_sel[val_idx])
            all_meta[val_idx] = meta_seeds.mean(axis=1)

        meta_oof = 1 - np.mean(np.abs(all_meta - y_avg))

        results[str(n_keep)] = {
            'n_features': n_kept,
            'balanced_seed15': round(student_oof, 5),
            'aggressive_seed15': round(aggr_student_oof, 5),
            'meta_oof': round(meta_oof, 5),
        }

        print(f"  Balanced(15 seeds): {student_oof:.5f} | Aggressive(15 seeds): {aggr_student_oof:.5f}")
        print(f"  Meta OOF: {meta_oof:.5f}")

    # Find best
    valid = results
    best_bal = min(valid.items(), key=lambda x: x[1]['balanced_seed15'])
    best_aggr = min(valid.items(), key=lambda x: x[1]['aggressive_seed15'])

    print(f"\n{'='*60}")
    print(f"BEST (balanced 15 seeds): {best_bal[0]} features → {best_bal[1]['balanced_seed15']}")
    print(f"BEST (aggressive 15 seeds): {best_aggr[0]} features → {best_aggr[1]['aggressive_seed15']}")
    print(f"{'='*60}")

    elapsed = time.time() - start_time

    ts = time.strftime("%Y%m%d_%H%M%S")
    result = {
        "version": "V484",
        "name": "Seed Averaging + Bottom-N Adversarial Features",
        "results": {str(k): v for k, v in valid.items()},
        "best_balanced": str(best_bal[0]),
        "best_aggressive": str(best_aggr[0]),
        "timestamp": ts,
        "total_time_s": int(elapsed)
    }

    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    with open(EXPERIMENTS / f"v484_{ts}.json", 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Saved: {EXPERIMENTS}/v484_{ts}.json")

if __name__ == '__main__':
    main()
