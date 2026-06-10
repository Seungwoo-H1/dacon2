#!/usr/bin/env python3
"""
V488 — Multi-Target OOF Stacking
가설: per-target student prediction을 feature로 추가해 meta learner가 학습하면
      student OOF를 낮출 수 있음.
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

def to_json(o):
    if isinstance(o, dict): return {str(k): to_json(v) for k,v in o.items()}
    if isinstance(o, (list,tuple)): return [to_json(x) for x in o]
    if isinstance(o,(np.integer,)): return int(o)
    if isinstance(o,(np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return o

def main():
    start_time = time.time()
    print("=" * 60)
    print("V488 — Multi-Target OOF Stacking")
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
    y_avg = train[TARGETS].mean(axis=1).values
    y_bin = (y_avg > 0.5).astype(int)

    from sklearn.model_selection import StratifiedKFold
    import lightgbm as lgb

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Fast adversarial
    X_test = test_feat[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    n_train = len(X_train)
    y_adv = np.array([0]*n_train + [1]*len(X_test))
    X_both = np.vstack([X_train, X_test])

    adv_models = []
    for seed in range(3):
        adv = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31,
            max_depth=5, min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.5, reg_lambda=1.0, random_state=42+seed, verbose=-1, n_jobs=-1)
        adv.fit(X_both, y_adv)
        adv_models.append(adv)

    mean_adv_imp = np.mean([m.feature_importances_ for m in adv_models], axis=0)
    adv_imp_norm = mean_adv_imp / (mean_adv_imp.max() + 1e-10)
    sorted_idx = np.argsort(-mean_adv_imp)

    n_keep_values = [21, 25, 30, 40, 68]
    results = {}

    for n_keep in n_keep_values:
        mask = np.zeros(len(feature_cols), dtype=bool)
        mask[sorted_idx[-n_keep:]] = True
        X_sel = X_train[:, mask]
        n_kept = mask.sum()

        print(f"\n{'='*60}")
        print(f"N_keep={n_kept}")
        print("=" * 60)

        # === Baseline: simple average of per-target student preds ===
        all_stu = np.zeros(len(X_sel))
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_sel, y_bin)):
            all_seeds = np.zeros((len(X_val) if False else len(X_sel[val_idx]), 15))
            stu_fold = np.zeros(len(X_sel[val_idx]))
            for t in TARGETS:
                y_t = (train.iloc[tr_idx][t] > 0.5).astype(int).values
                s = np.zeros((len(X_sel[val_idx]), 15))
                for seed in range(15):
                    m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
                        max_depth=5, min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
                        reg_alpha=0.5, reg_lambda=1.0, random_state=42+seed, verbose=-1, n_jobs=-1)
                    m.fit(X_sel[tr_idx], y_t)
                    s[:, seed] = m.predict_proba(X_sel[val_idx])[:,1]
                stu_fold += s.mean(axis=1) / len(TARGETS)
            all_stu[val_idx] = stu_fold

        stu_oof = 1 - np.mean(np.abs(all_stu - y_avg))
        print(f"  Baseline (avg per-target, 15 seeds): {stu_oof:.5f}")

        # === OOF Stacking ===
        # For each fold, get OOF student preds per target, then meta uses raw + student preds
        
        # Step 1: Get OOF student predictions per target (all samples)
        oof_stu_preds = np.zeros((len(X_sel), len(TARGETS)))
        for ti, t in enumerate(TARGETS):
            y_t = (train[t] > 0.5).astype(int).values
            for fold, (tr_idx, val_idx) in enumerate(skf.split(X_sel, y_bin)):
                y_t_tr = y_t[tr_idx]
                s = np.zeros((len(val_idx), 15))
                for seed in range(15):
                    m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
                        max_depth=5, min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
                        reg_alpha=0.5, reg_lambda=1.0, random_state=42+seed, verbose=-1, n_jobs=-1)
                    m.fit(X_sel[tr_idx], y_t_tr)
                    s[:, seed] = m.predict_proba(X_sel[val_idx])[:,1]
                oof_stu_preds[val_idx, ti] = s.mean(axis=1)

        # Step 2: Meta learner using raw features + OOF student preds
        all_meta = np.zeros(len(X_sel))
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_sel, y_bin)):
            X_tr_meta = np.hstack([X_sel[tr_idx], oof_stu_preds[tr_idx]])
            X_val_meta = np.hstack([X_sel[val_idx], oof_stu_preds[val_idx]])

            m_meta = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=25,
                max_depth=5, min_child_samples=30, subsample=0.8, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=3.0, random_state=99+fold, verbose=-1, n_jobs=-1)
            m_meta.fit(X_tr_meta, y_avg[tr_idx])
            all_meta[val_idx] = m_meta.predict(X_val_meta)

        meta_oof = 1 - np.mean(np.abs(all_meta - y_avg))
        
        # Average of student preds (same as baseline)
        stu_avg = np.mean(oof_stu_preds, axis=1)
        stu_avg_oof = 1 - np.mean(np.abs(stu_avg - y_avg))

        # Student avg vs meta
        gap = np.mean(all_meta - stu_avg)

        print(f"  Student avg OOF: {stu_avg_oof:.5f}")
        print(f"  Meta OOF (stacking): {meta_oof:.5f}")
        print(f"  Gap: {gap:.5f}")

        results[str(n_keep)] = {
            'n_features': n_kept,
            'student_avg_oof': round(stu_avg_oof, 5),
            'meta_stacking_oof': round(meta_oof, 5),
            'gap': round(gap, 5),
        }

        print(f"  Meta beats student: {meta_oof < stu_avg_oof}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k}: {to_json(v)}")
    print("=" * 60)

    elapsed = time.time() - start_time
    ts = time.strftime("%Y%m%d_%H%M%S")
    result = {
        "version": "V488",
        "name": "Multi-Target OOF Stacking",
        "results": to_json({k: v for k, v in results.items()}),
        "timestamp": ts,
        "total_time_s": int(elapsed)
    }

    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    with open(EXPERIMENTS / f"v488_{ts}.json", 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Saved: {EXPERIMENTS}/v488_{ts}.json")

if __name__ == '__main__':
    main()
