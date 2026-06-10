#!/usr/bin/env python3
"""
V487 — Binary Classification as Regression + LGBM Regressor
가설: V461/V466의 student OOF 0.607은 `LGBMClassifier` 기반.
      하지만 target이 binary (0/1)이면 `LGBMRegressor`가 더 나은 calibration을 제공할 수 있음.
      V466 code가 소멸되어 student config를 정확히 알 수 없으므로, 
      regression objective가 OOF를 낮추는지 테스트.
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
    print("V487 — Binary Classification as Regression")
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

    # Adversarial feature selection (fast)
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

    # Test multiple N_keep values
    n_keep_values = [21, 25, 30, 40, 54, 68]
    results = {}

    for n_keep in n_keep_values:
        mask = np.zeros(len(feature_cols), dtype=bool)
        mask[sorted_idx[-n_keep:]] = True
        X_sel = X_train[:, mask]
        n_kept = mask.sum()

        print(f"\n{'='*60}")
        print(f"N_keep={n_kept}")
        print("=" * 60)

        # === Classifier (baseline) ===
        all_stu_cls = np.zeros(len(X_sel))
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_sel, y_bin)):
            X_tr, X_val = X_sel[tr_idx], X_sel[val_idx]
            all_seeds = np.zeros((len(X_val), 15))
            for seed in range(15):
                stu = np.zeros(len(X_val))
                for t in TARGETS:
                    y_t = (train.iloc[tr_idx][t] > 0.5).astype(int).values
                    m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
                        max_depth=5, min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
                        reg_alpha=0.5, reg_lambda=1.0, random_state=42+seed, verbose=-1, n_jobs=-1)
                    m.fit(X_tr, y_t)
                    stu += m.predict_proba(X_val)[:,1] / len(TARGETS)
                all_seeds[:, seed] = stu
            all_stu_cls[val_idx] = all_seeds.mean(axis=1)

        cls_oof = 1 - np.mean(np.abs(all_stu_cls - y_avg))
        print(f"  Classifier (15 seeds): {cls_oof:.5f}")

        # === Regressor (binary targets) ===
        all_stu_reg = np.zeros(len(X_sel))
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_sel, y_bin)):
            X_tr, X_val = X_sel[tr_idx], X_sel[val_idx]
            all_seeds = np.zeros((len(X_val), 15))
            for seed in range(15):
                stu = np.zeros(len(X_val))
                for t in TARGETS:
                    y_t = (train.iloc[tr_idx][t] > 0.5).astype(int).values
                    m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31,
                        max_depth=5, min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
                        reg_alpha=0.5, reg_lambda=1.0,
                        objective='regression', random_state=42+seed, verbose=-1, n_jobs=-1)
                    m.fit(X_tr, y_t)
                    stu += m.predict(X_val) / len(TARGETS)
                all_seeds[:, seed] = stu
            all_stu_reg[val_idx] = all_seeds.mean(axis=1)

        reg_oof = 1 - np.mean(np.abs(all_stu_reg - y_avg))
        print(f"  Regressor (15 seeds): {reg_oof:.5f}")

        # === Regressor (continuous targets) ===
        all_stu_reg_cont = np.zeros(len(X_sel))
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_sel, y_bin)):
            X_tr, X_val = X_sel[tr_idx], X_sel[val_idx]
            all_seeds = np.zeros((len(X_val), 15))
            for seed in range(15):
                stu = np.zeros(len(X_val))
                for t in TARGETS:
                    y_t = train.iloc[tr_idx][t].values  # continuous!
                    m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31,
                        max_depth=5, min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
                        reg_alpha=0.5, reg_lambda=1.0,
                        objective='regression', random_state=42+seed, verbose=-1, n_jobs=-1)
                    m.fit(X_tr, y_t)
                    stu += m.predict(X_val) / len(TARGETS)
                all_seeds[:, seed] = stu
            all_stu_reg_cont[val_idx] = all_seeds.mean(axis=1)

        reg_cont_oof = 1 - np.mean(np.abs(all_stu_reg_cont - y_avg))
        print(f"  Regressor continuous (15 seeds): {reg_cont_oof:.5f}")

        # Meta (regression)
        all_meta = np.zeros(len(X_sel))
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_sel, y_bin)):
            meta_seeds = np.zeros((len(X_val), 5))
            for seed in range(5):
                m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=25,
                    max_depth=5, min_child_samples=30, subsample=0.8, colsample_bytree=0.7,
                    reg_alpha=1.0, reg_lambda=3.0, random_state=99+seed, verbose=-1, n_jobs=-1)
                m.fit(X_sel[tr_idx], y_avg[tr_idx])
                meta_seeds[:, seed] = m.predict(X_sel[val_idx])
            all_meta[val_idx] = meta_seeds.mean(axis=1)

        meta_oof = 1 - np.mean(np.abs(all_meta - y_avg))

        results[str(n_keep)] = {
            'n_features': n_kept,
            'classifier_15': round(cls_oof, 5),
            'regressor_binary_15': round(reg_oof, 5),
            'regressor_continuous_15': round(reg_cont_oof, 5),
            'meta_oof': round(meta_oof, 5),
        }

        print(f"  Meta: {meta_oof:.5f}")

    # Find best
    valid = results
    best_cls = min(valid.items(), key=lambda x: x[1]['classifier_15'])
    best_reg = min(valid.items(), key=lambda x: x[1]['regressor_binary_15'])
    best_reg_cont = min(valid.items(), key=lambda x: x[1]['regressor_continuous_15'])

    print(f"\n{'='*60}")
    print(f"BEST Classifier: {best_cls[0]} → {best_cls[1]['classifier_15']}")
    print(f"BEST Regressor (binary): {best_reg[0]} → {best_reg[1]['regressor_binary_15']}")
    print(f"BEST Regressor (continuous): {best_reg_cont[0]} → {best_reg_cont[1]['regressor_continuous_15']}")
    print("=" * 60)

    elapsed = time.time() - start_time
    ts = time.strftime("%Y%m%d_%H%M%S")
    result = {
        "version": "V487",
        "name": "Binary Classification as Regression",
        "results": to_json({k: v for k, v in valid.items()}),
        "timestamp": ts,
        "total_time_s": int(elapsed)
    }

    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    with open(EXPERIMENTS / f"v487_{ts}.json", 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Saved: {EXPERIMENTS}/v487_{ts}.json")

if __name__ == '__main__':
    main()
