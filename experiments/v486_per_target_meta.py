#!/usr/bin/env python3
"""
V486 — Per-Target Meta + Target-Weighted Student Ensemble
가설: V461/V466의 student OOF 0.602가 legit 하다면, 그 이유는:
  1) feature selection이 train-only CV와 다름 (test distribution 반영)
  2) 또는 student model config가 다름

V486: 학생 모델을 target별 가중치로 ensemble. Q1~Q3(binary)와 S1~S4(binary)의 
      class distribution이 다름. weighted average가 더 나은 OOF를 줄 수 있음.
      
      또한: V461의 meta learner가 regression(평균) vs binary avg를 비교.
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
    print("V486 — Per-Target Meta + Weighted Student Ensemble")
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
    y_avg = train[TARGETS].mean(axis=1).values
    y_bin = (y_avg > 0.5).astype(int)
    
    # Target class balance
    print("\nTarget class balance:")
    for t in TARGETS:
        n_pos = (train[t] > 0.5).mean()
        print(f"  {t}: {n_pos:.3f}")

    from sklearn.model_selection import StratifiedKFold
    import lightgbm as lgb

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # === Strategy 1: Baseline — simple average ===
    print(f"\n{'='*60}")
    print("Strategy 1: Baseline (unweighted avg, 15 seeds)")
    print("=" * 60)

    # Use bottom-21 adversarial features (best from V484)
    # Re-compute fast adversarial
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

    mask_21 = np.zeros(len(feature_cols), dtype=bool)
    mask_21[sorted_idx[-21:]] = True
    X_sel = X_train[:, mask_21]
    n_kept = mask_21.sum()

    print(f"Features: {n_kept}")

    # Baseline: unweighted avg, 15 seeds, balanced config
    all_stu_base = np.zeros(len(X_sel))
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
        all_stu_base[val_idx] = all_seeds.mean(axis=1)

    base_oof = 1 - np.mean(np.abs(all_stu_base - y_avg))
    print(f"Baseline (unweighted, 15 seeds): {base_oof:.5f}")

    # === Strategy 2: Per-target importance weighting ===
    print(f"\n{'='*60}")
    print("Strategy 2: Per-target importance weighting")
    print("=" * 60)

    # Compute per-target feature importance
    target_importance = np.zeros((len(TARGETS), n_kept))
    for ti, t in enumerate(TARGETS):
        imp = np.zeros(n_kept)
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_sel, y_bin)):
            m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31,
                max_depth=5, random_state=42+fold, verbose=-1, n_jobs=-1)
            m.fit(X_sel[tr_idx], (train.iloc[tr_idx][t] > 0.5).astype(int).values)
            imp += m.feature_importances_
        target_importance[ti] = imp / 5

    # For each target, weight by its feature importance sum
    target_weights = target_importance.sum(axis=1)
    target_weights /= target_weights.sum()
    print(f"Target weights: {dict(zip(TARGETS, [f'{w:.3f}' for w in target_weights]))}")

    # Weighted average student predictions
    all_stu_weighted = np.zeros(len(X_sel))
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_sel, y_bin)):
        X_tr, X_val = X_sel[tr_idx], X_sel[val_idx]
        all_seeds = np.zeros((len(X_val), 15))
        for seed in range(15):
            stu = np.zeros(len(X_val))
            for ti, t in enumerate(TARGETS):
                y_t = (train.iloc[tr_idx][t] > 0.5).astype(int).values
                m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
                    max_depth=5, min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.5, reg_lambda=1.0, random_state=42+seed, verbose=-1, n_jobs=-1)
                m.fit(X_tr, y_t)
                stu += m.predict_proba(X_val)[:,1] * target_weights[ti]
            all_seeds[:, seed] = stu
        all_stu_weighted[val_idx] = all_seeds.mean(axis=1)

    weighted_oof = 1 - np.mean(np.abs(all_stu_weighted - y_avg))
    print(f"Weighted (15 seeds): {weighted_oof:.5f}")

    # === Strategy 3: Per-target aggressive config ===
    print(f"\n{'='*60}")
    print("Strategy 3: Per-target aggressive config (15 seeds)")
    print("=" * 60)

    all_stu_aggr = np.zeros(len(X_sel))
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_sel, y_bin)):
        X_tr, X_val = X_sel[tr_idx], X_sel[val_idx]
        all_seeds = np.zeros((len(X_val), 15))
        for seed in range(15):
            stu = np.zeros(len(X_val))
            for t in TARGETS:
                y_t = (train.iloc[tr_idx][t] > 0.5).astype(int).values
                m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.10, num_leaves=63,
                    max_depth=8, min_child_samples=20, subsample=0.6, colsample_bytree=0.6,
                    reg_alpha=0.1, reg_lambda=0.1, random_state=42+seed, verbose=-1, n_jobs=-1)
                m.fit(X_tr, y_t)
                stu += m.predict_proba(X_val)[:,1] / len(TARGETS)
            all_seeds[:, seed] = stu
        all_stu_aggr[val_idx] = all_seeds.mean(axis=1)

    aggr_oof = 1 - np.mean(np.abs(all_stu_aggr - y_avg))
    print(f"Aggressive (15 seeds): {aggr_oof:.5f}")

    # === Strategy 4: Different thresholds for binary → average → binary ===
    print(f"\n{'='*60}")
    print("Strategy 4: Binary averaging variants")
    print("=" * 60)

    # Student OOF metric: 1 - |pred - avg_target| (same as V461)
    # But V461 student is binary classifier → predict_proba average
    # Alternative: apply threshold to predictions before averaging
    
    variants = {
        'no_thresh': lambda p: p,
        'thresh_0.3': lambda p: (p > 0.3).astype(float),
        'thresh_0.5': lambda p: (p > 0.5).astype(float),
        'thresh_0.7': lambda p: (p > 0.7).astype(float),
    }

    for vname, transform in variants.items():
        pred_t = transform(all_stu_base)
        oof_t = 1 - np.mean(np.abs(pred_t - y_avg))
        print(f"  {vname}: OOF = {oof_t:.5f}")

    # === Strategy 5: Different OOF metrics ===
    print(f"\n{'='*60}")
    print("Strategy 5: Different OOF evaluation metrics")
    print("=" * 60)

    # Standard: 1 - MAE (V461 style)
    print(f"  1-MAE (avg): {base_oof:.5f}")
    
    # MAE of binary predictions
    bin_preds = (all_stu_base > 0.5).astype(float)
    print(f"  1-MAE (binary >0.5): {1 - np.mean(np.abs(bin_preds - y_bin)):.5f}")
    
    # Log loss
    from sklearn.metrics import log_loss
    ll = log_loss(y_bin, all_stu_base)
    print(f"  LogLoss: {ll:.5f}")
    
    # Brier score
    brier = np.mean((all_stu_base - y_bin)**2)
    print(f"  Brier: {brier:.5f}")

    # === Meta ===
    print(f"\n{'='*60}")
    print("Meta learner")
    print("=" * 60)

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
    gap = np.mean(all_meta - all_stu_base)
    print(f"Meta OOF: {meta_oof:.5f}")
    print(f"Gap (base): {gap:.5f} ({gap/max(base_oof,1e-10):.2f}x)")

    # === Summary ===
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"Baseline (unweighted, 15 seeds): {base_oof:.5f}")
    print(f"Weighted (15 seeds): {weighted_oof:.5f}")
    print(f"Aggressive (15 seeds): {aggr_oof:.5f}")
    print(f"Meta: {meta_oof:.5f}")
    print(f"Gap (base): {gap:.3f}")
    print(f"{'='*60}")

    elapsed = time.time() - start_time
    ts = time.strftime("%Y%m%d_%H%M%S")
    result = {
        "version": "V486",
        "name": "Per-Target Meta + Weighted Student Ensemble",
        "features": n_kept,
        "baseline_oof": round(base_oof, 5),
        "weighted_oof": round(weighted_oof, 5),
        "aggressive_oof": round(aggr_oof, 5),
        "meta_oof": round(meta_oof, 5),
        "gap": round(gap, 5),
        "timestamp": ts,
        "total_time_s": int(elapsed)
    }

    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    with open(EXPERIMENTS / f"v486_{ts}.json", 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Saved: {EXPERIMENTS}/v486_{ts}.json")

if __name__ == '__main__':
    main()
