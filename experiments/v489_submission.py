#!/usr/bin/env python3
"""
V489 — Best Config Submission (V484 aggressive seed-averaging)
가장 낮은 train OOF(0.7653)을 기록한 config로 submission 파일 생성.
- Aggressive student config: lr=0.10, leaves=63, depth=8, min_child=20, sub=0.6, col=0.6
- Bottom-21 adversarial features
- 15 seeds averaging
- submission 파일만 생성 (승우さんが 수동 제출)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path('/home/mwoo423/.openclaw/workspace')
DATA_PROC = ROOT / 'data_processed'
DATA_RAW = ROOT / 'data_raw'
SUBMISSIONS = ROOT / 'submissions'

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']

def main():
    import time
    start_time = time.time()
    print("=" * 60)
    print("V489 — Best Config Submission")
    print("=" * 60)

    # Load
    train_feat = pd.read_parquet(DATA_PROC / 'train_features_clean_v60.parquet')
    test_feat = pd.read_parquet(DATA_PROC / 'test_features_clean_v60.parquet')
    train_csv = pd.read_csv(DATA_RAW / 'ch2026_metrics_train.csv')

    # Merge
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
    import lightgbm as lgb

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Adversarial feature selection
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
    sorted_idx = np.argsort(-mean_adv_imp)

    # Keep bottom 21 features
    mask = np.zeros(len(feature_cols), dtype=bool)
    mask[sorted_idx[-21:]] = True
    X_train_sel = X_train[:, mask]
    X_test_sel = X_test[:, mask]
    n_kept = mask.sum()

    print(f"Features: {n_kept}")
    print(f"Selected: {[feature_cols[i] for i in np.where(mask)[0]]}")

    # Aggressive config: lr=0.10, leaves=63, depth=8, min_child=20, sub=0.6, col=0.6
    # 15 seeds averaging
    print("\nTraining with 15 seeds...")
    
    all_test_preds = np.zeros((len(X_test_sel), 15, len(TARGETS)))
    
    for seed in range(15):
        print(f"  Seed {seed}/15...")
        for ti, t in enumerate(TARGETS):
            y_t = (train[t] > 0.5).astype(int).values
            m = lgb.LGBMClassifier(
                n_estimators=500, learning_rate=0.10, num_leaves=63,
                max_depth=8, min_child_samples=20,
                subsample=0.6, colsample_bytree=0.6,
                reg_alpha=0.1, reg_lambda=0.1,
                random_state=42 + seed, verbose=-1, n_jobs=-1
            )
            m.fit(X_train_sel, y_t)
            all_test_preds[:, seed, ti] = m.predict_proba(X_test_sel)[:, 1]

    # Average over seeds and targets
    final_preds = np.mean(all_test_preds, axis=(1, 2))

    # Create submission
    submission = pd.DataFrame({
        'subject_id': test_feat['subject_id'].values,
        'date': pd.to_datetime(test_feat['date']).dt.strftime('%Y-%m-%d'),
    })
    for i, t in enumerate(TARGETS):
        submission[t] = final_preds

    ts = time.strftime("%Y%m%d_%H%M%S")
    filename = SUBMISSIONS / f"submission_v489_aggressive_21features_{ts}.csv"
    submission.to_csv(filename, index=False)
    print(f"\nSubmission saved: {filename}")
    print(f"Shape: {submission.shape}")
    print(f"Columns: {list(submission.columns)}")
    
    print(f"\nPrediction stats:")
    for t in TARGETS:
        print(f"  {t}: mean={submission[t].mean():.4f}, std={submission[t].std():.4f}, min={submission[t].min():.4f}, max={submission[t].max():.4f}")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s")

if __name__ == '__main__':
    main()
