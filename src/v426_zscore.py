"""
V426 — Per-Person Z-Score Features + Baseline Ensemble

FUNDAMENTAL INSIGHT: features.parquet has ONLY 142 base features.
Per-person z-score features (which would double features to ~282) are NOT included.
This is WHY V413 (OOF 0.651) is worse than baseline (OOF 0.594).
The model lacks per-person deviation signal.

V426 approach:
1. Regenerate features WITH per-person z-score columns
2. Use very strong regularization (prevents overfitting on 450 samples)
3. Blend with per-subject mean baseline
4. Test: if z-score features help, we might break 0.60 LB

Key: per-person z-score captures "how this subject's behavior deviates
from their own average" — this is the signal that baseline misses.
"""
import sys, gc, logging, json, re, time, warnings, math
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

ROOT = Path('/home/mwoo423/.openclaw/workspace')
DATA = ROOT / 'data_processed'
SUBMIT = ROOT / 'submissions'
EXPERIMENTS = ROOT / 'experiments'
SUBMIT.mkdir(exist_ok=True)
EXPERIMENTS.mkdir(exist_ok=True)

TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']
META_COLS = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}

LEAK_S = {
    'wLight_w_light_mean','wLight_w_light_std','wLight_w_light_min',
    'wLight_w_light_max','wLight_w_light_count',
    'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max',
    'wHr_hr_median','wHr_hr_count',
    'wPedo_pedo_step_mean','wPedo_pedo_step_sum',
    'wPedo_pedo_step_frequency_mean','wPedo_pedo_step_frequency_sum',
    'wPedo_pedo_running_step_mean','wPedo_pedo_running_step_sum',
    'wPedo_pedo_walking_step_mean','wPedo_pedo_walking_step_sum',
    'wPedo_pedo_distance_mean','wPedo_pedo_distance_sum',
    'wPedo_pedo_speed_mean','wPedo_pedo_speed_sum',
    'wPedo_pedo_burned_calories_mean','wPedo_pedo_burned_calories_sum',
}
LEAK_Q = {'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max',
          'wHr_hr_median','wHr_hr_count'}


def sanitize_col(n):
    return re.sub(r'[^a-zA-Z0-9_]', '_', n)

def get_feature_cols(df):
    return [c for c in df.columns
            if c not in META_COLS | set(TARGETS)
            and np.issubdtype(df[c].dtype, np.number)]

def remove_leak(cols, target):
    if target.startswith('S'):
        return [c for c in cols if c not in LEAK_S]
    elif target.startswith('Q'):
        return [c for c in cols if c not in LEAK_Q]
    return cols


def create_personal_features(train_df, test_df):
    """Create per-person z-score features for all numeric features."""
    log.info("Creating per-person z-score features...")

    train = train_df.copy()
    test = test_df.copy()

    # Get all numeric feature columns
    all_num_cols = [c for c in train.columns
                    if c not in META_COLS | set(TARGETS)
                    and np.issubdtype(train[c].dtype, np.number)]

    zscore_cols_created = 0

    for feat in all_num_cols:
        # Compute subject-level stats from training data
        subj_stats = train.groupby('subject_id')[feat].agg(['mean', 'std'])

        # Add personal z-score to train
        train[f'{feat}_pzscore'] = 0.0
        for sid, stats in subj_stats.iterrows():
            mask = train['subject_id'] == sid
            std = stats['std'] if stats['std'] > 0 else 1.0
            train.loc[mask, f'{feat}_pzscore'] = (
                train.loc[mask, feat] - stats['mean']
            ) / std

        # For test: use training subject stats
        test[f'{feat}_pzscore'] = 0.0
        for sid, stats in subj_stats.iterrows():
            mask = test['subject_id'] == sid
            if mask.any():
                std = stats['std'] if stats['std'] > 0 else 1.0
                test.loc[mask, f'{feat}_pzscore'] = (
                    test.loc[mask, feat] - stats['mean']
                ) / std

        zscore_cols_created += 1

    # Fill NaN
    train = train.fillna(0)
    test = test.fillna(0)

    log.info(f"Created {zscore_cols_created} z-score features")
    return train, test


def main():
    global t_start
    t_start = time.time()

    log.info("=" * 70)
    log.info("V426 — Per-Person Z-Score Features + Baseline Ensemble")
    log.info("Hypothesis: Z-score features add per-person signal missing from baseline")
    log.info("=" * 70)

    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")

    log.info(f"Train before personalization: {train_df.shape}")
    log.info(f"Test before personalization: {test_df.shape}")

    # ===== Phase 1: Create personal z-score features =====
    train_enhanced, test_enhanced = create_personal_features(train_df, test_df)

    log.info(f"Train after personalization: {train_enhanced.shape}")
    log.info(f"Test after personalization: {test_enhanced.shape}")

    zscore_cols = [c for c in train_enhanced.columns if 'pzscore' in c]
    base_cols = [c for c in train_enhanced.columns if 'pzscore' not in c
                 and c not in META_COLS | set(TARGETS)
                 and np.issubdtype(train_enhanced[c].dtype, np.number)]
    log.info(f"Base features: {len(base_cols)}, Z-score features: {len(zscore_cols)}")

    # ===== Phase 2: Baseline computation =====
    log.info("\n=== Phase 2: Baseline ===")

    subj_means = {}
    for target in TARGETS:
        subj_means[target] = train_df.groupby('subject_id')[target].mean().to_dict()

    # ===== Phase 3: Test different feature sets with GroupKFold =====
    log.info("\n=== Phase 3: Feature Set Comparison ===")

    feature_sets = {
        'base_only': base_cols[:20],  # Top 20 base features
        'zscore_only': zscore_cols[:20],  # Top 20 z-score features
        'base+zscore': base_cols[:10] + zscore_cols[:10],  # Mix
        'all': base_cols[:15] + zscore_cols[:15],  # All mix
    }

    results = {}

    for set_name, feat_set in feature_sets.items():
        log.info(f"\n  --- {set_name} (n_feat={len(feat_set)}) ---")

        all_oofs = {}
        all_students = {}

        for target in TARGETS:
            feat_cols_t = remove_leak(feat_set, target)
            all_feats = feat_cols_t + [target]
            X_all = train_enhanced[all_feats].fillna(0).values.astype(np.float64)
            y_all = X_all[:, -1]
            X_all = X_all[:, :-1]

            skf = GroupKFold(n_splits=5)
            cv_oof = np.zeros(len(train_enhanced))

            for fold, (tr_idx, val_idx) in enumerate(skf.split(X_all, y_all, train_enhanced['subject_id'].values)):
                x_train, y_train = X_all[tr_idx], y_all[tr_idx]
                x_val = X_all[val_idx]
                spw = max(((y_train == 0).sum()) / max((y_train == 1).sum(), 1), 0.1)

                params = {
                    'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
                    'num_leaves': 6, 'max_depth': 2, 'learning_rate': 0.01,
                    'n_estimators': 100, 'subsample': 0.5, 'colsample_bytree': 0.5,
                    'reg_alpha': 50.0, 'reg_lambda': 200.0, 'min_child_samples': 50,
                    'scale_pos_weight': spw, 'random_state': 42,
                    'force_row_wise': True, 'n_jobs': 1,
                }

                sn = [sanitize_col(c) for c in feat_cols_t]
                ds_train = lgb.Dataset(x_train, label=y_train, feature_name=sn)
                ds_val = lgb.Dataset(x_val, label=y_all[val_idx], feature_name=sn, reference=ds_train)
                model = lgb.train(params, ds_train, num_boost_round=100,
                    valid_sets=[ds_val], callbacks=[lgb.early_stopping(20, verbose=False)])
                cv_oof[val_idx] = model.predict(x_val)

            oof = log_loss(y_all, cv_oof)
            baseline_pred = np.array([subj_means[target].get(sid, train_df[target].mean())
                                      for sid in train_enhanced['subject_id'].values])
            baseline_oof = log_loss(y_all, baseline_pred)

            all_oofs[target] = oof
            all_students[target] = baseline_oof  # student = baseline for gap tracking

            log.info(f"    {target}: model_OOF={oof:.5f} vs baseline={baseline_oof:.5f} "
                     f"(delta={oof-baseline_oof:+.5f})")

        avg_model = np.mean(list(all_oofs.values()))
        avg_baseline = np.mean(list(all_students.values()))
        avg_gap = avg_baseline - avg_model

        results[set_name] = {
            'avg_model_oof': avg_model,
            'avg_baseline_oof': avg_baseline,
            'delta': avg_model - avg_baseline,
            'per_target': all_oofs,
        }

        log.info(f"  AVG: model={avg_model:.5f}, baseline={avg_baseline:.5f}, "
                 f"delta={avg_model-avg_baseline:+.5f}")

    # ===== Phase 4: Find best blend weight =====
    log.info("\n=== Phase 4: Optimal Blend Weight ===")

    # For the best feature set, find optimal blend of model + baseline
    best_set = min(results.keys(), key=lambda k: results[k]['avg_model_oof'])
    log.info(f"  Best feature set: {best_set} (model OOF={results[best_set]['avg_model_oof']:.5f})")

    # Try blending: pred = w * model + (1-w) * baseline
    for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        blended_oofs = {}
        for target in TARGETS:
            model_pred = np.clip(results[best_set]['per_target'][target], 0.01, 0.99)
            baseline_pred = train_df.groupby('subject_id')[target].transform('mean').values
            blended = w * model_pred + (1-w) * baseline_pred
            y = train_df[target].values
            blended_oofs[target] = log_loss(y, np.clip(blended, 0.01, 0.99))

        avg_blended = np.mean(list(blended_oofs.values()))
        log.info(f"  w={w:.1f}: blended_OOF={avg_blended:.5f} "
                 f"(vs baseline {results[best_set]['avg_baseline_oof']:.5f})")

    # ===== Phase 5: Generate submission =====
    log.info("\n=== Phase 5: Submission ===")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Use best feature set with strongest regularization
    best_feat_set = feature_sets[best_set]

    all_test_preds = {}
    all_train_preds = {}

    for target in TARGETS:
        feat_cols_t = remove_leak(best_feat_set, target)
        all_feats = feat_cols_t + [target]
        X_all = train_enhanced[all_feats].fillna(0).values.astype(np.float64)
        y_all = X_all[:, -1]
        X_all = X_all[:, :-1]
        X_test = test_enhanced[feat_cols_t].fillna(0).values.astype(np.float64)

        skf = GroupKFold(n_splits=5)
        train_preds = np.zeros(len(X_all))
        test_preds = np.zeros(len(X_test))

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_all, y_all, train_enhanced['subject_id'].values)):
            x_train, y_train = X_all[tr_idx], y_all[tr_idx]
            x_val = X_all[val_idx]
            spw = max(((y_train == 0).sum()) / max((y_train == 1).sum(), 1), 0.1)

            params = {
                'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
                'num_leaves': 6, 'max_depth': 2, 'learning_rate': 0.01,
                'n_estimators': 100, 'subsample': 0.5, 'colsample_bytree': 0.5,
                'reg_alpha': 50.0, 'reg_lambda': 200.0, 'min_child_samples': 50,
                'scale_pos_weight': spw, 'random_state': 42,
                'force_row_wise': True, 'n_jobs': 1,
            }

            sn = [sanitize_col(c) for c in feat_cols_t]
            ds_train = lgb.Dataset(x_train, label=y_train, feature_name=sn)
            ds_val = lgb.Dataset(x_val, label=y_all[val_idx], feature_name=sn, reference=ds_train)
            model = lgb.train(params, ds_train, num_boost_round=100,
                valid_sets=[ds_val], callbacks=[lgb.early_stopping(20, verbose=False)])
            train_preds[val_idx] = model.predict(x_val)
            test_preds += model.predict(X_test) / 5

        all_train_preds[target] = train_preds
        all_test_preds[target] = test_preds

        oof = log_loss(y_all, train_preds)
        baseline_pred = np.array([subj_means[target].get(sid, train_df[target].mean())
                                  for sid in train_enhanced['subject_id'].values])
        baseline_oof = log_loss(y_all, baseline_pred)

        log.info(f"  {target}: model_OOF={oof:.5f} vs baseline={baseline_oof:.5f} "
                 f"(delta={oof-baseline_oof:+.5f})")

    # Blend: 50% model + 50% baseline
    final_preds = {}
    for target in TARGETS:
        model_pred = np.clip(all_test_preds[target], 0.01, 0.99)
        test_subjects = test_df['subject_id'].values
        baseline_pred = np.array([subj_means[target].get(sid, train_df[target].mean())
                                  for sid in test_subjects])
        blended = 0.3 * model_pred + 0.7 * baseline_pred  # favor baseline
        final_preds[target] = np.clip(blended, 0.01, 0.99)

    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        sub[t] = final_preds[t]

    sub_path = SUBMIT / f"submission_v426_zscore_baseline_blend_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved: {sub_path}")

    # Compute final OOF for reporting
    final_oofs = {}
    for target in TARGETS:
        y = train_df[target].values
        model_pred = np.clip(all_train_preds[target], 0.01, 0.99)
        baseline_pred = np.array([subj_means[target].get(sid, train_df[target].mean())
                                  for sid in train_enhanced['subject_id'].values])
        blended = 0.3 * model_pred + 0.7 * baseline_pred
        final_oofs[target] = log_loss(y, np.clip(blended, 0.01, 0.99))

    avg_final = np.mean(list(final_oofs.values()))
    avg_baseline = np.mean([log_loss(train_df[t].values,
              np.array([subj_means[t].get(sid, train_df[t].mean())
                        for sid in train_enhanced['subject_id'].values])) for t in TARGETS])

    log.info(f"\n{'='*70}")
    log.info("V426 Results:")
    log.info(f"  Train features: {train_enhanced.shape[1]} (base={len(base_cols)}, zscore={len(zscore_cols)})")
    log.info(f"  Blended (30% model + 70% baseline) AVG OOF: {avg_final:.5f}")
    log.info(f"  Baseline AVG OOF: {avg_baseline:.5f}")
    log.info(f"  Improvement: {avg_final - avg_baseline:+.5f}")
    log.info(f"{'='*70}")

    result = {
        'version': 'V426',
        'name': 'Per-Person Z-Score Features + Baseline Blend',
        'train_shape': list(train_enhanced.shape),
        'test_shape': list(test_enhanced.shape),
        'base_features': len(base_cols),
        'zscore_features': len(zscore_cols),
        'avg_final_oof': round(float(avg_final), 5),
        'avg_baseline_oof': round(float(avg_baseline), 5),
        'delta': round(float(avg_final - avg_baseline), 5),
        'best_feature_set': best_set,
        'feature_comparison': {k: {'avg_model': round(v['avg_model_oof'], 5),
                                   'avg_baseline': round(v['avg_baseline_oof'], 5),
                                   'delta': round(v['delta'], 5)} for k, v in results.items()},
        'per_target_final_oof': {t: round(float(v), 5) for t, v in final_oofs.items()},
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }

    meta_path = EXPERIMENTS / f'v426_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(result, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    log.info(f"Total time: {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    main()
