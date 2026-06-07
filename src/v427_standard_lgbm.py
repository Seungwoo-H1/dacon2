"""
V427 — LGBM Baseline: No Over-Reg, Standard Params

V424-V426의 핵심 발견:
1. Per-subject mean baseline avg OOF: 0.59363 (가장 좋음)
2. ML model (V413) avg OOF: 0.651 (baseline보다 나쁨)
3. Z-score features: model OOF 0.678 (baseline보다 더 나쁨)
4. Feature importance ALL ZERO → 과도한 regularization으로 model 학습 못 함

V427: standard LGBM params으로 다시 시도.
- reg_alpha=0, reg_lambda=1 (기본값)
- num_leaves=31, max_depth=-1 (deep learning 허용)
- lr=0.1, n_estimators=500 (더 강력하게 학습)
- colsample_bytree=0.8, subsample=0.8

이렇게 해서 baseline(0.594)보다 나은 모델이 나오면 features가 signal임을 확인.
아니면 baseline이 진짜 한계임을 확인.
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


def main():
    global t_start
    t_start = time.time()

    log.info("=" * 70)
    log.info("V427 — Standard LGBM (No Over-Reg)")
    log.info("=" * 70)

    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")

    groups_arr = train_df['subject_id'].values.astype(str)

    # Baseline
    subj_means = {}
    for target in TARGETS:
        subj_means[target] = train_df.groupby('subject_id')[target].mean().to_dict()

    avg_baseline = 0
    for target in TARGETS:
        y = train_df[target].values
        y_pred = np.array([subj_means[target].get(sid, train_df[target].mean())
                          for sid in groups_arr])
        avg_baseline += log_loss(y, y_pred)
    avg_baseline /= len(TARGETS)
    log.info(f"Baseline avg OOF: {avg_baseline:.5f}")

    # ===== Phase 1: Standard LGBM (V413-style) =====
    log.info("\n=== Phase 1: Standard LGBM ===")

    feat_cols = get_feature_cols(train_df)

    results = {}
    for target in TARGETS:
        feat_cols_t = remove_leak(feat_cols, target)
        y = train_df[target].values.astype(np.float64)
        X = train_df[feat_cols_t].fillna(0).values.astype(np.float64)
        spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)

        params = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
            'num_leaves': 63, 'max_depth': -1, 'learning_rate': 0.05,
            'n_estimators': 1000, 'bagging_fraction': 0.8,
            'feature_fraction': 0.8, 'bagging_freq': 5,
            'min_child_samples': 10, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
            'scale_pos_weight': spw, 'random_state': 42,
            'force_row_wise': True, 'n_jobs': 1,
        }

        sn = [sanitize_col(c) for c in feat_cols_t]
        ds = lgb.Dataset(X, label=y, feature_name=sn)
        skf = GroupKFold(n_splits=5)
        cv_oof = np.zeros(len(train_df))

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y, groups_arr)):
            ds_tr = lgb.Dataset(X[tr_idx], label=y[tr_idx], feature_name=sn)
            ds_val = lgb.Dataset(X[val_idx], label=y[val_idx], feature_name=sn, reference=ds_tr)
            model = lgb.train(params, ds_tr, num_boost_round=1000,
                valid_sets=[ds_val], callbacks=[lgb.early_stopping(50, verbose=False)])
            cv_oof[val_idx] = model.predict(X[val_idx])

        oof = log_loss(y, cv_oof)
        baseline_pred = np.array([subj_means[target].get(sid, train_df[target].mean())
                                  for sid in groups_arr])
        baseline_oof = log_loss(y, baseline_pred)

        results[target] = {
            'model_oof': oof, 'baseline_oof': baseline_oof,
            'delta': oof - baseline_oof,
            'best_iteration': model.best_iteration,
        }

        log.info(f"  {target}: model_OOF={oof:.5f}, baseline={baseline_oof:.5f}, "
                 f"delta={oof-baseline_oof:+.5f}, best_iter={model.best_iteration}")

    avg_model = np.mean([r['model_oof'] for r in results.values()])
    avg_delta = np.mean([r['delta'] for r in results.values()])

    log.info(f"\n  AVG model OOF: {avg_model:.5f}")
    log.info(f"  AVG baseline OOF: {avg_baseline:.5f}")
    log.info(f"  AVG delta: {avg_delta:+.5f}")

    # ===== Phase 2: Per-Config LGBM (like V413) =====
    log.info("\n=== Phase 2: Per-Target Config (V413-style) ===")

    configs = {
        'Q': {'num_leaves': 31, 'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 800,
              'min_child_samples': 15, 'reg_alpha': 1.0, 'reg_lambda': 10.0},
        'S': {'num_leaves': 127, 'max_depth': -1, 'learning_rate': 0.05, 'n_estimators': 1500,
              'min_child_samples': 5, 'reg_alpha': 0.0, 'reg_lambda': 1.0},
    }

    results2 = {}
    for target in TARGETS:
        feat_cols_t = remove_leak(feat_cols, target)
        y = train_df[target].values.astype(np.float64)
        X = train_df[feat_cols_t].fillna(0).values.astype(np.float64)
        spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
        prefix = 'Q' if target.startswith('Q') else 'S'
        cfg = configs[prefix]

        params = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
            'num_leaves': cfg['num_leaves'], 'max_depth': cfg['max_depth'],
            'learning_rate': cfg['learning_rate'], 'n_estimators': cfg['n_estimators'],
            'bagging_fraction': 0.8, 'feature_fraction': 0.8, 'bagging_freq': 5,
            'min_child_samples': cfg['min_child_samples'],
            'reg_alpha': cfg['reg_alpha'], 'reg_lambda': cfg['reg_lambda'],
            'scale_pos_weight': spw, 'random_state': 42,
            'force_row_wise': True, 'n_jobs': 1,
        }

        sn = [sanitize_col(c) for c in feat_cols_t]
        ds = lgb.Dataset(X, label=y, feature_name=sn)
        skf = GroupKFold(n_splits=5)
        cv_oof = np.zeros(len(train_df))

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y, groups_arr)):
            ds_tr = lgb.Dataset(X[tr_idx], label=y[tr_idx], feature_name=sn)
            ds_val = lgb.Dataset(X[val_idx], label=y[val_idx], feature_name=sn, reference=ds_tr)
            model = lgb.train(params, ds_tr, num_boost_round=cfg['n_estimators'],
                valid_sets=[ds_val], callbacks=[lgb.early_stopping(100, verbose=False)])
            cv_oof[val_idx] = model.predict(X[val_idx])

        oof = log_loss(y, cv_oof)
        baseline_pred = np.array([subj_means[target].get(sid, train_df[target].mean())
                                  for sid in groups_arr])
        baseline_oof = log_loss(y, baseline_pred)

        results2[target] = {
            'model_oof': oof, 'baseline_oof': baseline_oof,
            'delta': oof - baseline_oof,
            'best_iteration': model.best_iteration,
        }

        log.info(f"  {target}: model_OOF={oof:.5f}, baseline={baseline_oof:.5f}, "
                 f"delta={oof-baseline_oof:+.5f}, best_iter={model.best_iteration}")

    avg_model2 = np.mean([r['model_oof'] for r in results2.values()])
    avg_delta2 = np.mean([r['delta'] for r in results2.values()])

    log.info(f"\n  AVG model OOF: {avg_model2:.5f}")
    log.info(f"  AVG delta: {avg_delta2:+.5f}")

    # ===== Phase 3: Submission =====
    log.info("\n=== Phase 3: Submission (best config) ===")

    # Use better of the two approaches
    best_results = results if avg_delta < avg_delta2 else results2
    best_name = "standard" if avg_delta < avg_delta2 else "per_target"
    log.info(f"Best approach: {best_name} (avg_delta={min(avg_delta, avg_delta2):+.5f})")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate submission using best approach
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values

    use_results = results if best_name == "standard" else results2
    use_configs = configs

    for target in TARGETS:
        feat_cols_t = remove_leak(feat_cols, target)
        X_train = train_df[feat_cols_t].fillna(0).values.astype(np.float64)
        X_test = test_df[feat_cols_t].fillna(0).values.astype(np.float64)
        y = train_df[target].values.astype(np.float64)
        spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
        prefix = 'Q' if target.startswith('Q') else 'S'
        cfg = use_configs[prefix]

        params = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
            'num_leaves': cfg['num_leaves'], 'max_depth': cfg['max_depth'],
            'learning_rate': cfg['learning_rate'], 'n_estimators': cfg['n_estimators'],
            'bagging_fraction': 0.8, 'feature_fraction': 0.8, 'bagging_freq': 5,
            'min_child_samples': cfg['min_child_samples'],
            'reg_alpha': cfg['reg_alpha'], 'reg_lambda': cfg['reg_lambda'],
            'scale_pos_weight': spw, 'random_state': 42,
            'force_row_wise': True, 'n_jobs': 1,
        }

        sn = [sanitize_col(c) for c in feat_cols_t]
        ds = lgb.Dataset(X_train, label=y, feature_name=sn)

        # Use GroupKFold 5-seed ensemble
        skf = GroupKFold(n_splits=5)
        preds = np.zeros(len(X_test))

        for seed_offset in range(5):
            model = lgb.train(params, ds, num_boost_round=use_results[target]['best_iteration'],
                callbacks=[lgb.record_evaluation()])
            # Different seeds via different random_state offsets
            params_seeded = params.copy()
            params_seeded['random_state'] = 42 + seed_offset
            ds_seeded = lgb.Dataset(X_train, label=y, feature_name=sn,
                                   params={'seed': str(42 + seed_offset)})
            model = lgb.train(params_seeded, ds_seeded,
                num_boost_round=use_results[target]['best_iteration'])
            preds += model.predict(X_test) / 5

        sub[target] = np.clip(preds, 0.01, 0.99)

    sub_path = SUBMIT / f"submission_v427_standard_lgbm_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved: {sub_path}")

    # ===== Summary =====
    log.info(f"\n{'='*70}")
    log.info("V427 Summary:")
    log.info(f"  Baseline AVG OOF: {avg_baseline:.5f}")
    log.info(f"  Standard LGBM AVG OOF: {avg_model:.5f} (delta={avg_delta:+.5f})")
    log.info(f"  Per-target LGBM AVG OOF: {avg_model2:.5f} (delta={avg_delta2:+.5f})")
    log.info(f"  Best: {best_name} (delta={min(avg_delta, avg_delta2):+.5f})")
    log.info(f"{'='*70}")

    result = {
        'version': 'V427',
        'name': 'Standard LGBM (No Over-Reg)',
        'baseline_avg_oof': round(float(avg_baseline), 5),
        'standard_lgbm': {
            'avg_model_oof': round(float(avg_model), 5),
            'avg_delta': round(float(avg_delta), 5),
        },
        'per_target_lgbm': {
            'avg_model_oof': round(float(avg_model2), 5),
            'avg_delta': round(float(avg_delta2), 5),
        },
        'per_target': {t: {k: round(float(v), 5) if isinstance(v, (np.floating, float)) else v
                          for k, v in r.items()} for t, r in use_results.items()},
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }

    meta_path = EXPERIMENTS / f'v427_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(result, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    log.info(f"Total time: {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    main()
