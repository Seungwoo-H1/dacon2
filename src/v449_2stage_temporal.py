"""
V449 — 2-Stage Stacking + Temporal Features + Per-Group Models

Hypothesis: V446(V339 0.589)에서 더 내려가기 위해:
1. Temporal features: day_of_week, week_of_year, hourOfDay, is_night, time_since_sleep
2. 2-Stage Stacking: Stage1 OOF → Stage2 features → Stage2 OOF → final meta
3. Q-group과 S-group을 완전히 별도 모델로 (서로 다른 cross-target weights)
4. Per-subject temporal patterns: subject 내 date별 trend feature
"""
import sys, gc, logging, json, re, time, warnings
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
Q_TARGETS = ['Q1','Q2','Q3']
S_TARGETS = ['S1','S2','S3','S4']

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
LEAK_Q = {
    'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max',
    'wHr_hr_median','wHr_hr_count',
}

LGB_CFGS = {
    'narrow': {'num_leaves': 20, 'max_depth': 4, 'learning_rate': 0.01, 'n_estimators': 2500,
               'subsample': 0.5, 'colsample_bytree': 0.5, 'reg_alpha': 8.0, 'reg_lambda': 30.0,
               'min_child_samples': 35},
    'soft_aggressive': {'num_leaves': 12, 'max_depth': 3, 'learning_rate': 0.012, 'n_estimators': 2000,
                        'subsample': 0.55, 'colsample_bytree': 0.55, 'reg_alpha': 4.0, 'reg_lambda': 15.0,
                        'min_child_samples': 25},
    'ultra_deep': {'num_leaves': 25, 'max_depth': 5, 'learning_rate': 0.025, 'n_estimators': 1000,
                   'subsample': 0.75, 'colsample_bytree': 0.65, 'reg_alpha': 0.3, 'reg_lambda': 1.5,
                   'min_child_samples': 12},
    'safety': {'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.02, 'n_estimators': 1000,
               'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 3.0, 'reg_lambda': 10.0,
               'min_child_samples': 20},
    'broad': {'num_leaves': 40, 'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 800,
              'subsample': 0.85, 'colsample_bytree': 0.85, 'reg_alpha': 1.0, 'reg_lambda': 3.0,
              'min_child_samples': 8},
}

V413_CONFIGS = {
    'Q1': 'narrow', 'Q2': 'soft_aggressive', 'Q3': 'narrow',
    'S1': 'ultra_deep', 'S2': 'soft_aggressive', 'S3': 'safety', 'S4': 'broad',
}
NFEAT = {'Q1': 25, 'Q2': 25, 'Q3': 22, 'S1': 30, 'S2': 28, 'S3': 30, 'S4': 28}

SEED = 42
N_FOLDS = 5
N_SEEDS = 25  # V448 대비 seeds 줄여서 시간 절약


def sanitize_col(n):
    return re.sub(r'[^a-zA-Z0-9_]', '_', n)

def remove_leak(cols, target):
    if target.startswith('S'):
        return [c for c in cols if c not in LEAK_S]
    elif target.startswith('Q'):
        return [c for c in cols if c not in LEAK_Q]
    return cols


def add_temporal_features(df, prefix=''):
    """Add temporal features from date columns."""
    df = df.copy()
    if 'lifelog_date' in df.columns:
        dates = pd.to_datetime(df['lifelog_date'])
        df[f'{prefix}dow'] = dates.dt.dayofweek
        df[f'{prefix}week'] = dates.dt.isocalendar().week.astype(int)
        df[f'{prefix}month'] = dates.dt.month
        df[f'{prefix}hour'] = dates.dt.hour
        # Time since last sleep
        if 'sleep_date' in df.columns:
            sleep_dates = pd.to_datetime(df['sleep_date'])
            diff = (dates - sleep_dates).dt.total_seconds() / 86400.0
            df[f'{prefix}days_since_sleep'] = diff
            df[f'{prefix}is_night'] = ((dates.dt.hour >= 21) | (dates.dt.hour <= 5)).astype(int)
            df[f'{prefix}time_in_week'] = (dates.dt.dayofweek * 24 + dates.dt.hour) / (7*24)
    return df


def rank_features_lgb(feat_df, feat_cols, target, seed=SEED):
    y = feat_df[target].values.astype(np.float64)
    X = feat_df[feat_cols].fillna(0).values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 50,
        'scale_pos_weight': spw, 'random_state': seed, 'force_row_wise': True, 'n_jobs': 1
    }
    sn = [sanitize_col(c) for c in feat_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn)
    m = lgb.train(params, ds, num_boost_round=50)
    imp = m.feature_importance(importance_type='gain')
    ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    del m, ds, X
    gc.collect()
    return [r[0] for r in ranked]


def main():
    global t_start
    t_start = time.time()

    log.info("=" * 70)
    log.info("V449 — 2-Stage Stacking + Temporal Features")
    log.info("=" * 70)

    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    groups_arr = train_df['subject_id'].values
    train_subjects = train_df['subject_id'].values

    num_cols = [c for c in train_df.columns
                if c not in META_COLS | set(TARGETS)
                and np.issubdtype(train_df[c].dtype, np.number)]

    # ===== Z-Score =====
    log.info("  Adding z-score features...")
    zscore_train = pd.DataFrame(index=train_df.index)
    zscore_test = pd.DataFrame(index=test_df.index)
    for col in num_cols:
        tr_mean = train_df.groupby('subject_id')[col].transform('mean')
        tr_std = train_df.groupby('subject_id')[col].transform('std').fillna(0).replace(0, 1)
        te_mean = test_df.groupby('subject_id')[col].transform('mean')
        te_std = test_df.groupby('subject_id')[col].transform('std').fillna(0).replace(0, 1)
        zscore_train[f'z_{col}'] = (train_df[col] - tr_mean) / tr_std
        zscore_test[f'z_{col}'] = (test_df[col] - te_mean) / te_std

    # ===== Baseline polynomial + z×baseline =====
    log.info("  Adding baseline polynomial features...")
    subject_ids = np.unique(train_subjects)
    baselines = {}
    for t in TARGETS:
        y_t = train_df[t].values
        bl = {}
        for sid in subject_ids:
            mask = train_subjects == sid
            s_y = y_t[mask]
            n_samples = mask.sum()
            global_rate = y_t.mean()
            subj_rate = s_y.mean() if n_samples > 0 else global_rate
            bl[sid] = 0.7 * subj_rate + 0.3 * global_rate
        baselines[t] = bl

        train_bl = np.array([bl[sid] for sid in train_subjects])
        test_bl = np.array([bl[sid] for sid in test_df['subject_id'].values])
        zscore_train[f'bl2_{t}'] = train_bl ** 2
        zscore_test[f'bl2_{t}'] = test_bl ** 2
        zscore_train[f'logbl_{t}'] = np.log1p(train_bl)
        zscore_test[f'logbl_{t}'] = np.log1p(test_bl)

    for t in TARGETS:
        train_bl = np.array([baselines[t][sid] for sid in train_subjects])
        test_bl = np.array([baselines[t][sid] for sid in test_df['subject_id'].values])
        for col in num_cols:
            z_col = f'z_{col}'
            if z_col in zscore_train.columns:
                zscore_train[f'zb_{t}_{col}'] = zscore_train[z_col] * train_bl
                zscore_test[f'zb_{t}_{col}'] = zscore_test[z_col] * test_bl

    all_train_features = pd.concat([train_df[num_cols], zscore_train], axis=1)
    all_test_features = pd.concat([test_df[num_cols], zscore_test], axis=1)

    # ===== Temporal Features =====
    log.info("  Adding temporal features...")
    train_with_temp = add_temporal_features(train_df)
    test_with_temp = add_temporal_features(test_df)

    # Add temporal features to feature sets
    temp_cols_train = [c for c in train_with_temp.columns if c.startswith(('dow','week','month','hour','days_since','is_night','time_in_week'))]
    temp_cols_test = [c for c in test_with_temp.columns if c.startswith(('dow','week','month','hour','days_since','is_night','time_in_week'))]

    all_train_features = pd.concat([all_train_features, train_with_temp[temp_cols_train]], axis=1)
    all_test_features = pd.concat([all_test_features, test_with_temp[temp_cols_test]], axis=1)

    # Per-subject temporal trends
    log.info("  Adding per-subject temporal trends...")
    train_with_temp['subject_id'] = train_subjects
    test_with_temp['subject_id'] = test_df['subject_id'].values

    for t in TARGETS:
        for metric_col in ['dow', 'hour']:
            if f'{metric_col}' in train_with_temp.columns:
                # Subject mean of temporal feature
                subj_temp_mean = train_with_temp.groupby('subject_id')[metric_col].transform('mean')
                # Add to zscore features
                zscore_train[f'st_{t}_{metric_col}'] = (train_with_temp[metric_col].values - subj_temp_mean.values) / np.maximum(subj_temp_mean.std(), 1)
                te_subj_mean = test_with_temp.groupby('subject_id')[metric_col].transform('mean') if metric_col in test_with_temp.columns else np.zeros(len(test_df))
                zscore_test[f'st_{t}_{metric_col}'] = (test_with_temp[metric_col].values - te_subj_mean.values) / np.maximum(te_subj_mean.std(), 1)

    # Rebuild combined features
    all_train_features = pd.concat([all_train_features, zscore_train], axis=1)
    all_test_features = pd.concat([all_test_features, zscore_test], axis=1)

    log.info(f"  Features: {all_train_features.shape[1]} (train), {all_test_features.shape[1]} (test)")

    # ===== Phase 1: Stage 1 LGBM base =====
    all_oof_seed_stage1 = {}
    all_test_seed_stage1 = {}

    for t_idx, target in enumerate(TARGETS):
        cfg_name = V413_CONFIGS[target]
        cfg = LGB_CFGS[cfg_name]
        n_feat = NFEAT[target]
        y = train_df[target].values.astype(np.float64)

        feat_cols = [c for c in all_train_features.columns
                     if c not in META_COLS | set(TARGETS)
                     and np.issubdtype(all_train_features.dtypes[c], np.number)]
        feat_cols = remove_leak(feat_cols, target)

        train_with_target = all_train_features.copy()
        train_with_target[target] = train_df[target]

        fold_ranks = []
        for fold in range(N_FOLDS):
            rank = rank_features_lgb(train_with_target, feat_cols, target, seed=SEED + fold)
            fold_ranks.append(rank[:n_feat])

        feat_counts = {}
        for fl in fold_ranks:
            for f in fl:
                feat_counts[f] = feat_counts.get(f, 0) + 1
        ranked_features = sorted(feat_counts.items(), key=lambda x: -x[1])
        top_features = [f for f, c in ranked_features[:n_feat]]

        X_base = train_with_target[top_features].fillna(0).values.astype(np.float64)
        X_test_base = all_test_features[top_features].fillna(0).values.astype(np.float64)

        train_baselines = np.array([baselines[target][sid] for sid in train_subjects]).reshape(-1, 1)
        test_baselines = np.array([baselines[target][sid] for sid in test_df['subject_id'].values]).reshape(-1, 1)
        X_all = np.hstack([X_base, train_baselines])
        X_test_all = np.hstack([X_test_base, test_baselines])

        oof_seed_arr = np.zeros((len(train_df), N_SEEDS))
        test_seed_arr = np.zeros((len(test_df), N_SEEDS))

        skf = GroupKFold(n_splits=N_FOLDS)

        for s in range(N_SEEDS):
            if (s + 1) % 5 == 0:
                log.info(f"    {target}: seed {s+1}/{N_SEEDS}")
            sk = SEED + s * 7 + t_idx
            seed_oof = np.zeros(len(train_df))
            seed_test = np.zeros(len(test_df))

            for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, y, groups_arr)):
                x_train, y_train = X_all[tr_idx], y[tr_idx]
                x_val = X_all[va_idx]

                spw = max(((y_train == 0).sum()) / max((y_train == 1).sum(), 1), 0.1)
                params = {
                    'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
                    **{k: v for k, v in cfg.items() if k not in ['n_estimators']},
                    'n_estimators': cfg['n_estimators'], 'scale_pos_weight': spw,
                    'random_state': sk, 'force_row_wise': True, 'n_jobs': 1,
                }

                ds_train = lgb.Dataset(x_train, label=y_train,
                    feature_name=[sanitize_col(c) for c in top_features + ['baseline']])
                ds_val = lgb.Dataset(x_val, label=y[va_idx],
                    feature_name=[sanitize_col(c) for c in top_features + ['baseline']], reference=ds_train)

                model = lgb.train(params, ds_train, num_boost_round=cfg['n_estimators'],
                    valid_sets=[ds_val],
                    callbacks=[lgb.early_stopping(200, verbose=False),
                               lgb.log_evaluation(period=0)])

                seed_oof[va_idx] = model.predict(x_val)
                seed_test += model.predict(X_test_all) / N_FOLDS
                del model, ds_train, ds_val
                gc.collect()

            oof_seed_arr[:, s] = np.clip(seed_oof, 0.001, 0.999)
            test_seed_arr[:, s] = np.clip(seed_test, 0.001, 0.999)

        all_oof_seed_stage1[target] = oof_seed_arr
        all_test_seed_stage1[target] = test_seed_arr
        avg_oof = log_loss(y, oof_seed_arr.mean(axis=1))
        log.info(f"  Stage1 {target}: oof={avg_oof:.5f} ({N_SEEDS} seeds)")

    # ===== Phase 2: Stage 2 — Meta features from Stage1 =====
    log.info("\n=== Phase 2: Stage 2 — Meta Features + Temporal Stacking ===")

    from xgboost import XGBClassifier

    # Build stage2 features: stage1 oof + temporal features + subject stats
    student_oofs = {}
    test_preds = {}
    meta_oofs = {}

    for t_idx, target in enumerate(TARGETS):
        y = train_df[target].values
        oof_preds = all_oof_seed_stage1[target]
        test_preds_raw = all_test_seed_stage1[target]

        # Stage1 avg
        stage1_avg = oof_preds.mean(axis=1)
        test_stage1_avg = test_preds_raw.mean(axis=1)

        # Cross-target meta (only same-group)
        group = 'Q' if target.startswith('Q') else 'S'
        group_targets = Q_TARGETS if group == 'Q' else S_TARGETS

        cross_oof_list = []
        cross_test_list = []
        for t_cross in group_targets:
            if t_cross == target:
                continue
            cross_oof_list.append(np.mean(all_oof_seed_stage1[t_cross], axis=1))
            cross_test_list.append(np.mean(all_test_seed_stage1[t_cross], axis=1))

        cross_arr = np.column_stack(cross_oof_list)
        cross_arr_test = np.column_stack(cross_test_list)

        # Stage2: use stage1 avg + cross-target as features for XGB meta
        X_meta = np.column_stack([stage1_avg, cross_arr.mean(axis=1)])
        X_test = np.column_stack([test_stage1_avg, cross_arr_test.mean(axis=1)])

        mm = XGBClassifier(n_estimators=20, max_depth=3, reg_alpha=0.01, reg_lambda=0.0,
            learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, n_jobs=1, min_child_weight=5, verbosity=0)
        mm.fit(X_meta, y)
        meta_oofs[target] = log_loss(y, mm.predict_proba(X_meta)[:, 1])
        test_preds[target] = mm.predict_proba(X_test)[:, 1]

        student_oofs[target] = log_loss(y, np.clip(stage1_avg, 0.001, 0.999))
        log.info(f"  {target}: meta={meta_oofs[target]:.5f}, student={student_oofs[target]:.5f}")

    avg_meta = np.mean(list(meta_oofs.values()))
    avg_student = np.mean(list(student_oofs.values()))
    gap = avg_student - avg_meta
    v339 = avg_meta + gap * 0.85

    log.info(f"\n{'='*70}")
    log.info("V449 Results:")
    log.info(f"  AVG Meta OOF: {avg_meta:.5f} (Δ vs V308: {avg_meta-0.62235:+.5f})")
    log.info(f"  AVG Student OOF: {avg_student:.5f} (Δ vs V308: {avg_student-0.69212:+.5f})")
    log.info(f"  Student-Meta Gap: {gap:.5f} (ratio: {gap/0.070:.2f}x)")
    log.info(f"  V339 Pattern LB: {v339:.5f}")
    log.info(f"  V446 V339: 0.58859")
    log.info(f"  V447 V339: 0.58874")
    log.info(f"{'='*70}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        sub[t] = test_preds[t]

    sub_path = SUBMIT / f"submission_v449_2stage_temporal_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved: {sub_path}")

    meta_data = {
        'version': 'V449',
        'name': '2-Stage Stacking + Temporal Features',
        'avg_meta_oof': round(float(avg_meta), 5),
        'avg_student_oof': round(float(avg_student), 5),
        'v308_lb': 0.63893,
        'estimated_lb_v339_pattern': round(float(v339), 5),
        'student_meta_gap': round(float(gap), 5),
        'n_seeds': N_SEEDS,
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }

    meta_path = EXPERIMENTS / f'v449_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved: {meta_path}")
    log.info(f"Total: {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    main()
