"""
V444 — Z-Score + Baseline Poly + Domain-Specific Deep Features + Multi-Task Meta

Hypothesis: V443의 z-score+baseline poly가 student 0.604를 만들었음.
V444는:
1. Domain-specific interaction: light×hr, hr×pedo cross-domain features
2. Baseline polynomial × z-score (zb×z interaction)
3. Per-subject time features (day of week, week number, day index)
4. Multi-task meta: target group meta features (Q-group meta avg, S-group meta avg)
5. Target-group-specific meta regularization (Q targets: stronger reg)
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
V443_NFEAT = {'Q1': 25, 'Q2': 25, 'Q3': 22, 'S1': 30, 'S2': 28, 'S3': 30, 'S4': 28}

SEED = 42
N_FOLDS = 5
N_SEEDS = 15


def sanitize_col(n):
    return re.sub(r'[^a-zA-Z0-9_]', '_', n)

def remove_leak(cols, target):
    if target.startswith('S'):
        return [c for c in cols if c not in LEAK_S]
    elif target.startswith('Q'):
        return [c for c in cols if c not in LEAK_Q]
    return cols


def rank_features(feat_df, feat_cols, target, seed=SEED):
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
    log.info("V444 — Z-Score + Baseline Poly + Cross-Domain + Multi-Task Meta")
    log.info("=" * 70)

    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    groups_arr = train_df['subject_id'].values
    train_subjects = train_df['subject_id'].values

    # ===== Z-Score Features =====
    log.info("  Adding z-score features...")
    num_cols = [c for c in train_df.columns
                if c not in META_COLS | set(TARGETS)
                and np.issubdtype(train_df[c].dtype, np.number)]

    zscore_train = pd.DataFrame(index=train_df.index)
    zscore_test = pd.DataFrame(index=test_df.index)
    for col in num_cols:
        tr_mean = train_df.groupby('subject_id')[col].transform('mean')
        tr_std = train_df.groupby('subject_id')[col].transform('std').fillna(0).replace(0, 1)
        te_mean = test_df.groupby('subject_id')[col].transform('mean')
        te_std = test_df.groupby('subject_id')[col].transform('std').fillna(0).replace(0, 1)
        zscore_train[f'z_{col}'] = (train_df[col] - tr_mean) / tr_std
        zscore_test[f'z_{col}'] = (test_df[col] - te_mean) / te_std

    # Baseline features
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

    # Z × baseline interactions
    log.info("  Adding z×baseline interactions...")
    for t in TARGETS:
        train_bl = np.array([baselines[t][sid] for sid in train_subjects])
        test_bl = np.array([baselines[t][sid] for sid in test_df['subject_id'].values])
        for col in num_cols:
            z_col = f'z_{col}'
            if z_col in zscore_train.columns:
                zscore_train[f'zb_{t}_{col}'] = zscore_train[z_col] * train_bl
                zscore_test[f'zb_{t}_{col}'] = zscore_test[z_col] * test_bl

    # ===== Cross-domain interactions =====
    log.info("  Adding cross-domain interactions...")
    domains = {
        'light': ['wLight_w_light_mean', 'wLight_w_light_std', 'wLight_w_light_min', 'wLight_w_light_max'],
        'hr': ['wHr_hr_mean', 'wHr_hr_std', 'wHr_hr_min', 'wHr_hr_max', 'wHr_hr_median'],
        'pedo': ['wPedo_pedo_step_mean', 'wPedo_pedo_step_sum', 'wPedo_pedo_distance_mean',
                 'wPedo_pedo_speed_mean', 'wPedo_pedo_burned_calories_mean'],
    }
    domain_pairs = [('light', 'hr'), ('light', 'pedo'), ('hr', 'pedo')]

    for d1, d2 in domain_pairs:
        feats1 = [f for f in domains[d1] if f in train_df.columns]
        feats2 = [f for f in domains[d2] if f in train_df.columns]
        for f1 in feats1:
            for f2 in feats2:
                z1 = f'z_{f1}'
                z2 = f'z_{f2}'
                if z1 in zscore_train.columns and z2 in zscore_train.columns:
                    zscore_train[f'cross_z_{d1}_{d2}_{f1[:10]}_{f2[:10]}'] = zscore_train[z1] * zscore_train[z2]
                    zscore_test[f'cross_z_{d1}_{d2}_{f1[:10]}_{f2[:10]}'] = zscore_test[z1] * zscore_test[z2]

    # ===== Time features =====
    log.info("  Adding time features...")
    if 'lifelog_date' in train_df.columns:
        try:
            dates = pd.to_datetime(train_df['lifelog_date'])
            test_dates = pd.to_datetime(test_df['lifelog_date'])
            zscore_train['dow'] = dates.dt.dayofweek
            zscore_test['dow'] = test_dates.dt.dayofweek
            zscore_train['week'] = dates.dt.isocalendar().week.astype(int)
            zscore_test['week'] = test_dates.dt.isocalendar().week.astype(int)
        except:
            pass

    all_train_features = pd.concat([train_df[num_cols], zscore_train], axis=1)
    all_test_features = pd.concat([test_df[num_cols], zscore_test], axis=1)

    log.info(f"  Features: {all_train_features.shape[1]} (train), {all_test_features.shape[1]} (test)")

    # ===== Phase 1: LGBM base =====
    all_oof_seed = {}
    all_test_seed = {}

    for t_idx, target in enumerate(TARGETS):
        cfg_name = V413_CONFIGS[target]
        cfg = LGB_CFGS[cfg_name]
        n_feat = V443_NFEAT[target]
        y = train_df[target].values.astype(np.float64)

        feat_cols = [c for c in all_train_features.columns
                     if c not in META_COLS | set(TARGETS)
                     and np.issubdtype(all_train_features[c].dtype, np.number)]
        feat_cols = remove_leak(feat_cols, target)

        train_with_target = all_train_features.copy()
        train_with_target[target] = train_df[target]
        test_no_target = all_test_features.copy()

        fold_ranks = []
        for fold in range(N_FOLDS):
            rank = rank_features(train_with_target, feat_cols, target, seed=SEED + fold)
            fold_ranks.append(rank[:n_feat])

        feat_counts = {}
        for fl in fold_ranks:
            for f in fl:
                feat_counts[f] = feat_counts.get(f, 0) + 1
        ranked_features = sorted(feat_counts.items(), key=lambda x: -x[1])
        top_features = [f for f, c in ranked_features[:n_feat]]

        X_base = train_with_target[top_features].fillna(0).values.astype(np.float64)
        X_test_base = test_no_target[top_features].fillna(0).values.astype(np.float64)

        train_baselines = np.array([baselines[target][sid] for sid in train_subjects]).reshape(-1, 1)
        test_baselines = np.array([baselines[target][sid] for sid in test_df['subject_id'].values]).reshape(-1, 1)

        X_all = np.hstack([X_base, train_baselines])
        X_test_all = np.hstack([X_test_base, test_baselines])

        oof_seed_arr = np.zeros((len(train_df), N_SEEDS))
        test_seed_arr = np.zeros((len(test_df), N_SEEDS))

        for s in range(N_SEEDS):
            sk = SEED + s * 7 + t_idx
            skf = GroupKFold(n_splits=N_FOLDS)
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

        all_oof_seed[target] = oof_seed_arr
        all_test_seed[target] = test_seed_arr
        avg_oof = log_loss(y, oof_seed_arr.mean(axis=1))
        log.info(f"{target}: cross_domain_oof={avg_oof:.5f}")

    # ===== Phase 2: Multi-Task Meta =====
    log.info("\n=== Phase 2: Multi-Task Meta ===")

    from xgboost import XGBClassifier

    # Target group meta averages
    Q_TARGETS = ['Q1', 'Q2', 'Q3']
    S_TARGETS = ['S1', 'S2', 'S3', 'S4']

    student_oofs = {}
    test_preds = {}
    meta_oofs = {}

    for t_idx, target in enumerate(TARGETS):
        y = train_df[target].values
        oof_preds = all_oof_seed[target]
        test_preds_raw = all_test_seed[target]

        means = np.mean(oof_preds, axis=1, keepdims=True)
        stds = np.std(oof_preds, axis=1, keepdims=True)
        mins = np.min(oof_preds, axis=1, keepdims=True)
        maxs = np.max(oof_preds, axis=1, keepdims=True)

        student_oofs[target] = log_loss(y, oof_preds.mean(axis=1))

        # Weighted cross-target + target-group meta avg
        cross_oof_list = []
        cross_test_list = []
        for t_cross in TARGETS:
            if t_cross == target:
                continue
            w = 1.0 if t_cross.startswith('Q') == target.startswith('Q') else 0.5
            cross_oof_list.append(np.mean(all_oof_seed[t_cross], axis=1) * w)
            cross_test_list.append(np.mean(all_test_seed[t_cross], axis=1) * w)

        # Add target-group meta average
        if target.startswith('Q'):
            group_meta = Q_TARGETS
        else:
            group_meta = S_TARGETS

        group_oofs_list = []
        group_tests_list = []
        for t in group_meta:
            if t != target:
                g_oof = np.mean(all_oof_seed[t], axis=1)
                g_test = np.mean(all_test_seed[t], axis=1)
                group_oofs_list.append(g_oof)
                group_tests_list.append(g_test)
        if len(group_oofs_list) > 0:
            group_oof = np.mean(group_oofs_list, axis=0)  # (N,)
            group_test = np.mean(group_tests_list, axis=0)  # (N_test,)
        else:
            group_oof = np.zeros(len(y))
            group_test = np.zeros(len(test_df))

        # Stack all into (N, n_cross) and (N_test, n_cross)
        cross_arr_oof = np.column_stack(cross_oof_list + [group_oof])
        cross_arr_test = np.column_stack(cross_test_list + [group_test])

        X_meta = np.hstack([oof_preds, means, stds, mins, maxs, cross_arr_oof])
        X_test = np.hstack([test_preds_raw,
            np.mean(test_preds_raw, axis=1, keepdims=True),
            np.std(test_preds_raw, axis=1, keepdims=True),
            np.min(test_preds_raw, axis=1, keepdims=True),
            np.max(test_preds_raw, axis=1, keepdims=True),
            cross_arr_test])

        # Per-target regularization: Q targets stronger reg
        if target.startswith('Q'):
            reg_alpha = 0.1
            reg_lambda = 1.0
        else:
            reg_alpha = 0.01
            reg_lambda = 0.0

        mm = XGBClassifier(n_estimators=15, max_depth=3, reg_alpha=reg_alpha, reg_lambda=reg_lambda,
            gamma=0.0, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, n_jobs=1, min_child_weight=5, verbosity=0)
        mm.fit(X_meta, y)
        meta_oofs[target] = log_loss(y, mm.predict_proba(X_meta)[:, 1])
        test_preds[target] = mm.predict_proba(X_test)[:, 1]

        log.info(f"  {target}: meta={meta_oofs[target]:.5f}, student={student_oofs[target]:.5f}, "
                 f"alpha={reg_alpha}, lambda={reg_lambda}")

    avg_meta = np.mean(list(meta_oofs.values()))
    avg_student = np.mean(list(student_oofs.values()))
    gap = avg_student - avg_meta
    v339 = avg_meta + gap * 0.85

    log.info(f"\n{'='*70}")
    log.info("V444 Results:")
    log.info(f"  AVG Meta OOF: {avg_meta:.5f} (Δ vs V308: {avg_meta-0.62235:+.5f})")
    log.info(f"  AVG Student OOF: {avg_student:.5f} (Δ vs V308: {avg_student-0.69212:+.5f})")
    log.info(f"  Student-Meta Gap: {gap:.5f} (ratio: {gap/0.070:.2f}x)")
    log.info(f"  V339 Pattern LB: {v339:.5f}")
    log.info(f"  V443 V339: 0.59208 (for comparison)")
    log.info(f"{'='*70}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        sub[t] = test_preds[t]

    sub_path = SUBMIT / f"submission_v444_multitask_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved: {sub_path}")

    meta_data = {
        'version': 'V444',
        'name': 'Z-Score + Baseline Poly + Cross-Domain + Multi-Task Meta',
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

    meta_path = EXPERIMENTS / f'v444_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved: {meta_path}")
    log.info(f"Total: {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    main()
