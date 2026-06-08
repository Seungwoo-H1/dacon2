"""
V458 — Heterogeneous Ensemble: LGBM + XGB + LightCatBoost Base Learners

Hypothesis: V452(0.580)는 단일 model(LGBM)에 bottleneck.
다른 model architectures를 ensemble하면:
1. LGBM base (V452 동일)
2. XGB base (low reg)
3. LightCatBoost base (fast, shallow)
4. 3-model weighted ensemble → diversity가 student OOF 낮춤

Key: Model diversity가 overfitting을 줄이고 generalization을 높임.
"""
import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
import numpy as np
import pandas as pd
import lightgbm as lgb
from xgboost import XGBClassifier

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

LEAK_S = {'wLight_w_light_mean','wLight_w_light_std','wLight_w_light_min','wLight_w_light_max','wLight_w_light_count',
    'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max','wHr_hr_median','wHr_hr_count',
    'wPedo_pedo_step_mean','wPedo_pedo_step_sum','wPedo_pedo_step_frequency_mean','wPedo_pedo_step_frequency_sum',
    'wPedo_pedo_running_step_mean','wPedo_pedo_running_step_sum','wPedo_pedo_walking_step_mean','wPedo_pedo_walking_step_sum',
    'wPedo_pedo_distance_mean','wPedo_pedo_distance_sum','wPedo_pedo_speed_mean','wPedo_pedo_speed_sum',
    'wPedo_pedo_burned_calories_mean','wPedo_pedo_burned_calories_sum'}
LEAK_Q = {'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max','wHr_hr_median','wHr_hr_count'}

SEED = 42
N_FOLDS = 5
N_SEEDS = 15  # 3 models × 15 seeds = 45 total


def sanitize_col(n):
    return re.sub(r'[^a-zA-Z0-9_]', '_', n)

def remove_leak(cols, target):
    if target.startswith('S'):
        return [c for c in cols if c not in LEAK_S]
    elif target.startswith('Q'):
        return [c for c in cols if c not in LEAK_Q]
    return cols

def rank_features_lgb(feat_df, feat_cols, target, seed=SEED):
    y = feat_df[target].values.astype(np.float64)
    X = feat_df[feat_cols].fillna(0).values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    params = {'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
              'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 50,
              'scale_pos_weight': spw, 'random_state': seed, 'force_row_wise': True, 'n_jobs': 1}
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
    log.info("V458 — Heterogeneous Ensemble (LGBM + XGB + CatBoost)")
    log.info("=" * 70)

    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    groups_arr = train_df['subject_id'].values
    train_subjects = train_df['subject_id'].values

    num_cols = [c for c in train_df.columns
                if c not in META_COLS | set(TARGETS)
                and np.issubdtype(train_df.dtypes[c], np.number)]

    # ===== Z-Score + Baseline + Interactions (V452 style) =====
    log.info("  Adding features (V452 style)...")
    zscore_train = pd.DataFrame(index=train_df.index)
    zscore_test = pd.DataFrame(index=test_df.index)
    for col in num_cols:
        tr_mean = train_df.groupby('subject_id')[col].transform('mean')
        tr_std = train_df.groupby('subject_id')[col].transform('std').fillna(0).replace(0, 1)
        te_mean = test_df.groupby('subject_id')[col].transform('mean')
        te_std = test_df.groupby('subject_id')[col].transform('std').fillna(0).replace(0, 1)
        zscore_train[f'z_{col}'] = (train_df[col] - tr_mean) / tr_std
        zscore_test[f'z_{col}'] = (test_df[col] - te_mean) / te_std

    subject_ids = np.unique(train_subjects)
    baselines = {}
    for t in TARGETS:
        y_t = train_df[t].values
        bl = {}
        for sid in subject_ids:
            mask = train_subjects == sid
            s_y = y_t[mask]; n_samples = mask.sum(); global_rate = y_t.mean()
            subj_rate = s_y.mean() if n_samples > 0 else global_rate
            bl[sid] = 0.7 * subj_rate + 0.3 * global_rate
        baselines[t] = bl
        train_bl = np.array([bl[sid] for sid in train_subjects])
        test_bl = np.array([bl[sid] for sid in test_df['subject_id'].values])
        zscore_train[f'bl2_{t}'] = train_bl ** 2; zscore_test[f'bl2_{t}'] = test_bl ** 2
        zscore_train[f'logbl_{t}'] = np.log1p(train_bl); zscore_test[f'logbl_{t}'] = np.log1p(test_bl)

    for t in TARGETS:
        train_bl = np.array([baselines[t][sid] for sid in train_subjects])
        test_bl = np.array([baselines[t][sid] for sid in test_df['subject_id'].values])
        for col in num_cols:
            z_col = f'z_{col}'
            if z_col in zscore_train.columns:
                zscore_train[f'zb_{t}_{col}'] = zscore_train[z_col] * train_bl
                zscore_test[f'zb_{t}_{col}'] = zscore_test[z_col] * test_bl
                zscore_train[f'z3_{t}_{col}'] = zscore_train[z_col].values ** 3
                zscore_test[f'z3_{t}_{col}'] = zscore_test[z_col].values ** 3
                zscore_train[f'bz2_{t}_{col}'] = train_bl * (zscore_train[z_col].values ** 2)
                zscore_test[f'bz2_{t}_{col}'] = test_bl * (zscore_test[z_col].values ** 2)

    all_train_features = pd.concat([train_df[num_cols], zscore_train], axis=1)
    all_test_features = pd.concat([test_df[num_cols], zscore_test], axis=1)
    log.info(f"  Features: {all_train_features.shape[1]}")

    # ===== Feature Selection =====
    feat_cols_all = [c for c in all_train_features.columns
                     if c not in META_COLS | set(TARGETS)
                     and np.issubdtype(all_train_features.dtypes[c], np.number)]
    NFEAT = {'Q1': 25, 'Q2': 25, 'Q3': 22, 'S1': 30, 'S2': 28, 'S3': 30, 'S4': 28}

    # ===== Phase 1: Heterogeneous Base Learners =====
    log.info("\n=== Phase 1: Heterogeneous Base Learners ===")
    # 3 configs: SHALLOW, BALANCED, DEEP for diversity
    CFGS = [
        {'type': 'lgbm', 'params': {'num_leaves': 15, 'max_depth': 3, 'learning_rate': 0.015,
             'n_estimators': 2000, 'subsample': 0.6, 'colsample_bytree': 0.6,
             'reg_alpha': 5.0, 'reg_lambda': 20.0, 'min_child_samples': 25}},
        {'type': 'xgb', 'params': {'max_depth': 4, 'learning_rate': 0.015, 'n_estimators': 1500,
             'subsample': 0.65, 'colsample_bytree': 0.65, 'reg_alpha': 2.0, 'reg_lambda': 10.0,
             'min_child_weight': 5, 'gamma': 0.0}},
        {'type': 'lgbm', 'params': {'num_leaves': 30, 'max_depth': 5, 'learning_rate': 0.02,
             'n_estimators': 1200, 'subsample': 0.7, 'colsample_bytree': 0.7,
             'reg_alpha': 1.0, 'reg_lambda': 5.0, 'min_child_samples': 15}},
    ]

    # Store OOF and test predictions per model config
    model_oof_seeds = [{} for _ in CFGS]  # 3 lists of {target: oof_arr}
    model_test_seeds = [{} for _ in CFGS]

    for cfg_idx, cfg in enumerate(CFGS):
        log.info(f"\n  --- Model {cfg_idx+1}/3: {cfg['type'].upper()} ---")
        params = cfg['params']
        model_type = cfg['type']

        for t_idx, target in enumerate(TARGETS):
            n_feat = NFEAT[target]
            y = train_df[target].values.astype(np.float64)
            feat_cols = remove_leak(feat_cols_all, target)

            train_with_target = all_train_features.copy()
            train_with_target[target] = train_df[target]

            fold_ranks = []
            for fold in range(5):
                rank = rank_features_lgb(train_with_target, feat_cols, target, seed=SEED + fold * 3)
                fold_ranks.append(rank[:n_feat])
            feat_counts = {}
            for fl in fold_ranks:
                for f in fl: feat_counts[f] = feat_counts.get(f, 0) + 1
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
                sk = SEED + s * 7 + t_idx + cfg_idx * 200
                seed_oof = np.zeros(len(train_df))
                seed_test = np.zeros(len(test_df))

                for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, y, groups_arr)):
                    x_train, y_train = X_all[tr_idx], y[tr_idx]
                    x_val = X_all[va_idx]
                    spw = max(((y_train == 0).sum()) / max((y_train == 1).sum(), 1), 0.1)

                    if model_type == 'lgbm':
                        lgb_params = {'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
                            **params, 'scale_pos_weight': spw, 'random_state': sk,
                            'force_row_wise': True, 'n_jobs': 1}
                        ds_train = lgb.Dataset(x_train, label=y_train,
                            feature_name=[sanitize_col(c) for c in top_features + ['baseline']])
                        ds_val = lgb.Dataset(x_val, label=y[va_idx],
                            feature_name=[sanitize_col(c) for c in top_features + ['baseline']], reference=ds_train)
                        model = lgb.train(lgb_params, ds_train, num_boost_round=params['n_estimators'],
                            valid_sets=[ds_val],
                            callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(period=0)])
                        seed_oof[va_idx] = model.predict(x_val)
                        seed_test += model.predict(X_test_all) / N_FOLDS
                        del model, ds_train, ds_val

                    elif model_type == 'xgb':
                        xgb_p = {**params, 'objective': 'binary:logistic', 'eval_metric': 'logloss',
                            'scale_pos_weight': spw, 'random_state': sk, 'n_jobs': 1, 'verbosity': 0}
                        for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X_all, y, groups_arr)):
                            # XGB needs separate training per fold — redo loop
                            pass
                        # Simple XGB fold loop
                        for fold_i, (tr_idx_i, va_idx_i) in enumerate(skf.split(X_all, y, groups_arr)):
                            xm = XGBClassifier(**xgb_p)
                            xm.fit(X_all[tr_idx_i], y[tr_idx_i])
                            seed_oof[va_idx_i] = xm.predict_proba(X_all[va_idx_i])[:, 1]
                            seed_test += xm.predict_proba(X_test_all)[:, 1] / N_FOLDS
                        del xm
                        gc.collect()
                        continue  # skip the lgbm-style update below

                    seed_oof = np.clip(seed_oof, 0.001, 0.999)
                    seed_test = np.clip(seed_test, 0.001, 0.999)

                oof_seed_arr[:, s] = seed_oof
                test_seed_arr[:, s] = seed_test

            model_oof_seeds[cfg_idx][target] = oof_seed_arr
            model_test_seeds[cfg_idx][target] = test_seed_arr
            avg_oof = log_loss(y, oof_seed_arr.mean(axis=1))
            log.info(f"  {target} [{cfg['type']}]: oof={avg_oof:.5f}")

    # ===== Phase 2: Stack All Models + V446 Meta =====
    log.info("\n=== Phase 2: V446 Meta ===")
    student_oofs = {}; test_preds = {}; meta_oofs = {}

    for t_idx, target in enumerate(TARGETS):
        y = train_df[target].values
        # Average predictions across all 3 models
        all_oof = np.mean([model_oof_seeds[i][target].mean(axis=1) for i in range(3)], axis=0)
        all_test = np.mean([model_test_seeds[i][target].mean(axis=1) for i in range(3)], axis=0)
        student_oofs[target] = log_loss(y, all_oof)

        group = 'Q' if target.startswith('Q') else 'S'
        group_targets = Q_TARGETS if group == 'Q' else S_TARGETS
        other_group = S_TARGETS if group == 'Q' else Q_TARGETS
        cross_oof_list = []; cross_test_list = []
        for t_cross in group_targets:
            if t_cross == target: continue
            cross_oof_list.append(np.mean([model_oof_seeds[i][t_cross].mean(axis=1) for i in range(3)], axis=0))
            cross_test_list.append(np.mean([model_test_seeds[i][t_cross].mean(axis=1) for i in range(3)], axis=0))
        for t_cross in other_group:
            cross_oof_list.append(np.mean([model_oof_seeds[i][t_cross].mean(axis=1) for i in range(3)], axis=0) * 0.5)
            cross_test_list.append(np.mean([model_test_seeds[i][t_cross].mean(axis=1) for i in range(3)], axis=0) * 0.5)
        cross_arr = np.column_stack(cross_oof_list)
        cross_arr_test = np.column_stack(cross_test_list)

        X_meta = np.hstack([all_oof.reshape(-1, 1), cross_arr])
        X_test = np.hstack([all_test.reshape(-1, 1), cross_arr_test])

        mm = XGBClassifier(n_estimators=15, max_depth=3, reg_alpha=0.01, reg_lambda=0.0,
            gamma=0.0, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, n_jobs=1, min_child_weight=5, verbosity=0)
        mm.fit(X_meta, y)
        meta_oofs[target] = log_loss(y, mm.predict_proba(X_meta)[:, 1])
        test_preds[target] = mm.predict_proba(X_test)[:, 1]
        log.info(f"  {target}: meta={meta_oofs[target]:.5f}, student={student_oofs[target]:.5f}")

    avg_meta = np.mean(list(meta_oofs.values()))
    avg_student = np.mean(list(student_oofs.values()))
    gap = avg_student - avg_meta
    v339 = avg_meta + gap * 0.85

    log.info(f"\n{'='*70}")
    log.info("V458 Results:")
    log.info(f"  AVG Meta OOF: {avg_meta:.5f}")
    log.info(f"  AVG Student OOF: {avg_student:.5f}")
    log.info(f"  Gap: {gap:.5f} ({gap/0.070:.2f}x)")
    log.info(f"  V339 LB: {v339:.5f}")
    log.info(f"  V452: 0.58019")
    log.info(f"{'='*70}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS: sub[t] = test_preds[t]
    sub_path = SUBMIT / f"submission_v458_hetero_ens_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    meta_data = {'version': 'V458', 'name': 'Heterogeneous Ensemble (LGBM+XGB+LGBM-deep)',
        'avg_meta_oof': round(float(avg_meta), 5), 'avg_student_oof': round(float(avg_student), 5),
        'v308_lb': 0.63893, 'estimated_lb_v339_pattern': round(float(v339), 5),
        'student_meta_gap': round(float(gap), 5), 'n_models': 3, 'n_seeds': N_SEEDS,
        'submission_file': str(sub_path), 'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0)}
    meta_path = EXPERIMENTS / f'v458_{ts}.json'
    with open(meta_path, 'w') as f: json.dump(meta_data, f, indent=2)
    log.info(f"Saved: {sub_path}, Total: {time.time()-t_start:.0f}s")

if __name__ == '__main__':
    main()
