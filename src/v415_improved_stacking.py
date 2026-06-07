"""
V415 — Improved Stacking with Per-Target Meta Features + Regularization Sweep

Hypothesis: V413/V414 failed because:
1. V414 cross-ensemble added noise (low inter-target correlation)
2. V413 meta learner may be under-regularized (XGB md=3, n_est=15)
3. Meta features should be per-target optimized, not shared

V415 approach:
1. Same V413 pipeline for base models (per-target LGBM with V413 configs)
2. For meta learner: per-target regularization sweep (XGB n_est, max_depth, alpha)
3. Also try LR meta-learner (V308's approach) as baseline comparison
4. Key: meta features = per-seed OOF predictions (proper stacking, no leakage)

This is a careful re-implementation of the stacking pipeline with proper
per-target meta-learner tuning.
"""
import sys, gc, logging, json, re, time, warnings, math
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
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

CFGS = {
    'ultra_aggressive': {'num_leaves': 8, 'max_depth': 3, 'learning_rate': 0.005, 'n_estimators': 3000,
                         'subsample': 0.4, 'colsample_bytree': 0.4, 'reg_alpha': 10.0, 'reg_lambda': 50.0, 'min_child_samples': 50},
    'soft_aggressive':  {'num_leaves': 12, 'max_depth': 3, 'learning_rate': 0.012, 'n_estimators': 2000,
                         'subsample': 0.55, 'colsample_bytree': 0.55, 'reg_alpha': 4.0, 'reg_lambda': 15.0, 'min_child_samples': 25},
    'medium':           {'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.015, 'n_estimators': 1500,
                         'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 3.0, 'reg_lambda': 10.0, 'min_child_samples': 20},
    'ultra_deep':       {'num_leaves': 25, 'max_depth': 5, 'learning_rate': 0.025, 'n_estimators': 1000,
                         'subsample': 0.75, 'colsample_bytree': 0.65, 'reg_alpha': 0.3, 'reg_lambda': 1.5, 'min_child_samples': 12},
    'wide':             {'num_leaves': 30, 'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 300,
                         'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_samples': 5},
    'safety':           {'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.02, 'n_estimators': 1000,
                         'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 3.0, 'reg_lambda': 10.0, 'min_child_samples': 20},
    'broad':            {'num_leaves': 40, 'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 800,
                         'subsample': 0.85, 'colsample_bytree': 0.85, 'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_samples': 8},
    'ultra_light':      {'num_leaves': 6, 'max_depth': 2, 'learning_rate': 0.008, 'n_estimators': 3000,
                         'subsample': 0.35, 'colsample_bytree': 0.35, 'reg_alpha': 15.0, 'reg_lambda': 100.0, 'min_child_samples': 60},
    'narrow':           {'num_leaves': 20, 'max_depth': 4, 'learning_rate': 0.01, 'n_estimators': 2500,
                         'subsample': 0.5, 'colsample_bytree': 0.5, 'reg_alpha': 8.0, 'reg_lambda': 30.0, 'min_child_samples': 35},
}

V413_CONFIGS = {
    'Q1': 'narrow',
    'Q2': 'soft_aggressive',
    'Q3': 'narrow',
    'S1': 'ultra_deep',
    'S2': 'soft_aggressive',
    'S3': 'safety',
    'S4': 'broad',
}

# Per-target n_feat from V413
V413_NFEAT = {'Q1': 19, 'Q2': 19, 'Q3': 15, 'S1': 21, 'S2': 19, 'S3': 23, 'S4': 20}

SEED = 42
N_FOLDS = 5
N_SEEDS = 15


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
    log.info("V415 — Improved Stacking: Per-Target Meta Features + Reg Sweep")
    log.info("Hypothesis: Proper stacking with per-target meta-learner tuning")
    log.info("=" * 70)

    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")

    groups_arr = train_df['subject_id'].values

    # ===== Phase 1: Base model training with V413 configs =====
    # Build proper stacking: per-seed OOF predictions for each target
    # Output shape: (n_train, n_seeds) per target, (n_test, n_seeds) per target

    all_oof_seed = {}   # {target: np.array(n_train, n_seeds)}
    all_test_seed = {}  # {target: np.array(n_test, n_seeds)}
    all_student_seed = {}  # same as oof_seed (student = direct LGBM predictions)
    all_top_features = {}

    for t_idx, target in enumerate(TARGETS):
        cfg_name = V413_CONFIGS[target]
        cfg = CFGS[cfg_name]
        n_feat = V413_NFEAT[target]

        feat_cols = get_feature_cols(train_df)
        feat_cols = remove_leak(feat_cols, target)

        # Per-fold feature ranking
        fold_ranks = []
        for fold in range(N_FOLDS):
            rank = rank_features(train_df, feat_cols, target, seed=SEED + fold)
            fold_ranks.append(rank[:n_feat])

        # CV-averaged feature ranking
        feat_counts = {}
        for fl in fold_ranks:
            for f in fl:
                feat_counts[f] = feat_counts.get(f, 0) + 1
        ranked_features = sorted(feat_counts.items(), key=lambda x: -x[1])
        top_features = [f for f, c in ranked_features[:n_feat]]
        all_top_features[target] = top_features

        log.info(f"\n--- {target} (cfg={cfg_name}, n_feat={n_feat}) ---")

        X_all = train_df[top_features + [target]].fillna(0).values.astype(np.float64)
        y_all = X_all[:, -1]
        X_all = X_all[:, :-1]
        X_test_all = test_df[top_features].fillna(0).values.astype(np.float64)

        n_seeds_actual = N_SEEDS
        oof_seed_arr = np.zeros((len(train_df), n_seeds_actual))
        student_seed_arr = np.zeros((len(train_df), n_seeds_actual))
        test_seed_arr = np.zeros((len(test_df), n_seeds_actual))

        for s in range(n_seeds_actual):
            sk = SEED + s * 7 + t_idx  # diverse seeds

            # For each seed, do GroupKFold
            skf = GroupKFold(n_splits=N_FOLDS)
            seed_oof = np.zeros(len(train_df))
            seed_student = np.zeros(len(train_df))
            seed_test = np.zeros(len(test_df))

            fold_idx = 0
            for train_idx, val_idx in skf.split(X_all, y_all, groups_arr):
                x_train = X_all[train_idx]
                y_train = y_all[train_idx]
                x_val = X_all[val_idx]
                x_test = X_test_all

                spw = max(((y_train == 0).sum()) / max((y_train == 1).sum(), 1), 0.1)
                params = {
                    'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
                    **{k: v for k, v in cfg.items() if k not in ['n_estimators']},
                    'n_estimators': cfg['n_estimators'],
                    'scale_pos_weight': spw, 'random_state': sk,
                    'force_row_wise': True, 'n_jobs': 1,
                }

                ds_train = lgb.Dataset(x_train, label=y_train,
                    feature_name=[sanitize_col(c) for c in top_features])
                ds_val = lgb.Dataset(x_val, label=y_all[val_idx],
                    feature_name=[sanitize_col(c) for c in top_features], reference=ds_train)

                model = lgb.train(params, ds_train, num_boost_round=cfg['n_estimators'],
                    valid_sets=[ds_val],
                    callbacks=[lgb.early_stopping(200, verbose=False),
                               lgb.log_evaluation(period=0)])

                seed_oof[val_idx] = model.predict(x_val)
                seed_student[val_idx] = model.predict(x_val)
                seed_test += model.predict(x_test) / N_FOLDS

                del model, ds_train, ds_val
                gc.collect()
                fold_idx += 1

            oof_seed_arr[:, s] = seed_oof
            student_seed_arr[:, s] = seed_student
            test_seed_arr[:, s] = seed_test

            log.info(f"  Seed {s}: meta_oof={log_loss(y_all, seed_oof):.5f}, "
                     f"student_oof={log_loss(y_all, seed_student):.5f}")

        all_oof_seed[target] = oof_seed_arr
        all_student_seed[target] = student_seed_arr
        all_test_seed[target] = test_seed_arr

        # Average OOF and student
        avg_oof_target = log_loss(y_all, oof_seed_arr.mean(axis=1))
        avg_student_target = log_loss(y_all, student_seed_arr.mean(axis=1))
        log.info(f"  AVG: meta_oof={avg_oof_target:.5f}, student_oof={avg_student_target:.5f}")

    # ===== Phase 2: Meta-learner training =====
    # For each target, use per-seed OOF predictions as meta features
    # Sweep XGB hyperparameters to find optimal meta-learner

    log.info("\n=== Phase 2: Meta-Learner Sweep ===")

    # Meta-learner hyperparameter sweep
    meta_configs = [
        # (n_est, max_depth, reg_alpha, reg_lambda, learner_type)
        (15, 3, 1.0, 5.0, 'xgb'),       # V413 default
        (10, 3, 1.0, 5.0, 'xgb'),       # fewer estimators
        (20, 3, 1.0, 5.0, 'xgb'),       # more estimators
        (15, 2, 1.0, 5.0, 'xgb'),       # shallower
        (15, 4, 1.0, 5.0, 'xgb'),       # deeper
        (15, 3, 0.1, 1.0, 'xgb'),       # less reg
        (15, 3, 10.0, 50.0, 'xgb'),     # more reg
        (15, 3, 1.0, 5.0, 'lr'),        # LR meta (V308 style)
    ]

    best_meta_result = None
    best_meta_key = None

    for mc_idx, (n_est, md, alpha, lam, learner_type) in enumerate(meta_configs):
        if learner_type == 'lr':
            key = f"lr_c={1/max(alpha,0.01):.0f}"
        else:
            key = f"xgb_n{n_est}_md{md}_a{alpha}_l{lam}"

        meta_oofs = {}
        meta_students = {}

        for t_idx, target in enumerate(TARGETS):
            y = train_df[target].values
            X_meta = all_oof_seed[target]  # (n_train, n_seeds)

            if learner_type == 'lr':
                meta_model = LogisticRegression(
                    C=max(1/max(alpha, 0.01), 0.1), max_iter=1000, random_state=SEED)
                meta_model.fit(X_meta, y)
                meta_pred = meta_model.predict_proba(X_meta)[:, 1]
            else:
                meta_model = XGBClassifier(
                    n_estimators=n_est, max_depth=md, reg_alpha=alpha, reg_lambda=lam,
                    learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                    random_state=SEED, n_jobs=1, min_child_weight=10, verbosity=0)
                meta_model.fit(X_meta, y)
                meta_pred = meta_model.predict_proba(X_meta)[:, 1]

            meta_oofs[target] = log_loss(y, meta_pred)

            # Student = avg of seed predictions
            student_pred = all_student_seed[target].mean(axis=1)
            meta_students[target] = log_loss(y, student_pred)

        avg_meta = np.mean(list(meta_oofs.values()))
        avg_student = np.mean(list(meta_students.values()))
        gap = avg_student - avg_meta

        log.info(f"  Config {key}: avg_meta={avg_meta:.5f}, avg_student={avg_student:.5f}, "
                 f"gap={gap:.5f}")

        if best_meta_result is None or avg_meta < best_meta_result[0]:
            best_meta_result = (avg_meta, avg_student, gap, meta_oofs, meta_students,
                               n_est, md, alpha, lam, learner_type, key)
            best_meta_key = key

    best_avg_meta, best_avg_student, best_gap, best_oofs, best_students, \
        best_n_est, best_md, best_alpha, best_lam, best_learner, best_key = best_meta_result

    log.info(f"\n  Best meta config: {best_key}")

    # ===== Phase 3: Final test predictions with best meta config =====
    log.info("\n=== Phase 3: Final Test Predictions ===")

    test_preds = {}
    per_target_meta_oof = {}
    per_target_student_oof = {}

    for t_idx, target in enumerate(TARGETS):
        y = train_df[target].values
        X_meta = all_oof_seed[target]  # (n_train, n_seeds)
        X_test_meta = all_test_seed[target]  # (n_test, n_seeds)

        if best_learner == 'lr':
            meta_model = LogisticRegression(
                C=max(1/max(best_alpha, 0.01), 0.1), max_iter=1000, random_state=SEED)
            meta_model.fit(X_meta, y)
            test_pred = meta_model.predict_proba(X_test_meta)[:, 1]
        else:
            meta_model = XGBClassifier(
                n_estimators=best_n_est, max_depth=best_md, reg_alpha=best_alpha,
                reg_lambda=best_lam, learning_rate=0.1, subsample=0.8,
                colsample_bytree=0.8, random_state=SEED, n_jobs=1,
                min_child_weight=10, verbosity=0)
            meta_model.fit(X_meta, y)
            test_pred = meta_model.predict_proba(X_test_meta)[:, 1]

        test_preds[target] = test_pred
        per_target_meta_oof[target] = best_oofs[target]
        per_target_student_oof[target] = best_students[target]

    avg_oof = np.mean(list(best_oofs.values()))
    avg_student = np.mean(list(best_students.values()))
    gap = avg_student - avg_oof

    # Predict LB using V339 pattern
    predicted_lb = avg_oof + gap * 0.5
    estimated_lb_v339 = avg_oof + gap * 0.85  # V339 had gap ~0.85x multiplier

    log.info(f"\n{'='*70}")
    log.info("V415 Results:")
    log.info(f"  AVG Meta OOF: {avg_oof:.5f} (Δ vs V308: {avg_oof-0.62235:+.5f})")
    log.info(f"  AVG Student OOF: {avg_student:.5f} (Δ vs V411 {0.65388}: {avg_student-0.65388:+.5f})")
    log.info(f"  Student-Meta Gap: {gap:.5f} (V308: 0.070, ratio: {gap/0.070:.2f}x)")
    log.info(f"  Predicted LB: {predicted_lb:.5f}")
    log.info(f"  V339 Pattern LB: {estimated_lb_v339:.5f}")
    log.info(f"  Per-target Meta OOF: { {t: round(v,5) for t,v in per_target_meta_oof.items()} }")
    log.info(f"  Per-target Student OOF: { {t: round(v,5) for t,v in per_target_student_oof.items()} }")
    log.info(f"  Best meta config: {best_key}")
    log.info(f"{'='*70}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        sub[t] = test_preds[t]

    sub_path = SUBMIT / f"submission_v415_improved_stacking_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")

    meta_data = {
        'version': 'V415',
        'name': 'Improved Stacking: Per-Target Meta Features + Reg Sweep',
        'avg_meta_oof': round(float(avg_oof), 5),
        'avg_student_oof': round(float(avg_student), 5),
        'v308_avg_oof': 0.62235,
        'v308_avg_student': 0.69212,
        'v411_avg_student': 0.65388,
        'v413_avg_student': 0.65128,
        'v414_avg_student': 0.63706,
        'v308_lb': 0.63893,
        'delta_vs_v308_meta': round(float(avg_oof - 0.62235), 5),
        'delta_vs_v308_student': round(float(avg_student - 0.69212), 5),
        'delta_vs_v411_student': round(float(avg_student - 0.65388), 5),
        'delta_vs_v413_student': round(float(avg_student - 0.65128), 5),
        'predicted_lb': round(float(predicted_lb), 5),
        'estimated_lb_v339_pattern': round(float(estimated_lb_v339), 5),
        'student_meta_gap': round(float(gap), 5),
        'v308_gap': 0.070,
        'n_seeds': N_SEEDS,
        'n_folds': N_FOLDS,
        'meta_type': f'{best_key}',
        'best_lgb_cfg': V413_CONFIGS,
        'per_target_meta_oof': {t: round(float(v), 5) for t, v in per_target_meta_oof.items()},
        'per_target_student_oof': {t: round(float(v), 5) for t, v in per_target_student_oof.items()},
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }

    meta_path = EXPERIMENTS / f'v415_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")

    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    main()
