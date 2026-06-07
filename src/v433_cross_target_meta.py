"""
V433 — Cross-Target Feature Sharing in Meta-Learner

Hypothesis: Q targets (sleep quality metrics) and S targets (sleep timing metrics)
are correlated but processed independently. The XGB meta-learner only sees predictions
from ONE target's seeds, missing cross-target signal.

Cross-target features:
1. For each target, add other targets' predictions as meta features
2. This allows the meta-learner to use e.g. S1 prediction quality to inform Q1 prediction

Key insight: If S1 and Q1 share underlying signal (sleep quality affects both),
cross-target features should help the meta-learner resolve ambiguity.

Architecture:
- Level 0: 15 LGBM seeds per target (same as V429)
- Level 1 (meta): XGB with [15 self-seed preds + 15 cross-target avg preds] = 30 features
  - Cross-target avg: mean of other targets' seed predictions
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
    log.info("V433 — Cross-Target Feature Sharing in Meta-Learner")
    log.info("Hypothesis: Cross-target signal can improve meta predictions")
    log.info("=" * 70)

    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    groups_arr = train_df['subject_id'].values

    # ===== Phase 1: LGBM base models per target =====
    all_oof_seed = {}
    all_test_seed = {}

    for t_idx, target in enumerate(TARGETS):
        cfg_name = V413_CONFIGS[target]
        cfg = LGB_CFGS[cfg_name]
        n_feat = V413_NFEAT[target]

        feat_cols = get_feature_cols(train_df)
        feat_cols = remove_leak(feat_cols, target)

        fold_ranks = []
        for fold in range(N_FOLDS):
            rank = rank_features(train_df, feat_cols, target, seed=SEED + fold)
            fold_ranks.append(rank[:n_feat])

        feat_counts = {}
        for fl in fold_ranks:
            for f in fl:
                feat_counts[f] = feat_counts.get(f, 0) + 1
        ranked_features = sorted(feat_counts.items(), key=lambda x: -x[1])
        top_features = [f for f, c in ranked_features[:n_feat]]

        X_all = train_df[top_features + [target]].fillna(0).values.astype(np.float64)
        y_all = X_all[:, -1]
        X_all = X_all[:, :-1]
        X_test_all = test_df[top_features].fillna(0).values.astype(np.float64)

        oof_seed_arr = np.zeros((len(train_df), N_SEEDS))
        test_seed_arr = np.zeros((len(test_df), N_SEEDS))

        for s in range(N_SEEDS):
            sk = SEED + s * 7 + t_idx
            skf = GroupKFold(n_splits=N_FOLDS)
            seed_oof = np.zeros(len(train_df))
            seed_test = np.zeros(len(test_df))

            for train_idx, val_idx in skf.split(X_all, y_all, groups_arr):
                x_train, y_train = X_all[train_idx], y_all[train_idx]
                x_val, x_test = X_all[val_idx], X_test_all

                spw = max(((y_train == 0).sum()) / max((y_train == 1).sum(), 1), 0.1)
                params = {
                    'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
                    **{k: v for k, v in cfg.items() if k not in ['n_estimators']},
                    'n_estimators': cfg['n_estimators'], 'scale_pos_weight': spw,
                    'random_state': sk, 'force_row_wise': True, 'n_jobs': 1,
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
                seed_test += model.predict(x_test) / N_FOLDS
                del model, ds_train, ds_val
                gc.collect()

            oof_seed_arr[:, s] = seed_oof
            test_seed_arr[:, s] = seed_test

        all_oof_seed[target] = oof_seed_arr
        all_test_seed[target] = test_seed_arr

        avg_oof = log_loss(y_all, oof_seed_arr.mean(axis=1))
        log.info(f"{target}: base_oof={avg_oof:.5f}")

    # ===== Phase 2: Cross-Target Meta =====
    log.info("\n=== Phase 2: Cross-Target Meta ===")

    from xgboost import XGBClassifier

    # Build cross-target meta features for each target
    # For target T: meta features = [15 self-seed preds, mean of other targets' preds]
    def build_cross_target_meta(oof_seed, test_seed, targets_oof, targets_test, targets_list, target):
        """
        Build meta features: self-seed preds + cross-target averages
        Returns: [n_samples, 15 + N_TARGETS-1] features
        """
        t_idx = targets_list.index(target)
        self_preds = oof_seed  # [450, 15]
        self_test = test_seed  # [250, 15]

        # Cross-target averages (excluding self)
        cross_oof = []
        cross_test = []
        for i, t in enumerate(targets_list):
            if i != t_idx:
                cross_oof.append(np.mean(targets_oof[t], axis=1))
                cross_test.append(np.mean(targets_test[t], axis=1))

        if len(cross_oof) > 0:
            cross_oof = np.column_stack(cross_oof)  # [450, 6]
            cross_test = np.column_stack(cross_test)  # [250, 6]
            X_meta = np.hstack([self_preds, cross_oof])  # [450, 21]
            X_test_meta = np.hstack([self_test, cross_test])  # [250, 21]
        else:
            X_meta = self_preds
            X_test_meta = self_test

        return X_meta, X_test_meta

    meta_oofs = {}
    student_oofs = {}
    test_preds = {}

    for t_idx, target in enumerate(TARGETS):
        y = train_df[target].values
        X_meta, X_test_meta = build_cross_target_meta(
            all_oof_seed[target], all_test_seed[target],
            all_oof_seed, all_test_seed, TARGETS, target)
        
        student_baseline = log_loss(y, all_oof_seed[target].mean(axis=1))

        # Also run baseline (self-only) for comparison
        self_only = all_oof_seed[target]
        self_oof = log_loss(y, XGBClassifier(
            n_estimators=15, max_depth=3, reg_alpha=0.01, reg_lambda=0.0,
            gamma=0.0, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, n_jobs=1, min_child_weight=5, verbosity=0
        ).fit(self_only, y).predict_proba(self_only)[:, 1])
        
        # Cross-target XGB
        ct_model = XGBClassifier(
            n_estimators=15, max_depth=3, reg_alpha=0.01, reg_lambda=0.0,
            gamma=0.0, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, n_jobs=1, min_child_weight=5, verbosity=0)
        ct_model.fit(X_meta, y)
        ct_meta_oof = log_loss(y, ct_model.predict_proba(X_meta)[:, 1])
        test_preds[target] = ct_model.predict_proba(X_test_meta)[:, 1]
        meta_oofs[target] = ct_meta_oof
        student_oofs[target] = student_baseline

        log.info(f"  {target}: self_oof={self_oof:.5f}, cross_oof={ct_meta_oof:.5f}, "
                 f"Δ={self_oof-ct_meta_oof:+.5f}, student={student_oofs[target]:.5f}")

    avg_meta = np.mean(list(meta_oofs.values()))
    avg_student = np.mean(list(student_oofs.values()))
    gap = avg_student - avg_meta

    predicted_lb = avg_meta + gap * 0.5
    estimated_lb_v339 = avg_meta + gap * 0.85

    log.info(f"\n{'='*70}")
    log.info("V433 Results:")
    log.info(f"  Meta features: 15 self + 6 cross-target = 21")
    log.info(f"  AVG Meta OOF: {avg_meta:.5f} (Δ vs V308: {avg_meta-0.62235:+.5f})")
    log.info(f"  AVG Student OOF: {avg_student:.5f} (Δ vs V308: {avg_student-0.69212:+.5f})")
    log.info(f"  Student-Meta Gap: {gap:.5f} (V308: 0.070, ratio: {gap/0.070:.2f}x)")
    log.info(f"  Predicted LB: {predicted_lb:.5f}")
    log.info(f"  V339 Pattern LB: {estimated_lb_v339:.5f}")
    log.info(f"{'='*70}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        sub[t] = test_preds[t]

    sub_path = SUBMIT / f"submission_v433_cross_target_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")

    meta_data = {
        'version': 'V433',
        'name': 'Cross-Target Feature Sharing in Meta-Learner',
        'avg_meta_oof': round(float(avg_meta), 5),
        'avg_student_oof': round(float(avg_student), 5),
        'v308_avg_oof': 0.62235, 'v308_avg_student': 0.69212,
        'v308_lb': 0.63893,
        'delta_vs_v308_meta': round(float(avg_meta - 0.62235), 5),
        'delta_vs_v308_student': round(float(avg_student - 0.69212), 5),
        'predicted_lb': round(float(predicted_lb), 5),
        'estimated_lb_v339_pattern': round(float(estimated_lb_v339), 5),
        'student_meta_gap': round(float(gap), 5),
        'n_seeds': N_SEEDS,
        'meta_type': 'xgb_cross_target_21feat',
        'meta_features': '15_self + 6_cross_target_mean',
        'per_target_meta_oof': {t: round(float(v), 5) for t, v in meta_oofs.items()},
        'per_target_student_oof': {t: round(float(v), 5) for t, v in student_oofs.items()},
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }

    meta_path = EXPERIMENTS / f'v433_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    log.info(f"Total time: {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    main()
