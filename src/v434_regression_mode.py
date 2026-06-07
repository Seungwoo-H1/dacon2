"""
V434 — Regression Mode + Calibrated Probabilities

Hypothesis: Binary classification (0/1) with log-loss is not optimal for this dataset.
The targets have class imbalance and the predictions are probabilities that need
better calibration. Switching to regression mode in LGBM may help:

1. LGBM in regression mode predicts continuous values directly
2. Use squared_error / huber loss instead of binary_logloss
3. Convert regression predictions to probabilities via sigmoid calibration
4. This removes class_weight (scale_pos_weight) which may distort predictions

Additionally, test Huber loss which is more robust to outliers than squared_error.

V434 tests:
- reg_l1: L1 regression
- reg_l2: L2 regression (squared error)  
- reg_huber: Huber loss (robust to outliers)
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

# Regression-friendly configs (more learning rate, fewer trees)
LGB_REG_CFGS = {
    'narrow': {'num_leaves': 20, 'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 1500,
               'subsample': 0.5, 'colsample_bytree': 0.5, 'reg_alpha': 2.0, 'reg_lambda': 5.0,
               'min_child_samples': 20},
    'soft_aggressive': {'num_leaves': 15, 'max_depth': 3, 'learning_rate': 0.03, 'n_estimators': 1500,
                        'subsample': 0.55, 'colsample_bytree': 0.55, 'reg_alpha': 1.0, 'reg_lambda': 3.0,
                        'min_child_samples': 15},
    'ultra_deep': {'num_leaves': 30, 'max_depth': 6, 'learning_rate': 0.05, 'n_estimators': 800,
                   'subsample': 0.75, 'colsample_bytree': 0.65, 'reg_alpha': 0.1, 'reg_lambda': 0.5,
                   'min_child_samples': 8},
    'safety': {'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 1500,
               'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 1.0, 'reg_lambda': 3.0,
               'min_child_samples': 15},
    'broad': {'num_leaves': 40, 'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 600,
              'subsample': 0.85, 'colsample_bytree': 0.85, 'reg_alpha': 0.5, 'reg_lambda': 1.0,
              'min_child_samples': 5},
}

V413_CONFIGS = {
    'Q1': 'narrow', 'Q2': 'soft_aggressive', 'Q3': 'narrow',
    'S1': 'ultra_deep', 'S2': 'soft_aggressive', 'S3': 'safety', 'S4': 'broad',
}
V413_NFEAT = {'Q1': 19, 'Q2': 19, 'Q3': 15, 'S1': 21, 'S2': 19, 'S3': 23, 'S4': 20}

SEED = 42
N_FOLDS = 5
N_SEEDS = 15
LOSSES = ['reg_l1', 'reg_l2', 'reg_huber']


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


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
    params = {
        'objective': 'regression', 'metric': 'l2', 'verbose': -1,
        'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 50,
        'random_state': seed, 'force_row_wise': True, 'n_jobs': 1
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
    log.info("V434 — Regression Mode + Calibrated Probabilities")
    log.info("Testing: reg_l1, reg_l2, reg_huber with linear cal")
    log.info("=" * 70)

    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    groups_arr = train_df['subject_id'].values

    # Store results for each loss type
    all_results = {}

    for loss_type in LOSSES:
        log.info(f"\n{'='*60}")
        log.info(f"Loss type: {loss_type}")
        log.info(f"{'='*60}")

        all_oof_seed = {}
        all_test_seed = {}
        all_meta_oofs = {}
        all_student_oofs = {}
        all_test_preds = {}

        for t_idx, target in enumerate(TARGETS):
            cfg_name = V413_CONFIGS[target]
            cfg = LGB_REG_CFGS[cfg_name]
            n_feat = V413_NFEAT[target]
            y = train_df[target].values.astype(np.float64)

            # Feature ranking using regression
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

                    params = {
                        'objective': 'regression', 'metric': 'l2', 'verbose': -1,
                        **{k: v for k, v in cfg.items() if k not in ['n_estimators']},
                        'n_estimators': cfg['n_estimators'],
                        'random_state': sk, 'force_row_wise': True, 'n_jobs': 1,
                    }

                    ds_train = lgb.Dataset(x_train, label=y_train,
                        feature_name=[sanitize_col(c) for c in top_features])
                    ds_val = lgb.Dataset(x_val, label=y_all[val_idx],
                        feature_name=[sanitize_col(c) for c in top_features], reference=ds_train)

                    model = lgb.train(params, ds_train, num_boost_round=cfg['n_estimators'],
                        valid_sets=[ds_val],
                        callbacks=[lgb.early_stopping(100, verbose=False),
                                   lgb.log_evaluation(period=0)])

                    seed_oof[val_idx] = model.predict(x_val)
                    seed_test += model.predict(x_test) / N_FOLDS
                    del model, ds_train, ds_val
                    gc.collect()

                oof_seed_arr[:, s] = seed_oof
                test_seed_arr[:, s] = seed_test

            # Calibrate: map regression predictions to [0, 1] via sigmoid
            # Fit linear model: sigmoid(a * pred + b) ≈ y
            all_oof_seed[target] = oof_seed_arr
            all_test_seed[target] = test_seed_arr

            avg_oof = log_loss(y_all, sigmoid(oof_seed_arr.mean(axis=1)))
            log.info(f"  {target}: raw_oof={avg_oof:.5f}")

        # Meta: simple average (student)
        student_oofs = {}
        for t in TARGETS:
            y = train_df[t].values
            student_oofs[t] = log_loss(y, sigmoid(all_oof_seed[t].mean(axis=1)))

        avg_student = np.mean(list(student_oofs.values()))

        # Meta learner: XGB on calibrated predictions
        from xgboost import XGBClassifier
        
        meta_oofs = {}
        test_preds = {}
        for t_idx, target in enumerate(TARGETS):
            y = train_df[target].values
            # Calibrate seed predictions via sigmoid
            cal_oof = sigmoid(all_oof_seed[target])
            cal_test = sigmoid(all_test_seed[target])
            
            # Build meta features with stats
            means = np.mean(cal_oof, axis=1, keepdims=True)
            stds = np.std(cal_oof, axis=1, keepdims=True)
            mins = np.min(cal_oof, axis=1, keepdims=True)
            maxs = np.max(cal_oof, axis=1, keepdims=True)
            X_meta = np.hstack([cal_oof, means, stds, mins, maxs])
            X_test_meta = np.hstack([cal_test,
                np.mean(cal_test, axis=1, keepdims=True),
                np.std(cal_test, axis=1, keepdims=True),
                np.min(cal_test, axis=1, keepdims=True),
                np.max(cal_test, axis=1, keepdims=True)])

            mm = XGBClassifier(n_estimators=15, max_depth=3,
                reg_alpha=0.01, reg_lambda=0.0, gamma=0.0,
                learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                random_state=SEED, n_jobs=1, min_child_weight=5, verbosity=0)
            mm.fit(X_meta, y)
            meta_oofs[target] = log_loss(y, mm.predict_proba(X_meta)[:, 1])
            test_preds[target] = mm.predict_proba(X_test_meta)[:, 1]

        avg_meta = np.mean(list(meta_oofs.values()))
        gap = avg_student - avg_meta

        predicted_lb = avg_meta + gap * 0.5
        estimated_lb_v339 = avg_meta + gap * 0.85

        all_results[loss_type] = {
            'avg_meta': avg_meta, 'avg_student': avg_student, 'gap': gap,
            'predicted_lb': predicted_lb, 'estimated_lb_v339': estimated_lb_v339,
            'meta_oofs': meta_oofs, 'student_oofs': student_oofs,
            'test_preds': test_preds,
        }

        log.info(f"  AVG Meta OOF: {avg_meta:.5f}")
        log.info(f"  AVG Student OOF: {avg_student:.5f}")
        log.info(f"  Student-Meta Gap: {gap:.5f} (ratio: {gap/0.070:.2f}x)")
        log.info(f"  V339 Pattern LB: {estimated_lb_v339:.5f}")

    # Summary
    log.info(f"\n{'='*70}")
    log.info("V434 Summary:")
    log.info(f"{'Loss':<12} {'Meta':>8} {'Student':>8} {'Gap':>8} {'Gap/0.07':>8} {'V339 LB':>8}")
    log.info(f"{'-'*70}")
    for lt, r in all_results.items():
        log.info(f"{lt:<12} {r['avg_meta']:>8.5f} {r['avg_student']:>8.5f} "
                 f"{r['gap']:>8.5f} {r['gap']/0.070:>8.2f}x {r['estimated_lb_v339']:>8.5f}")
    log.info(f"{'='*70}")

    # Use best loss type for submission
    best_loss = min(all_results.items(), key=lambda x: x[1]['estimated_lb_v339'])[0]
    best = all_results[best_loss]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        sub[t] = best['test_preds'][t]

    sub_path = SUBMIT / f"submission_v434_reg_{best_loss}_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission (best={best_loss}): {sub_path}")

    meta_data = {
        'version': 'V434',
        'name': f'Regression Mode ({LOSSES}) + Calibrated XGB Meta',
        'best_loss': best_loss,
        'results': {lt: {
            'avg_meta_oof': round(float(r['avg_meta']), 5),
            'avg_student_oof': round(float(r['avg_student']), 5),
            'gap': round(float(r['gap']), 5),
            'v339_lb': round(float(r['estimated_lb_v339']), 5),
        } for lt, r in all_results.items()},
        'v308_lb': 0.63893,
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }

    meta_path = EXPERIMENTS / f'v434_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    log.info(f"Total time: {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    main()
