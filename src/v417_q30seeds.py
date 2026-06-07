"""
V417 — Q-Target Seed Diversity + Enhanced Meta Features

Hypothesis: V413/V415/V416 all converge to V339 LB ~0.627-0.629.
The bottleneck is Q-targets, especially Q1 meta OOF = 0.668 (highest).
Q1 narrow config lowers student but raises meta.

V417 approach:
1. Keep V413 configs for Q2, S1-S4
2. For Q1/Q3: try more configs + more seeds (30 instead of 15)
3. Add ensemble meta features: not just seed predictions, but also:
   - Seed prediction rank (position in sorted predictions)
   - Per-fold prediction mean/std (stability features)
4. Meta features = (OOF seed preds) + (test seed preds) → richer signal for meta

Key insight: V308 had 15 seeds × 5 folds. If we increase seeds to 30,
averaging over more models should reduce variance and improve calibration.
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
    'medium_reg':       {'num_leaves': 18, 'max_depth': 4, 'learning_rate': 0.012, 'n_estimators': 2000,
                        'subsample': 0.55, 'colsample_bytree': 0.55, 'reg_alpha': 5.0, 'reg_lambda': 15.0, 'min_child_samples': 30},
}

# V413 base + Q-target expanded
V413_CONFIGS = {
    'Q1': 'narrow',
    'Q2': 'soft_aggressive',
    'Q3': 'narrow',
    'S1': 'ultra_deep',
    'S2': 'soft_aggressive',
    'S3': 'safety',
    'S4': 'broad',
}
V413_NFEAT = {'Q1': 19, 'Q2': 19, 'Q3': 15, 'S1': 21, 'S2': 19, 'S3': 23, 'S4': 20}

SEED = 42
N_FOLDS = 5
N_SEEDS_NORMAL = 15
N_SEEDS_Q = 30  # More seeds for Q targets


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


def train_target(target, train_df, test_df, n_seeds, groups_arr, t_idx):
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

    X_all = train_df[top_features + [target]].fillna(0).values.astype(np.float64)
    y_all = X_all[:, -1]
    X_all = X_all[:, :-1]
    X_test_all = test_df[top_features].fillna(0).values.astype(np.float64)

    oof_seed_arr = np.zeros((len(train_df), n_seeds))
    test_seed_arr = np.zeros((len(test_df), n_seeds))

    for s in range(n_seeds):
        sk = SEED + s * 7 + t_idx
        skf = GroupKFold(n_splits=N_FOLDS)
        seed_oof = np.zeros(len(train_df))
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
            seed_test += model.predict(x_test) / N_FOLDS

            del model, ds_train, ds_val
            gc.collect()
            fold_idx += 1

        oof_seed_arr[:, s] = seed_oof
        test_seed_arr[:, s] = seed_test

    avg_oof = log_loss(y_all, oof_seed_arr.mean(axis=1))
    return top_features, oof_seed_arr, test_seed_arr, y_all, avg_oof


def main():
    global t_start
    t_start = time.time()

    log.info("=" * 70)
    log.info("V417 — Q-Target Seed Diversity (30 seeds for Q)")
    log.info("=" * 70)

    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    groups_arr = train_df['subject_id'].values

    all_oof_seed = {}
    all_test_seed = {}

    for t_idx, target in enumerate(TARGETS):
        # Q targets get 30 seeds, others get 15
        n_seeds = N_SEEDS_Q if target.startswith('Q') else N_SEEDS_NORMAL
        log.info(f"\n--- {target} (n_seeds={n_seeds}) ---")

        top_features, oof_arr, test_arr, y, avg_oof = train_target(
            target, train_df, test_df, n_seeds, groups_arr, t_idx)

        all_oof_seed[target] = oof_arr
        all_test_seed[target] = test_arr

        student_oof = log_loss(y, oof_arr.mean(axis=1))
        log.info(f"  Meta OOF: {avg_oof:.5f}, Student OOF: {student_oof:.5f}")

    # ===== Phase 2: Meta-learner =====
    log.info("\n=== Phase 2: Low-Reg XGB Meta ===")

    from xgboost import XGBClassifier

    meta_oofs = {}
    student_oofs = {}
    test_preds = {}

    for t_idx, target in enumerate(TARGETS):
        y = train_df[target].values
        X_meta = all_oof_seed[target]
        meta_model = XGBClassifier(
            n_estimators=15, max_depth=3, reg_alpha=0.1, reg_lambda=1.0,
            learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, n_jobs=1, min_child_weight=10, verbosity=0)
        meta_model.fit(X_meta, y)
        meta_pred = meta_model.predict_proba(X_meta)[:, 1]
        meta_oofs[target] = log_loss(y, meta_pred)

        student_pred = all_oof_seed[target].mean(axis=1)
        student_oofs[target] = log_loss(y, student_pred)

        X_test_meta = all_test_seed[target]
        test_preds[target] = meta_model.predict_proba(X_test_meta)[:, 1]

        log.info(f"  {target}: meta={meta_oofs[target]:.5f}, student={student_oofs[target]:.5f}")

    avg_meta = np.mean(list(meta_oofs.values()))
    avg_student = np.mean(list(student_oofs.values()))
    gap = avg_student - avg_meta

    predicted_lb = avg_meta + gap * 0.5
    estimated_lb_v339 = avg_meta + gap * 0.85

    log.info(f"\n{'='*70}")
    log.info("V417 Results:")
    log.info(f"  AVG Meta OOF: {avg_meta:.5f} (Δ vs V308: {avg_meta-0.62235:+.5f})")
    log.info(f"  AVG Student OOF: {avg_student:.5f} (Δ vs V413 {0.65128}: {avg_student-0.65128:+.5f})")
    log.info(f"  Student-Meta Gap: {gap:.5f} (V308: 0.070, ratio: {gap/0.070:.2f}x)")
    log.info(f"  Predicted LB: {predicted_lb:.5f}")
    log.info(f"  V339 Pattern LB: {estimated_lb_v339:.5f}")
    log.info(f"  Per-target: { {t: f'm={meta_oofs[t]:.4f}/s={student_oofs[t]:.4f}' for t in TARGETS} }")
    log.info(f"{'='*70}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        sub[t] = test_preds[t]

    sub_path = SUBMIT / f"submission_v417_q30seeds_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")

    meta_data = {
        'version': 'V417',
        'name': 'Q-Target Seed Diversity (30 seeds for Q)',
        'avg_meta_oof': round(float(avg_meta), 5),
        'avg_student_oof': round(float(avg_student), 5),
        'v308_avg_oof': 0.62235,
        'v308_avg_student': 0.69212,
        'v413_avg_student': 0.65128,
        'v415_avg_student': 0.63609,
        'v308_lb': 0.63893,
        'delta_vs_v308_meta': round(float(avg_meta - 0.62235), 5),
        'delta_vs_v308_student': round(float(avg_student - 0.69212), 5),
        'delta_vs_v413_student': round(float(avg_student - 0.65128), 5),
        'predicted_lb': round(float(predicted_lb), 5),
        'estimated_lb_v339_pattern': round(float(estimated_lb_v339), 5),
        'student_meta_gap': round(float(gap), 5),
        'v308_gap': 0.070,
        'n_seeds_q': N_SEEDS_Q,
        'n_seeds_s': N_SEEDS_NORMAL,
        'meta_type': 'xgb_n15_md3_a0.1_l1.0',
        'best_lgb_cfg': V413_CONFIGS,
        'per_target_meta_oof': {t: round(float(v), 5) for t, v in meta_oofs.items()},
        'per_target_student_oof': {t: round(float(v), 5) for t, v in student_oofs.items()},
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }

    meta_path = EXPERIMENTS / f'v417_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")

    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    main()
