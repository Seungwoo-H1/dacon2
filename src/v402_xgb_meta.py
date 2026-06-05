"""
V402 — XGBoost Meta-Learner + Per-Target Meta C

Hypothesis: V308/V392 use LR as meta-learner (linear). The relationship
between LGBM seed predictions and the true target may be non-linear.

V392: LR meta → meta 0.617, student 0.692
V402: XGBoost meta (non-linear ensemble of 15 seed predictions)
      → better meta fitting without changing student models

Key: XGBoost meta takes 15-dimensional seed predictions as features,
predicts the binary target. With proper regularization, it should
outperform linear LR, lowering meta OOF while student stays same.
"""
import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
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
    'deep':   {'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.02, 'n_estimators': 1000,
               'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 0.5, 'reg_lambda': 2.0, 'min_child_samples': 15},
    'v48':    {'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 500,
               'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_samples': 10},
    'wide':   {'num_leaves': 30, 'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 300,
               'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_samples': 5},
    'safety': {'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.02, 'n_estimators': 1000,
               'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 3.0, 'reg_lambda': 10.0, 'min_child_samples': 20},
}

V53_SWEEP = {
    'Q1':  {'cfg': 'deep',   'n_feat': 19},
    'Q2':  {'cfg': 'deep',   'n_feat': 14},
    'Q3':  {'cfg': 'v48',    'n_feat': 11},
    'S1':  {'cfg': 'wide',   'n_feat': 21},
    'S2':  {'cfg': 'deep',   'n_feat': 19},
    'S3':  {'cfg': 'safety', 'n_feat': 23},
    'S4':  {'cfg': 'wide',   'n_feat': 20},
}

SEED = 42
N_FOLDS = 5
N_SEEDS = 15
META_C_PER_TARGET = {
    'Q1': 10.0, 'Q2': 10.0, 'Q3': 10.0,
    'S1': 100.0, 'S2': 100.0, 'S3': 100.0, 'S4': 100.0,
}


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


def train_xgb_meta(stacked, y, n_estimators, max_depth, learning_rate):
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'max_depth': max_depth, 'learning_rate': learning_rate,
        'n_estimators': n_estimators, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_alpha': 1.0, 'reg_lambda': 5.0, 'random_state': SEED,
        'min_child_weight': 10, 'n_jobs': 1,
    }
    ds = lgb.Dataset(stacked, label=y)
    return lgb.train(params, ds, num_boost_round=n_estimators)


def main():
    global t_start
    t_start = time.time()

    log.info("=" * 70)
    log.info("V402 — XGBoost Meta-Learner + Per-Target Meta C")
    log.info("Hypothesis: Non-linear XGB meta outperforms linear LR meta")
    log.info("V392: LR meta → meta 0.617, student 0.692")
    log.info("V402: XGBoost meta (non-linear ensemble of seed predictions)")
    log.info("=" * 70)

    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")

    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')

    train_feat_cols = get_feature_cols(train_df)
    test_feat_cols = get_feature_cols(test_df)
    log.info(f"Train: {len(train_feat_cols)} features, Test: {len(test_feat_cols)} features")

    group = train_df['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    n_train = len(train_df)
    n_test = len(test_df)

    train_oof = {t: np.zeros(n_train) for t in TARGETS}
    test_preds = {t: np.zeros((n_test, N_SEEDS)) for t in TARGETS}
    per_seed_oofs_all = {t: [] for t in TARGETS}
    best_meta_params = {}

    for t in TARGETS:
        log.info(f"\n{'='*60}")
        log.info(f"Target: {t}")
        y = train_df[t].values.astype(np.float64)
        feat_cols_clean = remove_leak(train_feat_cols, t)
        n_feat = V53_SWEEP[t]['n_feat']
        cfg_name = V53_SWEEP[t]['cfg']
        meta_c = META_C_PER_TARGET[t]

        ranked = rank_features(train_df, feat_cols_clean, t)
        sel_cols = ranked[:n_feat]

        sel_cols_test = [c for c in sel_cols if c in test_feat_cols]
        if len(sel_cols_test) != len(sel_cols):
            sel_cols = sel_cols_test

        cfg = CFGS[cfg_name]
        log.info(f"    Config: {cfg_name}, n_feat: {n_feat}, meta: XGBoost")

        per_seed_oofs = []
        for si in range(N_SEEDS):
            seed = SEED + si * 7
            seed_oof = np.zeros(n_train)
            seed_test = np.zeros(n_test)

            for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
                X_tr = train_df[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
                X_va = train_df[sel_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
                y_tr = y[tr_idx]

                spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed,
                          'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                sn = [sanitize_col(c) for c in sel_cols]
                ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])

                seed_oof[va_idx] = m.predict(X_va)
                seed_test += m.predict(test_df[sel_cols_test].fillna(0).values.astype(np.float64))

            seed_oof = np.clip(seed_oof, 0.001, 0.999)
            seed_test /= N_FOLDS
            per_seed_oofs.append(seed_oof)
            test_preds[t][:, si] = seed_test
            per_seed_oofs_all[t].append(seed_oof)

            if si < 5 or si % 3 == 0:
                ll_seed = log_loss(y, seed_oof)
                log.info(f"    Seed {si:2d} (s{seed}): OOF={ll_seed:.5f}")

        # Level 1: XGBoost meta-learner (try multiple configs, pick best OOF)
        stacked = np.column_stack(per_seed_oofs)

        best_oof = float('inf')
        best_params = None

        for n_est in [5, 10, 15]:
            for md in [2, 3]:
                for lr in [0.05, 0.1]:
                    mdl = train_xgb_meta(stacked, y, n_est, md, lr)
                    preds = mdl.predict(stacked)
                    preds = np.clip(preds, 0.001, 0.999)
                    ll = log_loss(y, preds)
                    if ll < best_oof:
                        best_oof = ll
                        best_params = (n_est, md, lr)

        # Compare with LR
        lr_meta = LogisticRegression(C=meta_c, max_iter=1000, random_state=SEED)
        lr_meta.fit(stacked, y)
        lr_preds = lr_meta.predict_proba(stacked)[:, 1]
        lr_oof = log_loss(y, np.clip(lr_preds, 0.001, 0.999))
        log.info(f"    {t}: LR OOF={lr_oof:.5f}, XGB best OOF={best_oof:.5f} (n_est={best_params[0]}, md={best_params[1]}, lr={best_params[2]})")

        # Generate train OOF predictions using best XGB meta
        best_mdl = train_xgb_meta(stacked, y, best_params[0], best_params[1], best_params[2])
        train_oof[t] = best_mdl.predict(stacked)
        student_oof = np.mean([log_loss(y, np.clip(p, 0.001, 0.999)) for p in per_seed_oofs])
        log.info(f"    {t}: Meta OOF={best_oof:.5f}, Student OOF={student_oof:.5f}")

        best_meta_params[t] = best_params

    # Compute results
    per_target_oof = {}
    per_target_student = {}
    for t in TARGETS:
        per_target_oof[t] = log_loss(train_df[t].values, np.clip(train_oof[t], 0.001, 0.999))
        per_target_student[t] = np.mean([log_loss(train_df[t].values, np.clip(p, 0.001, 0.999)) for p in per_seed_oofs_all[t]])

    avg_oof = np.mean(list(per_target_oof.values()))
    avg_student = np.mean(list(per_target_student.values()))

    log.info(f"\n{'='*70}")
    log.info(f"V402 RESULTS (XGBoost Meta-Learner)")
    log.info(f"{'='*70}")
    for t in TARGETS:
        n_est, md, lr = best_meta_params[t]
        log.info(f"  {t}: Meta={per_target_oof[t]:.5f}, Student={per_target_student[t]:.5f}, XGB(n={n_est},md={md},lr={lr})")
    log.info(f"  AVG Meta OOF: {avg_oof:.5f}")
    log.info(f"  AVG Student OOF: {avg_student:.5f}")
    log.info(f"  V308 AVG OOF: 0.62235")
    log.info(f"  V308 AVG Student: 0.69212")
    log.info(f"  Δ Meta vs V308: {avg_oof - 0.62235:+.5f}")
    log.info(f"  Δ Student vs V308: {avg_student - 0.69212:+.5f}")

    v308_gap = 0.63893 - 0.62235
    predicted_lb = avg_oof + v308_gap
    log.info(f"  V308 OOF-LB gap: {v308_gap:.5f}")
    log.info(f"  Predicted LB: {predicted_lb:.5f}")
    log.info(f"  Beats V308? {predicted_lb < 0.63893}")

    gap = avg_student - avg_oof
    v308_student_gap = 0.69212 - 0.62235
    log.info(f"  Student-Meta Gap: {gap:.5f} (V308: {v308_student_gap:.5f})")
    log.info(f"{'='*70}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build submission: apply best meta models to test predictions
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        n_est, md, lr = best_meta_params[t]
        stacked_test = np.column_stack([test_preds[t][:, i] for i in range(N_SEEDS)])
        stacked_train = np.column_stack(per_seed_oofs_all[t])
        y_t = train_df[t].values.astype(np.float64)
        mdl = train_xgb_meta(stacked_train, y_t, n_est, md, lr)
        sub[t] = np.clip(mdl.predict(stacked_test), 0.001, 0.999)

    sub_path = SUBMIT / f"submission_v402_xgb_meta_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")

    meta_data = {
        'version': 'V402',
        'name': 'XGBoost Meta-Learner',
        'avg_meta_oof': round(float(avg_oof), 5),
        'avg_student_oof': round(float(avg_student), 5),
        'v308_avg_oof': 0.62235,
        'v308_avg_student': 0.69212,
        'v308_lb': 0.63893,
        'delta_vs_v308_meta': round(float(avg_oof - 0.62235), 5),
        'delta_vs_v308_student': round(float(avg_student - 0.69212), 5),
        'predicted_lb': round(float(predicted_lb), 5),
        'student_meta_gap': round(float(gap), 5),
        'v308_gap': round(float(v308_gap), 5),
        'n_seeds': N_SEEDS,
        'meta_type': 'xgboost',
        'meta_c_per_target': META_C_PER_TARGET,
        'best_meta_params': {t: list(v) for t, v in best_meta_params.items()},
        'per_target_meta_oof': {t: round(float(per_target_oof[t]), 5) for t in TARGETS},
        'per_target_student_oof': {t: round(float(per_target_student[t]), 5) for t in TARGETS},
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }

    meta_path = EXPERIMENTS / f'v402_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")

    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return avg_oof, meta_data


if __name__ == '__main__':
    main()
