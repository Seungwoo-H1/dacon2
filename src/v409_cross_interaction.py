"""
V409 — Feature Cross-Interaction for Q Targets

Hypothesis: Q targets (Q1, Q3) have very high Student OOF (~0.82) and are
the main bottleneck. The current approach uses only base features + z-score.
Q targets may benefit from cross-interaction features between HR and pedometry
that capture lifestyle-health relationships.

V409 adds cross-interaction features:
- HR × pedometry interactions (activity-heart rate coupling)
- Light × sleep interactions
- Temporal ratio features (active minutes / total minutes)
- These are added on top of the existing 141 base features

The hypothesis is that these interaction features will help Q targets more
than S targets (which already have good OOF).
"""
import sys, gc, logging, json, re, time, warnings, itertools
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


def add_cross_features(df, feature_cols, target):
    """Add cross-interaction features to df in-place."""
    # HR × pedometry interactions
    hr_cols = [c for c in feature_cols if 'wHr_hr_mean' in c or 'wHr_hr_std' in c]
    pedo_cols = [c for c in feature_cols if 'wPedo_pedo_step_mean' in c or 'wPedo_pedo_distance_mean' in c or 'wPedo_pedo_burned_calories_mean' in c]

    df['hr_x_pedo_step'] = 0.0
    df['hr_x_pedo_dist'] = 0.0
    df['hr_x_pedo_cal'] = 0.0
    if hr_cols and pedo_cols:
        df['hr_x_pedo_step'] = df[hr_cols[0]] * df[pedo_cols[0]]
        df['hr_x_pedo_dist'] = df[hr_cols[0]] * df[pedo_cols[1]] if len(pedo_cols) > 1 else df[hr_cols[0]]
        df['hr_x_pedo_cal'] = df[hr_cols[0]] * df[pedo_cols[2]] if len(pedo_cols) > 2 else df[hr_cols[0]]

    # Light × sleep interactions
    light_mean = [c for c in feature_cols if 'wLight_w_light_mean' in c]
    if light_mean and 'hr_x_pedo_step' in df.columns:
        df['light_x_hr'] = df[light_mean[0]] * df[hr_cols[0]] if hr_cols else 0.0

    # Temporal ratios
    if 'hr_x_pedo_step' in df.columns:
        hr_std = [c for c in feature_cols if 'wHr_hr_std' in c]
        if hr_std:
            df['hr_cv'] = df[hr_std[0]] / (df['hr_x_pedo_step'] + 1e-8)

    # Active ratio: step_mean / distance_mean
    step_mean = [c for c in feature_cols if 'wPedo_pedo_step_mean' in c]
    dist_mean = [c for c in feature_cols if 'wPedo_pedo_distance_mean' in c]
    if step_mean and dist_mean:
        df['step_to_dist'] = df[step_mean[0]] / (df[dist_mean[0]] + 1e-8)

    return df


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
    log.info("V409 — Feature Cross-Interaction")
    log.info("Hypothesis: Cross features help Q targets reduce student OOF")
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

    # Add cross-interaction features
    log.info("Adding cross-interaction features...")
    add_cross_features(train_df, train_feat_cols, None)
    add_cross_features(test_df, test_feat_cols, None)

    # Update feature columns
    train_feat_cols_ext = get_feature_cols(train_df)
    test_feat_cols_ext = get_feature_cols(test_df)
    log.info(f"Train extended: {len(train_feat_cols_ext)} features, Test extended: {len(test_feat_cols_ext)} features")

    group = train_df['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    n_train = len(train_df)
    n_test = len(test_df)

    train_oof = {t: np.zeros(n_train) for t in TARGETS}
    test_preds = {t: np.zeros(n_test) for t in TARGETS}
    per_seed_student_oofs = {t: [] for t in TARGETS}

    XGB_PARAMS = {'n_estimators': 15, 'max_depth': 3, 'learning_rate': 0.1}

    for t in TARGETS:
        log.info(f"\n{'='*60}")
        log.info(f"Target: {t}")
        y = train_df[t].values.astype(np.float64)
        feat_cols_clean = remove_leak(train_feat_cols_ext, t)
        n_feat = V53_SWEEP[t]['n_feat']
        cfg_name = V53_SWEEP[t]['cfg']

        ranked = rank_features(train_df, feat_cols_clean, t)
        sel_cols = ranked[:n_feat]

        sel_cols_test = [c for c in sel_cols if c in test_feat_cols_ext]
        if len(sel_cols_test) != len(sel_cols):
            sel_cols = sel_cols_test

        cfg = CFGS[cfg_name]
        log.info(f"    Config: {cfg_name}, n_feat: {n_feat}")

        per_seed_oofs = []
        per_seed_test = np.zeros((n_test, N_SEEDS))
        seed_lls = []

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
            per_seed_test[:, si] = seed_test

            ll = log_loss(y, seed_oof)
            seed_lls.append(ll)
            per_seed_student_oofs[t].append(ll)

            if si < 5 or si % 3 == 0:
                log.info(f"    Seed {si:2d} (s{seed}): OOF={ll:.5f}")

        seed_lls = np.array(seed_lls)
        student_oof = np.mean(seed_lls)

        # XGB meta
        stacked = np.column_stack(per_seed_oofs)
        mdl = train_xgb_meta(stacked, y, XGB_PARAMS['n_estimators'],
                           XGB_PARAMS['max_depth'], XGB_PARAMS['learning_rate'])
        train_oof[t] = np.clip(mdl.predict(stacked), 0.001, 0.999)
        test_preds[t] = per_seed_test.mean(axis=1)
        test_preds[t] = np.clip(test_preds[t], 0.001, 0.999)

        meta_oof_final = log_loss(y, train_oof[t])
        gap_final = student_oof - meta_oof_final
        log.info(f"    {t}: Meta={meta_oof_final:.5f}, Student={student_oof:.5f}, Gap={gap_final:.5f}")

    # Compute results
    per_target_oof = {}
    per_target_student = {}
    for t in TARGETS:
        per_target_oof[t] = log_loss(train_df[t].values, np.clip(train_oof[t], 0.001, 0.999))
        per_target_student[t] = np.mean(per_seed_student_oofs[t])

    avg_oof = np.mean(list(per_target_oof.values()))
    avg_student = np.mean(list(per_target_student.values()))

    log.info(f"\n{'='*70}")
    log.info(f"V409 RESULTS (Feature Cross-Interaction)")
    log.info(f"{'='*70}")
    for t in TARGETS:
        log.info(f"  {t}: Meta={per_target_oof[t]:.5f}, Student={per_target_student[t]:.5f}")
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

    v339_actual_gap = 0.033
    v339_ratio = v339_actual_gap / v308_gap
    gap_ratio = v308_student_gap / v308_gap
    estimated_actual_gap = min(gap_ratio, v339_ratio * (gap / v308_student_gap)) * v308_gap
    estimated_lb = avg_oof + estimated_actual_gap
    log.info(f"  Estimated actual LB (V339 pattern): {estimated_lb:.5f}")
    log.info(f"{'='*70}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        sub[t] = test_preds[t]

    sub_path = SUBMIT / f"submission_v409_cross_feat_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")

    meta_data = {
        'version': 'V409',
        'name': 'Feature Cross-Interaction',
        'avg_meta_oof': round(float(avg_oof), 5),
        'avg_student_oof': round(float(avg_student), 5),
        'v308_avg_oof': 0.62235,
        'v308_avg_student': 0.69212,
        'v308_lb': 0.63893,
        'delta_vs_v308_meta': round(float(avg_oof - 0.62235), 5),
        'delta_vs_v308_student': round(float(avg_student - 0.69212), 5),
        'predicted_lb': round(float(predicted_lb), 5),
        'estimated_lb_v339_pattern': round(float(estimated_lb), 5),
        'student_meta_gap': round(float(gap), 5),
        'v308_gap': round(float(v308_gap), 5),
        'n_seeds': N_SEEDS,
        'meta_type': 'xgb_15_md3_lr0.1',
        'cross_features_added': True,
        'per_target_meta_oof': {t: round(float(per_target_oof[t]), 5) for t in TARGETS},
        'per_target_student_oof': {t: round(float(per_target_student[t]), 5) for t in TARGETS},
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }

    meta_path = EXPERIMENTS / f'v409_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")

    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return avg_oof, meta_data


if __name__ == '__main__':
    main()
