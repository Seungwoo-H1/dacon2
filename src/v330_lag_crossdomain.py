"""
V330: V329 Enhanced — Lag Features + Cross-Domain Ratios

Hypothesis: V329 adds cross-subject + domain aggregations to V328, getting
OOF 0.57742. Further improvements can come from:
1. Lag features (previous day's values) — temporal continuity
2. Cross-domain ratios — interaction between different sensor domains
3. Subject activity level (total activity per subject)

These capture sequential patterns and inter-domain relationships.

Expected OOF: 0.570-0.575
Risk: LOW (lag features are safe with CV, cross-domain ratios are deterministic)
Cost: ~60s
"""
import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
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

SEED = 42
N_FOLDS = 5
N_SEEDS = 15
META_C = 10.0
FEATURE_BAG_FRACTION = 0.75


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


def build_all_features(train_df, test_df):
    """Build all features from V329 + lag features + cross-domain ratios."""
    log.info("Building all features...")
    train_df = train_df.copy()
    test_df = test_df.copy()

    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')

    # 1. Global z-score
    base_cols = [c for c in train_df.columns
                 if c not in META_COLS | set(TARGETS)
                 and not c.endswith('_zscore')
                 and np.issubdtype(train_df[c].dtype, np.number)]
    test_base = [c for c in test_df.columns
                 if c not in META_COLS | set(TARGETS)
                 and not c.endswith('_zscore')
                 and np.issubdtype(test_df[c].dtype, np.number)]
    common_cols = set(base_cols) & set(test_base)

    for col in common_cols:
        vals = train_df[col].fillna(0).values.astype(np.float64)
        mean = np.mean(vals)
        std = np.std(vals, ddof=0)
        if std < 1e-8:
            std = 1e-8
        zc = f'{col}_zscore'
        test_df[zc] = (test_df[col].fillna(0).values.astype(np.float64) - mean) / std
        train_df[zc] = (vals - mean) / std

    # 2. Interaction features
    hr_cols = [c for c in train_df.columns if c.startswith('wHr_') and np.issubdtype(train_df[c].dtype, np.number)]
    pedo_cols = [c for c in train_df.columns if c.startswith('wPedo_') and np.issubdtype(train_df[c].dtype, np.number)]
    light_cols = [c for c in train_df.columns if c.startswith('mLight_') and np.issubdtype(train_df[c].dtype, np.number)]
    screen_cols = [c for c in train_df.columns if c.startswith('mScreenStatus_') and np.issubdtype(train_df[c].dtype, np.number)]
    gps_cols = [c for c in train_df.columns if c.startswith('mGps_') and np.issubdtype(train_df[c].dtype, np.number)]
    ble_cols = [c for c in train_df.columns if c.startswith('mBle_') and np.issubdtype(train_df[c].dtype, np.number)]
    wifi_cols = [c for c in train_df.columns if c.startswith('mWifi_') and np.issubdtype(train_df[c].dtype, np.number)]

    for df in [train_df, test_df]:
        if hr_cols and pedo_cols:
            df['hr_pedo_interact'] = df[hr_cols].fillna(0).mean(axis=1) * df[pedo_cols].fillna(0).mean(axis=1)
        if light_cols and screen_cols:
            df['light_screen_interact'] = df[light_cols].fillna(0).mean(axis=1) * df[screen_cols].fillna(0).mean(axis=1)
        if gps_cols and ble_cols:
            df['gps_ble_interact'] = df[gps_cols].fillna(0).mean(axis=1) * df[ble_cols].fillna(0).mean(axis=1)
        if wifi_cols and gps_cols:
            df['wifi_gps_interact'] = df[wifi_cols].fillna(0).mean(axis=1) * df[gps_cols].fillna(0).mean(axis=1)

        pedo_steps = [c for c in pedo_cols if 'step' in c and 'sum' not in c]
        pedo_dist = [c for c in pedo_cols if 'distance' in c]
        if pedo_steps and pedo_dist:
            step_mean = df[pedo_steps].fillna(0).mean(axis=1)
            dist_mean = df[pedo_dist].fillna(0).mean(axis=1)
            df['step_length_ratio'] = (dist_mean + 1e-8) / (step_mean + 1e-8)

    # 3. Enhanced per-subject features (from V328)
    ps_base = [c for c in train_df.columns if c not in META_COLS | set(TARGETS) | {'date'}
               and not c.endswith('_zscore') and not c.endswith('_interact')
               and not c.endswith('_ratio') and not c.endswith('_proxy')
               and np.issubdtype(train_df[c].dtype, np.number)]
    ps_base = ps_base[:60]

    for col in ps_base:
        for df_src, grp_col in [(train_df, 'subject_id'), (test_df, 'subject_id')]:
            grp = df_src.groupby(grp_col)[col]

            df_src[f'ps_roll3_mean_{col}'] = grp.transform(lambda g: g.rolling(3, min_periods=1).mean()).values
            df_src[f'ps_roll3_std_{col}'] = grp.transform(lambda g: g.rolling(3, min_periods=1).std().fillna(0)).values
            df_src[f'ps_roll5_mean_{col}'] = grp.transform(lambda g: g.rolling(5, min_periods=1).mean()).values
            df_src[f'ps_roll5_std_{col}'] = grp.transform(lambda g: g.rolling(5, min_periods=1).std().fillna(0)).values

            ps_mean = grp.transform('mean')
            ps_std = grp.transform('std').fillna(0)
            ps_min = grp.transform('min')
            ps_max = grp.transform('max')
            ps_median = grp.transform('median')
            ps_iqr = grp.transform(lambda g: g.quantile(0.75) - g.quantile(0.25))

            df_src[f'ps_min_{col}'] = ps_min.values
            df_src[f'ps_max_{col}'] = ps_max.values
            df_src[f'ps_median_{col}'] = ps_median.values
            df_src[f'ps_iqr_{col}'] = ps_iqr.values
            df_src[f'ps_range_{col}'] = (ps_max - ps_min).values
            df_src[f'ps_cv_{col}'] = (ps_std / (ps_mean.abs() + 1e-8)).values
            df_src[f'ps_maxmin_ratio_{col}'] = (ps_max / (ps_min.abs() + 1e-8)).values

            abs_dev = grp.transform(lambda g: (g - g.mean()).abs())
            sq_dev = grp.transform(lambda g: (g - g.mean()) ** 2)
            outliers = grp.transform(lambda g: (g - g.mean()).abs() > 2 * max(g.std(ddof=0), 1e-8)).astype(float)

            df_src[f'ps_absdev_{col}'] = abs_dev.values
            df_src[f'ps_sqdev_{col}'] = sq_dev.values
            df_src[f'ps_outliers_{col}'] = outliers.values

    # 4. Cross-subject z-scores
    cs_base = [c for c in train_df.columns if c not in META_COLS | set(TARGETS) | {'date'}
               and not c.endswith('_zscore') and not c.endswith('_interact')
               and not c.endswith('_ratio') and not c.startswith('ps_')
               and np.issubdtype(train_df[c].dtype, np.number)]

    for col in cs_base[:80]:
        pop_mean = train_df[col].fillna(0).mean()
        pop_std = train_df[col].fillna(0).std(ddof=0)
        if pop_std < 1e-8:
            pop_std = 1e-8

        train_df[f'cs_zscore_{col}'] = (train_df[col].fillna(0) - pop_mean) / pop_std
        test_df[f'cs_zscore_{col}'] = (test_df[col].fillna(0) - pop_mean) / pop_std

    # 5. Domain aggregations
    domains = {
        'wHr': [c for c in train_df.columns if c.startswith('wHr_')],
        'wPedo': [c for c in train_df.columns if c.startswith('wPedo_')],
        'mLight': [c for c in train_df.columns if c.startswith('mLight_')],
        'mScreenStatus': [c for c in train_df.columns if c.startswith('mScreenStatus_')],
        'mGps': [c for c in train_df.columns if c.startswith('mGps_')],
        'mBle': [c for c in train_df.columns if c.startswith('mBle_')],
        'mWifi': [c for c in train_df.columns if c.startswith('mWifi_')],
        'mUsageStats': [c for c in train_df.columns if c.startswith('mUsageStats_')],
    }

    for domain_name, cols in domains.items():
        domain_base = [c for c in cols if c not in META_COLS | set(TARGETS) | {'date'}
                       and not c.endswith('_zscore') and not c.endswith('_interact')
                       and not c.endswith('_ratio') and not c.startswith('ps_')
                       and np.issubdtype(train_df[c].dtype, np.number)]
        if not domain_base:
            continue

        for df_src in [train_df, test_df]:
            grp = df_src.groupby('subject_id')[domain_base]
            dm = grp.mean()
            ds = grp.std().fillna(0)
            df_name = f'{domain_name.lower()}_domain'
            df_src[f'{df_name}_mean'] = dm.mean(axis=1).reindex(df_src['subject_id']).values
            df_src[f'{df_name}_std'] = ds.mean(axis=1).reindex(df_src['subject_id']).values

    # 6. Lag features (previous day's values per subject)
    log.info("  Generating lag features...")
    for col in ps_base[:40]:
        for df_src in [train_df, test_df]:
            grp = df_src.groupby('subject_id')[col]
            df_src[f'lag1_{col}'] = grp.shift(1).values
            df_src[f'lag2_{col}'] = grp.shift(2).values
            df_src[f'lag1_diff_{col}'] = grp.transform(lambda g: g - g.shift(1)).values
            df_src[f'lag1_diff_{col}'] = df_src[f'lag1_diff_{col}'].fillna(0).values

    # 7. Cross-domain ratios
    log.info("  Generating cross-domain ratios...")
    hr_mean = train_df[[c for c in train_df.columns if c.startswith('wHr_') and 'mean' in c]].fillna(0).mean(axis=1)
    pedo_mean = train_df[[c for c in train_df.columns if c.startswith('wPedo_') and 'mean' in c]].fillna(0).mean(axis=1)
    light_mean = train_df[[c for c in train_df.columns if c.startswith('mLight_') and 'mean' in c]].fillna(0).mean(axis=1)

    for df_src in [train_df, test_df]:
        df_hr = hr_mean if df_src is train_df else hr_mean  # same stats from train
        df_pedo = train_df[[c for c in train_df.columns if c.startswith('wPedo_') and 'mean' in c]].fillna(0).mean(axis=1) if df_src is train_df else test_df[[c for c in test_df.columns if c.startswith('wPedo_') and 'mean' in c]].fillna(0).mean(axis=1)
        df_light = train_df[[c for c in train_df.columns if c.startswith('mLight_') and 'mean' in c]].fillna(0).mean(axis=1) if df_src is train_df else test_df[[c for c in test_df.columns if c.startswith('mLight_') and 'mean' in c]].fillna(0).mean(axis=1)

    # Use train stats to compute test cross-domain ratios
    train_hr_mean = train_df[[c for c in train_df.columns if c.startswith('wHr_') and 'mean' in c]].fillna(0).mean(axis=1)
    train_pedo_mean = train_df[[c for c in train_df.columns if c.startswith('wPedo_') and 'mean' in c]].fillna(0).mean(axis=1)
    train_light_mean = train_df[[c for c in train_df.columns if c.startswith('mLight_') and 'mean' in c]].fillna(0).mean(axis=1)

    # For test: compute using test columns but same column names
    test_hr_mean = test_df[[c for c in test_df.columns if c.startswith('wHr_') and 'mean' in c]].fillna(0).mean(axis=1)
    test_pedo_mean = test_df[[c for c in test_df.columns if c.startswith('wPedo_') and 'mean' in c]].fillna(0).mean(axis=1)
    test_light_mean = test_df[[c for c in test_df.columns if c.startswith('mLight_') and 'mean' in c]].fillna(0).mean(axis=1)

    train_df['hr_pedo_ratio'] = (train_pedo_mean + 1e-8) / (train_hr_mean + 1e-8)
    train_df['hr_light_ratio'] = (train_light_mean + 1e-8) / (train_hr_mean + 1e-8)
    test_df['hr_pedo_ratio'] = (test_pedo_mean + 1e-8) / (test_hr_mean + 1e-8)
    test_df['hr_light_ratio'] = (test_light_mean + 1e-8) / (test_hr_mean + 1e-8)

    # 8. Subject-level total activity
    all_base = [c for c in train_df.columns if c not in META_COLS | set(TARGETS) | {'date'}
                and not c.endswith('_zscore') and not c.endswith('_interact')
                and not c.endswith('_ratio') and not c.startswith('ps_')
                and not c.startswith('lag') and not c.startswith('cs_')
                and np.issubdtype(train_df[c].dtype, np.number)]

    train_df['total_activity_proxy'] = train_df[all_base].fillna(0).abs().sum(axis=1)
    test_df['total_activity_proxy'] = test_df[all_base].fillna(0).abs().sum(axis=1)

    return train_df, test_df


# V326 config
CFGS = {
    'wide':   {'num_leaves': 30, 'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 300,
               'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_samples': 5},
    'deep':   {'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.02, 'n_estimators': 1000,
               'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 0.5, 'reg_lambda': 2.0, 'min_child_samples': 15},
    'v48':    {'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 500,
               'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_samples': 10},
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


def main():
    global t_start
    t_start = time.time()

    log.info("=" * 70)
    log.info("V330 — V329 Enhanced: Lag Features + Cross-Domain Ratios")
    log.info("=" * 70)

    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")

    train_df, test_df = build_all_features(train_df, test_df)

    train_feat_cols = get_feature_cols(train_df)
    test_feat_cols = get_feature_cols(test_df)

    log.info(f"Feature counts:")
    log.info(f"  Total: {len(train_feat_cols)}")

    group = train_df['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    n_train = len(train_df)
    n_test = len(test_df)

    test_preds = {t: np.zeros((n_test, N_SEEDS)) for t in TARGETS}
    all_seed_oofs = {t: [] for t in TARGETS}

    for t in TARGETS:
        log.info(f"\n{'='*60}")
        log.info(f"Target: {t}")
        y = train_df[t].values.astype(np.float64)
        feat_cols_clean = remove_leak(train_feat_cols, t)
        n_feat = V53_SWEEP[t]['n_feat']
        cfg_name = V53_SWEEP[t]['cfg']
        cfg = CFGS[cfg_name]

        ranked = rank_features(train_df, feat_cols_clean, t)
        candidate_feats = ranked

        for si in range(N_SEEDS):
            seed = SEED + si * 7

            rng = np.random.RandomState(seed)
            n_bag = max(int(len(candidate_feats) * FEATURE_BAG_FRACTION), n_feat)
            bag = rng.choice(candidate_feats, size=n_bag, replace=False)
            bag_set = set(bag)
            bag_feats = [f for f in ranked if f in bag_set][:n_feat]

            if len(bag_feats) < n_feat:
                remaining = [f for f in ranked if f not in bag_set][:n_feat - len(bag_feats)]
                bag_feats.extend(remaining)

            sel_cols = [c for c in bag_feats if c in test_feat_cols]

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
                seed_test += m.predict(test_df[sel_cols].fillna(0).values.astype(np.float64))

            seed_oof = np.clip(seed_oof, 0.001, 0.999)
            seed_test /= N_FOLDS
            all_seed_oofs[t].append(seed_oof.copy())
            test_preds[t][:, si] = seed_test

            if si < 3 or si == N_SEEDS - 1:
                s_oof = log_loss(y, seed_oof)
                log.info(f"    Seed {si:2d} (s{seed}): OOF={s_oof:.5f}")

    # LR meta-learner
    target_oofs = {}
    student_avg_oofs = {}
    meta_weights_info = {}

    for t in TARGETS:
        y = train_df[t].values.astype(np.float64)
        oof_matrix = np.column_stack(all_seed_oofs[t])

        meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta.fit(oof_matrix, y)

        train_pred = meta.predict_proba(oof_matrix)[:, 1]
        target_oofs[t] = log_loss(y, np.clip(train_pred, 0.001, 0.999))
        student_avg_oofs[t] = np.mean([log_loss(y, p) for p in all_seed_oofs[t]])
        meta_weights_info[t] = meta.coef_[0]

    avg_oof = np.mean(list(target_oofs.values()))
    avg_student = np.mean(list(student_avg_oofs.values()))

    log.info(f"\n{'='*70}")
    log.info(f"V330 RESULTS (V329 Enhanced — Lag + Cross-Domain)")
    log.info(f"{'='*70}")

    for t in TARGETS:
        gap = student_avg_oofs[t] - target_oofs[t]
        w_sum = np.sum(np.abs(meta_weights_info[t]))
        log.info(f"  {t}: OOF={target_oofs[t]:.5f} (student={student_avg_oofs[t]:.5f}, gap={gap:+.4f}, |W|={w_sum:.3f})")
    log.info(f"  AVG OOF: {avg_oof:.5f}")
    log.info(f"  AVG Student: {avg_student:.5f}")
    log.info(f"  V321: 0.60569 | V326: 0.59159 | V328: 0.58050 | V329: 0.57742")
    log.info(f"  Δ vs V321: {avg_oof - 0.60569:+.5f}")
    log.info(f"  Δ vs V326: {avg_oof - 0.59159:+.5f}")
    log.info(f"  Δ vs V329: {avg_oof - 0.57742:+.5f}")

    pred_lb = avg_oof + 0.019
    log.info(f"  Predicted LB: {pred_lb:.5f}")
    log.info(f"{'='*70}")

    # Build submission
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values

    for t in TARGETS:
        y = train_df[t].values.astype(np.float64)
        oof_matrix = np.column_stack(all_seed_oofs[t])
        meta_t = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta_t.fit(oof_matrix, y)
        sub[t] = meta_t.predict_proba(test_preds[t])[:, 1]

    sub_path = SUBMIT / f"submission_v330_lag_crossdomain_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")

    meta_data = {
        'version': 'V330',
        'name': 'V329 Enhanced — Lag Features + Cross-Domain Ratios',
        'avg_oof': round(float(avg_oof), 5),
        'avg_student_oof': round(float(avg_student), 5),
        'n_features_total': len(train_feat_cols),
        'n_seeds': N_SEEDS,
        'meta_c': META_C,
        'feature_bag_fraction': FEATURE_BAG_FRACTION,
        'v321_avg_oof': 0.60569,
        'v326_avg_oof': 0.59159,
        'v328_avg_oof': 0.58050,
        'v329_avg_oof': 0.57742,
        'delta_vs_v321': round(float(avg_oof - 0.60569), 5),
        'delta_vs_v326': round(float(avg_oof - 0.59159), 5),
        'delta_vs_v329': round(float(avg_oof - 0.57742), 5),
        'per_target_oof': {t: round(float(target_oofs[t]), 5) for t in TARGETS},
        'student_oof_avg': {t: round(float(student_avg_oofs[t]), 5) for t in TARGETS},
        'predicted_lb': round(float(pred_lb), 5),
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
        'key_difference': 'V329 + lag features (lag1, lag2, lag_diff) + cross-domain ratios + total activity proxy',
    }

    meta_path = EXPERIMENTS / f'v330_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")

    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return avg_oof, meta_data


if __name__ == '__main__':
    main()
