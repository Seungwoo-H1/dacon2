"""
V336: Deep Feature Interactions (per-subject, domain-grouped)

Hypothesis: V329 has 2047 features but they're mostly additive (z-scores, rolling stats).
Real signal may come from interactions between features within the same domain.
E.g., pedo_mean × hr_mean captures activity-heart rate coupling.
V335 failed because it shared features across targets. V336 keeps per-target ranking
but adds domain-specific pairwise interactions.

Design:
1. Per domain (pedo, hr, light, screen, gps, ble), create pairwise interactions
   only between features within the same domain (not cross-domain).
2. Add domain cross-product: mean_pedo × mean_hr etc.
3. Keep all V329 features, just ADD these interactions.
4. Use per-target feature ranking to select best interactions.

Expected: Student OOF improves because interactions capture non-linear coupling.
Risk: LOW — same approach, just more interactions.
Cost: ~90s
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


def generate_base_features(train_df, test_df):
    """Generate V329-style features."""
    train_base = [c for c in train_df.columns
                  if c not in META_COLS | set(TARGETS)
                  and not c.endswith('_zscore')
                  and np.issubdtype(train_df[c].dtype, np.number)]
    test_base = [c for c in test_df.columns
                 if c not in META_COLS | set(TARGETS)
                 and not c.endswith('_zscore')
                 and np.issubdtype(test_df[c].dtype, np.number)]
    common_cols = set(train_base) & set(test_base)

    for col in common_cols:
        vals = train_df[col].fillna(0).values.astype(np.float64)
        mean = np.mean(vals)
        std = np.std(vals, ddof=0)
        if std < 1e-8:
            std = 1e-8
        zc = f'{col}_zscore'
        test_df = test_df.copy()
        test_df[zc] = (test_df[col].fillna(0).values.astype(np.float64) - mean) / std
        train_df = train_df.copy()
        train_df[zc] = (vals - mean) / std

    def add_interactions(df):
        df = df.copy()
        hr_cols = [c for c in df.columns if c.startswith('wHr_') and np.issubdtype(df[c].dtype, np.number)]
        pedo_cols = [c for c in df.columns if c.startswith('wPedo_') and np.issubdtype(df[c].dtype, np.number)]
        light_cols = [c for c in df.columns if c.startswith('mLight_') and np.issubdtype(df[c].dtype, np.number)]
        screen_cols = [c for c in df.columns if c.startswith('mScreenStatus_') and np.issubdtype(df[c].dtype, np.number)]
        gps_cols = [c for c in df.columns if c.startswith('mGps_') and np.issubdtype(df[c].dtype, np.number)]
        ble_cols = [c for c in df.columns if c.startswith('mBle_') and np.issubdtype(df[c].dtype, np.number)]
        wifi_cols = [c for c in df.columns if c.startswith('mWifi_') and np.issubdtype(df[c].dtype, np.number)]

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

        return df

    train_df = add_interactions(train_df)
    test_df = add_interactions(test_df)

    base_cols = [c for c in train_df.columns if c not in META_COLS | set(TARGETS) | {'date'}
                 and not c.endswith('_zscore') and not c.endswith('_interact')
                 and not c.endswith('_ratio')
                 and np.issubdtype(train_df[c].dtype, np.number)]
    base_cols = base_cols[:60]

    for col in base_cols:
        grp_train = train_df.groupby('subject_id')[col]
        grp_test = test_df.groupby('subject_id')[col]

        train_df[f'ps_roll3_mean_{col}'] = grp_train.transform(lambda g: g.rolling(3, min_periods=1).mean()).values
        train_df[f'ps_roll3_std_{col}'] = grp_train.transform(lambda g: g.rolling(3, min_periods=1).std().fillna(0)).values
        train_df[f'ps_roll5_mean_{col}'] = grp_train.transform(lambda g: g.rolling(5, min_periods=1).mean()).values
        train_df[f'ps_roll5_std_{col}'] = grp_train.transform(lambda g: g.rolling(5, min_periods=1).std().fillna(0)).values

        ps_mean = grp_train.transform('mean')
        ps_std = grp_train.transform('std').fillna(0)
        ps_min = grp_train.transform('min')
        ps_max = grp_train.transform('max')
        ps_median = grp_train.transform('median')
        ps_iqr = grp_train.transform(lambda g: g.quantile(0.75) - g.quantile(0.25))

        train_df[f'ps_min_{col}'] = ps_min.values
        train_df[f'ps_max_{col}'] = ps_max.values
        train_df[f'ps_median_{col}'] = ps_median.values
        train_df[f'ps_iqr_{col}'] = ps_iqr.values
        train_df[f'ps_range_{col}'] = (ps_max - ps_min).values
        train_df[f'ps_cv_{col}'] = (ps_std / (ps_mean.abs() + 1e-8)).values
        train_df[f'ps_maxmin_ratio_{col}'] = (ps_max / (ps_min.abs() + 1e-8)).values

        abs_dev = grp_train.transform(lambda g: (g - g.mean()).abs())
        sq_dev = grp_train.transform(lambda g: (g - g.mean()) ** 2)
        outliers = grp_train.transform(lambda g: (g - g.mean()).abs() > 2 * max(g.std(ddof=0), 1e-8)).astype(float)

        train_df[f'ps_absdev_{col}'] = abs_dev.values
        train_df[f'ps_sqdev_{col}'] = sq_dev.values
        train_df[f'ps_outliers_{col}'] = outliers.values

        test_df[f'ps_roll3_mean_{col}'] = grp_test.transform(lambda g: g.rolling(3, min_periods=1).mean()).values
        test_df[f'ps_roll3_std_{col}'] = grp_test.transform(lambda g: g.rolling(3, min_periods=1).std().fillna(0)).values
        test_df[f'ps_roll5_mean_{col}'] = grp_test.transform(lambda g: g.rolling(5, min_periods=1).mean()).values
        test_df[f'ps_roll5_std_{col}'] = grp_test.transform(lambda g: g.rolling(5, min_periods=1).std().fillna(0)).values

        ps_mean_t = grp_test.transform('mean')
        ps_std_t = grp_test.transform('std').fillna(0)
        ps_min_t = grp_test.transform('min')
        ps_max_t = grp_test.transform('max')
        ps_median_t = grp_test.transform('median')
        ps_iqr_t = grp_test.transform(lambda g: g.quantile(0.75) - g.quantile(0.25))

        test_df[f'ps_min_{col}'] = ps_min_t.values
        test_df[f'ps_max_{col}'] = ps_max_t.values
        test_df[f'ps_median_{col}'] = ps_median_t.values
        test_df[f'ps_iqr_{col}'] = ps_iqr_t.values
        test_df[f'ps_range_{col}'] = (ps_max_t - ps_min_t).values
        test_df[f'ps_cv_{col}'] = (ps_std_t / (ps_mean_t.abs() + 1e-8)).values
        test_df[f'ps_maxmin_ratio_{col}'] = (ps_max_t / (ps_min_t.abs() + 1e-8)).values

        abs_dev_t = grp_test.transform(lambda g: (g - g.mean()).abs())
        sq_dev_t = grp_test.transform(lambda g: (g - g.mean()) ** 2)
        outliers_t = grp_test.transform(lambda g: (g - g.mean()).abs() > 2 * max(g.std(ddof=0), 1e-8)).astype(float)

        test_df[f'ps_absdev_{col}'] = abs_dev_t.values
        test_df[f'ps_sqdev_{col}'] = sq_dev_t.values
        test_df[f'ps_outliers_{col}'] = outliers_t.values

    return train_df, test_df


def add_cross_subject_features(train_df, test_df):
    """Add cross-subject z-scores and domain aggregations."""
    log.info("Generating cross-subject features...")
    train_df = train_df.copy()
    test_df = test_df.copy()

    base_cols = [c for c in train_df.columns if c not in META_COLS | set(TARGETS) | {'date'}
                 and not c.endswith('_zscore') and not c.endswith('_interact')
                 and not c.endswith('_ratio')
                 and not c.startswith('ps_')
                 and np.issubdtype(train_df[c].dtype, np.number)]

    for col in base_cols[:80]:
        pop_mean = train_df[col].fillna(0).mean()
        pop_std = train_df[col].fillna(0).std(ddof=0)
        if pop_std < 1e-8:
            pop_std = 1e-8
        test_df[f'cs_zscore_{col}'] = (test_df[col].fillna(0) - pop_mean) / pop_std
        train_df[f'cs_zscore_{col}'] = (train_df[col].fillna(0) - pop_mean) / pop_std

    domains_dict = {
        'wHr': [c for c in train_df.columns if c.startswith('wHr_')],
        'wPedo': [c for c in train_df.columns if c.startswith('wPedo_')],
        'mLight': [c for c in train_df.columns if c.startswith('mLight_')],
        'mScreenStatus': [c for c in train_df.columns if c.startswith('mScreenStatus_')],
        'mGps': [c for c in train_df.columns if c.startswith('mGps_')],
        'mBle': [c for c in train_df.columns if c.startswith('mBle_')],
        'mWifi': [c for c in train_df.columns if c.startswith('mWifi_')],
        'mUsageStats': [c for c in train_df.columns if c.startswith('mUsageStats_')],
    }

    for domain_name, cols in domains_dict.items():
        if not cols:
            continue
        domain_base = [c for c in cols if c not in META_COLS | set(TARGETS) | {'date'}
                       and not c.endswith('_zscore') and not c.endswith('_interact')
                       and not c.endswith('_ratio') and not c.startswith('ps_')
                       and np.issubdtype(train_df[c].dtype, np.number)]
        if not domain_base:
            continue
        grp_train = train_df.groupby('subject_id')[domain_base]
        grp_test = test_df.groupby('subject_id')[domain_base]
        dm_train = grp_train.mean()
        ds_train = grp_train.std().fillna(0)
        dm_test = grp_test.mean()
        ds_test = grp_test.std().fillna(0)
        df_name = f'{domain_name.lower()}_domain'
        train_df[f'{df_name}_mean'] = dm_train.mean(axis=1).reindex(train_df['subject_id']).values
        train_df[f'{df_name}_std'] = ds_train.mean(axis=1).reindex(train_df['subject_id']).values
        test_df[f'{df_name}_mean'] = dm_test.mean(axis=1).reindex(test_df['subject_id']).values
        test_df[f'{df_name}_std'] = ds_test.mean(axis=1).reindex(test_df['subject_id']).values

    log.info(f"  Cross-subject z-scores: {len(base_cols[:80])}")
    log.info(f"  Domain aggregations: {len(domains_dict)}")
    return train_df, test_df


def add_domain_pairwise_interactions(train_df, test_df):
    """V336 KEY ADDITION: Domain-grouped pairwise interactions."""
    log.info("Generating domain pairwise interactions...")
    train_df = train_df.copy()
    test_df = test_df.copy()

    # Domain definitions
    domains = {
        'pedo': [c for c in train_df.columns if c.startswith('wPedo_') and np.issubdtype(train_df[c].dtype, np.number)],
        'hr': [c for c in train_df.columns if c.startswith('wHr_') and np.issubdtype(train_df[c].dtype, np.number)],
        'light': [c for c in train_df.columns if c.startswith('mLight_') and np.issubdtype(train_df[c].dtype, np.number)],
        'screen': [c for c in train_df.columns if c.startswith('mScreenStatus_') and np.issubdtype(train_df[c].dtype, np.number)],
        'gps': [c for c in train_df.columns if c.startswith('mGps_') and np.issubdtype(train_df[c].dtype, np.number)],
        'ble': [c for c in train_df.columns if c.startswith('mBle_') and np.issubdtype(train_df[c].dtype, np.number)],
        'wifi': [c for c in train_df.columns if c.startswith('mWifi_') and np.issubdtype(train_df[c].dtype, np.number)],
        'usage': [c for c in train_df.columns if c.startswith('mUsageStats_') and np.issubdtype(train_df[c].dtype, np.number)],
        'ac': [c for c in train_df.columns if c.startswith('mACStatus_') and np.issubdtype(train_df[c].dtype, np.number)],
        'activity': [c for c in train_df.columns if c.startswith('mActivity_') and np.issubdtype(train_df[c].dtype, np.number)],
    }

    n_new = 0
    for domain_name, cols in domains.items():
        if len(cols) < 2:
            continue
        # Pairwise: only top features per domain (limit to avoid explosion)
        # Take mean of domain features first, then pairwise between domains
        domain_mean_train = train_df[cols].fillna(0).mean(axis=1)
        domain_mean_test = test_df[cols].fillna(0).mean(axis=1)

        # Pairwise between all domain pairs
        for dname_a, dcols_a in domains.items():
            for dname_b, dcols_b in domains.items():
                if dname_a >= dname_b:
                    continue
                if not dcols_a or not dcols_b:
                    continue
                mean_a = train_df[dcols_a].fillna(0).mean(axis=1)
                mean_b = train_df[dcols_b].fillna(0).mean(axis=1)
                mean_a_t = test_df[dcols_b].fillna(0).mean(axis=1) if dcols_b else 0
                mean_b_t = test_df[dcols_a].fillna(0).mean(axis=1) if dcols_a else 0

                # Interaction: product of domain means
                inter_name = f'inter_{dname_a}_{dname_b}'
                train_df[inter_name] = (mean_a * mean_b).values
                test_df[inter_name] = (test_df[dcols_a].fillna(0).mean(axis=1) * test_df[dcols_b].fillna(0).mean(axis=1)).values
                n_new += 1

                # Ratio: mean_a / mean_b
                ratio_name = f'ratio_{dname_a}_{dname_b}'
                train_df[ratio_name] = (mean_a / (mean_b.abs() + 1e-8)).values
                test_df[ratio_name] = (test_df[dcols_a].fillna(0).mean(axis=1) / (test_df[dcols_b].fillna(0).mean(axis=1).abs() + 1e-8)).values
                n_new += 1

    log.info(f"  Domain pairwise interactions: {n_new}")
    return train_df, test_df


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
    log.info("V336 — Deep Feature Interactions (domain-grouped pairwise)")
    log.info("=" * 70)

    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")

    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')

    log.info("Generating base features (V329 level)...")
    train_df, test_df = generate_base_features(train_df, test_df)
    train_df, test_df = add_cross_subject_features(train_df, test_df)
    train_df, test_df = add_domain_pairwise_interactions(train_df, test_df)

    all_feat_cols = get_feature_cols(train_df)
    test_feat_cols = get_feature_cols(test_df)

    base_cols = [c for c in all_feat_cols if '_zscore' not in c
                 and 'ps_' not in c and '_interact' not in c and 'ratio' not in c
                 and 'domain' not in c and 'inter_' not in c]

    log.info(f"\nFeature counts:")
    log.info(f"  Base: {len(base_cols)}")
    log.info(f"  Total: {len(all_feat_cols)}")

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
        feat_cols_clean = remove_leak(all_feat_cols, t)
        n_feat = V53_SWEEP[t]['n_feat']
        cfg_name = V53_SWEEP[t]['cfg']
        cfg = CFGS[cfg_name]

        ranked = rank_features(train_df, feat_cols_clean, t)

        for si in range(N_SEEDS):
            seed = SEED + si * 7

            rng = np.random.RandomState(seed)
            n_bag = max(int(len(ranked) * FEATURE_BAG_FRACTION), n_feat)
            bag = rng.choice(ranked, size=n_bag, replace=False)
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
    log.info(f"V336 RESULTS — Deep Feature Interactions (domain pairwise)")
    log.info(f"{'='*70}")

    for t in TARGETS:
        gap = student_avg_oofs[t] - target_oofs[t]
        w_sum = np.sum(np.abs(meta_weights_info[t]))
        log.info(f"  {t}: OOF={target_oofs[t]:.5f} (student={student_avg_oofs[t]:.5f}, gap={gap:+.4f}, |W|={w_sum:.3f})")
    log.info(f"  AVG OOF: {avg_oof:.5f}")
    log.info(f"  AVG Student: {avg_student:.5f}")
    log.info(f"  V329_cross_ps: 0.54365 | V329_cross_ps student: 0.64698")
    log.info(f"  Δ vs V329: {avg_oof - 0.54365:+.5f}")
    log.info(f"  Δ student vs V329: {avg_student - 0.64698:+.5f}")

    pred_lb = avg_oof + 0.019
    log.info(f"  Predicted LB: {pred_lb:.5f}")
    log.info(f"{'='*70}")

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

    sub_path = SUBMIT / f"submission_v336_domain_interactions_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")

    meta_data = {
        'version': 'V336',
        'name': 'Deep Feature Interactions — Domain-grouped pairwise',
        'avg_oof': round(float(avg_oof), 5),
        'avg_student_oof': round(float(avg_student), 5),
        'n_features_total': len(all_feat_cols),
        'n_features_base': len(base_cols),
        'n_seeds': N_SEEDS,
        'meta_c': META_C,
        'feature_bag_fraction': FEATURE_BAG_FRACTION,
        'delta_vs_v329': round(float(avg_oof - 0.54365), 5),
        'delta_student_vs_v329': round(float(avg_student - 0.64698), 5),
        'per_target_oof': {t: round(float(target_oofs[t]), 5) for t in TARGETS},
        'student_oof_avg': {t: round(float(student_avg_oofs[t]), 5) for t in TARGETS},
        'predicted_lb': round(float(pred_lb), 5),
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
        'key_difference': 'V329 + domain pairwise interactions (product + ratio between all domain pairs)',
    }

    meta_path = EXPERIMENTS / f'v336_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")

    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return avg_oof, meta_data


if __name__ == '__main__':
    main()
