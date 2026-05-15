"""
V31 — Improved ensemble: V10 per-target tuning + V25 rolling windows + XGBoost diversity

Key improvements over V10 (cal OOF 0.6038 / LB 0.66):
1. Rolling windows (3, 7) from V25 — captures temporal dynamics
2. V10-style per-target hyperparameter search (better fitting)
3. LightGBM + XGBoost ensemble — diverse tree structures
4. Same leakage fix, personalization, calibration as V10 (proven safe)

Expected: 0.55-0.58 test score
"""

import sys
import re
import json
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

sys.path.insert(0, 'src')
from config import TARGETS, DATA_PROCESSED, MODEL_DIR, SUBMIT_DIR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data_raw"

TARGET_COLS = TARGETS
META_COLS = {"subject_id", "lifelog_date", "sleep_date", "date"}

LEAKAGE_FEATURES_S = {
    'wLight_w_light_mean', 'wLight_w_light_std', 'wLight_w_light_min', 'wLight_w_light_max', 'wLight_w_light_count',
    'wHr_hr_mean', 'wHr_hr_std', 'wHr_hr_min', 'wHr_hr_max', 'wHr_hr_median', 'wHr_hr_count',
    'wPedo_pedo_step_mean', 'wPedo_pedo_step_sum',
    'wPedo_pedo_step_frequency_mean', 'wPedo_pedo_step_frequency_sum',
    'wPedo_pedo_running_step_mean', 'wPedo_pedo_running_step_sum',
    'wPedo_pedo_walking_step_mean', 'wPedo_pedo_walking_step_sum',
    'wPedo_pedo_distance_mean', 'wPedo_pedo_distance_sum',
    'wPedo_pedo_speed_mean', 'wPedo_pedo_speed_sum',
    'wPedo_pedo_burned_calories_mean', 'wPedo_pedo_burned_calories_sum',
}
LEAKAGE_FEATURES_Q = {
    'wHr_hr_mean', 'wHr_hr_std', 'wHr_hr_min', 'wHr_hr_max', 'wHr_hr_median', 'wHr_hr_count',
}

RANDOM_SEEDS = [42, 123, 456, 789, 1024, 1337, 2048, 3037, 4096, 5001,
                6000, 7123, 8001, 9000, 10000, 11111, 12000, 13001, 14000, 15001]
N_SEEDS = len(RANDOM_SEEDS)
N_SPLITS = 5

LGB_PARAMS = {
    'objective': 'binary', 'metric': 'binary_logloss',
    'num_leaves': 15, 'max_depth': 4,
    'learning_rate': 0.03, 'n_estimators': 500,
    'subsample': 0.7, 'colsample_bytree': 0.7,
    'reg_alpha': 1.0, 'reg_lambda': 3.0,
    'min_child_samples': 10,
    'force_row_wise': True, 'n_jobs': -1,
    'verbose': -1,
}

CONSTANT_COLS = [
    'mACStatus_m_charging_min', 'mACStatus_m_charging_max',
    'mLight_m_light_min',
    'mScreenStatus_m_screen_use_min', 'mScreenStatus_m_screen_use_max',
    'wPedo_pedo_running_step_mean', 'wPedo_pedo_running_step_sum',
    'wPedo_pedo_walking_step_mean', 'wPedo_pedo_walking_step_sum',
    'mGps_gps_has_speed_mean', 'mGps_gps_has_speed_std',
    'mGps_gps_has_speed_max', 'mGps_gps_has_speed_min',
    'mUsageStats_usage_major_ratio_min', 'mUsageStats_usage_game_ratio_min',
]
COLLINEAR_DROP = [
    'wPedo_pedo_step_frequency_mean', 'wPedo_pedo_step_frequency_sum',
    'mBle_ble_device_count_mean', 'mBle_ble_device_count_std', 'mBle_ble_device_count_max',
    'mWifi_wifi_bssid_count_mean', 'mWifi_wifi_bssid_count_std', 'mWifi_wifi_bssid_count_max',
]


def sanitize(name):
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)


# ── Rolling windows ─────────────────────────────────────
def add_rolling_windows(df, cols, windows=[3, 7]):
    df = df.copy().sort_values(['subject_id', 'date'])
    new_cols = []
    for col in cols:
        grp = df.groupby('subject_id')[col]
        for w in windows:
            rm = grp.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
            rs = grp.rolling(w, min_periods=1).std().fillna(0).reset_index(level=0, drop=True)
            col_rm, col_rs = f'{col}_rm{w}', f'{col}_rs{w}'
            df[col_rm] = rm.values
            df[col_rs] = rs.values
            new_cols.extend([col_rm, col_rs])
    return df, new_cols


# ── Seasonal ────────────────────────────────────────────
def add_seasonal(df):
    df = df.copy()
    dates = pd.to_datetime(df['date'])
    df['day_of_year'] = dates.dt.dayofyear
    df['is_weekend'] = (dates.dt.dayofweek >= 5).astype(int)
    df['month'] = dates.dt.month
    df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    return df


# ── Personalization (z-score per subject) ───────────────
def add_personalization(df, cols, stats=None):
    df = df.copy()
    new_cols = []
    computed_stats = {}
    for col in cols:
        if stats is None:
            subj_stats = df.groupby('subject_id')[col].agg(['mean', 'std']).reset_index()
            subj_stats.columns = ['subject_id', f'{col}_sm', f'{col}_ss']
            df = df.merge(subj_stats, on='subject_id', how='left')
            computed_stats[col] = {'mean': float(df[f'{col}_sm'].mean()), 'std': float(df[f'{col}_ss'].std())}
        else:
            sm = stats[col]['mean']
            ss = max(stats[col]['std'], 1e-6)
            df[f'{col}_sm'] = sm
            df[f'{col}_ss'] = ss

        mask = df[f'{col}_ss'] > 0
        df.loc[mask, f'{col}_z'] = (df.loc[mask, col] - df.loc[mask, f'{col}_sm']) / df.loc[mask, f'{col}_ss']
        df.loc[~mask, f'{col}_z'] = 0
        new_cols.append(f'{col}_z')

        df_sorted = df.sort_values(['subject_id', 'date'])
        df_sorted[f'{col}_d'] = df_sorted.groupby('subject_id')[col].diff()
        df[f'{col}_d'] = df_sorted[f'{col}_d'].fillna(0).values
        new_cols.append(f'{col}_d')

        global_mean = df[col].mean()
        df[f'{col}_gmd'] = df[f'{col}_sm'] - global_mean
        new_cols.append(f'{col}_gmd')
    return df, new_cols, computed_stats


# ── Feature ranking ─────────────────────────────────────
def rank_features(feat, cols, target):
    y = feat[target].values
    X = feat[cols].fillna(0).values
    n_pos = max((y == 1).sum(), 1)
    n_neg = (y == 0).sum()
    spw = n_neg / n_pos
    params = {**LGB_PARAMS, 'num_leaves': 15, 'max_depth': 4, 'n_estimators': 100,
              'scale_pos_weight': spw, 'random_state': 42}
    sanitized = [sanitize(c) for c in cols]
    ds = lgb.Dataset(X, label=y, feature_name=sanitized, params={'verbose': '-1'})
    mdl = lgb.train(params, ds, num_boost_round=100)
    imp = mdl.feature_importance(importance_type='gain')
    return sorted(zip(cols, imp), key=lambda x: -x[1])


# ── LightGBM CV ─────────────────────────────────────────
def lgb_cv(feat, cols, target, seeds):
    y = feat[target].values
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_full = np.zeros((len(y), len(seeds)))
    spw = ((y == 0).sum()) / max((y == 1).sum(), 1)
    sanitized = [sanitize(c) for c in cols]

    for si, seed in enumerate(seeds):
        cfg = {**LGB_PARAMS, 'random_state': seed, 'scale_pos_weight': spw}
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(feat, y, feat['subject_id'])):
            X_tr = feat.iloc[tr_idx][cols].fillna(0).values
            X_va = feat.iloc[va_idx][cols].fillna(0).values
            ds = lgb.Dataset(X_tr, label=y[tr_idx], feature_name=sanitized, params={'verbose': '-1'})
            vad = lgb.Dataset(X_va, label=y[va_idx], feature_name=sanitized, reference=ds, params={'verbose': '-1'})
            mdl = lgb.train(cfg, ds, num_boost_round=500, valid_sets=[vad],
                           callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            pred = mdl.predict(X_va)
            oof_full[va_idx, si] = pred

    oof_avg = oof_full.mean(axis=1)
    return oof_avg


# ── Calibration ─────────────────────────────────────────
def simple_mean_match(pred, target_rate):
    shift = target_rate - pred.mean()
    return np.clip(pred + shift, 0.0001, 0.9999)


# ── Per-target hyperparameter search (V10-style) ────────
def tune_target(feat, feature_cols, target):
    configs = [
        {'name': 'C1', 'nl': 8, 'md': 3, 'lr': 0.02, 'ne': 200, 'ss': 0.6, 'cst': 0.6, 'ra': 2.0, 'rl': 5.0, 'mc': 15},
        {'name': 'C2', 'nl': 10, 'md': 3, 'lr': 0.03, 'ne': 300, 'ss': 0.7, 'cst': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
        {'name': 'C3', 'nl': 12, 'md': 4, 'lr': 0.03, 'ne': 200, 'ss': 0.7, 'cst': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
        {'name': 'C4', 'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 500, 'ss': 0.7, 'cst': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
        {'name': 'C5', 'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 300, 'ss': 0.7, 'cst': 0.7, 'ra': 0.5, 'rl': 2.0, 'mc': 8},
        {'name': 'C6', 'nl': 6, 'md': 2, 'lr': 0.02, 'ne': 200, 'ss': 0.5, 'cst': 0.5, 'ra': 5.0, 'rl': 10.0, 'mc': 20},
    ]

    best_config = None
    best_cv = float('inf')
    best_oof = None
    best_selected_cols = None

    ranked = rank_features(feat, feature_cols, target)

    for n_feat in [10, 20, 30]:
        selected_cols = [r[0] for r in ranked[:n_feat]]

        y = feat[target].values
        n_pos = max((y == 1).sum(), 1)
        n_neg = (y == 0).sum()
        spw = n_neg / n_pos

        for cfg in configs:
            test_cfg = {**LGB_PARAMS,
                       'num_leaves': cfg['nl'], 'max_depth': cfg['md'],
                       'learning_rate': cfg['lr'], 'n_estimators': cfg['ne'],
                       'subsample': cfg['ss'], 'colsample_bytree': cfg['cst'],
                       'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
                       'min_child_samples': cfg['mc'],
            }

            # Quick CV with subset of seeds to save time
            quick_seeds = RANDOM_SEEDS[:10]
            oof_avg = lgb_cv(feat, selected_cols, target, quick_seeds)

            oof_loss = log_loss(y, oof_avg, labels=[0, 1])
            score = oof_loss

            if score < best_cv:
                best_cv = score
                best_config = {**cfg, '_n_feats': n_feat}
                best_oof = oof_avg
                best_selected_cols = selected_cols

    return best_config, best_selected_cols, best_oof


# ── Main pipeline ───────────────────────────────────────
def main():
    log.info("=" * 70)
    log.info("V31 — V10 tuning + V25 rolling + XGBoost ensemble")
    log.info("=" * 70)

    # ── 1. Load features ───────────────────────────────────
    feat = pd.read_parquet(DATA_PROCESSED / "features.parquet")
    log.info(f"Loaded: {feat.shape}")

    all_num = [c for c in feat.columns
               if c not in META_COLS | set(TARGET_COLS)
               and feat[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]

    # wHr fix
    bad = (feat['wHr_hr_mean'] < 20) | (feat['wHr_hr_mean'] > 180)
    feat.loc[bad, 'wHr_hr_mean'] = np.nan
    feat.loc[bad, 'wHr_hr_std'] = np.nan

    # Clean cols
    clean_cols = [c for c in all_num if c not in CONSTANT_COLS and c not in COLLINEAR_DROP]
    log.info(f"Base clean cols: {len(clean_cols)}")

    # ── 2. Add rolling windows ────────────────────────────
    feat, roll_cols = add_rolling_windows(feat, clean_cols)
    log.info(f"Rolling cols: {len(roll_cols)}")

    # ── 3. Add personalization ────────────────────────────
    feat, pers_cols, pers_stats = add_personalization(feat, clean_cols)
    log.info(f"Personalization cols: {len(pers_cols)}")

    # ── 4. Add seasonal ───────────────────────────────────
    feat = add_seasonal(feat)

    # ── 5. Missing indicators ─────────────────────────────
    all_avail = [c for c in feat.columns
                 if c not in META_COLS | set(TARGET_COLS)
                 and feat[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]
    miss_cols = [c for c in all_avail if feat[c].isnull().mean() > 0.05]
    for col in miss_cols:
        feat[f'{col}_missing'] = feat[col].isnull().astype(int)

    feat = feat.fillna(0)
    train_rate = {t: feat[t].mean() for t in TARGET_COLS}
    log.info(f"Target rates: {train_rate}")

    # ── 6. Per-target tuning + CV ─────────────────────────
    log.info("\n=== Per-target CV + XGBoost ===")

    # --- LightGBM with V10-style tuning ---
    lgb_oof = {}
    lgb_sel = {}
    lgb_config = {}
    for target in TARGET_COLS:
        log.info(f"\n  --- {target} ---")
        leak = LEAKAGE_FEATURES_S if target.startswith('S') else LEAKAGE_FEATURES_Q
        avail = [c for c in all_avail if c not in leak]

        best_config, best_cols, best_oof = tune_target(feat, avail, target)
        lgb_oof[target] = best_oof
        lgb_sel[target] = best_cols
        lgb_config[target] = best_config

        cal = simple_mean_match(best_oof, train_rate[target])
        cal_loss = log_loss(feat[target], cal, labels=[0, 1])
        log.info(f"    LGBM config: {best_config}")
        log.info(f"    LGBM cal OOF: {cal_loss:.4f}, shift: {cal.mean()-train_rate[target]:+.4f}")

    # --- XGBoost OOF ---
    try:
        import xgboost as xgb
        USE_XGB = True
        log.info("\n  XGBoost available")
    except ImportError:
        USE_XGB = False
        log.info("\n  XGBoost NOT available")

    xgb_oof = {}
    if USE_XGB:
        XGB_PARAMS = {
            'tree_method': 'hist', 'objective': 'binary:logistic',
            'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 500,
            'subsample': 0.7, 'colsample_bytree': 0.7,
            'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_weight': 10,
            'random_state': 42, 'verbosity': 0, 'n_jobs': -1,
        }
        gkf = GroupKFold(n_splits=N_SPLITS)

        for target in TARGET_COLS:
            leak = LEAKAGE_FEATURES_S if target.startswith('S') else LEAKAGE_FEATURES_Q
            avail = [c for c in all_avail if c not in leak]
            ranked = rank_features(feat, avail, target)
            sel = [r[0] for r in ranked[:30]]

            y = feat[target].values
            X = feat[sel].fillna(0).values
            oof_xgb = np.zeros(len(y))
            spw = ((y == 0).sum()) / max((y == 1).sum(), 1)

            for seed_i, seed in enumerate(RANDOM_SEEDS):
                cfg = {**XGB_PARAMS, 'random_state': seed, 'scale_pos_weight': spw}
                for fold, (tr_idx, va_idx) in enumerate(gkf.split(feat, y, feat['subject_id'])):
                    clf = xgb.XGBClassifier(**cfg)
                    clf.fit(X[tr_idx], y[tr_idx],
                           eval_set=[(X[va_idx], y[va_idx])], verbose=False)
                    oof_xgb[va_idx] += clf.predict_proba(X[va_idx])[:, 1]
                oof_xgb /= N_SPLITS

            xgb_oof[target] = oof_xgb / N_SEEDS
            cal = simple_mean_match(xgb_oof[target], train_rate[target])
            cal_loss = log_loss(feat[target], cal, labels=[0, 1])
            log.info(f"    XGB {target} cal OOF: {cal_loss:.4f}")

    # ── 7. Ensemble weight search ─────────────────────────
    log.info("\n=== Ensemble search ===")
    best_weight = 0.5
    best_score = float('inf')

    if USE_XGB:
        for w_lgb in [0.3, 0.4, 0.5, 0.6, 0.7]:
            score = 0
            for target in TARGET_COLS:
                ens = w_lgb * lgb_oof[target] + (1 - w_lgb) * xgb_oof[target]
                cal = simple_mean_match(ens, train_rate[target])
                score += log_loss(feat[target], cal, labels=[0, 1])
            score /= len(TARGET_COLS)
            log.info(f"    w_lgb={w_lgb:.1f}: avg cal OOF = {score:.4f}")
            if score < best_score:
                best_score = score
                best_weight = w_lgb

        final_oof = {t: best_weight * lgb_oof[t] + (1-best_weight) * xgb_oof[t] for t in TARGET_COLS}
        final_cal = {t: simple_mean_match(final_oof[t], train_rate[t]) for t in TARGET_COLS}
        log.info(f"  Best: LGBM={best_weight:.1f}, XGB={1-best_weight:.1f}, OOF={best_score:.4f}")
    else:
        final_oof = {t: lgb_oof[t] for t in TARGET_COLS}
        final_cal = {t: simple_mean_match(lgb_oof[t], train_rate[t]) for t in TARGET_COLS}
        best_score = np.mean([log_loss(feat[t], final_cal[t], labels=[0,1]) for t in TARGET_COLS])
        log.info(f"  No XGBoost, using LGBM only, OOF={best_score:.4f}")

    # ── 8. Generate submission ─────────────────────────────
    log.info("\n=== Generating submission ===")
    import importlib.util

    spec = importlib.util.spec_from_file_location("01_load_data", Path('src/01_load_data.py'))
    ld_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ld_mod)

    sample = pd.read_csv('data_raw/ch2026_submission_sample.csv')
    sample['lifelog_date'] = pd.to_datetime(sample['lifelog_date']).dt.date
    sample['sleep_date'] = pd.to_datetime(sample['sleep_date']).dt.date
    test_dates = set(sample["sleep_date"].astype(str).tolist() + sample["lifelog_date"].astype(str).tolist())

    parquet_names = {
        "mACStatus": "ch2025_mACStatus.parquet", "mActivity": "ch2025_mActivity.parquet",
        "mAmbience": "ch2025_mAmbience.parquet", "mBle": "ch2025_mBle.parquet",
        "mGps": "ch2025_mGps.parquet", "mLight": "ch2025_mLight.parquet",
        "mScreenStatus": "ch2025_mScreenStatus.parquet", "mUsageStats": "ch2025_mUsageStats.parquet",
        "mWifi": "ch2025_mWifi.parquet", "wHr": "ch2025_wHr.parquet",
        "wLight": "ch2025_wLight.parquet", "wPedo": "ch2025_wPedo.parquet",
    }
    parquet_dfs = {}
    for name, fname in parquet_names.items():
        path = DATA_RAW / "ch2025_data_items" / fname
        if path.exists():
            df = pd.read_parquet(path)
            df = ld_mod.build_merge_key(df)
            df = df[df["date"].astype(str).isin(test_dates)]
            parquet_dfs[name] = df

    spec2 = importlib.util.spec_from_file_location("02_feature_engineering", Path('src/02_feature_engineering.py'))
    feat_eng = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(feat_eng)

    test_features = feat_eng.create_day_features(parquet_dfs, sample)
    log.info(f"Test features: {test_features.shape}")

    test_cols = [c for c in test_features.columns
                 if c not in META_COLS | set(TARGET_COLS)
                 and test_features[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]
    test_cols = [c for c in test_cols if c not in CONSTANT_COLS]

    # Apply same transformations
    test_features = add_rolling_windows(test_features, test_cols)
    test_features, _, _ = add_personalization(test_features, clean_cols, pers_stats)
    test_features = add_seasonal(test_features)
    for col in miss_cols:
        if col in test_features.columns:
            test_features[f'{col}_missing'] = test_features[col].isnull().astype(int)
    test_features = test_features.fillna(0)

    # Predict
    predictions = test_features[['subject_id', 'sleep_date', 'lifelog_date']].copy()

    # LightGBM
    lgb_test_preds = {}
    for target in TARGET_COLS:
        sel = lgb_sel[target]
        y_all = feat[target].values
        X_all = feat[sel].fillna(0).values
        test_X = test_features[sel].fillna(0).values
        sanitized = [sanitize(c) for c in sel]
        spw = ((y_all == 0).sum()) / max((y_all == 1).sum(), 1)
        cfg = lgb_config[target]
        lgb_p = {**LGB_PARAMS, 'num_leaves': cfg['nl'], 'max_depth': cfg['md'],
                 'learning_rate': cfg['lr'], 'n_estimators': cfg['ne'],
                 'subsample': cfg['ss'], 'colsample_bytree': cfg['cst'],
                 'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
                 'min_child_samples': cfg['mc'], 'scale_pos_weight': spw}

        all_preds = np.zeros(len(test_X))
        for seed in RANDOM_SEEDS:
            params = {**lgb_p, 'random_state': seed}
            ds = lgb.Dataset(X_all, label=y_all, feature_name=sanitized, params={'verbose': '-1'})
            mdl = lgb.train(params, ds, num_boost_round=cfg['ne'])
            all_preds += mdl.predict(test_X)
        all_preds /= N_SEEDS
        lgb_test_preds[target] = all_preds

    # XGBoost
    xgb_test_preds = {}
    if USE_XGB:
        XGB_FULL = {**XGB_PARAMS, 'n_jobs': -1}
        for target in TARGET_COLS:
            sel = lgb_sel[target]  # use same features
            y_all = feat[target].values
            X_all = feat[sel].fillna(0).values
            test_X = test_features[sel].fillna(0).values
            spw = ((y_all == 0).sum()) / max((y_all == 1).sum(), 1)

            all_preds = np.zeros(len(test_X))
            for seed in RANDOM_SEEDS:
                cfg = {**XGB_FULL, 'random_state': seed, 'scale_pos_weight': spw}
                clf = xgb.XGBClassifier(**cfg)
                clf.fit(X_all, y_all, verbose=False)
                all_preds += clf.predict_proba(test_X)[:, 1]
            all_preds /= N_SEEDS
            xgb_test_preds[target] = all_preds

    # Ensemble
    if USE_XGB:
        for target in TARGET_COLS:
            ens = best_weight * lgb_test_preds[target] + (1-best_weight) * xgb_test_preds[target]
            cal = simple_mean_match(ens, train_rate[target])
            predictions[target] = cal
    else:
        for target in TARGET_COLS:
            cal = simple_mean_match(lgb_test_preds[target], train_rate[target])
            predictions[target] = cal

    # ── Save ───────────────────────────────────────────────
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    sub_path = SUBMIT_DIR / f'submission_v31_{timestamp}.csv'
    predictions.to_csv(sub_path, index=False)
    log.info(f"\n✅ Submission saved: {sub_path}")

    meta = {
        'version': 'v31',
        'submission_file': str(sub_path),
        'timestamp': timestamp,
        'n_samples': len(predictions),
        'models': ['LightGBM (per-target tuned)' + (' + XGBoost' if USE_XGB else '')],
        'ensemble_weight_lgbm': best_weight if USE_XGB else 1.0,
        'n_seeds': N_SEEDS, 'n_splits': N_SPLITS,
        'features': {'base': len(clean_cols), 'rolling': len(roll_cols), 'personalization': len(pers_cols),
                     'seasonal': 6, 'missing': len(miss_cols)},
        'calibration': 'simple mean-matching + clip',
        'cal_oof_score': float(best_score),
        'per_target': {},
    }
    for t in TARGET_COLS:
        meta['per_target'][t] = {
            'config': lgb_config[t],
            'n_features': len(lgb_sel[t]),
            'cal_oof_loss': float(log_loss(feat[t], final_cal[t], labels=[0, 1])),
            'cal_mean': float(predictions[t].mean()),
            'train_rate': float(train_rate[t]),
        }
        log.info(f"  {t}: cal_OOF={meta['per_target'][t]['cal_oof_loss']:.4f}, mean={predictions[t].mean():.4f}")

    meta_path = sub_path.parent / f'meta_v31_{timestamp}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)

    log.info(f"\n{'='*70}")
    log.info("V31 FINAL")
    log.info(f"{'='*70}")
    log.info(f"Submission: {sub_path}")
    log.info(f"{'Target':<6} {'Cal OOF':<12} {'Test Mean':<12} {'Train Rate':<12} {'Shift'}")
    for t in TARGET_COLS:
        cal_oof = log_loss(feat[t], final_cal[t], labels=[0,1])
        log.info(f"{t:<6} {cal_oof:<12.4f} {predictions[t].mean():<12.4f} {train_rate[t]:<12.3f} {predictions[t].mean()-train_rate[t]:+.4f}")
    log.info(f"  AVG Cal OOF: {best_score:.4f}")


if __name__ == "__main__":
    main()
