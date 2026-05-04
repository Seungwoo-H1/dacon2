"""
V25-GPU — Rolling Window + Season + Personalization (Final, GPU-accelerated)

GPU-accelerated version: device='gpu' in LGB_PARAMS.
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
N_TOP = 30

LGB_PARAMS = {
    'objective': 'binary', 'metric': 'binary_logloss',
    'num_leaves': 15, 'max_depth': 4,
    'learning_rate': 0.03, 'n_estimators': 500,
    'subsample': 0.7, 'colsample_bytree': 0.7,
    'reg_alpha': 1.0, 'reg_lambda': 3.0,
    'min_child_samples': 10,
    'force_row_wise': True, 'n_jobs': -1,
    'device': 'gpu',
    'verbose': -1,
}


def sanitize(name):
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)


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


def add_personalization(df, cols, stats=None):
    df = df.copy()
    new_cols = []
    computed_stats = {}
    for col in cols:
        if stats is None:
            subj_stats = df.groupby('subject_id')[col].agg(['mean', 'std']).reset_index()
            subj_stats.columns = ['subject_id', f'{col}_sm', f'{col}_ss']
            df = df.merge(subj_stats, on='subject_id', how='left')
            computed_stats[col] = {'mean': df[f'{col}_sm'].mean(), 'std': df[f'{col}_ss']}
        else:
            sm = stats[col]['mean']
            ss = stats[col]['std'].replace(0, 1e-6)
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

        global_mean = col in df.columns and df[col].mean() if computed_stats is None else df[col].mean()
        df[f'{col}_gmd'] = df[f'{col}_sm'] - df[col].mean()
        new_cols.append(f'{col}_gmd')
    return df, new_cols, computed_stats


def add_seasonal(df):
    df = df.copy()
    dates = pd.to_datetime(df['date'])
    df['day_of_year'] = dates.dt.dayofyear
    df['is_weekend'] = (dates.dt.dayofweek >= 5).astype(int)
    df['month'] = dates.dt.month
    df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    return df


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


def lgb_cv(feat, cols, target, seeds):
    y = feat[target].values
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_full = np.zeros((len(y), len(seeds)))
    all_fold_losses = {i: [] for i in range(N_SPLITS)}
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
            all_fold_losses[fold].append(log_loss(y[va_idx], pred))

    oof_avg = oof_full.mean(axis=1)
    return oof_avg


def simple_mean_match(pred, target_rate):
    shift = target_rate - pred.mean()
    return np.clip(pred + shift, 0.0001, 0.9999)


def main():
    log.info("=" * 70)
    log.info("V25 — Rolling Window + Season + Personalization")
    log.info("=" * 70)

    # ── Load ──
    feat = pd.read_parquet(DATA_PROCESSED / "features.parquet")
    log.info(f"Loaded: {feat.shape}")

    raw_cols = [c for c in feat.columns
                if c not in META_COLS | set(TARGET_COLS)
                and feat[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]

    # ── Clean ──
    clean_cols = [c for c in raw_cols if c not in CONSTANT_COLS and c not in COLLINEAR_DROP]
    log.info(f"Clean cols: {len(clean_cols)}")

    # wHr fix
    bad = (feat['wHr_hr_mean'] < 20) | (feat['wHr_hr_mean'] > 180)
    feat.loc[bad, 'wHr_hr_mean'] = np.nan
    feat.loc[bad, 'wHr_hr_std'] = np.nan

    # ── Rolling windows ──
    feat, roll_cols = add_rolling_windows(feat, clean_cols)
    log.info(f"Rolling cols: {len(roll_cols)}")

    # ── Personalization ──
    feat, pers_cols, pers_stats = add_personalization(feat, clean_cols)
    log.info(f"Personal cols: {len(pers_cols)}")

    # ── Seasonal ──
    feat = add_seasonal(feat)

    # Missing indicators
    all_num = [c for c in feat.columns
               if c not in META_COLS | set(TARGET_COLS)
               and feat[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]
    miss_cols = [c for c in all_num if feat[c].isnull().mean() > 0.05]
    for col in miss_cols:
        feat[f'{col}_missing'] = feat[col].isnull().astype(int)

    feat = feat.fillna(0)
    train_rate = {t: feat[t].mean() for t in TARGET_COLS}

    # ── Per-target CV with top-30 ──
    log.info("\n=== Per-target CV ===")
    all_oof = {}
    all_sel_cols = {}

    for target in TARGET_COLS:
        log.info(f"\n--- {target} ---")
        leak = LEAKAGE_FEATURES_S if target.startswith('S') else LEAKAGE_FEATURES_Q
        avail = [c for c in all_num if c not in leak]

        # Feature ranking
        ranked = rank_features(feat, avail, target)

        # Select top-30
        sel = [r[0] for r in ranked[:N_TOP]]
        log.info(f"  Selected {len(sel)} features")

        # CV
        oof = lgb_cv(feat, sel, target, RANDOM_SEEDS)

        # Calibrate
        cal = simple_mean_match(oof, train_rate[target])
        cal_loss = log_loss(feat[target], cal, labels=[0, 1])
        all_oof[target] = oof
        all_sel_cols[target] = sel

        log.info(f"  Cal OOF: {cal_loss:.4f}, mean shift: {cal.mean() - train_rate[target]:+.4f}")

    # ── Summary ──
    avg_cal = np.mean([log_loss(feat[t], simple_mean_match(all_oof[t], train_rate[t]), labels=[0, 1]) for t in TARGET_COLS])
    log.info(f"\n=== V25 Cal OOF Avg: {avg_cal:.4f} ===")

    # ── Submission ──
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

    # Load base features for test
    spec2 = importlib.util.spec_from_file_location("02_feature_engineering", Path('src/02_feature_engineering.py'))
    feat_eng = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(feat_eng)
    test_features = feat_eng.create_day_features(parquet_dfs, sample)

    # Apply same transformations
    test_cols = [c for c in test_features.columns
                 if c not in META_COLS | set(TARGET_COLS)
                 and test_features[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]
    test_cols = [c for c in test_cols if c not in CONSTANT_COLS]

    # Add rolling windows
    test_features, _ = add_rolling_windows(test_features, test_cols)

    # Add personalization using train stats
    for col in clean_cols:
        zcol = f'{col}_z'
        dcol = f'{col}_d'
        gcol = f'{col}_gmd'
        if zcol in test_features.columns:
            sm = pers_stats[col]['mean']
            ss = max(pers_stats[col]['std'], 1e-6)
            test_features[zcol] = (test_features[col] - sm) / ss
        if dcol in test_features.columns:
            test_features[dcol] = 0  # no history for test
        if gcol in test_features.columns:
            test_features[gcol] = pers_stats[col]['mean'] - pers_stats[col]['mean']  # = 0

    test_features = add_seasonal(test_features)
    for col in miss_cols:
        if col in test_features.columns:
            test_features[f'{col}_missing'] = test_features[col].isnull().astype(int)
    test_features = test_features.fillna(0)

    # Predict
    predictions = test_features[['subject_id', 'sleep_date', 'lifelog_date']].copy()

    for target in TARGET_COLS:
        sel = all_sel_cols[target]
        y_all = feat[target].values
        X_all = feat[sel].fillna(0).values
        test_X = test_features[sel].fillna(0).values
        sanitized = [sanitize(c) for c in sel]
        spw = ((y_all == 0).sum()) / max((y_all == 1).sum(), 1)

        all_preds = np.zeros(len(test_X))
        for seed in RANDOM_SEEDS:
            params = {**LGB_PARAMS, 'random_state': seed, 'scale_pos_weight': spw}
            ds = lgb.Dataset(X_all, label=y_all, feature_name=sanitized, params={'verbose': '-1'})
            mdl = lgb.train(params, ds, num_boost_round=500)
            all_preds += mdl.predict(test_X)

        all_preds /= N_SEEDS
        cal = simple_mean_match(all_preds, train_rate[target])
        predictions[target] = cal
        log.info(f"  {target}: mean={cal.mean():.4f}, train_rate={train_rate[target]:.3f}, shift={cal.mean()-train_rate[target]:+.4f}")

    # Save
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    sub_path = SUBMIT_DIR / f'submission_v25_{timestamp}.csv'
    predictions.to_csv(sub_path, index=False)
    log.info(f"✅ Saved: {sub_path}")

    meta = {'version': 'v25', 'submission_file': str(sub_path), 'timestamp': timestamp,
            'n_samples': len(predictions), 'n_seeds': N_SEEDS, 'n_splits': N_SPLITS,
            'features': {'base_cleaned': len(clean_cols), 'rolling_window': len(roll_cols),
                         'personalization': len(pers_cols), 'seasonal': 6,
                         'missing_indicator': len(miss_cols), 'n_top_per_target': N_TOP},
            'calibration': 'simple mean-matching + clip',
            'per_target': {}}
    for t in TARGET_COLS:
        cal_oof = log_loss(feat[t], simple_mean_match(all_oof[t], train_rate[t]), labels=[0, 1])
        meta['per_target'][t] = {'n_features': len(all_sel_cols[t]), 'cal_oof_loss': float(cal_oof),
                                  'cal_mean': float(predictions[t].mean()), 'train_rate': float(train_rate[t])}

    meta_path = SUBMIT_DIR / f'meta_v25_{timestamp}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)

    log.info(f"\n{'='*70}")
    log.info("V25 FINAL SUMMARY")
    log.info(f"{'='*70}")
    log.info(f"Submission: {sub_path}")
    log.info(f"{'Target':<6} {'Cal OOF':<12} {'Test Mean':<12} {'Train Rate':<12} {'Shift'}")
    for t in TARGET_COLS:
        cal_oof = log_loss(feat[t], simple_mean_match(all_oof[t], train_rate[t]), labels=[0, 1])
        log.info(f"{t:<6} {cal_oof:<12.4f} {predictions[t].mean():<12.4f} {train_rate[t]:<12.3f} {predictions[t].mean()-train_rate[t]:+.4f}")
    log.info(f"  Avg Cal OOF: {avg_cal:.4f}")

    return predictions


if __name__ == "__main__":
    main()
