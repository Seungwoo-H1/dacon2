"""
V26 — Exhaustive Ablation Study (Fast)

Factors:
  A: base_only
  B: +rolling (3d, 7d)
  C: +personal (z, delta, gmd)
  D: +seasonal (is_weekend, month, doy_sin, doy_cos)

Configs: all 2^4 = 16 combos (empty config skipped)
Top-N: 20, 30, 50 only
Seeds: 10
Spds: 5-fold CV

Optimized: feature ranking uses n_estimators=50 (fast), CV uses 5 seeds for ranking, 10 for scoring.
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
from config import TARGETS, DATA_PROCESSED, SUBMIT_DIR

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
LEAKAGE_S = {
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
LEAKAGE_Q = {
    'wHr_hr_mean', 'wHr_hr_std', 'wHr_hr_min', 'wHr_hr_max', 'wHr_hr_median', 'wHr_hr_count',
}

SEEDS = [42, 123, 456, 789, 1024, 1337, 2048, 3037, 4096, 5001]
N_SEEDS = len(SEEDS)
N_SPLITS = 5
N_TOP_VALUES = [20, 30, 50]

LGB_BASE = {
    'objective': 'binary', 'metric': 'binary_logloss',
    'num_leaves': 15, 'max_depth': 4,
    'learning_rate': 0.03, 'n_estimators': 500,
    'subsample': 0.7, 'colsample_bytree': 0.7,
    'reg_alpha': 1.0, 'reg_lambda': 3.0,
    'min_child_samples': 10,
    'force_row_wise': True, 'n_jobs': -1,
    'verbose': -1,
}


def sanitize(name):
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)


def add_rolling(df, cols):
    """Add rolling mean/std for 3-day and 7-day windows."""
    df = df.copy().sort_values(['subject_id', 'date'])
    new = []
    for c in cols:
        grp = df.groupby('subject_id')[c]
        for w in [3, 7]:
            rm = grp.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
            rs = grp.rolling(w, min_periods=1).std().fillna(0).reset_index(level=0, drop=True)
            n_rm, n_rs = f'{c}_rm{w}', f'{c}_rs{w}'
            df[n_rm] = rm.values
            df[n_rs] = rs.values
            new.extend([n_rm, n_rs])
    return df, new


def add_personal(df, cols):
    """Add per-subject z-score, delta, global-mean-deviation."""
    df = df.copy()
    new = []
    for c in cols:
        st = df.groupby('subject_id')[c].agg(['mean', 'std']).reset_index()
        st.columns = ['subject_id', f'{c}_sm', f'{c}_ss']
        df = df.merge(st, on='subject_id', how='left')
        mask = df[f'{c}_ss'] > 0
        df.loc[mask, f'{c}_z'] = (df.loc[mask, c] - df.loc[mask, f'{c}_sm']) / df.loc[mask, f'{c}_ss']
        df.loc[~mask, f'{c}_z'] = 0
        new.append(f'{c}_z')
        ds = df.sort_values(['subject_id', 'date'])
        ds[f'{c}_d'] = ds.groupby('subject_id')[c].diff()
        df[f'{c}_d'] = ds[f'{c}_d'].fillna(0).values
        new.append(f'{c}_d')
        gm = df[c].mean()
        df[f'{c}_gmd'] = df[f'{c}_sm'] - gm
        new.append(f'{c}_gmd')
    return df, new


def add_seasonal(df):
    """Add seasonal features."""
    df = df.copy()
    d = pd.to_datetime(df['date'])
    df['is_weekend'] = (d.dt.dayofweek >= 5).astype(int)
    df['month'] = d.dt.month
    df['doy_sin'] = np.sin(2 * np.pi * d.dt.dayofyear / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * d.dt.dayofyear / 365.25)
    return df


def rank_feat(feat, cols, target):
    """Quick feature ranking (50 iterations)."""
    y = feat[target].values
    X = feat[cols].fillna(0).values
    n_pos = max((y == 1).sum(), 1)
    n_neg = (y == 0).sum()
    spw = n_neg / n_pos
    params = {**LGB_BASE, 'num_leaves': 15, 'max_depth': 4, 'n_estimators': 50,
              'scale_pos_weight': spw, 'random_state': 42}
    sn = [sanitize(c) for c in cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn, params={'verbose': '-1'})
    m = lgb.train(params, ds, num_boost_round=50)
    imp = m.feature_importance(importance_type='gain')
    return sorted(zip(cols, imp), key=lambda x: -x[1])


def lgb_cv(feat, cols, target):
    """5-fold CV with 10 seeds."""
    y = feat[target].values
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_full = np.zeros((len(y), N_SEEDS))
    spw = ((y == 0).sum()) / max((y == 1).sum(), 1)
    sn = [sanitize(c) for c in cols]
    X = feat[cols].fillna(0).values
    for si, seed in enumerate(SEEDS):
        cfg = {**LGB_BASE, 'random_state': seed, 'scale_pos_weight': spw}
        for fold, (tr_i, va_i) in enumerate(gkf.split(feat, y, feat['subject_id'])):
            ds = lgb.Dataset(X[tr_i], label=y[tr_i], feature_name=sn, params={'verbose': '-1'})
            vad = lgb.Dataset(X[va_i], label=y[va_i], feature_name=sn, reference=ds, params={'verbose': '-1'})
            m = lgb.train(cfg, ds, num_boost_round=500, valid_sets=[vad],
                         callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            oof_full[va_i, si] = m.predict(X[va_i])
    return oof_full.mean(axis=1)


def mean_match(pred, rate):
    return np.clip(pred + (rate - pred.mean()), 0.0001, 0.9999)


# ── Main ──
def main():
    log.info("=" * 70)
    log.info("V26 — Exhaustive Ablation Study (Fast)")
    log.info("=" * 70)

    feat = pd.read_parquet(DATA_PROCESSED / "features.parquet")
    raw = [c for c in feat.columns if c not in META_COLS | set(TARGET_COLS)
           and feat[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]

    base = [c for c in raw if c not in CONSTANT_COLS and c not in COLLINEAR_DROP]
    log.info(f"Base cleaned: {len(base)}")

    # Fix wHr
    feat = feat.copy()
    bad = (feat['wHr_hr_mean'] < 20) | (feat['wHr_hr_mean'] > 180)
    feat.loc[bad, 'wHr_hr_mean'] = np.nan
    feat.loc[bad, 'wHr_hr_std'] = np.nan

    # Build progressive feature sets
    feat_roll, r_cols = add_rolling(feat, base)
    feat_pers, p_cols = add_personal(feat_roll, base)
    feat_full = add_seasonal(feat_pers)

    # Missing indicators
    all_num = [c for c in feat_full.columns if c not in META_COLS | set(TARGET_COLS)
               and feat_full[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]
    miss = [c for c in all_num if feat_full[c].isnull().mean() > 0.05]
    for c in miss:
        feat_full[f'{c}_missing'] = feat_full[c].isnull().astype(int)
    feat_full = feat_full.fillna(0)

    # Precompute feature columns for each factor
    season_cols = ['is_weekend', 'month', 'doy_sin', 'doy_cos']
    factor_cols = {
        'base': base,
        'rolling': r_cols,
        'personal': p_cols,
        'seasonal': season_cols,
    }

    # Build all non-empty configs (2^4 - 1 = 15)
    factors = ['base', 'rolling', 'personal', 'seasonal']
    configs = []
    for mask in range(1, 16):
        parts = [factors[i] for i in range(4) if mask & (1 << i)]
        cfg_name = '+'.join(parts)
        cfg_cols = []
        for p in parts:
            cfg_cols.extend(factor_cols[p])
        # Filter to columns that exist in feat_full
        cfg_cols = list(dict.fromkeys(cfg_cols))  # preserve order, dedupe
        cfg_cols = [c for c in cfg_cols if c in feat_full.columns]
        configs.append((cfg_name, cfg_cols))

    train_rate = {t: feat_full[t].mean() for t in TARGET_COLS}

    all_results = []

    for cfg_name, cfg_cols in configs:
        # Check if any feature exists
        if len(cfg_cols) == 0:
            continue

        log.info(f"\n{'='*60}")
        log.info(f"Config: {cfg_name} ({len(cfg_cols)} features)")
        log.info(f"{'='*60}")

        for n_top in N_TOP_VALUES:
            log.info(f"  --- Top-{n_top} ---")
            results_per_target = {}
            for target in TARGET_COLS:
                leak = LEAKAGE_S if target.startswith('S') else LEAKAGE_Q
                avail = [c for c in cfg_cols if c not in leak]
                if len(avail) < n_top:
                    log.warning(f"    {target}: only {len(avail)} available, skip")
                    results_per_target[target] = float('inf')
                    continue

                ranked = rank_feat(feat_full, avail, target)
                sel = [r[0] for r in ranked[:n_top]]

                oof = lgb_cv(feat_full, sel, target)
                cal = mean_match(oof, train_rate[target])
                loss = log_loss(feat_full[target], cal, labels=[0, 1])
                results_per_target[target] = loss

            avg = np.mean([v for v in results_per_target.values() if v != float('inf')])
            all_results.append({
                'config': cfg_name, 'n_top': n_top, 'avg_cal_oof': float(avg),
                'n_features': len(cfg_cols),
                'per_target': {k: float(v) if v != float('inf') else -1 for k, v in results_per_target.items()},
            })
            log.info(f"    AVG Cal OOF: {avg:.4f}")

    # Sort by best
    all_results.sort(key=lambda x: x['avg_cal_oof'])

    log.info(f"\n{'='*70}")
    log.info("ALL RESULTS (sorted by best Cal OOF)")
    log.info(f"{'='*70}")
    log.info(f"{'Rank':<5} {'Config':<30} {'N':<4} {'Features':<10} {'Avg Cal OOF':<14} {'Δ vs V10'}")
    for i, r in enumerate(all_results):
        d = r['avg_cal_oof'] - 0.6038
        sign = '+' if d >= 0 else ''
        log.info(f"{i+1:<5} {r['config']:<30} {r['n_top']:<4} {r['n_features']:<10} {r['avg_cal_oof']:<14.4f} {sign}{d:.4f}")

    best = all_results[0]
    log.info(f"\n🏆 BEST: {best['config']} Top-{best['n_top']} → AVG={best['avg_cal_oof']:.4f} (features={best['n_features']})")
    log.info(f"{'='*70}")

    # Save
    meta_path = SUBMIT_DIR / 'meta_v26_ablation.json'
    with open(meta_path, 'w') as f:
        json.dump({'results': [{'config': r['config'], 'n_top': r['n_top'], 'n_features': r['n_features'], 'avg': r['avg_cal_oof'], 'per_target': r['per_target']} for r in all_results],
                   'best': {'config': best['config'], 'n_top': best['n_top'], 'n_features': best['n_features'], 'avg': best['avg_cal_oof']}}, f, indent=2)
    log.info(f"Results saved: {meta_path}")


if __name__ == "__main__":
    main()
