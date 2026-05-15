"""
V26 — Exhaustive Ablation Study
Tests ALL individual + combined feature improvements.

Factor 1: base_only (cleaned base features)
Factor 2: +rolling (3d, 7d rolling mean/std)
Factor 3: +personal (z-score, delta, gmd)
Factor 4: +seasonal (doy_sin, doy_cos, is_weekend, month)

Configurations:
  - A: base_only (baseline)
  - B: base + rolling
  - C: base + personal
  - D: base + seasonal
  - E: base + rolling + personal
  - F: base + rolling + seasonal
  - G: base + personal + seasonal
  - H: base + rolling + personal + seasonal (full)

Each: Top-20, Top-30, Top-40, Top-50 tested per target.
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
N_TOP_VALUES = [20, 30, 40, 50]

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


def add_rolling(df, cols, windows=[3, 7]):
    df = df.copy().sort_values(['subject_id', 'date'])
    new = []
    for c in cols:
        grp = df.groupby('subject_id')[c]
        for w in windows:
            rm = grp.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
            rs = grp.rolling(w, min_periods=1).std().fillna(0).reset_index(level=0, drop=True)
            n_rm, n_rs = f'{c}_rm{w}', f'{c}_rs{w}'
            df[n_rm] = rm.values
            df[n_rs] = rs.values
            new.extend([n_rm, n_rs])
    return df, new


def add_personal(df, cols):
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
    df = df.copy()
    d = pd.to_datetime(df['date'])
    df['is_weekend'] = (d.dt.dayofweek >= 5).astype(int)
    df['month'] = d.dt.month
    df['doy_sin'] = np.sin(2 * np.pi * d.dt.dayofyear / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * d.dt.dayofyear / 365.25)
    return df


def rank_feat(feat, cols, target):
    y = feat[target].values
    X = feat[cols].fillna(0).values
    n_pos = max((y == 1).sum(), 1)
    n_neg = (y == 0).sum()
    spw = n_neg / n_pos
    params = {**LGB_BASE, 'num_leaves': 15, 'max_depth': 4, 'n_estimators': 100,
              'scale_pos_weight': spw, 'random_state': 42}
    sn = [sanitize(c) for c in cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn, params={'verbose': '-1'})
    m = lgb.train(params, ds, num_boost_round=100)
    imp = m.feature_importance(importance_type='gain')
    return sorted(zip(cols, imp), key=lambda x: -x[1])


def lgb_cv(feat, cols, target):
    y = feat[target].values
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_full = np.zeros((len(y), N_SEEDS))
    spw = ((y == 0).sum()) / max((y == 1).sum(), 1)
    sn = [sanitize(c) for c in cols]
    for si, seed in enumerate(SEEDS):
        cfg = {**LGB_BASE, 'random_state': seed, 'scale_pos_weight': spw}
        for fold, (tr_i, va_i) in enumerate(gkf.split(feat, y, feat['subject_id'])):
            ds = lgb.Dataset(feat.iloc[tr_i][cols].fillna(0).values, label=y[tr_i], feature_name=sn, params={'verbose': '-1'})
            vad = lgb.Dataset(feat.iloc[va_i][cols].fillna(0).values, label=y[va_i], feature_name=sn, reference=ds, params={'verbose': '-1'})
            m = lgb.train(cfg, ds, num_boost_round=500, valid_sets=[vad],
                         callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            oof_full[va_i, si] = m.predict(feat.iloc[va_i][cols].fillna(0).values)
    return oof_full.mean(axis=1)


def mean_match(pred, rate):
    return np.clip(pred + (rate - pred.mean()), 0.0001, 0.9999)


# ── Main ──
def main():
    log.info("=" * 70)
    log.info("V26 — Exhaustive Ablation Study")
    log.info("=" * 70)

    feat = pd.read_parquet(DATA_PROCESSED / "features.parquet")
    raw = [c for c in feat.columns if c not in META_COLS | set(TARGET_COLS)
           and feat[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]

    # Clean base
    base = [c for c in raw if c not in CONSTANT_COLS and c not in COLLINEAR_DROP]
    log.info(f"Base cleaned: {len(base)}")

    # Fix wHr
    bad = (feat['wHr_hr_mean'] < 20) | (feat['wHr_hr_mean'] > 180)
    feat = feat.copy()
    feat.loc[bad, 'wHr_hr_mean'] = np.nan
    feat.loc[bad, 'wHr_hr_std'] = np.nan

    # Build feature sets
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

    train_rate = {t: feat_full[t].mean() for t in TARGET_COLS}

    # Config definitions
    configs = [
        ('A_base_only',     base, base, base, {}),
        ('B_base+rolling',  base, base+['dummy'], base, {}),
        ('C_base+personal', base, base, base, {}),
        ('D_base+seasonal', base, base, base, {}),
        ('E_base+rolling+personal', base, base+['dummy'], base, {}),
        ('F_base+rolling+seasonal', base, base+['dummy'], base, {}),
        ('G_base+personal+seasonal', base, base, base, {}),
        ('H_base+rolling+personal+seasonal', base, base+['dummy'], base, {}),
    ]

    # For each config, map config columns → actual columns from feat_full
    config_map = {
        'A_base_only': base,
        'B_base+rolling': base + r_cols,
        'C_base+personal': base + p_cols,
        'D_base+seasonal': base + ['is_weekend', 'month', 'doy_sin', 'doy_cos'],
        'E_base+rolling+personal': base + r_cols + p_cols,
        'F_base+rolling+seasonal': base + r_cols + ['is_weekend', 'month', 'doy_sin', 'doy_cos'],
        'G_base+personal+seasonal': base + p_cols + ['is_weekend', 'month', 'doy_sin', 'doy_cos'],
        'H_base+rolling+personal+seasonal': base + r_cols + p_cols + ['is_weekend', 'month', 'doy_sin', 'doy_cos'],
    }

    all_results = []

    for cfg_name in ['A_base_only', 'B_base+rolling', 'C_base+personal', 'D_base+seasonal',
                      'E_base+rolling+personal', 'F_base+rolling+seasonal',
                      'G_base+personal+seasonal', 'H_base+rolling+personal+seasonal']:
        feat_cols = config_map[cfg_name]
        # Filter to only columns that exist
        feat_cols = [c for c in feat_cols if c in feat_full.columns]
        log.info(f"\n{'='*60}")
        log.info(f"Config: {cfg_name} ({len(feat_cols)} features)")
        log.info(f"{'='*60}")

        for n_top in N_TOP_VALUES:
            log.info(f"\n  --- Top-{n_top} ---")
            results_per_target = {}
            for target in TARGET_COLS:
                leak = LEAKAGE_S if target.startswith('S') else LEAKAGE_Q
                avail = [c for c in feat_cols if c not in leak]
                ranked = rank_feat(feat_full, avail, target)
                sel = [r[0] for r in ranked[:n_top]]
                if len(sel) == 0:
                    results_per_target[target] = float('inf')
                    continue

                oof = lgb_cv(feat_full, sel, target)
                cal = mean_match(oof, train_rate[target])
                loss = log_loss(feat_full[target], cal, labels=[0, 1])
                results_per_target[target] = loss
                log.info(f"    {target}: {loss:.4f} (top: {sel[0]}, {sel[1]}, {sel[2]}...)")

            avg = np.mean([v for v in results_per_target.values() if v != float('inf')])
            all_results.append({
                'config': cfg_name, 'n_top': n_top, 'avg_cal_oof': float(avg),
                'per_target': results_per_target,
            })
            log.info(f"  AVG Cal OOF: {avg:.4f}")

    # ── Sort and show best ──
    all_results.sort(key=lambda x: x['avg_cal_oof'])

    log.info(f"\n{'='*70}")
    log.info("ALL RESULTS (sorted by best Cal OOF)")
    log.info(f"{'='*70}")
    log.info(f"{'Rank':<5} {'Config':<35} {'N':<5} {'Avg Cal OOF':<14} {'Δ vs V10'}")
    for i, r in enumerate(all_results):
        d = r['avg_cal_oof'] - 0.6038  # vs V10
        sign = '+' if d >= 0 else ''
        log.info(f"{i+1:<5} {r['config']:<35} {r['n_top']:<5} {r['avg_cal_oof']:<14.4f} {sign}{d:.4f}")

    best = all_results[0]
    log.info(f"\n🏆 BEST: {best['config']} Top-{best['n_top']} → AVG={best['avg_cal_oof']:.4f}")
    log.info(f"{'='*70}")

    # ── Save best config details ──
    meta_path = SUBMIT_DIR / 'meta_v26_ablation.json'
    with open(meta_path, 'w') as f:
        json.dump({'results': [{'config': r['config'], 'n_top': r['n_top'], 'avg': r['avg_cal_oof']} for r in all_results],
                   'best': {'config': best['config'], 'n_top': best['n_top'], 'avg': best['avg_cal_oof']}}, f, indent=2)

    log.info(f"Results saved: {meta_path}")


if __name__ == "__main__":
    main()
