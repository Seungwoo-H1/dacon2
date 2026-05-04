"""
08_v11_extended_features.py — V11: Extended feature engineering

Expands the original feature set (02_feature_engineering.py) with:
1. Rolling window aggregations (1,3,5,7,14 day windows)
2. Temporal features (hour-of-day, day-of-week, weekend, month, quarter)
3. Rhythm features (sleep timing consistency, activity regularity)
4. Cross-source interaction features
5. Rate of change (current vs previous window)
6. Percentile rank within subject
7. Missing data indicators
8. Pairwise interaction features

Pipeline:
1. Load raw parquet data (via 01_load_data)
2. Create day-level features (via 02_feature_engineering base)
3. Add extended features
4. Save to data_processed/features_v11.parquet
"""

import sys
import logging
import re
import warnings
import importlib
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from config import TARGETS, DATA_PROCESSED

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data_raw"

META_COLS = {"subject_id", "lifelog_date", "sleep_date", "date"}


def get_feature_cols(df):
    """Get non-meta, non-target columns."""
    return [c for c in df.columns
            if c not in META_COLS | set(TARGETS)
            and df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]


def load_base_features():
    """Load existing features."""
    log.info("Loading base features...")
    feat_path = DATA_PROCESSED / "features.parquet"
    if feat_path.exists():
        feat = pd.read_parquet(feat_path)
        log.info(f"  Loaded features: {feat.shape}")
    else:
        log.info("  Generating base features from scratch...")
        feat_eng = importlib.import_module("02_feature_engineering")
        ld_mod = importlib.import_module("01_load_data")
        parquet_dfs, labels = ld_mod.main()
        feat = feat_eng.create_day_features(parquet_dfs, labels)
    return feat


# ── 1. Rolling window aggregations ──────────────────────────

def add_rolling_windows(feat):
    feat = feat.copy()
    feat['date_dt'] = pd.to_datetime(feat['lifelog_date'])
    numeric_cols = get_feature_cols(feat)
    all_new_names = set()
    windows = [1, 3, 5, 7, 14]

    for sid in feat['subject_id'].unique():
        mask = feat['subject_id'] == sid
        sub = feat.loc[mask].sort_values('date_dt')

        for col in numeric_cols:
            vals = sub[col].ffill().bfill()
            for w in windows:
                rm = vals.rolling(window=w, min_periods=1).mean()
                rs = vals.rolling(window=w, min_periods=min(2, w)).std().fillna(0)
                rn = vals.rolling(window=w, min_periods=1).min()
                rx = vals.rolling(window=w, min_periods=1).max()
                rr = rx - rn

                for name, val in [
                    (f'{col}_roll_mean_{w}d', rm),
                    (f'{col}_roll_std_{w}d', rs),
                    (f'{col}_roll_min_{w}d', rn),
                    (f'{col}_roll_max_{w}d', rx),
                    (f'{col}_roll_range_{w}d', rr),
                ]:
                    feat.loc[sub.index, name] = val.values
                    all_new_names.add(name)

    added = all_new_names - set(feat.columns) - set(numeric_cols)
    # Also remove any already-existing base columns
    already = all_new_names - added
    log.info(f"  Rolling window: {len(all_new_names)} attempted, {len(added)} new")
    return feat, list(added)


# ── 2. Temporal features ──────────────────────────────

def add_temporal_features(feat):
    feat = feat.copy()
    feat['date_dt'] = pd.to_datetime(feat['lifelog_date'])

    temporal = [
        'hour', 'dow', 'is_weekend', 'month', 'quarter',
        'day_of_year', 'week_of_year',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    ]
    feat['hour'] = feat['date_dt'].dt.hour
    feat['dow'] = feat['date_dt'].dt.dayofweek
    feat['is_weekend'] = (feat['dow'] >= 5).astype(float)
    feat['month'] = feat['date_dt'].dt.month
    feat['quarter'] = feat['date_dt'].dt.quarter
    feat['day_of_year'] = feat['date_dt'].dt.dayofyear
    feat['week_of_year'] = feat['date_dt'].dt.isocalendar().week.astype(float)
    feat['hour_sin'] = np.sin(2 * np.pi * feat['hour'] / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * feat['hour'] / 24)
    feat['dow_sin'] = np.sin(2 * np.pi * feat['dow'] / 7)
    feat['dow_cos'] = np.cos(2 * np.pi * feat['dow'] / 7)

    new = [c for c in temporal if c not in feat.columns]
    log.info(f"  Temporal features: {len(new)}")
    return feat, new


# ── 3. Rhythm features ──────────────────────────────

def add_rhythm_features(feat):
    feat = feat.copy()
    feat['date_dt'] = pd.to_datetime(feat['lifelog_date'])
    all_new_names = set()
    numeric_cols = get_feature_cols(feat)

    activity_like = [c for c in numeric_cols
                     if any(k in c.lower() for k in ['activity', 'pedo', 'step'])
                     and '_mean' in c.lower()]

    for sid in feat['subject_id'].unique():
        mask = feat['subject_id'] == sid
        sub = feat.loc[mask].sort_values('date_dt')

        for col in activity_like:
            vals = sub[col].ffill().fillna(0)
            if vals.std() == 0:
                continue
            for w in [3, 5, 7]:
                rstd = vals.rolling(window=w, min_periods=2).std().fillna(0)
                name = f'{col}_rhythm_{w}d'
                feat.loc[sub.index, name] = rstd.values
                all_new_names.add(name)

    # Step-to-distance ratio
    if 'wPedo_pedo_step_mean' in feat.columns and 'wPedo_pedo_distance_mean' in feat.columns:
        steps = feat['wPedo_pedo_step_mean'].ffill().fillna(0)
        dist = feat['wPedo_pedo_distance_mean'].ffill().fillna(0)
        ratio = steps / (dist + 1)
        for w in [3, 5]:
            rr = ratio.groupby(feat['subject_id']).transform(
                lambda x: x.rolling(window=w, min_periods=2).mean().fillna(0)
            )
            name = f'step_dist_ratio_roll_{w}d'
            feat[name] = rr
            all_new_names.add(name)

    new = all_new_names - set(feat.columns)
    log.info(f"  Rhythm features: {len(new)}")
    return feat, list(new)


# ── 4. Cross-source interaction features ──────────────────

def add_cross_source_features(feat):
    feat = feat.copy()
    numeric_cols = get_feature_cols(feat)
    all_new_names = set()

    # Activity + HR
    act_means = [c for c in numeric_cols if 'mActivity' in c and '_mean' in c]
    hr_cols = [c for c in numeric_cols if 'wHr' in c]
    for ac in act_means:
        for hc in hr_cols:
            name = f'cross_{ac.split("_")[-1]}_hr'
            feat[name] = feat[ac] * feat[hc] / (feat[hc].mean() + 1e-8)
            all_new_names.add(name)

    # Light + Screen
    light_means = [c for c in numeric_cols if 'Light' in str(c.split('_')[0]) and '_mean' in c]
    screen_cols = [c for c in numeric_cols if 'mScreenStatus' in c]
    for lc in light_means:
        for sc in screen_cols:
            name = f'cross_{lc.split("_")[-1]}_screen'
            feat[name] = feat[lc] * feat[sc] / (feat[sc].mean() + 1e-8)
            all_new_names.add(name)

    # Step + Distance
    if all(c in feat.columns for c in ['wPedo_pedo_step_mean', 'wPedo_pedo_distance_mean']):
        feat['cross_step_distance'] = (
            feat['wPedo_pedo_step_mean'] * feat['wPedo_pedo_distance_mean'] /
            (feat['wPedo_pedo_distance_mean'].mean() + 1e-8)
        )
        all_new_names.add('cross_step_distance')

    # HR + Step
    if all(c in feat.columns for c in ['wHr_hr_mean', 'wPedo_pedo_step_mean']):
        feat['cross_hr_step'] = (
            feat['wHr_hr_mean'] * feat['wPedo_pedo_step_mean'] /
            (feat['wPedo_pedo_step_mean'].mean() + 1e-8)
        )
        all_new_names.add('cross_hr_step')

    # Ambience + Screen
    amb_cols = [c for c in numeric_cols if 'mAmbience' in c]
    if amb_cols and screen_cols:
        for amb in amb_cols:
            for sc in screen_cols:
                name = f'cross_{amb.split("_")[-1]}_screen'
                feat[name] = feat[amb] * feat[sc]
                all_new_names.add(name)

    # Usage + Screen
    use_cols = [c for c in numeric_cols if 'mUsageStats' in c]
    if use_cols and screen_cols:
        for uc in use_cols:
            for sc in screen_cols:
                name = f'cross_{uc.split("_")[-1]}_screen'
                feat[name] = feat[uc] * feat[sc]
                all_new_names.add(name)

    # WiFi + BLE co-occurrence
    wifi_cnt = [c for c in numeric_cols if 'mWifi' in c and 'count' in c]
    ble_cnt = [c for c in numeric_cols if 'mBle' in c and 'count' in c]
    for wc in wifi_cnt:
        for bc in ble_cnt:
            feat['cross_wifi_ble'] = feat[wc] * feat[bc]
            all_new_names.add('cross_wifi_ble')

    new = all_new_names - set(feat.columns)
    log.info(f"  Cross-source features: {len(new)}")
    return feat, list(new)


# ── 5. Rate of change ─────────────────────────────────

def add_rate_of_change(feat):
    feat = feat.copy()
    feat['date_dt'] = pd.to_datetime(feat['lifelog_date'])
    numeric_cols = get_feature_cols(feat)

    # Base features only
    base_cols = [c for c in numeric_cols
                if all(k not in c for k in [
                    'roll_', 'rhythm_', 'hour', 'dow', 'is_weekend',
                    'month', 'quarter', 'day_of_year', 'week_of_year',
                    'sin', 'cos', 'cross_', 'roc', 'dev', 'pctile', 'missing', 'pair_'
                ])]

    all_new_names = set()

    for sid in feat['subject_id'].unique():
        mask = feat['subject_id'] == sid
        sub = feat.loc[mask].sort_values('date_dt')

        for col in base_cols:
            vals = sub[col].ffill().fillna(0)
            if vals.std() == 0:
                continue

            for lag in [1, 3]:
                roc = vals.diff(lag).fillna(0)
                name = f'{col}_roc{lag}d'
                feat.loc[sub.index, name] = roc.values
                all_new_names.add(name)

            # Deviation from 3-day rolling mean
            rmean = vals.rolling(window=3, min_periods=1).mean()
            dev = vals - rmean
            name = f'{col}_dev3d'
            feat.loc[sub.index, name] = dev.values
            all_new_names.add(name)

    new = all_new_names - set(feat.columns)
    log.info(f"  Rate of change features: {len(new)}")
    return feat, list(new)


# ── 6. Percentile rank within subject ──────────────────

def add_percentile_rank(feat):
    feat = feat.copy()
    numeric_cols = get_feature_cols(feat)

    base_cols = [c for c in numeric_cols
                if all(k not in c for k in [
                    'roll_', 'rhythm_', 'hour', 'dow', 'is_weekend',
                    'month', 'quarter', 'day_of_year', 'week_of_year',
                    'sin', 'cos', 'cross_', 'roc', 'dev', 'pctile', 'missing', 'pair_'
                ])]

    all_new_names = set()

    for sid in feat['subject_id'].unique():
        mask = feat['subject_id'] == sid
        sub_idx = feat.loc[mask].index

        for col in base_cols:
            vals = feat.loc[sub_idx, col]
            if vals.isnull().all():
                continue
            ranked = vals.rank(pct=True, method='average')
            name = f'{col}_pctile'
            feat.loc[sub_idx, name] = ranked.values
            all_new_names.add(name)

    new = all_new_names - set(feat.columns)
    log.info(f"  Percentile rank features: {len(new)}")
    return feat, list(new)


# ── 7. Missing data indicators ──────────────────────

def add_missing_indicators(feat):
    feat = feat.copy()
    numeric_cols = get_feature_cols(feat)

    sources = ['mActivity', 'mLight', 'wLight', 'wPedo', 'mAmbience',
               'mBle', 'mGps', 'mScreenStatus', 'mUsageStats', 'mWifi',
               'mACStatus', 'wHr']

    all_new_names = set()

    for src in sources:
        src_cols = [c for c in numeric_cols if c.startswith(src)]
        if not src_cols:
            continue
        mc = feat[src_cols].isnull().sum(axis=1)
        total = max(len(src_cols), 1)
        feat[f'{src}_missing_count'] = mc.values
        feat[f'{src}_missing_ratio'] = (mc / total).values
        all_new_names.add(f'{src}_missing_count')
        all_new_names.add(f'{src}_missing_ratio')

    # Total missing ratio
    total_cols = max(len(numeric_cols), 1)
    feat['total_missing_ratio'] = (feat[numeric_cols].isnull().sum(axis=1) / total_cols).values
    all_new_names.add('total_missing_ratio')

    new = all_new_names - set(feat.columns)
    log.info(f"  Missing indicators: {len(new)}")
    return feat, list(new)


# ── 8. Pairwise interactions ──────────────────────

def add_pairwise_interactions(feat):
    feat = feat.copy()
    import lightgbm as lgb

    numeric_cols = get_feature_cols(feat)
    all_new_names = set()

    for target in TARGETS:
        y = feat[target].values
        X = feat[numeric_cols].fillna(0).values
        n_pos = max((y == 1).sum(), 1)
        n_neg = (y == 0).sum()
        spw = n_neg / n_pos

        safe_names = [re.sub(r'[^a-zA-Z0-9_]', '_', c) for c in numeric_cols]
        params = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
            'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.03,
            'n_estimators': 50, 'subsample': 0.7, 'colsample_bytree': 0.7,
            'reg_alpha': 1.0, 'reg_lambda': 3.0,
            'scale_pos_weight': spw, 'random_state': 42,
            'min_child_samples': 10, 'force_row_wise': True, 'n_jobs': -1,
        }
        ds = lgb.Dataset(X, label=y, feature_name=safe_names, params={'verbose': '-1'})
        model = lgb.train(params, ds, num_boost_round=50)
        imp = model.feature_importance(importance_type="gain")
        ranked = sorted(zip(numeric_cols, imp), key=lambda x: -x[1])

        top_feats = [r[0] for r in ranked[:15] if r[1] > 0]

        for i in range(len(top_feats)):
            for j in range(i + 1, len(top_feats)):
                fi, fj = top_feats[i], top_feats[j]
                name = f'pair_{fi.split("_")[-1]}_x_{fj.split("_")[-1]}'
                feat[name] = (feat[fi] / (feat[fi].mean() + 1e-8)) * (feat[fj] / (feat[fj].mean() + 1e-8))
                all_new_names.add(name)

    new = all_new_names - set(feat.columns)
    log.info(f"  Pairwise interactions: {len(new)}")
    return feat, list(new)


# ── Main pipeline ─────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("V11 Extended Feature Engineering")
    log.info("=" * 70)

    feat = load_base_features()
    log.info(f"  Base shape: {feat.shape}")

    steps = [
        ("Rolling windows", add_rolling_windows),
        ("Temporal", add_temporal_features),
        ("Rhythm", add_rhythm_features),
        ("Cross-source", add_cross_source_features),
        ("Rate of change", add_rate_of_change),
        ("Percentile rank", add_percentile_rank),
        ("Missing indicators", add_missing_indicators),
        ("Pairwise interactions", add_pairwise_interactions),
    ]

    total_new = 0
    for name, fn in steps:
        feat, added = fn(feat)
        total_new += len(added)
        log.info(f"  → {name}: +{len(added)} new features")

    feat_path = DATA_PROCESSED / "features_v11.parquet"
    feat.to_parquet(feat_path, index=False)
    log.info(f"\n✅ Saved {feat_path}: {feat.shape}")

    nc = get_feature_cols(feat)
    base_count = sum(1 for c in nc
                     if all(k not in c for k in [
                         'roll_', 'rhythm_', 'hour', 'dow', 'is_weekend',
                         'month', 'quarter', 'day_of_year', 'week_of_year',
                         'sin', 'cos', 'cross_', 'roc', 'dev', 'pctile', 'missing', 'pair_'
                     ]))
    log.info(f"\n  Base features: {base_count}")
    log.info(f"  Total features: {len(nc)}")
    log.info(f"  New features total: {total_new}")

    nulls = feat[nc].isnull().mean()
    high_null = nulls[nulls > 0.1]
    if len(high_null) > 0:
        log.info(f"\n  >10% null features: {len(high_null)}")
        for c, v in high_null.sort_values(ascending=False).head(10).items():
            log.info(f"    {c}: {v:.1%}")

    log.info(f"\n  Target rates:")
    for t in TARGETS:
        if t in feat.columns:
            log.info(f"    {t}: mean={feat[t].mean():.3f}")

    return feat


if __name__ == "__main__":
    main()
