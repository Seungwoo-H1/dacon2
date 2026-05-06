"""
V56 — Rich Feature Engineering + Multi-Strategy Ensemble

New features vs V53:
1. Proper HR parsing (list of values -> per-row stats)
2. Nighttime vs daytime separation for HR/light
3. Rolling statistics (3, 7, 14, 28 day windows)
4. Rate of change features
5. Day-of-week deviation features (behavioral rhythm)
6. Weekend/weekday differentiation
7. Activity ratio features (step/running/walking)
8. BLE/WiFi device richness features
9. Ambience category ratios
10. GPS mobility features
11. Screen usage timing features (morning/afternoon/evening/night)
12. Charging pattern features
13. Usage stats category ratios

Then train per-target LGBM models with:
- Target-specific feature selection
- Calibration
- Multiple seeds ensemble
"""

import sys, re, gc, time, warnings, logging, json, os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = ROOT / "data_processed"
DATA_RAW = ROOT / "data_raw"
TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
META = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}


# ── Helper: parse list-valued columns ──

def parse_array_stats(series, col_name):
    """Parse ndarray/list column into mean, std, min, max, count."""
    means, stds, mins, maxs, counts = [], [], [], [], []
    for val in series:
        if isinstance(val, (np.ndarray, list)):
            arr = np.array(val, dtype=float)
            means.append(arr.mean())
            stds.append(arr.std())
            mins.append(arr.min())
            maxs.append(arr.max())
            counts.append(len(arr))
        elif isinstance(val, (int, float)):
            means.append(float(val))
            stds.append(0.0)
            mins.append(float(val))
            maxs.append(float(val))
            counts.append(1)
        else:
            means.append(np.nan)
            stds.append(np.nan)
            mins.append(np.nan)
            maxs.append(np.nan)
            counts.append(0)
    prefix = col_name.rstrip('_')
    return pd.DataFrame({
        f'{prefix}_mean': means, f'{prefix}_std': stds,
        f'{prefix}_min': mins, f'{prefix}_max': maxs, f'{prefix}_count': counts
    })


def parse_ambience_probs(series):
    """Parse ambience ndarray of [category, prob] pairs into category sums."""
    categories = ['outside,_urban_or_manmade', 'inside,_domestic_or_personal',
                   'inside,_office_or_institutional', 'inside,_store_or_shop',
                   'inside,_public_or_crowded', 'inside,_vehicle_interior',
                   'inside,_building_entrance_or_lobby', 'Music', 'Nature',
                   'Human_sounds', 'Appliances_and_hvac', 'Vehicle',
                   'Animal', 'Horse', 'Water', 'Wind_and_other_audible_effects']
    result = pd.DataFrame({f'ambience_{c}': 0.0 for c in categories}, index=series.index)
    for i, val in enumerate(series):
        if isinstance(val, (np.ndarray, list)):
            arr = np.array(val)
            if arr.ndim == 2:
                for row in arr:
                    if len(row) >= 2:
                        cat = str(row[0])
                        prob = float(row[1])
                        if cat in result.columns:
                            result.loc[i, cat] = prob
    return result


# ── Feature Engineering ──

def build_features(subjects, labels_df):
    """Build rich features from raw parquet files."""
    parquet_dir = DATA_RAW / "ch2025_data_items"
    all_rows = []

    for subj in subjects:
        log.info(f"Processing {subj}...")
        subj_labels = labels_df[labels_df['subject_id'] == subj].copy()

        # ── 1. mLight (indoor light sensor) ──
        ml = pd.read_parquet(parquet_dir / "ch2025_mLight.parquet")
        ml = ml[ml['subject_id'] == subj].copy()
        ml['date'] = ml['timestamp'].dt.date
        ml_feat = ml.groupby('date')['m_light'].agg(['mean','std','max','min','count']).reset_index()
        ml_feat.columns = ['date', 'light_mean','light_std','light_max','light_min','light_count']

        # ── 2. wLight (wearable light) ──
        wl = pd.read_parquet(parquet_dir / "ch2025_wLight.parquet")
        wl = wl[wl['subject_id'] == subj].copy()
        wl['date'] = wl['timestamp'].dt.date
        wl_feat = wl.groupby('date')['w_light'].agg(['mean','std','max','min','count']).reset_index()
        wl_feat.columns = ['date', 'wlight_mean','wlight_std','wlight_max','wlight_min','wlight_count']

        # Nighttime light (22:00-06:00)
        wl['hour'] = wl['timestamp'].dt.hour
        wl['night'] = wl['hour'].between(22, 24) | wl['hour'].between(0, 6)
        wl_night = wl[wl['night']].groupby('date')['w_light'].agg(['mean','std','max','min','count']).reset_index()
        wl_night.columns = ['date', 'wlight_night_mean','wlight_night_std','wlight_night_max','wlight_night_min','wlight_night_count']

        # ── 3. wPedo (step/walking/running) ──
        wp = pd.read_parquet(parquet_dir / "ch2025_wPedo.parquet")
        wp = wp[wp['subject_id'] == subj].copy()
        wp['date'] = wp['timestamp'].dt.date
        wp_feat = wp.groupby('date').agg(
            step_mean=('step','mean'), step_std=('step','std'), step_max=('step','max'),
            step_sum=('step','sum'), step_count=('step','count'),
            running_step_sum=('running_step','sum'), walking_step_sum=('walking_step','sum'),
            distance_sum=('distance','sum'), speed_mean=('speed','mean'),
            burned_cal_sum=('burned_calories','sum')
        ).reset_index()

        # Activity ratios
        wp_feat['running_ratio'] = wp_feat['running_step_sum'] / (wp_feat['step_sum'] + 1e-9)
        wp_feat['walking_ratio'] = wp_feat['walking_step_sum'] / (wp_feat['step_sum'] + 1e-9)

        # ── 4. wHr (heart rate - array values!) ──
        wh = pd.read_parquet(parquet_dir / "ch2025_wHr.parquet")
        wh = wh[wh['subject_id'] == subj].copy()
        wh['date'] = wh['timestamp'].dt.date
        hr_stats = parse_array_stats(wh['heart_rate'], 'heart_rate')
        wh = pd.concat([wh, hr_stats], axis=1)

        # Daily HR stats
        hr_daily = wh.groupby('date').agg(
            hr_mean=('heart_rate_mean','mean'), hr_std=('heart_rate_std','mean'),
            hr_min_all=('heart_rate_min','min'), hr_max_all=('heart_rate_max','max'),
            hr_total_count=('heart_rate_count','sum')
        ).reset_index()

        # Nighttime HR (22:00-06:00)
        wh['hour'] = wh['timestamp'].dt.hour
        wh['night'] = wh['hour'].between(22, 24) | wh['hour'].between(0, 6)
        hr_night = wh[wh['night']].groupby('date').agg(
            hr_night_mean=('heart_rate_mean','mean'), hr_night_std=('heart_rate_std','mean'),
            hr_night_min=('heart_rate_min','min'), hr_night_max=('heart_rate_max','max'),
            hr_night_count=('heart_rate_count','sum')
        ).reset_index()

        # ── 5. mActivity ──
        ma = pd.read_parquet(parquet_dir / "ch2025_mActivity.parquet")
        ma = ma[ma['subject_id'] == subj].copy()
        ma['date'] = ma['timestamp'].dt.date
        act_feat = ma.groupby('date')['m_activity'].agg(['mean','std','max','min','count']).reset_index()
        act_feat.columns = ['date','activity_mean','activity_std','activity_max','activity_min','activity_count']

        # ── 6. mScreenStatus ──
        ms = pd.read_parquet(parquet_dir / "ch2025_mScreenStatus.parquet")
        ms = ms[ms['subject_id'] == subj].copy()
        ms['date'] = ms['timestamp'].dt.date
        screen_feat = ms.groupby('date')['m_screen_use'].agg(['mean','std','max','min','count']).reset_index()
        screen_feat.columns = ['date','screen_mean','screen_std','screen_max','screen_min','screen_count']

        # Screen time by time of day
        ms['hour'] = ms['timestamp'].dt.hour
        for label, hour_range in [('morning', (6,12)), ('afternoon', (12,18)), ('evening', (18,22)), ('night', (22,24))]:
            mask = ms['hour'].between(*hour_range)
            sub = ms[mask].groupby('date')['m_screen_use'].agg(screen_mean='mean', screen_count='count').reset_index()
            sub.columns = ['date', f'screen_{label}_mean', f'screen_{label}_count']
            screen_feat = screen_feat.merge(sub, on='date', how='left')

        # ── 7. mACStatus (charging) ──
        mc = pd.read_parquet(parquet_dir / "ch2025_mACStatus.parquet")
        mc = mc[mc['subject_id'] == subj].copy()
        mc['date'] = mc['timestamp'].dt.date
        charge_feat = mc.groupby('date')['m_charging'].agg(['mean','std','max','min','count']).reset_index()
        charge_feat.columns = ['date','charge_mean','charge_std','charge_max','charge_min','charge_count']

        # ── 8. mWifi ──
        mw = pd.read_parquet(parquet_dir / "ch2025_mWifi.parquet")
        mw = mw[mw['subject_id'] == subj].copy()
        mw['date'] = mw['timestamp'].dt.date
        wifi_feat = mw.groupby('date').agg(
            wifi_count=('timestamp','count'),
        ).reset_index()

        # WiFi RSSI stats
        if 'm_wifi' in mw.columns and mw['m_wifi'].dtype == object:
            wifi_rssi = []
            for val in mw['m_wifi']:
                if isinstance(val, np.ndarray) and val.ndim == 2 and val.shape[1] >= 3:
                    rssi = val[:, 2].astype(float)
                    wifi_rssi.append({'rssi_mean': rssi.mean(), 'rssi_std': rssi.std(),
                                     'rssi_min': rssi.min(), 'rssi_max': rssi.max()})
                else:
                    wifi_rssi.append({'rssi_mean': np.nan, 'rssi_std': np.nan,
                                     'rssi_min': np.nan, 'rssi_max': np.nan})
            wifi_rssi_df = pd.DataFrame(wifi_rssi, index=mw.index)
            wifi_daily_rssi = wifi_rssi_df.groupby(mw['date']).agg(
                wifi_rssi_mean=('rssi_mean','mean'), wifi_rssi_std=('rssi_std','mean'),
                wifi_rssi_min=('rssi_min','min'), wifi_rssi_max=('rssi_max','max')
            ).reset_index()
            wifi_daily_rssi.columns = ['date','wifi_rssi_mean','wifi_rssi_std','wifi_rssi_min','wifi_rssi_max']
            wifi_feat = wifi_feat.merge(wifi_daily_rssi, on='date', how='left')

        # ── 9. mBle ──
        mb = pd.read_parquet(parquet_dir / "ch2025_mBle.parquet")
        mb = mb[mb['subject_id'] == subj].copy()
        mb['date'] = mb['timestamp'].dt.date
        ble_feat = mb.groupby('date').agg(
            ble_count=('timestamp','count'),
        ).reset_index()

        # BLE RSSI stats
        if 'm_ble' in mb.columns and mb['m_ble'].dtype == object:
            ble_rssi = []
            for val in mb['m_ble']:
                if isinstance(val, np.ndarray) and val.ndim == 2 and val.shape[1] >= 3:
                    rssi = val[:, 2].astype(float)
                    ble_rssi.append({'rssi_mean': rssi.mean(), 'rssi_std': rssi.std(),
                                     'rssi_min': rssi.min(), 'rssi_max': rssi.max()})
                else:
                    ble_rssi.append({'rssi_mean': np.nan, 'rssi_std': np.nan,
                                     'rssi_min': np.nan, 'rssi_max': np.nan})
            ble_rssi_df = pd.DataFrame(ble_rssi, index=mb.index)
            ble_daily_rssi = ble_rssi_df.groupby(mb['date']).agg(
                ble_rssi_mean=('rssi_mean','mean'), ble_rssi_std=('rssi_std','mean'),
                ble_rssi_min=('rssi_min','min'), ble_rssi_max=('rssi_max','max')
            ).reset_index()
            ble_daily_rssi.columns = ['date','ble_rssi_mean','ble_rssi_std','ble_rssi_min','ble_rssi_max']
            ble_feat = ble_feat.merge(ble_daily_rssi, on='date', how='left')

        # ── 10. mAmbience ──
        ma2 = pd.read_parquet(parquet_dir / "ch2025_mAmbience.parquet")
        ma2 = ma2[ma2['subject_id'] == subj].copy()
        ma2['date'] = ma2['timestamp'].dt.date
        amb_feat = parse_ambience_probs(ma2['m_ambience'])
        amb_feat['date'] = ma2['date'].values
        amb_count = ma2.groupby('date')['m_ambience'].count().reset_index()
        amb_count.columns = ['date','ambience_count']
        amb_feat = amb_feat.merge(amb_count, on='date', how='left')

        # ── 11. mGps ──
        mg = pd.read_parquet(parquet_dir / "ch2025_mGps.parquet")
        mg = mg[mg['subject_id'] == subj].copy()
        mg['date'] = mg['timestamp'].dt.date
        gps_feat = mg.groupby('date')['m_gps'].agg(['count']).reset_index()
        gps_feat.columns = ['date','gps_count']

        # GPS mobility features
        if 'm_gps' in mg.columns and mg['m_gps'].dtype == object:
            gps_stats = []
            for val in mg['m_gps']:
                if isinstance(val, np.ndarray) and val.ndim == 2 and val.shape[1] >= 5:
                    speed = val[:, 3].astype(float)
                    alt = val[:, 4].astype(float)
                    gps_stats.append({
                        'gps_speed_mean': speed.mean(), 'gps_speed_max': speed.max(),
                        'gps_speed_std': speed.std(), 'gps_alt_mean': alt.mean(),
                        'gps_alt_range': alt.max() - alt.min()
                    })
                else:
                    gps_stats.append({'gps_speed_mean': np.nan, 'gps_speed_max': np.nan,
                                     'gps_speed_std': np.nan, 'gps_alt_mean': np.nan,
                                     'gps_alt_range': np.nan})
            gps_stats_df = pd.DataFrame(gps_stats, index=mg.index)
            gps_mobility = gps_stats_df.groupby(mg['date']).agg({
                'gps_speed_mean': 'mean', 'gps_speed_max': 'max',
                'gps_speed_std': 'mean', 'gps_alt_mean': 'mean',
                'gps_alt_range': 'max'
            }).reset_index()
            gps_feat = gps_feat.merge(gps_mobility, on='date', how='left')

        # ── 12. mUsageStats ──
        mu = pd.read_parquet(parquet_dir / "ch2025_mUsageStats.parquet")
        mu = mu[mu['subject_id'] == subj].copy()
        mu['date'] = mu['timestamp'].dt.date
        usage_feat = mu.groupby('date')['m_usage_stats'].agg(['count']).reset_index()
        usage_feat.columns = ['date','usage_count']

        # Usage stats categories
        if 'm_usage_stats' in mu.columns and mu['m_usage_stats'].dtype == object:
            usage_cats = []
            for val in mu['m_usage_stats']:
                if isinstance(val, np.ndarray) and val.ndim == 2 and val.shape[1] >= 3:
                    # First column: category index
                    cats = val[:, 0].astype(int)
                    cats_dict = {}
                    for c in cats:
                        cats_dict[c] = cats_dict.get(c, 0) + 1
                    total = len(cats)
                    # Top categories
                    for i in range(min(5, len(cats_dict))):
                        cats_dict[f'top_{i+1}'] = cats_dict.get(i, 0) / total
                    # Fill remaining
                    for i in range(5):
                        if f'top_{i+1}' not in cats_dict:
                            cats_dict[f'top_{i+1}'] = 0.0
                    usage_cats.append(cats_dict)
                else:
                    usage_cats.append({f'top_{i+1}': 0.0 for i in range(5)})
            usage_cats_df = pd.DataFrame(usage_cats, index=mu.index)
            usage_cats_df['date'] = mu['date'].values
            usage_daily_cats = usage_cats_df.groupby('date').mean(numeric_only=True).reset_index()
            usage_feat = usage_feat.merge(usage_daily_cats, on='date', how='left')

        # ── Merge all features ──
        df_feat = ml_feat
        for other in [wl_feat, wl_night, wp_feat, hr_daily, hr_night, act_feat,
                     screen_feat, charge_feat, wifi_feat, ble_feat, amb_feat,
                     gps_feat, usage_feat]:
            df_feat = df_feat.merge(other, on='date', how='left')

        # ── Add temporal features ──
        dates_parsed = pd.to_datetime(df_feat['date'])
        df_feat['dayofweek'] = dates_parsed.dt.dayofweek
        df_feat['dayofyear'] = dates_parsed.dt.dayofyear
        df_feat['is_weekend'] = df_feat['dayofweek'].isin([5, 6]).astype(int)

        # Merge labels
        df_feat = df_feat.merge(
            subj_labels[['lifelog_date','date','Q1','Q2','Q3','S1','S2','S3','S4']],
            on='date', how='left'
        )

        # ── Rolling features ──
        df_feat = df_feat.sort_values('date').reset_index(drop=True)
        numeric_cols = ['light_mean','wlight_mean','step_mean','hr_mean','activity_mean',
                       'screen_mean','charge_mean','wlight_night_mean','hr_night_mean',
                       'wlight_night_mean']
        numeric_cols = [c for c in numeric_cols if c in df_feat.columns]

        for col in numeric_cols:
            for w in [3, 7]:
                rm = df_feat[col].rolling(w, min_periods=1).mean()
                rs = df_feat[col].rolling(w, min_periods=1).std().fillna(0)
                df_feat[f'{col}_rm{w}'] = rm.values
                df_feat[f'{col}_rs{w}'] = rs.values
            # Rate of change
            df_feat[f'{col}_diff1'] = df_feat[col].diff(1).fillna(0)
            df_feat[f'{col}_diff3'] = df_feat[col].diff(3).fillna(0)

        # ── Day-of-week deviation (rhythm) ──
        for col in numeric_cols:
            if col in df_feat.columns:
                dow_means = df_feat.groupby('dayofweek')[col].mean()
                df_feat[f'{col}_dow_dev'] = df_feat.apply(
                    lambda r: r[col] - dow_means.get(r['dayofweek'], r[col]), axis=1
                )

        # ── Personalization: per-subject z-score ──
        feat_cols = [c for c in df_feat.columns if c not in META | set(TARGETS)
                    and df_feat[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int, np.float64, np.float32]]
        feat_cols = [c for c in feat_cols if df_feat[c].nunique() > 2]

        # Compute per-column mean/std for z-score
        col_stats = {}
        for c in feat_cols:
            mean_val = df_feat[c].mean()
            std_val = df_feat[c].std()
            col_stats[c] = (mean_val, std_val if std_val > 0 else 1.0)

        for c in feat_cols:
            mean_val, std_val = col_stats[c]
            df_feat[f'{c}_zscore'] = (df_feat[c] - mean_val) / std_val

        all_rows.append(df_feat)

    combined = pd.concat(all_rows, ignore_index=True)
    return combined


# ── Model Training ──

def train_per_target(train_df, seed, fold):
    """Train models for all targets on one fold."""
    fold_dir = DATA_PROCESSED / f"v56_seed{seed}_fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    feat_cols = [c for c in train_df.columns if c not in META | set(TARGETS)
                and train_df[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int, np.float64, np.float32]]
    feat_cols = [c for c in feat_cols if train_df[c].nunique() > 2]

    # Drop constant cols
    feat_cols = [c for c in feat_cols if train_df[c].nunique() > 1]

    for tgt in TARGETS:
        train_tgt = train_df.dropna(subset=[tgt])
        if len(train_tgt) == 0:
            continue

        X = train_tgt[feat_cols].values
        y = train_tgt[tgt].values
        target_mean = y.mean()

        # LGBM config
        cfg = {
            'objective': 'binary', 'metric': 'binary_logloss',
            'num_leaves': 8, 'max_depth': 3,
            'learning_rate': 0.02, 'n_estimators': 200,
            'subsample': 0.6, 'colsample_bytree': 0.6,
            'reg_alpha': 2.0, 'reg_lambda': 5.0,
            'min_child_samples': 15, 'verbose': -1, 'seed': seed
        }

        model = lgb.LGBMClassifier(**cfg)
        model.fit(X, y)

        oof_pred = model.predict_proba(X)[:, 1]
        oof_pred = np.clip(oof_pred + (target_mean - oof_pred.mean()), 0.0001, 0.9999)
        oof_loss = log_loss(y, oof_pred)

        results[tgt] = {
            'model': model, 'oof_loss': oof_loss,
            'oof_pred': oof_pred, 'target_mean': target_mean,
            'feat_cols': feat_cols
        }

    # Save models
    for tgt, res in results.items():
        joblib_path = fold_dir / f'{tgt}_model.pkl'
        import joblib
        joblib.dump(res['model'], joblib_path)

    return results


def main():
    import joblib

    start_time = time.time()
    log.info("=" * 60)
    log.info("V56 — Rich Feature Engineering + Ensemble")
    log.info("=" * 60)

    # Load labels
    labels = pd.read_csv(DATA_RAW / "ch2026_metrics_train.csv", parse_dates=['sleep_date', 'lifelog_date'])
    subjects = sorted(labels['subject_id'].unique())
    log.info(f"Labels: {labels.shape}, Subjects: {len(subjects)}")

    # Build features
    log.info("Building rich features...")
    features = build_features(subjects, labels)
    log.info(f"Features shape: {features.shape}")

    # Save features
    features.to_parquet(DATA_PROCESSED / "features_v56_rich.parquet")
    log.info("Features saved")

    # Evaluate with GroupKFold
    log.info("Evaluating with GroupKFold (5 folds)...")
    gkf = GroupKFold(n_splits=5)

    feat_cols = [c for c in features.columns if c not in META | set(TARGETS)
                and features[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int, np.float64, np.float32]]
    feat_cols = [c for c in feat_cols if features[c].nunique() > 1]

    all_oof = {tgt: np.zeros(len(features)) for tgt in TARGETS}
    oof_counts = {tgt: np.zeros(len(features)) for tgt in TARGETS}

    for fold, (train_idx, val_idx) in enumerate(gkf.split(features, groups=features['subject_id'])):
        log.info(f"\nFold {fold+1}/{5}")
        fold_train = features.iloc[train_idx]
        fold_val = features.iloc[val_idx]

        for tgt in TARGETS:
            train_tgt = fold_train.dropna(subset=[tgt])
            if len(train_tgt) == 0:
                continue

            X_train = train_tgt[feat_cols].fillna(0).values
            y_train = train_tgt[tgt].values
            target_mean = y_train.mean()

            X_val = fold_val.loc[val_idx, feat_cols].fillna(0).values

            cfg = {
                'objective': 'binary', 'metric': 'binary_logloss',
                'num_leaves': 8, 'max_depth': 3,
                'learning_rate': 0.02, 'n_estimators': 200,
                'subsample': 0.6, 'colsample_bytree': 0.6,
                'reg_alpha': 2.0, 'reg_lambda': 5.0,
                'min_child_samples': 15, 'verbose': -1, 'seed': 42
            }

            model = lgb.LGBMClassifier(**cfg)
            model.fit(X_train, y_train)

            oof_pred = model.predict_proba(X_val)[:, 1]
            oof_pred = np.clip(oof_pred + (target_mean - oof_pred.mean()), 0.0001, 0.9999)

            all_oof[tgt][val_idx] = oof_pred
            oof_counts[tgt][val_idx] = 1

    # Compute OOF loss
    log.info("\n=== OOF CV Scores ===")
    total_loss = 0
    for tgt in TARGETS:
        mask = oof_counts[tgt] > 0
        if mask.sum() > 0:
            y_true = features.loc[mask, tgt].values
            y_pred = all_oof[tgt][mask]
            loss = log_loss(y_true, y_pred)
            total_loss += loss
            log.info(f"  {tgt}: OOF_logloss={loss:.4f}, pred_mean={all_oof[tgt][mask].mean():.4f}, rate={y_true.mean():.4f}")
    avg_loss = total_loss / len(TARGETS)
    log.info(f"\n  Average OOF log_loss: {avg_loss:.4f}")

    log.info(f"\nTotal time: {time.time() - start_time:.0f}s")


if __name__ == "__main__":
    main()
