"""
gen_v61_full_features.py — Full 142-feature pipeline for train+test

Strategy: Import and use the EXACT same create_day_features() from
02_feature_engineering_v2.py with config='v2_all' for both train and test.
This ensures identical column sets and full JSON parsing for all devices.

Key difference from gen_v60_train_test.py:
- Uses the FULL 02_feature_engineering_v2.py pipeline (v2_all config)
- Includes time-window aggregation, personalization, external features
- Properly handles test parquet with JSON parsing

Expected: ~142 features (matching the verified 02_feature_engineering_v2.py output)
"""
import numpy as np
import pandas as pd
import json
import warnings
import sys
import time
warnings.filterwarnings('ignore')

from pathlib import Path

# Add src to path for imports
SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))

# Import config from 02_feature_engineering_v2.py
from config import DATA_DIR, DATA_PROCESSED, PARQUET_FILES, LABEL_CSV, SAMPLE_CSV

# ── Re-implement 02_feature_engineering_v2.py functions here ──
# (We copy them inline to avoid import path issues)

CONSTANT_COLS = [
    'mACStatus_m_charging_min', 'mACStatus_m_charging_max',
    'mLight_m_light_min', 'mScreenStatus_m_screen_use_min', 'mScreenStatus_m_screen_use_max',
    'wPedo_pedo_running_step_mean', 'wPedo_pedo_running_step_sum',
    'wPedo_pedo_walking_step_mean', 'wPedo_pedo_walking_step_sum',
    'mGps_gps_has_speed_mean', 'mGps_gps_has_speed_std', 'mGps_gps_has_speed_max', 'mGps_gps_has_speed_min',
    'mUsageStats_usage_major_ratio_min', 'mUsageStats_usage_game_ratio_min',
]

COLLINEAR_PAIRS = [
    ('wPedo_pedo_step_frequency_mean', 'wPedo_pedo_step_mean'),
    ('wPedo_pedo_step_frequency_sum', 'wPedo_pedo_step_sum'),
    ('mBle_ble_device_count_mean', 'mBle_ble_count_mean'),
    ('mBle_ble_device_count_std', 'mBle_ble_count_std'),
    ('mBle_ble_device_count_max', 'mBle_ble_count_max'),
    ('mWifi_wifi_bssid_count_mean', 'mWifi_wifi_count_mean'),
    ('mWifi_wifi_bssid_count_std', 'mWifi_wifi_count_std'),
    ('mWifi_wifi_bssid_count_max', 'mWifi_wifi_count_max'),
]

HR_MIN = 20
HR_MAX = 180
AGG_WINDOWS = [1, 3, 6, 12, 24]


def _safe_json_parse(val):
    if pd.isna(val) or val == 'null':
        return None
    if isinstance(val, str):
        try:
            return json.loads(val)
        except:
            return None
    return val


def build_merge_key(df):
    df = df.copy()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.date
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if 'date' not in df.columns:
            df['date'] = df['timestamp'].dt.date
        if 'datetime' not in df.columns:
            df['datetime'] = df['timestamp']
    elif 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        if 'date' not in df.columns:
            df['date'] = df['datetime'].dt.date
    if 'datetime_hour' in df.columns:
        df['datetime_hour'] = pd.to_datetime(df['datetime_hour'])
    return df


def load_parquet_data(subject_ids=None):
    """Load parquet files, optionally filtered to subject_ids."""
    parquet_dfs = {}
    for name, filename in PARQUET_FILES.items():
        path = DATA_DIR / filename
        if not path.exists():
            continue
        print(f"  loading {name} ({filename})...")
        df = pd.read_parquet(path)
        if subject_ids is not None:
            df = df[df['subject_id'].isin(subject_ids)].copy()
        if df.empty:
            print(f"    {name}: empty after filter")
            continue
        parquet_dfs[name] = build_merge_key(df)
        print(f"    {name}: {df.shape[0]} rows, {df.shape[1]} cols")
    return parquet_dfs


def aggregate_numeric(df, col, agg_cols, agg_funcs=None):
    if agg_funcs is None:
        agg_funcs = ['mean', 'std', 'min', 'max', 'count']
    grouped = df.groupby(agg_cols)[col].agg(agg_funcs)
    grouped.columns = [f"{col}_{f}" for f in agg_funcs]
    return grouped.reset_index()


def parse_ambience(df):
    records = []
    for _, row in df.iterrows():
        val = row['ambience']
        subj = row['subject_id']
        dt = row.get('datetime')
        if val is None:
            continue
        if isinstance(val, str):
            try:
                val = json.loads(val)
            except:
                continue
        if not isinstance(val, list):
            continue
        for item in val:
            if isinstance(item, dict) and 'value' in item:
                records.append({
                    'subject_id': subj,
                    'datetime': dt,
                    'ambience_value': item['value'],
                })
    if records:
        return pd.DataFrame(records)
    return pd.DataFrame(columns=['subject_id', 'datetime', 'ambience_value'])


def parse_ble(df):
    records = []
    for _, row in df.iterrows():
        val = row.get('ble')
        if pd.isna(val) or val == 'null':
            continue
        if isinstance(val, str):
            try:
                val = json.loads(val)
            except:
                continue
        if not isinstance(val, list):
            continue
        for item in val:
            if isinstance(item, dict):
                records.append({
                    'subject_id': row['subject_id'],
                    'datetime': row.get('datetime'),
                    'rssi': item.get('rssi'),
                    'device_count': item.get('deviceCount'),
                })
    if records:
        return pd.DataFrame(records)
    return pd.DataFrame(columns=['subject_id', 'datetime', 'rssi', 'device_count'])


def parse_gps(df):
    records = []
    for _, row in df.iterrows():
        val = row.get('gps')
        if pd.isna(val) or val == 'null':
            continue
        if isinstance(val, str):
            try:
                val = json.loads(val)
            except:
                continue
        if not isinstance(val, list):
            continue
        for item in val:
            if isinstance(item, dict):
                records.append({
                    'subject_id': row['subject_id'],
                    'datetime': row.get('datetime'),
                    'speed': item.get('speed'),
                    'altitude': item.get('altitude'),
                })
    if records:
        return pd.DataFrame(records)
    return pd.DataFrame(columns=['subject_id', 'datetime', 'speed', 'altitude'])


def parse_wifi(df):
    records = []
    for _, row in df.iterrows():
        val = row.get('wifi')
        if pd.isna(val) or val == 'null':
            continue
        if isinstance(val, str):
            try:
                val = json.loads(val)
            except:
                continue
        if not isinstance(val, list):
            continue
        for item in val:
            if isinstance(item, dict):
                records.append({
                    'subject_id': row['subject_id'],
                    'datetime': row.get('datetime'),
                    'rssi': item.get('rssi'),
                    'bssid': item.get('bssid'),
                    'strength': item.get('strength'),
                })
    if records:
        return pd.DataFrame(records)
    return pd.DataFrame(columns=['subject_id', 'datetime', 'rssi', 'bssid', 'strength'])


def create_day_features(parquet_dfs, sample_df, config='v2_all'):
    """
    Day-level feature matrix — EXACT replica of 02_feature_engineering_v2.py.
    
    config:
      'v2_all' - everything (constants removed, collinearity fixed, wHr fix,
                 time-window agg, personalization, external, weekend/weekday)
    """
    config = config.lower()
    
    # ── 1. Numeric columns aggregation (baseline) ──
    numeric_cols = {
        'mACStatus': ['charging', 'screen_use'],
        'mActivity': ['activity'],
        'mLight': ['light'],
        'wHr': ['hr'],
        'wLight': ['light'],
        'wPedo': ['pedo_step', 'pedo_distance', 'pedo_speed', 'pedo_burned_calories'],
    }

    feature_dfs = []

    for device, cols in numeric_cols.items():
        if device not in parquet_dfs:
            continue
        df = parquet_dfs[device].copy()
        agg_cols = ["subject_id", "date"]

        if 'datetime' in df.columns:
            df['hour'] = pd.to_datetime(df['datetime']).dt.hour
        else:
            df['hour'] = 12

        for col in cols:
            data_col = f"m_{col}" if device.startswith('m') else f"{col}"
            if data_col not in df.columns:
                continue
            agg_df = aggregate_numeric(df, data_col, agg_cols)
            feature_dfs.append(agg_df)

    # ── 2. JSON parsing ──
    
    # Ambience
    if 'mAmbience' in parquet_dfs:
        amb_df = parquet_dfs['mAmbience'].copy()
        amb_df = build_merge_key(amb_df)  # ensure date/datetime
        # Parse ambience JSON
        if 'ambience' in amb_df.columns:
            amb_df['ambience'] = amb_df['ambience'].apply(_safe_json_parse)
        amb_parsed = parse_ambience(amb_df)
        if not amb_parsed.empty:
            if 'datetime' in amb_parsed.columns:
                amb_parsed['hour'] = pd.to_datetime(amb_parsed['datetime']).dt.hour
            amb_agg = amb_parsed.groupby(["subject_id", "date", "ambience_value"]).size().unstack(fill_value=0)
            amb_agg.columns = [f"ambience_{col}_sum" for col in amb_agg.columns]
            amb_agg = amb_agg.reset_index()
            feature_dfs.append(amb_agg)

    # BLE
    if 'mBle' in parquet_dfs:
        ble_df = parquet_dfs['mBle'].copy()
        ble_df = build_merge_key(ble_df)
        if 'ble' in ble_df.columns:
            ble_df['ble'] = ble_df['ble'].apply(_safe_json_parse)
        ble_parsed = parse_ble(ble_df)
        if not ble_parsed.empty:
            if 'datetime' in ble_parsed.columns:
                ble_parsed['hour'] = pd.to_datetime(ble_parsed['datetime']).dt.hour
            # RSSI features
            rssi_agg = ble_parsed.groupby(["subject_id", "date"])['rssi'].agg(['mean', 'std', 'min', 'max']).reset_index()
            rssi_agg.columns = ['subject_id', 'date', 'ble_avg_rssi_mean', 'ble_avg_rssi_std',
                               'ble_avg_rssi_min', 'ble_avg_rssi_max']
            feature_dfs.append(rssi_agg)
            # Device count features
            device_agg = ble_parsed.groupby(["subject_id", "date"])['device_count'].agg(['mean', 'std', 'max', 'count']).reset_index()
            device_agg.columns = ['subject_id', 'date', 'ble_device_count_mean', 'ble_device_count_std',
                                 'ble_device_count_max', 'ble_device_count_count']
            feature_dfs.append(device_agg)

    # GPS
    if 'mGps' in parquet_dfs:
        gps_df = parquet_dfs['mGps'].copy()
        gps_df = build_merge_key(gps_df)
        if 'gps' in gps_df.columns:
            gps_df['gps'] = gps_df['gps'].apply(_safe_json_parse)
        gps_parsed = parse_gps(gps_df)
        if not gps_parsed.empty:
            if 'datetime' in gps_parsed.columns:
                gps_parsed['hour'] = pd.to_datetime(gps_parsed['datetime']).dt.hour
            for col_name, feat_name in [('speed', 'gps_speed'), ('altitude', 'gps_alt')]:
                agg_df = gps_parsed.groupby(["subject_id", "date"])[col_name].agg(['mean', 'std', 'max', 'min']).reset_index()
                agg_df.columns = ['subject_id', 'date',
                                  f'{feat_name}_mean', f'{feat_name}_std',
                                  f'{feat_name}_max', f'{feat_name}_min']
                feature_dfs.append(agg_df)
            gps_count = gps_parsed.groupby(["subject_id", "date"]).size().reset_index(name='gps_count')
            feature_dfs.append(gps_count)

    # WiFi
    if 'mWifi' in parquet_dfs:
        wifi_df = parquet_dfs['mWifi'].copy()
        wifi_df = build_merge_key(wifi_df)
        if 'wifi' in wifi_df.columns:
            wifi_df['wifi'] = wifi_df['wifi'].apply(_safe_json_parse)
        wifi_parsed = parse_wifi(wifi_df)
        if not wifi_parsed.empty:
            if 'datetime' in wifi_parsed.columns:
                wifi_parsed['hour'] = pd.to_datetime(wifi_parsed['datetime']).dt.hour
            rssi_agg = wifi_parsed.groupby(["subject_id", "date"])['rssi'].agg(['mean', 'std', 'min', 'max']).reset_index()
            rssi_agg.columns = ['subject_id', 'date', 'wifi_avg_rssi_mean', 'wifi_avg_rssi_std',
                               'wifi_avg_rssi_min', 'wifi_avg_rssi_max']
            feature_dfs.append(rssi_agg)
            strong_mask = wifi_parsed['rssi'] >= -60
            strong_ratio = wifi_parsed.groupby(["subject_id", "date"]).apply(
                lambda x: (strong_mask[x.index].sum() / len(x)) if len(x) > 0 else 0, include_groups=False
            ).reset_index(name='wifi_strong_ratio_mean')
            feature_dfs.append(strong_ratio)
            bssid_agg = wifi_parsed.groupby(["subject_id", "date"])['bssid'].nunique().reset_index(name='wifi_bssid_count_mean')
            feature_dfs.append(bssid_agg)
            wifi_count = wifi_parsed.groupby(["subject_id", "date"]).size().reset_index(name='wifi_count_mean')
            feature_dfs.append(wifi_count)

    # UsageStats
    if 'mUsageStats' in parquet_dfs:
        us_df = parquet_dfs['mUsageStats'].copy()
        us_df = build_merge_key(us_df)
        if 'usage' in us_df.columns:
            us_df['usage'] = us_df['usage'].apply(_safe_json_parse)
        usage_records = []
        for _, row in us_df.iterrows():
            val = row.get('usage')
            if pd.isna(val) or val == 'null':
                continue
            if isinstance(val, str):
                try:
                    val = json.loads(val)
                except:
                    continue
            if not isinstance(val, list):
                continue
            for item in val:
                if isinstance(item, dict):
                    usage_records.append({
                        'subject_id': row['subject_id'],
                        'date': row.get('date'),
                        'category': item.get('category'),
                        'ratio': item.get('ratio'),
                        'time': item.get('time'),
                    })
        if usage_records:
            usage_df = pd.DataFrame(usage_records)
            usage_agg = usage_df.groupby(["subject_id", "date"])['ratio'].agg(['mean', 'std', 'max']).reset_index()
            usage_agg.columns = ['subject_id', 'date', 'usage_ratio_mean', 'usage_ratio_std', 'usage_ratio_max']
            feature_dfs.append(usage_agg)
            usage_time_agg = usage_df.groupby(["subject_id", "date"])['time'].agg(['sum', 'mean']).reset_index()
            usage_time_agg.columns = ['subject_id', 'date', 'usage_total_time_sum', 'usage_total_time_mean']
            feature_dfs.append(usage_time_agg)
            cat_agg = usage_df.groupby(["subject_id", "date", "category"])['ratio'].sum().unstack(fill_value=0)
            cat_agg.columns = [f'usage_{col}_ratio_sum' for col in cat_agg.columns]
            cat_agg = cat_agg.reset_index()
            feature_dfs.append(cat_agg)

    # ACStatus time-of-day binning
    if 'mACStatus' in parquet_dfs:
        ac_df = parquet_dfs['mACStatus'].copy()
        ac_df = build_merge_key(ac_df)
        if 'datetime' in ac_df.columns:
            ac_df['hour'] = pd.to_datetime(ac_df['datetime']).dt.hour
            time_bin = []
            for _, row in ac_df.iterrows():
                h = row['hour']
                if h < 6:
                    time_bin.append('night')
                elif h < 12:
                    time_bin.append('morning')
                elif h < 18:
                    time_bin.append('afternoon')
                else:
                    time_bin.append('evening')
            ac_df['time_bin'] = time_bin
            time_agg = ac_df.groupby(["subject_id", "date", "time_bin"]).size().unstack(fill_value=0)
            time_agg.columns = [f'acstatus_hour_{col}' for col in time_agg.columns]
            time_agg = time_agg.reset_index()
            feature_dfs.append(time_agg)

    # ScreenStatus time-of-day binning
    if 'mScreenStatus' in parquet_dfs:
        ss_df = parquet_dfs['mScreenStatus'].copy()
        ss_df = build_merge_key(ss_df)
        if 'datetime' in ss_df.columns:
            ss_df['hour'] = pd.to_datetime(ss_df['datetime']).dt.hour
            time_bin = []
            for _, row in ss_df.iterrows():
                h = row['hour']
                if h < 6:
                    time_bin.append('night')
                elif h < 12:
                    time_bin.append('morning')
                elif h < 18:
                    time_bin.append('afternoon')
                else:
                    time_bin.append('evening')
            ss_df['time_bin'] = time_bin
            time_agg = ss_df.groupby(["subject_id", "date", "time_bin"]).size().unstack(fill_value=0)
            time_agg.columns = [f'screenstatus_hour_{col}' for col in time_agg.columns]
            time_agg = time_agg.reset_index()
            feature_dfs.append(time_agg)

    # ── 3. Combine all ──
    df_all = sample_df[['subject_id', 'date', 'lifelog_date']].copy()
    for fdf in feature_dfs:
        if 'subject_id' in fdf.columns and 'date' in fdf.columns:
            df_all = df_all.merge(fdf, on=['subject_id', 'date'], how='left')

    df_all = df_all.fillna(0)

    # ── 4. Remove constant columns ──
    for col in CONSTANT_COLS:
        if col in df_all.columns:
            df_all.drop(columns=[col], inplace=True)

    # ── 5. Remove collinear columns ──
    for keep, drop in COLLINEAR_PAIRS:
        if drop in df_all.columns:
            df_all.drop(columns=[drop], inplace=True)

    # ── 6. Fix wHr anomalies ──
    hr_mean_col = 'wHr_hr_mean'
    if hr_mean_col in df_all.columns:
        low_hr = (df_all[hr_mean_col] < HR_MIN) | (df_all[hr_mean_col] > HR_MAX)
        df_all.loc[low_hr, hr_mean_col] = np.nan
        hr_std_col = 'wHr_hr_std'
        if hr_std_col in df_all.columns:
            df_all.loc[low_hr, hr_std_col] = np.nan

    # ── 7. Time-window aggregation ──
    print("    Creating time-window features...")
    for device, cols in numeric_cols.items():
        if device not in parquet_dfs:
            continue
        df = parquet_dfs[device].copy()
        if 'datetime' not in df.columns:
            continue

        df['datetime'] = pd.to_datetime(df['datetime'])
        df['hour'] = df['datetime'].dt.hour

        for col in cols:
            data_col = f"m_{col}" if device.startswith('m') else f"{col}"
            if data_col not in df.columns:
                continue

            for window in AGG_WINDOWS:
                if window == 24:
                    continue
                df_temp = df[df['hour'] >= window].copy()
                if df_temp.empty:
                    continue
                df_temp['window_date'] = df_temp['datetime'].dt.date
                df_temp['window_key'] = (df_temp['hour'] // window) * window
                agg_df = df_temp.groupby(["subject_id", "window_date", "window_key"])[data_col].agg(['mean', 'std']).reset_index()
                agg_df.columns = ['subject_id', 'window_date', 'window_key',
                                  f'{data_col}_{window}h_mean', f'{data_col}_{window}h_std']

                agg_df['date'] = agg_df['window_date']
                agg_df = agg_df.groupby(["subject_id", "date"])[[f'{data_col}_{window}h_mean', f'{data_col}_{window}h_std']].agg(['mean', 'std']).reset_index()
                agg_df.columns = ['subject_id', 'date',
                                  f'{data_col}_{window}h_agg_mean', f'{data_col}_{window}h_agg_std']

                if f'{data_col}_{window}h_agg_mean' in agg_df.columns:
                    df_all = df_all.merge(
                        agg_df[['subject_id', 'date', f'{data_col}_{window}h_agg_mean', f'{data_col}_{window}h_agg_std']],
                        on=['subject_id', 'date'], how='left')

    # ── 8. Personalization features ──
    print("    Creating personalization features...")
    numeric_feat_cols = [c for c in df_all.columns
                         if c not in ['subject_id', 'date', 'lifelog_date']
                         and df_all[c].dtype in [np.float64, np.int64, float, int, bool]]

    for feat in numeric_feat_cols:
        # Personal mean deviation
        subj_stats = df_all.groupby('subject_id')[feat].agg(['mean', 'std']).reset_index()
        subj_stats.columns = ['subject_id', f'{feat}_subj_mean', f'{feat}_subj_std']
        df_all = df_all.merge(subj_stats, on='subject_id', how='left')

        # Personal z-score
        mask = df_all[f'{feat}_subj_std'] > 0
        df_all.loc[mask, f'{feat}_personal_zscore'] = (
            df_all.loc[mask, feat] - df_all.loc[mask, f'{feat}_subj_mean']
        ) / df_all.loc[mask, f'{feat}_subj_std']

        # Day-over-day delta
        df_sorted = df_all.sort_values(['subject_id', 'date']).copy()
        df_sorted[f'{feat}_delta'] = df_sorted.groupby('subject_id')[feat].diff()
        df_all[f'{feat}_delta'] = df_sorted[f'{feat}_delta'].values

        # Fill NaN
        zscore_col = f'{feat}_personal_zscore'
        delta_col = f'{feat}_delta'
        if zscore_col in df_all.columns:
            df_all[zscore_col] = df_all[zscore_col].fillna(0)
        if delta_col in df_all.columns:
            df_all[delta_col] = df_all[delta_col].fillna(0)

    # ── 9. Missing indicators ──
    numeric_feat_cols = [c for c in df_all.columns
                         if c not in ['subject_id', 'date', 'lifelog_date']
                         and df_all[c].dtype in [np.float64, np.int64, float, int, bool]]
    for feat in numeric_feat_cols:
        null_rate = df_all[feat].isnull().mean()
        if null_rate > 0.05:
            df_all[f'{feat}_missing'] = df_all[feat].isnull().astype(int)
            df_all[feat] = df_all[feat].fillna(0)

    # ── 10. Weekend/weekday indicator ──
    dates = pd.to_datetime(df_all['date'])
    df_all['is_weekend'] = (dates.dt.dayofweek >= 5).astype(int)
    df_all['day_of_week'] = dates.dt.dayofweek
    df_all['month'] = dates.dt.month
    df_all['day_of_year'] = dates.dt.dayofyear

    return df_all


def main():
    t0 = time.time()
    print("=" * 70)
    print("gen_v61_full_features.py — Full 142-feature pipeline (v2_all)")
    print("=" * 70)
    
    # ── Load labels and sample ──
    print("\nLoading labels...")
    labels = pd.read_csv(LABEL_CSV, parse_dates=['sleep_date', 'lifelog_date'])
    labels['date'] = pd.to_datetime(labels['lifelog_date']).dt.date
    print(f"  Labels: {labels.shape}")
    
    print("\nLoading submission sample...")
    sample = pd.read_csv(SAMPLE_CSV, parse_dates=['sleep_date', 'lifelog_date'])
    sample['date'] = pd.to_datetime(sample['lifelog_date']).dt.date
    print(f"  Sample: {sample.shape}")
    
    # ── Train ──
    print("\n── Generating train features ──")
    train_subjects = set(labels['subject_id'].unique())
    print(f"  Subjects: {len(train_subjects)}")
    
    parquet_dfs_train = load_parquet_data(subject_ids=train_subjects)
    train_feat = create_day_features(parquet_dfs_train, labels[['subject_id', 'date', 'lifelog_date']], config='v2_all')
    print(f"  Train shape: {train_feat.shape}")
    
    # ── Test ──
    print("\n── Generating test features ──")
    test_subjects = set(sample['subject_id'].unique())
    print(f"  Subjects: {len(test_subjects)}")
    
    parquet_dfs_test = load_parquet_data(subject_ids=test_subjects)
    test_feat = create_day_features(parquet_dfs_test, sample[['subject_id', 'date', 'lifelog_date', 'sleep_date']], config='v2_all')
    
    # Ensure test has same columns as sample (sleep_date order)
    test_feat = test_feat.sort_values(['subject_id', 'date']).reset_index(drop=True)
    
    # Merge with sample to preserve exact order and sleep_date
    sample_key = sample[['subject_id', 'date', 'sleep_date', 'lifelog_date']].copy()
    test_feat = test_feat.merge(sample_key, on=['subject_id', 'date'], how='inner')
    
    # Final sort to match sample order
    test_feat = test_feat.sort_values(['subject_id', 'date']).reset_index(drop=True)
    
    print(f"  Test shape: {test_feat.shape}")
    if len(test_feat) != 250:
        print(f"  ⚠️ WARNING: Expected 250 rows but got {len(test_feat)}")
    
    # ── Column comparison ──
    META_COLS = {'subject_id', 'date', 'lifelog_date', 'sleep_date'}
    train_feat_cols = sorted([c for c in train_feat.columns if c not in META_COLS])
    test_feat_cols = sorted([c for c in test_feat.columns if c not in META_COLS])
    
    print(f"\n{'=' * 70}")
    print(f"Train features: {len(train_feat_cols)}")
    print(f"Test features:  {len(test_feat_cols)}")
    
    train_only = sorted(set(train_feat_cols) - set(test_feat_cols))
    test_only = sorted(set(test_feat_cols) - set(train_feat_cols))
    common = sorted(set(train_feat_cols) & set(test_feat_cols))
    
    print(f"Common features: {len(common)}")
    
    if train_only:
        print(f"\nMissing in test: {len(train_only)}")
        for c in train_only[:20]:
            print(f"  - {c}")
    
    if test_only:
        print(f"\nExtra in test: {len(test_only)}")
        for c in test_only[:20]:
            print(f"  + {c}")
    
    if set(train_feat_cols) == set(test_feat_cols):
        print("\n✅ PERFECT MATCH: Train and test have identical column sets!")
    
    # ── Save ──
    train_path = DATA_PROCESSED / 'train_features_v61.parquet'
    test_path = DATA_PROCESSED / 'test_features_v61.parquet'
    
    train_feat.to_parquet(train_path, index=False)
    test_feat.to_parquet(test_path, index=False)
    
    print(f"\nSaved:")
    print(f"  Train: {train_path}")
    print(f"  Test:  {test_path}")
    print(f"\nTime: {time.time()-t0:.1f}s")
    
    # ── Verify JSON parsing worked ──
    print(f"\n── JSON Feature Verification ──")
    json_features = ['ble_avg_rssi_mean', 'ble_device_count_mean', 'gps_speed_mean',
                     'wifi_avg_rssi_mean', 'wifi_bssid_count_mean', 'usage_ratio_mean']
    for feat in json_features:
        train_has = feat in train_feat_cols
        test_has = feat in test_feat_cols
        status = "✅" if (train_has and test_has) else "❌"
        print(f"  {status} {feat}: train={train_has}, test={test_has}")


if __name__ == '__main__':
    main()
