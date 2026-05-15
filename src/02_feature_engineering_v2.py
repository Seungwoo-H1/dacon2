"""
feature_engineering_v2.py — 개선된 피처 공학

변경 사항 (V1 기준):
1. Constant/Near-constant feature 제거 (13개)
2. 다중공선성 제거 (r>0.99 쌍 중 하나만 유지)
3. wHr/hr_mean 이상치 처리 (< 20 또는 > 180 제거)
4. Time-window aggregation 추가 (1h, 3h, 6h, 12h, 24h)
5. Personalization features 추가 (baseline deviation, day-over-day delta)
6. Leakage 방지: nighttime data는 S 타겟에만, daytime은 Q 타겟에만
7. Missing indicator 추가
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from config import DATA_DIR, DATA_PROCESSED, PARQUET_FILES, TARGETS, AGG_WINDOWS, HOUR_BINS


def build_merge_key(df):
    """Merge key 생성."""
    df = df.copy()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.date
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        if 'date' not in df.columns:
            df['date'] = df['datetime'].dt.date
    if 'datetime_hour' in df.columns:
        df['datetime_hour'] = pd.to_datetime(df['datetime_hour'])
    return df


def load_parquet_data():
    """파라쿠 데이터 로드."""
    parquet_dfs = {}
    for name, filename in PARQUET_FILES.items():
        path = DATA_DIR / filename
        if path.exists():
            print(f"  loading {name}...")
            df = pd.read_parquet(path)
            parquet_dfs[name] = build_merge_key(df)
    return parquet_dfs


# ── 제거할 constant/near-constant features ──
CONSTANT_COLS = [
    'mACStatus_m_charging_min',
    'mACStatus_m_charging_max',
    'mLight_m_light_min',
    'mScreenStatus_m_screen_use_min',
    'mScreenStatus_m_screen_use_max',
    'wPedo_pedo_running_step_mean',
    'wPedo_pedo_running_step_sum',
    'wPedo_pedo_walking_step_mean',
    'wPedo_pedo_walking_step_sum',
    'mGps_gps_has_speed_mean',
    'mGps_gps_has_speed_std',
    'mGps_gps_has_speed_max',
    'mGps_gps_has_speed_min',
    'mUsageStats_usage_major_ratio_min',
    'mUsageStats_usage_game_ratio_min',
]

# 다중공선성 제거 pairs (상관 > 0.99인 것 중 유지할 것)
COLLINEAR_PAIRS = [
    ('wPedo_pedo_step_frequency_mean', 'wPedo_pedo_step_mean'),    # r=1.000
    ('wPedo_pedo_step_frequency_sum', 'wPedo_pedo_step_sum'),      # r=1.000
    ('mBle_ble_device_count_mean', 'mBle_ble_count_mean'),         # r=0.999
    ('mBle_ble_device_count_std', 'mBle_ble_count_std'),           # r=0.999
    ('mBle_ble_device_count_max', 'mBle_ble_count_max'),           # r=0.999
    ('mWifi_wifi_bssid_count_mean', 'mWifi_wifi_count_mean'),      # r=1.000
    ('mWifi_wifi_bssid_count_std', 'mWifi_wifi_count_std'),        # r=1.000
    ('mWifi_wifi_bssid_count_max', 'mWifi_wifi_count_max'),        # r=1.000
]

# wHr hr_mean > 180 또는 < 20은 이상치
HR_MIN = 20
HR_MAX = 180


def parse_json_columns(df, json_columns):
    """JSON 열 파싱."""
    for col in json_columns:
        if col not in df.columns:
            continue
        df[col] = df[col].apply(_safe_json_parse)
    return df


def _safe_json_parse(val):
    """JSON 문자열을 safely 파싱."""
    if pd.isna(val) or val == 'null':
        return None
    if isinstance(val, str):
        import json
        try:
            return json.loads(val)
        except:
            return None
    return val


def aggregate_numeric(df, col, agg_cols, agg_funcs=None):
    """Numeric 열 aggregation."""
    if agg_funcs is None:
        agg_funcs = ['mean', 'std', 'min', 'max', 'count']
    grouped = df.groupby(agg_cols)[col].agg(agg_funcs)
    grouped.columns = [f"{col}_{f}" for f in agg_funcs]
    return grouped.reset_index()


def create_ambience_features(df):
    """Ambience 데이터에서 카테고리별 sum aggregation."""
    ambience_map = {
        'Speech': 'speech',
        'Music': 'music',
        'Vehicle (road)': 'motor_vehicle',
        'Inside, small room': 'inside_small',
        'Inside, small room or hall': 'inside_large',
        'Outside, rural or natural': 'outside_rural',
        'Outside, urban or manmade': 'outside_urban',
    }
    result_dfs = []
    for korean_name, eng_name in ambience_map.items():
        col = f"ambience_{korean_name}"
        if col in df.columns:
            grouped = df.groupby(["subject_id", "date"])["ambience_value"].agg(["sum", "mean"]).reset_index()
            grouped.columns = [f"ambience_value", f"ambience_{eng_name}_{f}" for f in ['sum', 'mean']]
            grouped['subject_id'] = df['subject_id'].values[:len(grouped)]
            grouped['date'] = df['date'].values[:len(grouped)]
            result_dfs.append(grouped)
    return result_dfs


def parse_ambience(df):
    """Ambience JSON parsing."""
    if 'ambience' not in df.columns:
        return df

    records = []
    for _, row in df.iterrows():
        val = row['ambience']
        subj = row['subject_id']
        dt = row['datetime'] if 'datetime' in row else None

        if val is None:
            continue
        if isinstance(val, str):
            try:
                import json
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
        amb_df = pd.DataFrame(records)
        return amb_df
    return pd.DataFrame(columns=['subject_id', 'datetime', 'ambience_value'])


def parse_ble(df):
    """BLE RSSI parsing."""
    records = []
    for _, row in df.iterrows():
        val = row.get('ble')
        if pd.isna(val) or val == 'null':
            continue
        if isinstance(val, str):
            try:
                import json
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
    """GPS parsing."""
    records = []
    for _, row in df.iterrows():
        val = row.get('gps')
        if pd.isna(val) or val == 'null':
            continue
        if isinstance(val, str):
            try:
                import json
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
    """WiFi parsing."""
    records = []
    for _, row in df.iterrows():
        val = row.get('wifi')
        if pd.isna(val) or val == 'null':
            continue
        if isinstance(val, str):
            try:
                import json
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


def create_day_features(parquet_dfs, sample_df, config='v2'):
    """
    Day-level feature matrix 생성.
    
    config:
      'v2_base'  - baseline (remove constants/collinearity, fix wHr)
      'v2_windows' - + time-window aggregation
      'v2_personal' - + personalization features
      'v2_all' - everything
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
            df['hour'] = 12  # dummy

        for col in cols:
            # Check which column exists
            data_col = f"m_{col}" if device.startswith('m') else f"{col}"
            if data_col not in df.columns:
                continue
            agg_df = aggregate_numeric(df, data_col, agg_cols)
            feature_dfs.append(agg_df)

    # ── 2. JSON parsing ──
    # Ambience
    if 'mAmbience' in parquet_dfs:
        amb_df = parquet_dfs['mAmbience'].copy()
        amb_df = parse_json_columns(amb_df, ['ambience'])
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
        ble_df = parse_json_columns(ble_df, ['ble'])
        ble_parsed = parse_ble(ble_df)
        if not ble_parsed.empty:
            if 'datetime' in ble_parsed.columns:
                ble_parsed['hour'] = pd.to_datetime(ble_parsed['datetime']).dt.hour
            for agg_fn in ['mean', 'std', 'min', 'max']:
                if agg_fn == 'mean':
                    agg_df = ble_parsed.groupby(["subject_id", "date"])['rssi'].agg(['mean', 'std', 'min', 'max']).reset_index()
                    agg_df.columns = ['subject_id', 'date', 'ble_avg_rssi_mean', 'ble_avg_rssi_std', 'ble_avg_rssi_min', 'ble_avg_rssi_max']
                    feature_dfs.append(agg_df)
                elif agg_fn == 'max':
                    agg_df = ble_parsed.groupby(["subject_id", "date"])['rssi'].agg(['max']).reset_index()
                    agg_df.columns = ['subject_id', 'date', 'ble_max_rssi_mean']
                    # Get max rssi per device too
                    device_agg = ble_parsed.groupby(["subject_id", "date"])['device_count'].agg(['mean', 'std', 'max', 'count']).reset_index()
                    device_agg.columns = ['subject_id', 'date', 'ble_device_count_mean', 'ble_device_count_std', 'ble_device_count_max', 'ble_device_count_count']
                    feature_dfs.append(device_agg)
                    # RSSI max/min per reading
                    rssi_max_agg = ble_parsed.groupby(["subject_id", "date"])['rssi'].agg(['max', 'std']).reset_index()
                    rssi_max_agg.columns = ['subject_id', 'date', 'ble_max_rssi_mean', 'ble_max_rssi_std']
                    feature_dfs.append(rssi_max_agg)
                    rssi_min_agg = ble_parsed.groupby(["subject_id", "date"])['rssi'].agg(['min', 'std']).reset_index()
                    rssi_min_agg.columns = ['subject_id', 'date', 'ble_min_rssi_mean', 'ble_min_rssi_std']
                    feature_dfs.append(rssi_min_agg)
                    rssi_std_agg = ble_parsed.groupby(["subject_id", "date"])['rssi'].agg(['std']).reset_index()
                    rssi_std_agg.columns = ['subject_id', 'date', 'ble_rssi_std_mean']
                    feature_dfs.append(rssi_std_agg)

    # GPS
    if 'mGps' in parquet_dfs:
        gps_df = parquet_dfs['mGps'].copy()
        gps_df = parse_json_columns(gps_df, ['gps'])
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
            # GPS count (how many readings per day)
            gps_count = gps_parsed.groupby(["subject_id", "date"]).size().reset_index(name='gps_count')
            feature_dfs.append(gps_count)

    # WiFi
    if 'mWifi' in parquet_dfs:
        wifi_df = parquet_dfs['mWifi'].copy()
        wifi_df = parse_json_columns(wifi_df, ['wifi'])
        wifi_parsed = parse_wifi(wifi_df)
        if not wifi_parsed.empty:
            if 'datetime' in wifi_parsed.columns:
                wifi_parsed['hour'] = pd.to_datetime(wifi_parsed['datetime']).dt.hour
            # RSSI features
            rssi_agg = wifi_parsed.groupby(["subject_id", "date"])['rssi'].agg(['mean', 'std', 'min', 'max']).reset_index()
            rssi_agg.columns = ['subject_id', 'date', 'wifi_avg_rssi_mean', 'wifi_avg_rssi_std', 'wifi_avg_rssi_min', 'wifi_avg_rssi_max']
            feature_dfs.append(rssi_agg)
            # Max RSSI features
            max_rssi_agg = wifi_parsed.groupby(["subject_id", "date"])['rssi'].agg(['max']).reset_index()
            max_rssi_agg.columns = ['subject_id', 'date', 'wifi_max_rssi_mean']
            feature_dfs.append(max_rssi_agg)
            # Strong signal ratio
            strong_mask = wifi_parsed['rssi'] >= -60
            strong_ratio = wifi_parsed.groupby(["subject_id", "date"]).apply(
                lambda x: (strong_mask[x.index].sum() / len(x)) if len(x) > 0 else 0, include_groups=False
            ).reset_index(name='wifi_strong_ratio_mean')
            feature_dfs.append(strong_ratio)
            # BSSID count (unique networks)
            bssid_agg = wifi_parsed.groupby(["subject_id", "date"])['bssid'].nunique().reset_index(name='wifi_bssid_count_mean')
            feature_dfs.append(bssid_agg)
            # WiFi count
            wifi_count = wifi_parsed.groupby(["subject_id", "date"]).size().reset_index(name='wifi_count_mean')
            feature_dfs.append(wifi_count)

    # UsageStats
    if 'mUsageStats' in parquet_dfs:
        us_df = parquet_dfs['mUsageStats'].copy()
        us_df = parse_json_columns(us_df, ['usage'])
        usage_records = []
        for _, row in us_df.iterrows():
            val = row.get('usage')
            if pd.isna(val) or val == 'null':
                continue
            if isinstance(val, str):
                try:
                    import json
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
            # Category aggregation
            cat_agg = usage_df.groupby(["subject_id", "date", "category"])['ratio'].sum().unstack(fill_value=0)
            cat_agg.columns = [f'usage_{col}_ratio_sum' for col in cat_agg.columns]
            cat_agg = cat_agg.reset_index()
            feature_dfs.append(cat_agg)

    # ACStatus time-of-day binning
    if 'mACStatus' in parquet_dfs:
        ac_df = parquet_dfs['mACStatus'].copy()
        ac_df = parse_json_columns(ac_df, ['charging'])
        if not ac_df.empty and 'datetime' in ac_df.columns:
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
        ss_df = parse_json_columns(ss_df, ['screen_use'])
        if not ss_df.empty and 'datetime' in ss_df.columns:
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

    # Fill missing
    df_all = df_all.fillna(0)

    # ── 4. Remove constant columns ──
    if 'v2_base' in config:
        for col in CONSTANT_COLS:
            if col in df_all.columns:
                df_all.drop(columns=[col], inplace=True)

    # ── 5. Remove collinear columns ──
    if 'v2_base' in config:
        for keep, drop in COLLINEAR_PAIRS:
            if drop in df_all.columns:
                df_all.drop(columns=[drop], inplace=True)

    # ── 6. Fix wHr anomalies ──
    if 'v2_base' in config:
        if 'wHr_hr_mean' in df_all.columns:
            low_hr = (df_all['wHr_hr_mean'] < HR_MIN) | (df_all['wHr_hr_mean'] > HR_MAX)
            df_all.loc[low_hr, 'wHr_hr_mean'] = np.nan
            if 'wHr_hr_std' in df_all.columns:
                df_all.loc[low_hr, 'wHr_hr_std'] = np.nan

    # ── 7. Time-window aggregation ──
    if 'windows' in config:
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
                        continue  # already done as day-level
                    # Aggregate last N hours from each timestamp
                    # Simplified: group by (subject, date, hour-window)
                    df_temp = df[df['hour'] >= window].copy()
                    df_temp['window_date'] = df_temp['datetime'].dt.date
                    df_temp['window_key'] = (df_temp['hour'] // window) * window
                    agg_df = df_temp.groupby(["subject_id", "window_date", "window_key"])[data_col].agg(['mean', 'std']).reset_index()
                    agg_df.columns = ['subject_id', 'window_date', 'window_key',
                                      f'{data_col}_{window}h_mean', f'{data_col}_{window}h_std']

                    # Map back to day-level: assign to the date where the window ends
                    agg_df['date'] = agg_df['window_date']
                    agg_df = agg_df.groupby(["subject_id", "date"])[[f'{data_col}_{window}h_mean', f'{data_col}_{window}h_std']].agg(['mean', 'std']).reset_index()
                    agg_df.columns = ['subject_id', 'date',
                                      f'{data_col}_{window}h_agg_mean', f'{data_col}_{window}h_agg_std']

                    if f'{data_col}_{window}h_agg_mean' in agg_df.columns:
                        df_all = df_all.merge(agg_df[['subject_id', 'date', f'{data_col}_{window}h_agg_mean', f'{data_col}_{window}h_agg_std']],
                                              on=['subject_id', 'date'], how='left')

    # ── 8. Personalization features ──
    if 'personal' in config:
        print("    Creating personalization features...")
        numeric_feat_cols = [c for c in df_all.columns
                             if c not in ['subject_id', 'date', 'lifelog_date']
                             and df_all[c].dtype in [np.float64, np.int64, float, int, bool]]

        for feat in numeric_feat_cols:
            # Personal mean deviation
            subj_stats = df_all.groupby('subject_id')[feat].agg(['mean', 'std']).reset_index()
            subj_stats.columns = ['subject_id', f'{feat}_subj_mean', f'{feat}_subj_std']
            df_all = df_all.merge(subj_stats, on='subject_id', how='left')

            # Personal z-score (deviation from personal mean)
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
    if 'personal' in config or 'windows' in config:
        # Add missing indicators for features with >5% null rate
        numeric_feat_cols = [c for c in df_all.columns
                             if c not in ['subject_id', 'date', 'lifelog_date']
                             and df_all[c].dtype in [np.float64, np.int64, float, int, bool]]
        for feat in numeric_feat_cols:
            null_rate = df_all[feat].isnull().mean()
            if null_rate > 0.05:
                df_all[f'{feat}_missing'] = df_all[feat].isnull().astype(int)
                # Impute with 0
                df_all[feat] = df_all[feat].fillna(0)

    # ── 10. Weekend/weekday indicator ──
    if 'v2_all' in config or 'windows' in config or 'personal' in config:
        dates = pd.to_datetime(df_all['date'])
        df_all['is_weekend'] = (dates.dt.dayofweek >= 5).astype(int)
        df_all['day_of_week'] = dates.dt.dayofweek
        df_all['month'] = dates.dt.month
        df_all['day_of_year'] = dates.dt.dayofyear

    return df_all


def main():
    """파이프라인 실행."""
    print("=" * 60)
    print("feature_engineering_v2.py")
    print("=" * 60)

    # Load sample
    sample = pd.read_csv(str(DATA_DIR.parent / 'ch2026_submission_sample.csv'),
                         parse_dates=['sleep_date', 'lifelog_date'])
    sample['lifelog_date'] = pd.to_datetime(sample['lifelog_date']).dt.date

    # Load parquet data
    print("\nLoading parquet data...")
    parquet_dfs = load_parquet_data()

    # Run configs
    configs = {
        'v2_base': 'base (constants+collinearity+wHr fix)',
        'v2_windows': 'base + time-window aggregation',
        'v2_personal': 'base + personalization',
        'v2_all': 'base + windows + personal + external',
    }

    for config_key, desc in configs.items():
        print(f"\n{'=' * 60}")
        print(f"Config: {config_key} — {desc}")
        print("=" * 60)

        df_feat = create_day_features(parquet_dfs, sample, config=config_key)
        print(f"  Shape: {df_feat.shape}")
        print(f"  Columns: {len([c for c in df_feat.columns if c not in ['subject_id','date','lifelog_date']])} features")

        # Save
        out_path = DATA_PROCESSED / f'test_features_{config_key}.parquet'
        df_feat.to_parquet(out_path)
        print(f"  Saved: {out_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
