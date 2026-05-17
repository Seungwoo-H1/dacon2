"""
gen_v62_full_train_test.py — Full feature pipeline for BOTH train and test

Strategy: Use the EXACT same logic as 02_feature_engineering.py (which produced
the verified features.parquet with 153 cols, 450 rows) for both train and test.

Key insight: The verified 02_feature_engineering.py uses numpy ndarray parsing
(most JSON columns are already parsed by pyarrow as ndarrays), NOT JSON string
parsing. This is the CORRECT approach.

Steps:
1. Load all parquet files (same as 02_feature_engineering.py)
2. Apply day-level aggregation (same as 02_feature_engineering.py)  
3. For train: merge with labels
4. For test: use sample_submission dates
5. Save both as parquet files
"""
import numpy as np
import pandas as pd
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path('/home/mwoo423/.openclaw/workspace')
DATA_RAW = ROOT / 'data_raw' / 'ch2025_data_items'
DATA_PROCESSED = ROOT / 'data_processed'
SAMPLE_CSV = DATA_RAW.parent / 'ch2026_submission_sample.csv'
LABEL_CSV = DATA_RAW.parent / 'ch2026_metrics_train.csv'

# ── Column names in parquet files ──
PARQUET_FILES = {
    'mACStatus':   'ch2025_mACStatus.parquet',
    'mActivity':   'ch2025_mActivity.parquet',
    'mAmbience':   'ch2025_mAmbience.parquet',
    'mBle':        'ch2025_mBle.parquet',
    'mGps':        'ch2025_mGps.parquet',
    'mLight':      'ch2025_mLight.parquet',
    'mScreenStatus': 'ch2025_mScreenStatus.parquet',
    'mUsageStats': 'ch2025_mUsageStats.parquet',
    'mWifi':       'ch2025_mWifi.parquet',
    'wHr':         'ch2025_wHr.parquet',
    'wLight':      'ch2025_wLight.parquet',
    'wPedo':       'ch2025_wPedo.parquet',
}

AGG_WINDOWS = [1, 3, 6, 12, 24]
TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']

# ── JSON parsers (from 02_feature_engineering.py) ──────────────────────────

def parse_ambience(value):
    """m_ambience: ndarray of [category, prob] → 확률 합산."""
    if not isinstance(value, (np.ndarray, list)):
        return {}
    scores = defaultdict(float)
    for item in value:
        if isinstance(item, (np.ndarray, list)) and len(item) >= 2:
            scores[str(item[0])] += float(item[1])
    return dict(scores)


def parse_ble(value):
    """m_ble: list of {address, device_class, rssi} → 통계."""
    if not isinstance(value, (np.ndarray, list)):
        return {"count": 0, "avg_rssi": np.nan, "max_rssi": np.nan}
    rssis = []
    devices = set()
    for item in value:
        if isinstance(item, dict):
            if "rssi" in item and item["rssi"] is not None:
                rssis.append(item["rssi"])
            if "address" in item:
                devices.add(item["address"][:10])
    return {
        "ble_count": len(value),
        "ble_device_count": len(devices),
        "ble_avg_rssi": np.mean(rssis) if rssis else np.nan,
        "ble_max_rssi": np.max(rssis) if rssis else np.nan,
        "ble_min_rssi": np.min(rssis) if rssis else np.nan,
        "ble_rssi_std": np.std(rssis) if len(rssis) > 1 else np.nan,
    }


def parse_gps(value):
    """m_gps: list of {lat, lon, alt, speed} → 통계."""
    if not isinstance(value, (np.ndarray, list)):
        return {"count": 0}
    speeds = []
    alts = []
    for item in value:
        if isinstance(item, dict):
            if "speed" in item and item["speed"] is not None:
                speeds.append(float(item["speed"]))
            if "altitude" in item and item["altitude"] is not None:
                alts.append(float(item["altitude"]))
    return {
        "gps_count": len(value),
        "gps_avg_speed": np.mean(speeds) if speeds else np.nan,
        "gps_max_speed": np.max(speeds) if speeds else np.nan,
        "gps_alt_range": (np.max(alts) - np.min(alts)) if alts else np.nan,
        "gps_has_speed": 1.0 if speeds else 0.0,
    }


def parse_usage_stats(value):
    """m_usage_stats: list of {app_name, total_time, category, ratio} → 통계."""
    if not isinstance(value, (np.ndarray, list)):
        return {"app_count": 0, "total_time": 0, "ratio": 0}
    apps = []
    total_time = 0
    categories = []
    ratios = []
    for item in value:
        if isinstance(item, dict):
            apps.append(item.get("app_name", ""))
            tt = item.get("total_time", 0)
            if tt is not None:
                total_time += float(tt)
            cat = item.get("category", "")
            if cat:
                categories.append(cat)
            r = item.get("ratio", 0)
            if r is not None:
                ratios.append(float(r))

    # Category ratio aggregation
    cat_ratio = defaultdict(float)
    for a in apps:
        if isinstance(a, str):
            if any(k in a.lower() for k in ["naver", "google", "카카오", "home", "browser"]):
                cat_ratio["major"] += 1
            elif "game" in a.lower():
                cat_ratio["game"] += 1
            else:
                cat_ratio["other"] += 1

    # Also aggregate by actual category names
    cat_name_ratio = defaultdict(float)
    for cat in categories:
        if isinstance(cat, str) and cat:
            cat_name_ratio[cat] += 1.0

    return {
        "usage_app_count": len(apps),
        "usage_total_time": total_time,
        "usage_major_ratio": cat_ratio.get("major", 0) / max(len(apps), 1),
        "usage_game_ratio": cat_ratio.get("game", 0) / max(len(apps), 1),
    }


def parse_wifi(value):
    """m_wifi: list of {bssid, rssi} → 통계."""
    if not isinstance(value, (np.ndarray, list)):
        return {"count": 0}
    rssis = []
    bssids = set()
    for item in value:
        if isinstance(item, dict):
            if "rssi" in item and item["rssi"] is not None:
                rssis.append(item["rssi"])
            if "bssid" in item:
                bssids.add(item["bssid"])
    strong = sum(1 for r in rssis if r > -60) if rssis else 0
    return {
        "wifi_count": len(value),
        "wifi_bssid_count": len(bssids),
        "wifi_avg_rssi": np.mean(rssis) if rssis else np.nan,
        "wifi_max_rssi": np.max(rssis) if rssis else np.nan,
        "wifi_strong_ratio": strong / max(len(rssis), 1),
    }


def parse_heart_rate(value):
    """wHr: 배열 → 기본 통계."""
    if not isinstance(value, (np.ndarray, list)):
        return {"count": 0}
    hr = np.array(value, dtype=float)
    return {
        "hr_count": len(hr),
        "hr_mean": np.mean(hr),
        "hr_std": np.std(hr),
        "hr_min": np.min(hr),
        "hr_max": np.max(hr),
        "hr_median": np.median(hr),
    }


# ── Load parquet data ──────────────────────────────────────────────────────

def load_parquet_data(subject_ids=None):
    """Load all parquet files, optionally filtered."""
    parquet_dfs = {}
    for name, filename in PARQUET_FILES.items():
        path = DATA_RAW / filename
        if not path.exists():
            print(f"  ⚠ {name}: {filename} not found")
            continue
        print(f"  loading {name}...")
        df = pd.read_parquet(path)
        if subject_ids is not None:
            df = df[df['subject_id'].isin(subject_ids)].copy()
        if df.empty:
            continue
        # Build date column
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['datetime'].dt.date
            df['hour'] = df['datetime'].dt.hour
        elif 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df['date'] = df['datetime'].dt.date
            df['hour'] = df['datetime'].dt.hour
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.date
            df['hour'] = 12  # dummy
        parquet_dfs[name] = df
        print(f"    {name}: {len(df)} rows")
    return parquet_dfs


# ── Feature creation ───────────────────────────────────────────────────────

def aggregate_numeric(df, col, prefix):
    """Aggregate numeric column."""
    grouped = df.groupby(["subject_id", "date"])[col].agg(["mean", "std", "min", "max", "count"])
    grouped.columns = [f"{prefix}_{c}" for c in grouped.columns]
    return grouped.reset_index()


def create_features_only(parquet_dfs):
    """
    Create day-level features ONLY (no label merge).
    This is the core logic from 02_feature_engineering.py.
    """
    all_features = None
    
    def _merge_feat(left, right):
        if left is None:
            return right.drop_duplicates(["subject_id", "date"])
        merged = left.merge(right, on=["subject_id", "date"], how="outer")
        return merged.drop_duplicates(["subject_id", "date"])

    print("    [1/3] Numeric aggregation")
    
    # mACStatus
    if 'mACStatus' in parquet_dfs:
        feat = aggregate_numeric(parquet_dfs['mACStatus'], 'm_charging', 'mACStatus_m_charging')
        all_features = _merge_feat(all_features, feat)
    
    # mActivity
    if 'mActivity' in parquet_dfs:
        feat = aggregate_numeric(parquet_dfs['mActivity'], 'm_activity', 'mActivity_m_activity')
        all_features = _merge_feat(all_features, feat)
    
    # mLight
    if 'mLight' in parquet_dfs:
        feat = aggregate_numeric(parquet_dfs['mLight'], 'm_light', 'mLight_m_light')
        all_features = _merge_feat(all_features, feat)
    
    # mScreenStatus
    if 'mScreenStatus' in parquet_dfs:
        feat = aggregate_numeric(parquet_dfs['mScreenStatus'], 'm_screen_use', 'mScreenStatus_m_screen_use')
        all_features = _merge_feat(all_features, feat)
    
    # wLight
    if 'wLight' in parquet_dfs:
        feat = aggregate_numeric(parquet_dfs['wLight'], 'w_light', 'wLight_w_light')
        all_features = _merge_feat(all_features, feat)
    
    # wPedo
    if 'wPedo' in parquet_dfs:
        pedo_cols = ['step', 'step_frequency', 'running_step', 'walking_step',
                     'distance', 'speed', 'burned_calories']
        grouped = parquet_dfs['wPedo'].groupby(["subject_id", "date"])[pedo_cols].agg(["mean", "sum"])
        grouped.columns = [f"wPedo_pedo_{col}_{stat}" for col, stat in grouped.columns]
        pedo_feat = grouped.reset_index()
        all_features = _merge_feat(all_features, pedo_feat)

    print("    [2/3] JSON parsing")
    
    # ── JSON parsing ──
    
    # mAmbience (special: groupby + apply)
    if 'mAmbience' in parquet_dfs:
        df = parquet_dfs['mAmbience']
        
        def _amb_group(g):
            scores = defaultdict(float)
            total_rows = 0
            for v in g['m_ambience'].dropna():
                if isinstance(v, (np.ndarray, list)):
                    parsed = parse_ambience(v)
                    for k, s in parsed.items():
                        scores[k] += s
                    total_rows += 1
            result = {}
            for cat in ["Speech", "Music", "Vehicle", "Motor vehicle (road)",
                         "Inside, large room or hall", "Inside, small room",
                         "Outside, urban or manmade", "Outside, rural or natural",
                         "Car", "Truck"]:
                key = f"ambience_{cat.lower().replace(' ', '_')}_sum"
                result[key] = scores.get(cat, 0.0) / max(total_rows, 1)
            all_scores = sorted(scores.values(), reverse=True)[:5]
            result["ambience_top5_sum"] = sum(all_scores) / max(total_rows, 1)
            result["ambience_max_cat"] = max(scores, key=scores.get) if scores else ""
            return pd.Series(result)
        
        amb_feats = df.groupby(["subject_id", "date"]).apply(_amb_group, include_groups=False)
        amb_feats = amb_feats.reset_index()
        amb_feats = amb_feats.rename(
            columns={c: f"mAmbience_{c}" for c in amb_feats.columns if c not in ("subject_id", "date")}
        )
        all_features = _merge_feat(all_features, amb_feats)

    # mBle
    if 'mBle' in parquet_dfs:
        df = parquet_dfs['mBle']
        parsed = df['m_ble'].apply(parse_ble)
        parsed_df = pd.DataFrame(parsed.tolist(), index=df.index)
        parsed_df["subject_id"] = df["subject_id"]
        parsed_df["date"] = df["date"]
        stat_cols = [c for c in parsed_df.columns if c not in ("subject_id", "date")]
        for sc in stat_cols:
            grouped = parsed_df.groupby(["subject_id", "date"])[sc].agg(["mean", "std", "max", "min"])
            grouped.columns = [f"mBle_{sc}_{s}" for s in grouped.columns]
            all_features = _merge_feat(all_features, grouped.reset_index())

    # mGps
    if 'mGps' in parquet_dfs:
        df = parquet_dfs['mGps']
        parsed = df['m_gps'].apply(parse_gps)
        parsed_df = pd.DataFrame(parsed.tolist(), index=df.index)
        parsed_df["subject_id"] = df["subject_id"]
        parsed_df["date"] = df["date"]
        stat_cols = [c for c in parsed_df.columns if c not in ("subject_id", "date")]
        for sc in stat_cols:
            grouped = parsed_df.groupby(["subject_id", "date"])[sc].agg(["mean", "std", "max", "min"])
            grouped.columns = [f"mGps_{sc}_{s}" for s in grouped.columns]
            all_features = _merge_feat(all_features, grouped.reset_index())

    # mWifi
    if 'mWifi' in parquet_dfs:
        df = parquet_dfs['mWifi']
        parsed = df['m_wifi'].apply(parse_wifi)
        parsed_df = pd.DataFrame(parsed.tolist(), index=df.index)
        parsed_df["subject_id"] = df["subject_id"]
        parsed_df["date"] = df["date"]
        stat_cols = [c for c in parsed_df.columns if c not in ("subject_id", "date")]
        for sc in stat_cols:
            grouped = parsed_df.groupby(["subject_id", "date"])[sc].agg(["mean", "std", "max", "min"])
            grouped.columns = [f"mWifi_{sc}_{s}" for s in grouped.columns]
            all_features = _merge_feat(all_features, grouped.reset_index())

    # mUsageStats
    if 'mUsageStats' in parquet_dfs:
        df = parquet_dfs['mUsageStats']
        parsed = df['m_usage_stats'].apply(parse_usage_stats)
        parsed_df = pd.DataFrame(parsed.tolist(), index=df.index)
        parsed_df["subject_id"] = df["subject_id"]
        parsed_df["date"] = df["date"]
        stat_cols = [c for c in parsed_df.columns if c not in ("subject_id", "date")]
        for sc in stat_cols:
            grouped = parsed_df.groupby(["subject_id", "date"])[sc].agg(["mean", "std", "max", "min"])
            grouped.columns = [f"mUsageStats_{sc}_{s}" for s in grouped.columns]
            all_features = _merge_feat(all_features, grouped.reset_index())

    # wHr
    if 'wHr' in parquet_dfs:
        df = parquet_dfs['wHr']
        hr_records = []
        for _, row in df.iterrows():
            sid = row["subject_id"]
            date_val = row["date"]
            hr_vals = []
            if isinstance(row['heart_rate'], (np.ndarray, list)):
                hr_vals = [float(v) for v in row['heart_rate'] if v is not None]
            hr_records.append({
                "subject_id": sid,
                "date": date_val,
                "wHr_hr_mean": np.mean(hr_vals) if hr_vals else np.nan,
                "wHr_hr_std": np.std(hr_vals) if len(hr_vals) > 1 else np.nan,
                "wHr_hr_count": len(hr_vals),
            })
        day_feat = pd.DataFrame(hr_records)
        all_features = _merge_feat(all_features, day_feat)

    print("    [3/3] Time-of-day features")
    
    # Time-of-day bins for mACStatus
    if 'mACStatus' in parquet_dfs:
        df = parquet_dfs['mACStatus']
        df = df.copy()
        df["hour_bin"] = df["hour"].apply(
            lambda h: "night" if h < 6 else "morning" if h < 12 else "afternoon" if h < 18 else "evening"
        )
        ratio = df.groupby(["subject_id", "date"])["hour_bin"].value_counts(normalize=True).unstack(fill_value=0)
        ratio.columns = [f"mACStatus_hour_{c}" for c in ratio.columns]
        all_features = _merge_feat(all_features, ratio.reset_index())

    # Time-of-day bins for mScreenStatus
    if 'mScreenStatus' in parquet_dfs:
        df = parquet_dfs['mScreenStatus']
        df = df.copy()
        df["hour_bin"] = df["hour"].apply(
            lambda h: "night" if h < 6 else "morning" if h < 12 else "afternoon" if h < 18 else "evening"
        )
        ratio = df.groupby(["subject_id", "date"])["hour_bin"].value_counts(normalize=True).unstack(fill_value=0)
        ratio.columns = [f"mScreenStatus_hour_{c}" for c in ratio.columns]
        all_features = _merge_feat(all_features, ratio.reset_index())

    # Fill NaN with 0
    if all_features is not None:
        numeric_cols = all_features.select_dtypes(include=[np.number]).columns
        all_features[numeric_cols] = all_features[numeric_cols].fillna(0)

    return all_features


def main():
    t0 = time.time()
    print("=" * 70)
    print("gen_v62_full_train_test.py — Full 02_feature_engineering pipeline")
    print("=" * 70)

    # Load labels
    print("\nLoading labels...")
    labels = pd.read_csv(LABEL_CSV, parse_dates=['sleep_date', 'lifelog_date'])
    print(f"  Labels: {labels.shape}")

    # Load sample
    print("\nLoading sample...")
    sample = pd.read_csv(SAMPLE_CSV, parse_dates=['sleep_date', 'lifelog_date'])
    print(f"  Sample: {sample.shape}")

    # ── Generate train features ──
    print("\n── Generating train features ──")
    train_subjects = set(labels['subject_id'].unique())
    print(f"  Subjects: {len(train_subjects)}")
    
    train_parquet = load_parquet_data(subject_ids=train_subjects)
    train_feat = create_features_only(train_parquet)
    print(f"  Feature rows: {len(train_feat)}")
    
    # Merge with labels
    labels_day = labels[['subject_id', 'lifelog_date', 'sleep_date'] + TARGETS].copy()
    labels_day['date'] = pd.to_datetime(labels_day['lifelog_date']).dt.date
    
    # Ensure date types match (convert to date objects for merge)
    train_feat = train_feat.copy()
    if not isinstance(train_feat['date'].iloc[0], (str, type(pd.NaT.date))):
        train_feat['date'] = pd.to_datetime(train_feat['date']).dt.date
    
    merged = labels_day.merge(train_feat, on=['subject_id', 'date'], how='left')
    merged = merged.sort_values(['subject_id', 'date']).reset_index(drop=True)
    
    print(f"  Train merged: {merged.shape}")

    # ── Generate test features ──
    print("\n── Generating test features ──")
    test_subjects = set(sample['subject_id'].unique())
    print(f"  Subjects: {len(test_subjects)}")
    
    test_parquet = load_parquet_data(subject_ids=test_subjects)
    test_feat = create_features_only(test_parquet)
    print(f"  Feature rows: {len(test_feat)}")
    
    # Filter to sample dates and merge
    sample_key = sample[['subject_id', 'lifelog_date', 'sleep_date']].copy()
    sample_key['date'] = pd.to_datetime(sample_key['lifelog_date']).dt.date
    sample_key = sample_key.drop_duplicates()
    
    # Ensure date types match
    test_feat = test_feat.copy()
    if not isinstance(test_feat['date'].iloc[0], (str, type(pd.NaT.date))):
        test_feat['date'] = pd.to_datetime(test_feat['date']).dt.date
    
    test_merged = sample_key.merge(test_feat, on=['subject_id', 'date'], how='left')
    test_merged = test_merged.sort_values(['subject_id', 'date']).reset_index(drop=True)
    
    print(f"  Test merged: {test_merged.shape}")
    
    if len(test_merged) != 250:
        print(f"  ⚠️ WARNING: Expected 250 rows but got {len(test_merged)}")

    # ── Column comparison ──
    META_COLS = {'subject_id', 'date', 'lifelog_date', 'sleep_date'}
    train_cols = sorted([c for c in merged.columns if c not in META_COLS and c in TARGETS])
    test_cols = sorted([c for c in test_merged.columns if c not in META_COLS and c not in TARGETS])
    
    # Actually compare all non-meta columns
    train_all_cols = sorted([c for c in merged.columns if c not in META_COLS])
    test_all_cols = sorted([c for c in test_merged.columns if c not in META_COLS])
    
    print(f"\n{'=' * 70}")
    print(f"Train features: {len(train_all_cols)}")
    print(f"Test features:  {len(test_cols)}")
    
    train_only = sorted(set(train_all_cols) - set(test_all_cols))
    test_only = sorted(set(test_all_cols) - set(train_all_cols))
    common = sorted(set(train_all_cols) & set(test_all_cols))
    
    print(f"Common features: {len(common)}")
    
    if train_only:
        print(f"\nMissing in test: {len(train_only)}")
        for c in train_only[:20]:
            print(f"  - {c}")
    
    if test_only:
        print(f"\nExtra in test: {len(test_only)}")
        for c in test_only[:20]:
            print(f"  + {c}")
    
    if set(train_all_cols) == set(test_all_cols):
        print("\n✅ PERFECT MATCH: Train and test have identical column sets!")

    # ── Verify JSON features ──
    print(f"\n── JSON Feature Verification ──")
    json_features = [
        'mBle_ble_ble_avg_rssi_mean', 'mBle_ble_ble_max_rssi_mean',
        'mGps_gps_gps_avg_speed_mean', 'mGps_gps_gps_has_speed_mean',
        'mWifi_wifi_wifi_avg_rssi_mean', 'mWifi_wifi_wifi_bssid_count_mean',
        'mUsageStats_usage_usage_app_count_mean',
        'mAmbience_mAmbience_ambience_music_sum',
        'wHr_wHr_hr_mean',
    ]
    for feat in json_features:
        train_has = feat in train_all_cols
        test_has = feat in test_all_cols
        status = "✅" if (train_has and test_has) else "❌"
        print(f"  {status} {feat}: train={train_has}, test={test_has}")

    # ── Save ──
    print(f"\n── Saving ──")
    train_path = DATA_PROCESSED / 'train_features_v62.parquet'
    test_path = DATA_PROCESSED / 'test_features_v62.parquet'
    
    merged.to_parquet(train_path, index=False)
    test_merged.to_parquet(test_path, index=False)
    
    print(f"  Train: {train_path} ({len(merged)} rows, {len(merged.columns)} cols)")
    print(f"  Test:  {test_path} ({len(test_merged)} rows, {len(test_merged.columns)} cols)")
    print(f"\nTime: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
