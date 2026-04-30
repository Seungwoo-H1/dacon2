import logging

import numpy as np
import pandas as pd

from pathlib import Path

from config import DATA_PROCESSED, TARGETS

import importlib

importlib.invalidate_caches()
import importlib
import importlib.util

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def build_feature_eng_pipeline():
    """Return a dict of functions that each return a DataFrame of features."""
    return {
        "numeric_agg": _numeric_agg,
        "pedo_agg": _pedo_agg,
        "json_agg": _json_agg,
        "hour_ratio": _hour_ratio,
        "activity_density": _activity_density,
        "sensor_density": _sensor_density,
        "circadian": _circadian_features,
        "daily_patterns": _daily_patterns,
    }


# ─── helpers ─────────────────────────────────────────────

def _agg_numeric(df, col):
    """Aggregate numeric column by subject_id + date."""
    grouped = df.groupby(["subject_id", "date"])[col].agg(
        ["mean", "std", "min", "max", "count"]
    )
    grouped.columns = [f"{col}_{s}" for s in grouped.columns]
    return grouped.reset_index()


def _safe_merge(left, right):
    """Outer merge on subject_id + date, drop duplicates."""
    if left is None:
        return right.drop_duplicates(["subject_id", "date"])
    merged = left.merge(right, on=["subject_id", "date"], how="outer")
    return merged.drop_duplicates(["subject_id", "date"])


def _rename_except_meta(df, prefix):
    """Prefix all columns except subject_id, date."""
    rename_map = {
        c: f"{prefix}_{c}" for c in df.columns if c not in ("subject_id", "date")
    }
    return df.rename(columns=rename_map)


# ─── feature builders ────────────────────────────────────

def _numeric_agg(df_dict):
    """mACStatus, mActivity, mLight, mScreenStatus, wLight."""
    all_feat = None
    for src, col in [
        ("mACStatus", "m_charging"),
        ("mActivity", "m_activity"),
        ("mLight", "m_light"),
        ("mScreenStatus", "m_screen_use"),
        ("wLight", "w_light"),
    ]:
        feat = _agg_numeric(df_dict[src], col)
        feat = _rename_except_meta(feat, src)
        all_feat = _safe_merge(all_feat, feat)
    return all_feat


def _pedo_agg(df_dict):
    """wPedo aggregation."""
    WPEO_COLS = [
        "step", "step_frequency", "running_step", "walking_step",
        "distance", "speed", "burned_calories",
    ]
    grouped = df_dict["wPedo"].groupby(["subject_id", "date"])[WPEO_COLS].agg(
        ["mean", "sum"]
    )
    grouped.columns = [f"pedo_{col}_{stat}" for col, stat in grouped.columns]
    pedo_feat = grouped.reset_index()
    pedo_feat = _rename_except_meta(pedo_feat, "wPedo")
    return _safe_merge(None, pedo_feat)


def _json_agg(df_dict):
    """JSON column parsing & aggregation."""

    # --- mAmbience ---
    def _extract_ambience_features(grp):
        scores = {}
        total = 0
        for v in grp["m_ambience"].dropna():
            if isinstance(v, (np.ndarray, list)) and len(v) >= 10:
                parsed = [
                    ("Speech", v[0]), ("Music", v[1]), ("Vehicle", v[2]),
                    ("Motor vehicle (road)", v[3]),
                    ("Inside, large room or hall", v[4]),
                    ("Inside, small room", v[5]),
                    ("Outside, urban or manmade", v[6]),
                    ("Outside, rural or natural", v[7]),
                    ("Car", v[8]), ("Truck", v[9]),
                ]
                for cat, s in parsed:
                    try:
                        val = float(np.asarray(s).flatten()[0]) if hasattr(s, '__iter__') else float(s)
                    except (ValueError, TypeError, IndexError):
                        val = 0.0
                    scores[cat] = scores.get(cat, 0) + val
                total += 1
        result = {}
        for cat in scores:
            key = f"ambience_{cat.lower().replace(' ', '_')}_sum"
            result[key] = scores[cat] / max(total, 1)
        top5 = sorted(scores.values(), reverse=True)[:5]
        result["ambience_top5_sum"] = sum(top5) / max(total, 1)
        result["ambience_max_cat"] = max(scores, key=scores.get) if scores else ""
        return pd.Series(result)

    amb_df = df_dict["mAmbience"].groupby(["subject_id", "date"]).apply(
        _extract_ambience_features, include_groups=False
    ).reset_index()
    amb_df = _rename_except_meta(amb_df, "mAmbience")

    # --- wHr ---
    hr_records = []
    for _, row in df_dict["wHr"].iterrows():
        hr_vals = []
        if isinstance(row["heart_rate"], (np.ndarray, list)):
            hr_vals = [float(v) for v in row["heart_rate"] if v is not None]
        hr_records.append({
            "subject_id": row["subject_id"],
            "date": row["date"],
            "wHr_hr_mean": np.mean(hr_vals) if hr_vals else np.nan,
            "wHr_hr_std": np.std(hr_vals) if len(hr_vals) > 1 else np.nan,
            "wHr_hr_count": len(hr_vals),
        })
    hr_feat = pd.DataFrame(hr_records)

    # --- JSON stat cols (mBle, mGps, mUsageStats, mWifi) ---
    parsers = {
        "mBle": lambda v: v if isinstance(v, list) and len(v) > 0 else [[]],
        "mGps": lambda v: v if isinstance(v, list) and len(v) > 0 else [[]],
        "mUsageStats": lambda v: (v if isinstance(v, dict) else {}),
        "mWifi": lambda v: v if isinstance(v, list) and len(v) > 0 else [[]],
    }

    json_feat = None
    for src, parser in parsers.items():
        df_src = df_dict[src]
        parsed = df_src["m_ble" if src == "mBle" else
                         "m_gps" if src == "mGps" else
                         "m_usage_stats" if src == "mUsageStats" else
                         "m_wifi"].apply(parser)
        parsed_df = pd.DataFrame(parsed.tolist(), index=df_src.index)
        parsed_df["subject_id"] = df_src["subject_id"]
        parsed_df["date"] = df_src["date"]
        stat_cols = [c for c in parsed_df.columns if c not in ("subject_id", "date")]
        for sc in stat_cols:
            grouped = parsed_df.groupby(["subject_id", "date"])[sc].agg(
                ["mean", "std", "max", "min"]
            )
            grouped.columns = [f"{src}_{sc}_{s}" for s in grouped.columns]
            json_feat = _safe_merge(json_feat, grouped.reset_index())

    # Merge all JSON features
    result = _safe_merge(None, amb_df)
    result = _safe_merge(result, hr_feat)
    result = _safe_merge(result, json_feat)
    return result


def _hour_ratio(df_dict):
    """Time-of-day hour ratio features."""
    result = None
    for src, _ in [("mACStatus", "m_charging"), ("mScreenStatus", "m_screen_use")]:
        df_src = df_dict[src].copy()
        df_src["hour_bin"] = df_src["hour"].apply(
            lambda h: "night" if h < 6 else "morning" if h < 12
                      else "afternoon" if h < 18 else "evening"
        )
        ratio = df_src.groupby(["subject_id", "date"])["hour_bin"].value_counts(
            normalize=True
        ).unstack(fill_value=0)
        ratio.columns = [f"{src}_hour_{c}" for c in ratio.columns]
        ratio = ratio.reset_index()
        result = _safe_merge(result, ratio)
    return result


def _activity_density(df_dict):
    """Activity density: steps/movement per hour of data collection."""
    result = None

    # wPedo density
    pedo = df_dict["wPedo"].copy()
    pedo["timestamp"] = pd.to_datetime(pedo["timestamp"])
    time_span = (pedo["timestamp"].max() - pedo["timestamp"].min())
    hours = time_span.total_seconds() / 3600
    pedo["step_per_hour"] = pedo["step"] / max(hours, 1)
    density = _agg_numeric(pedo, "step_per_hour")
    density = _rename_except_meta(density, "pedo_density")
    result = _safe_merge(result, density)

    # Activity density
    act = df_dict["mActivity"].copy()
    act["timestamp"] = pd.to_datetime(act["timestamp"])
    act["active_duration"] = act["m_activity"].apply(
        lambda x: 1 if isinstance(x, (int, float)) and x > 0 else 0
    )
    act_dense = _agg_numeric(act, "active_duration")
    act_dense = _rename_except_meta(act_dense, "act_density")
    result = _safe_merge(result, act_dense)

    return result


def _sensor_density(df_dict):
    """Records per day from each sensor source."""
    result = None
    for src in ["mBle", "mGps", "mWifi", "mUsageStats"]:
        df_src = df_dict[src]
        counts = df_src.groupby(["subject_id", "date"]).size().reset_index(name=f"{src}_record_count")
        result = _safe_merge(result, counts)
    return result


def _circadian_features(df_dict):
    """Circadian / sleep-related features."""
    result = None

    # Charging hour ratio (proxy for charging at night = sleeping)
    ac = df_dict["mACStatus"].copy()
    ac["is_charging"] = ac["m_charging"].apply(
        lambda x: 1 if isinstance(x, (int, float)) and x == 1 else 0
    )
    night_charging = ac[ac["hour"].between(22, 6) | ac["hour"].between(0, 4)]
    charging_stats = _agg_numeric(ac, "is_charging")
    charging_stats = _rename_except_meta(charging_stats, "night_charge")
    result = _safe_merge(result, charging_stats)

    # Screen use timing: late-night screen use
    screen = df_dict["mScreenStatus"].copy()
    screen["late_night"] = screen["hour"].apply(
        lambda h: 1 if h >= 23 or h <= 4 else 0
    )
    late_screen = _agg_numeric(screen, "late_night")
    late_screen = _rename_except_meta(late_screen, "late_screen")
    result = _safe_merge(result, late_screen)

    return result


def _daily_patterns(df_dict):
    """Additional daily pattern features."""
    result = None

    # Light variability: ratio of night light to total light
    light = df_dict["mLight"].copy()
    light["night_light"] = light["hour"].apply(
        lambda h: 1 if h >= 21 or h <= 5 else 0
    )
    light_stats = _agg_numeric(light, "m_light")
    light_stats = _rename_except_meta(light_stats, "mLight")
    result = _safe_merge(result, light_stats)

    # GPS mobility: max speed as proxy for outdoor activity
    gps = df_dict["mGps"].copy()
    if "max_speed" in gps.columns:
        gps_stats = _agg_numeric(gps, "max_speed")
        gps_stats = _rename_except_meta(gps_stats, "mGps")
        result = _safe_merge(result, gps_stats)

    # WiFi connectivity: count of BSSIDs / unique networks
    wifi = df_dict["mWifi"].copy()
    if "bssid_count" in wifi.columns:
        wifi_stats = _agg_numeric(wifi, "bssid_count")
        wifi_stats = _rename_except_meta(wifi_stats, "mWifi")
        result = _safe_merge(result, wifi_stats)

    return result


# ─── main ────────────────────────────────────────────────

def main():
    """Create features, save to parquet."""
    logger.info("=== 02_feature_engineering.py ===")
    logger.info("Loading data...")

    import importlib
    load_mod = importlib.import_module("01_load_data")
    parquet_dfs, labels = load_mod.main()

    # Build features
    pipeline = build_feature_eng_pipeline()
    all_features = None

    # Step 1: numeric aggregation
    logger.info("[1/7] Numeric aggregation")
    feat = _numeric_agg(parquet_dfs)
    all_features = _safe_merge(all_features, feat)

    # Step 2: wPedo
    logger.info("[2/7] wPedo aggregation")
    feat = _pedo_agg(parquet_dfs)
    all_features = _safe_merge(all_features, feat)

    # Step 3: JSON columns
    logger.info("[3/7] JSON parsing & aggregation")
    feat = _json_agg(parquet_dfs)
    all_features = _safe_merge(all_features, feat)

    # Step 4: hour ratio
    logger.info("[4/7] Hour ratio features")
    feat = _hour_ratio(parquet_dfs)
    all_features = _safe_merge(all_features, feat)

    # Step 5: activity density
    logger.info("[5/7] Activity density")
    feat = _activity_density(parquet_dfs)
    all_features = _safe_merge(all_features, feat)

    # Step 6: sensor density
    logger.info("[6/7] Sensor density")
    feat = _sensor_density(parquet_dfs)
    all_features = _safe_merge(all_features, feat)

    # Step 7: circadian & daily patterns
    logger.info("[7/7] Circadian & daily patterns")
    feat = _circadian_features(parquet_dfs)
    all_features = _safe_merge(all_features, feat)
    feat = _daily_patterns(parquet_dfs)
    all_features = _safe_merge(all_features, feat)

    # Merge with labels
    logger.info("Merging with labels...")
    labels_day = labels[["subject_id", "lifelog_date", "sleep_date"] + TARGETS].copy()
    labels_day["date"] = pd.to_datetime(labels_day["lifelog_date"]).dt.date.astype(str)

    if all_features is not None and "date" in all_features.columns:
        if all_features["date"].dtype != object:
            all_features["date"] = all_features["date"].dt.date.astype(str)
        elif isinstance(all_features["date"].iloc[0], (str,)):
            pass  # already str

    merged = labels_day.merge(all_features, on=["subject_id", "date"], how="left")
    logger.info(f"  Merged shape: {merged.shape}")

    # Save
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED / "features.parquet"
    merged.to_parquet(out_path, index=False)
    logger.info(f"Saved to {out_path} ({merged.shape[0]} rows, {merged.shape[1]} cols)")

    # Feature count & coverage
    feat_cols = [c for c in merged.columns if c not in ("subject_id","lifelog_date","sleep_date","date") + TARGETS]
    coverage = merged[feat_cols].notna().mean() * 100
    n_sparse = (coverage < 50).sum()
    n_dense = (coverage >= 50).sum()
    logger.info(f"Features: {len(feat_cols)} ({n_dense} ≥50% coverage, {n_sparse} <50%)")
    logger.info(f"Missing feature values: {merged[feat_cols].isnull().sum().sum()}")

    # Per-target distribution
    logger.info("\n--- Per-target distribution ---")
    for t in TARGETS:
        mean_val = merged[t].mean()
        missing = merged[t].isna().sum()
        logger.info(f"  {t}: mean={mean_val:.3f}, missing={missing}")


if __name__ == "__main__":
    main()
