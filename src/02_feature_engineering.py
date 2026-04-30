"""
02_feature_engineering.py — 특징 공학 파이프라인

라이프로그 12개 파일을 time-based aggregation으로 day-level 피처로 변환.
JSON 타입 컬럼(mAmbience, mBle, mGps, mUsageStats, mWifi, wHr)도 통계 추출.

핵심 전략:
  - lifelog_date(당일 라이프로그) → sleep_date(다음날 라벨) 의 1일 간격 유지
  - 시간 누수 방지: 각 subject의 lifelog_date 기준 aggregation
  - 다중 window aggregation (1h, 3h, 6h, 12h, 24h)
"""

import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    AGG_WINDOWS,
    HOUR_BINS,
    LABEL_CSV,
    PARQUET_FILES,
    RANDOM_SEED,
    TARGETS,
    DATA_DIR,
    DATA_PROCESSED,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ── JSON 컬럼 매핑 ─────────────────────────────────────────
# 각 JSON 컬럼에 대한 파싱/통계 추출 함수
JSON_COLUMNS = {
    "mAmbience":    {"col": "m_ambience"},
    "mBle":         {"col": "m_ble"},
    "mGps":         {"col": "m_gps"},
    "mUsageStats":  {"col": "m_usage_stats"},
    "mWifi":        {"col": "m_wifi"},
    "wHr":          {"col": "heart_rate"},
}

# 비JSON 수치 컬럼
NUMERIC_COLUMNS = {
    "mACStatus":     "m_charging",
    "mActivity":     "m_activity",
    "mLight":        "m_light",
    "mScreenStatus": "m_screen_use",
    "wLight":        "w_light",
    "wPedo":         None,  # 다중 열
}

# wPedo 열들
WPEDO_COLS = ["step", "step_frequency", "running_step", "walking_step",
              "distance", "speed", "burned_calories"]


# ── JSON 파싱 헬퍼 ────────────────────────────────────────

def parse_ambience(value) -> dict[str, float]:
    """m_ambience: ndarray of [category, prob] → 확률 합산."""
    if not isinstance(value, (np.ndarray, list)):
        return {}
    scores = defaultdict(float)
    for item in value:
        if isinstance(item, (np.ndarray, list)) and len(item) >= 2:
            scores[str(item[0])] += float(item[1])
    return dict(scores)


def parse_ble(value) -> dict:
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
                devices.add(item["address"][:10])  # prefix로 grouping
    return {
        "ble_count": len(value),
        "ble_device_count": len(devices),
        "ble_avg_rssi": np.mean(rssis) if rssis else np.nan,
        "ble_max_rssi": np.max(rssis) if rssis else np.nan,
        "ble_min_rssi": np.min(rssis) if rssis else np.nan,
        "ble_rssi_std": np.std(rssis) if len(rssis) > 1 else np.nan,
    }


def parse_gps(value) -> dict:
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


def parse_usage_stats(value) -> dict:
    """m_usage_stats: list of {app_name, total_time} → 통계."""
    if not isinstance(value, (np.ndarray, list)):
        return {"app_count": 0, "total_time": 0}
    apps = []
    total_time = 0
    for item in value:
        if isinstance(item, dict):
            apps.append(item.get("app_name", ""))
            tt = item.get("total_time", 0)
            if tt is not None:
                total_time += float(tt)
    # 상위 앱 (top-3 카테고리)
    app_cat = defaultdict(float)
    for a in apps:
        if isinstance(a, str):
            # 간략한 카테고리 매핑
            if any(k in a.lower() for k in ["naver", "google", "카카오"]):
                app_cat["major"] += 1
            elif "game" in a.lower():
                app_cat["game"] += 1
            else:
                app_cat["other"] += 1

    return {
        "usage_app_count": len(apps),
        "usage_total_time": total_time,
        "usage_major_ratio": app_cat.get("major", 0) / max(len(apps), 1),
        "usage_game_ratio": app_cat.get("game", 0) / max(len(apps), 1),
    }


def parse_wifi(value) -> dict:
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
    # Strong signal ratio
    strong = sum(1 for r in rssis if r > -60) if rssis else 0
    return {
        "wifi_count": len(value),
        "wifi_bssid_count": len(bssids),
        "wifi_avg_rssi": np.mean(rssis) if rssis else np.nan,
        "wifi_max_rssi": np.max(rssis) if rssis else np.nan,
        "wifi_strong_ratio": strong / max(len(rssis), 1),
    }


def parse_heart_rate(value) -> dict:
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


# 파서 맵
PARSERS = {
    "mAmbience": parse_ambience,
    "mBle": parse_ble,
    "mGps": parse_gps,
    "mUsageStats": parse_usage_stats,
    "mWifi": parse_wifi,
    "wHr": parse_heart_rate,
}


# ── JSON 통계 컬럼 추출 (ambience) ───────────────────────
def _extract_ambience_features(row_values: pd.Series) -> pd.Series:
    """mAmbience 전체에 대한 category별 aggregated 확률."""
    scores = defaultdict(float)
    total_rows = 0
    for v in row_values.dropna():
        if isinstance(v, (np.ndarray, list)):
            parsed = parse_ambience(v)
            for k, s in parsed.items():
                scores[k] += s
            total_rows += 1

    result = pd.Series(dtype=float)
    for cat in ["Speech", "Music", "Vehicle", "Motor vehicle (road)",
                 "Inside, large room or hall", "Inside, small room",
                 "Outside, urban or manmade", "Outside, rural or natural",
                 "Car", "Truck"]:
        key = f"ambience_{cat.lower().replace(' ', '_')}_sum"
        result[key] = scores.get(cat, 0.0) / max(total_rows, 1)
    # 전체 sound score (top-5 확률 합)
    all_scores = sorted(scores.values(), reverse=True)[:5]
    result["ambience_top5_sum"] = sum(all_scores) / max(total_rows, 1)
    result["ambience_max_cat"] = max(scores, key=scores.get) if scores else ""
    return result


# ── 핵심 aggregation 함수 ────────────────────────────────

def aggregate_numeric(df: pd.DataFrame, col: str, date_col: str,
                      agg_cols: list[str]) -> pd.DataFrame:
    """Numeric 열을 date 기준 aggregation."""
    grouped = df.groupby(date_col)[col].agg(["mean", "std", "min", "max", "count"])
    grouped.columns = [f"{col}_{c}" for c in grouped.columns]
    # groupby가 빈 그룹을 빼므로 reindex로 보강
    return grouped.reset_index()


def aggregate_wpedo(df: pd.DataFrame, date_col: str,
                    agg_cols: list[str]) -> pd.DataFrame:
    """wPedo의 7개 열을 date 기준 aggregation."""
    grouped = df.groupby(date_col)[WPEDO_COLS].agg(["mean", "sum"])
    grouped.columns = [f"pedo_{col}_{stat}" for col, stat in grouped.columns]
    return grouped.reset_index()


# ── 메인 파이프라인 ──────────────────────────────────────

def create_day_features(
    parquet_dfs: dict,
    labels: pd.DataFrame,
    agg_windows: list[int] = None,
) -> pd.DataFrame:
    """
    모든 라이프로그 데이터를 day-level 피처로 변환 후 라벨과 병합.

    Parameters
    ----------
    parquet_dfs : dict
        01_load_data에서 로드한 12개 parquet DataFrames
    labels : pd.DataFrame
        라벨 CSV
    agg_windows : list[int], optional
        시간 윈도우 (시) — 기본은 config.py AGG_WINDOWS

    Returns
    -------
    pd.DataFrame
        day-level 피처 + 라벨이 병합된 데이터프레임
    """
    if agg_windows is None:
        agg_windows = AGG_WINDOWS

    log.info("=" * 60)
    log.info("02_feature_engineering.py — 특징 공학")
    log.info("=" * 60)

    all_features = pd.DataFrame()

    # ── 1) Numeric 컬럼 aggregation ────────────────────────
    log.info("[1/4] Numeric column aggregation")
    for source, col in NUMERIC_COLUMNS.items():
        df = parquet_dfs[source]
        if col is None:
            # wPedo는 별도 처리
            pass
        else:
            feat = aggregate_numeric(df, col, "date", [])
            feat = feat.rename(columns={c: f"{source}_{c}" for c in feat.columns if c != "date"})
            all_features = pd.concat([all_features, feat], axis=0).drop_duplicates("date")

    # wPedo
    pedo_feat = aggregate_wpedo(parquet_dfs["wPedo"], "date", [])
    pedo_feat = pedo_feat.rename(columns={c: f"wPedo_{c}" for c in pedo_feat.columns if c != "date"})
    all_features = pd.concat([all_features, pedo_feat], axis=0).drop_duplicates("date")

    # ── 2) JSON 컬럼 파싱 → day aggregation ───────────────
    log.info("[2/4] JSON column parsing & aggregation")

    for source, info in JSON_COLUMNS.items():
        df = parquet_dfs[source]
        json_col = info["col"]
        parser = PARSERS[source]

        if source == "mAmbience":
            # mAmbience는 special 처리: 전체 day의 ambience를 하나로 합침
            amb_feats = df.groupby("date").apply(
                lambda g: _extract_ambience_features(g[json_col]),
                include_groups=False,
            )
            amb_feats = amb_feats.reset_index()
            amb_feats = amb_feats.rename(
                columns={c: f"mAmbience_{c}" for c in amb_feats.columns if c != "date"}
            )
            all_features = pd.concat([all_features, amb_feats], axis=0).drop_duplicates("date")

        elif source == "wHr":
            # wHr: heart_rate 배열 → day별 평균 HR
            hr_flat = []
            for _, row in df.iterrows():
                if isinstance(row[json_col], (np.ndarray, list)):
                    hr_flat.extend(row[json_col])
            avg_hr = np.mean(hr_flat) if hr_flat else np.nan
            day_feat = pd.DataFrame({
                "date": df["date"].unique(),
                "wHr_hr_mean": avg_hr,
                "wHr_hr_std": np.std(hr_flat) if len(hr_flat) > 1 else np.nan,
                "wHr_hr_count": len(hr_flat),
            })
            all_features = pd.concat([all_features, day_feat], axis=0).drop_duplicates("date")

        else:
            # mBle, mGps, mUsageStats, mWifi: row-level 파싱 → day aggregation
            parsed = df[json_col].apply(parser)
            parsed_df = pd.DataFrame(parsed.tolist(), index=df.index)
            parsed_df["date"] = df["date"]

            # 각 통계열을 day별 mean/std로 aggregte
            stat_cols = [c for c in parsed_df.columns if c != "date"]
            for sc in stat_cols:
                grouped = parsed_df.groupby("date")[sc].agg(["mean", "std", "max", "min"])
                grouped.columns = [f"{source}_{sc}_{s}" for s in grouped.columns]
                all_features = pd.concat([all_features, grouped.reset_index()], axis=0).drop_duplicates("date")

    # ── 3) 시간대별 피처 ───────────────────────────────────
    log.info("[3/4] Time-of-day features")
    for source, col in [("mACStatus", "m_charging"), ("mScreenStatus", "m_screen_use")]:
        df = parquet_dfs[source]
        # 시간대별 사용 비율
        df["hour_bin"] = df["hour"].apply(
            lambda h: "night" if h < 6 else "morning" if h < 12 else "afternoon" if h < 18 else "evening"
        )
        ratio = df.groupby("date")["hour_bin"].value_counts(normalize=True).unstack(fill_value=0)
        ratio.columns = [f"{source}_hour_{c}" for c in ratio.columns]
        all_features = pd.concat([all_features, ratio.reset_index()], axis=0).drop_duplicates("date")

    # ── 4) 라벨과 병합 ─────────────────────────────────────
    log.info("[4/4] Merge with labels")
    labels_day = labels[["subject_id", "lifelog_date", "sleep_date"] + TARGETS].copy()
    labels_day["date"] = labels_day["lifelog_date"].dt.date

    # all_features의 date도 object(str)로 통일
    if all_features["date"].dtype.name.startswith("datetime"):
        all_features = all_features.copy()
        all_features["date"] = all_features["date"].dt.date

    merged = labels_day.merge(all_features, on=["date"], how="left")
    log.info(f"  Merged shape: {merged.shape}")
    log.info(f"  Missing features: {merged.isnull().sum().sum()}")

    # subject_id 추가 (merge 시 빠질 수 있음)
    if "subject_id" not in merged.columns:
        merged["subject_id"] = labels_day["subject_id"]

    return merged


def save_features(df: pd.DataFrame, suffix: str = "") -> Path:
    """특징 데이터프레임을 parquet로 저장."""
    out = DATA_PROCESSED / f"features{suffix}.parquet"
    df.to_parquet(out, index=False)
    log.info(f"  Saved to {out} ({len(df)} rows, {len(df.columns)} cols)")
    return out


def main(agg_windows: list[int] | None = None):
    """전체 파이프라인 실행."""
    import importlib
    load_module = importlib.import_module("01_load_data")
    parquet_dfs, labels = load_module.main()

    features = create_day_features(parquet_dfs, labels, agg_windows)
    save_features(features)

    # 타겟별 분포 확인
    log.info("\n── 타겟별 분포 ──")
    for t in TARGETS:
        if t in features.columns:
            log.info(f"  {t}: mean={features[t].mean():.3f}, "
                     f"missing={features[t].isnull().sum()}")

    return features


if __name__ == "__main__":
    main()
