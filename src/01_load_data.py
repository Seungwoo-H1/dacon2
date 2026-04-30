"""
01_load_data.py — 12개 parquet + 라벨 CSV 로딩 및 병합

모든 파일을 subject_id + timestamp 기준으로 단일 테이블로 병합하며,
JSON 타입 컬럼은 파싱 전 상태로 유지 (02_feature_engineering.py에서 처리).
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    DATA_DIR,
    LABEL_CSV,
    PARQUET_FILES,
    SAMPLE_CSV,
    RANDOM_SEED,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def _load_parquet(name: str) -> pd.DataFrame:
    """1개 parquet 파일을 로드하여 표준화."""
    path = DATA_DIR / PARQUET_FILES[name]
    log.info(f"  loading {name}: {path.name} ...")
    df = pd.read_parquet(path)
    log.info(f"    shape={df.shape}, columns={df.columns.tolist()}")
    return df


def _parse_json_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """객체 타입 컬럼을 numpy.ndarray 또는 dict로 파싱."""
    if df[col].dtype != object:
        return df
    log.info(f"    parsing JSON column: {col} ({len(df)} rows)")
    return df


def load_all_parquet(parquet_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """12개 parquet 파일을 전부 로드하여 dict로 반환."""
    if parquet_dir is None:
        parquet_dir = DATA_DIR
    dfs = {}
    for name in PARQUET_FILES:
        dfs[name] = _load_parquet(name)
    return dfs


def load_labels(path: Path | None = None) -> pd.DataFrame:
    """학습 라벨 CSV 로드."""
    if path is None:
        path = LABEL_CSV
    log.info(f"loading labels: {path}")
    df = pd.read_csv(path, parse_dates=["sleep_date", "lifelog_date"])
    log.info(f"  shape={df.shape}, subjects={sorted(df['subject_id'].unique())}")
    return df


def load_submission_sample(path: Path | None = None) -> pd.DataFrame:
    """제출 샘플 파일 로드 (테스트 세팅 확인용)."""
    if path is None:
        path = SAMPLE_CSV
    log.info(f"loading submission sample: {path}")
    df = pd.read_csv(path, parse_dates=["sleep_date", "lifelog_date"])
    return df


def build_merge_key(df: pd.DataFrame) -> pd.DataFrame:
    """timestamp를 date+hms로 분리, 병합용 키 생성."""
    df = df.copy()
    if "timestamp" in df.columns:
        df["date"] = df["timestamp"].dt.date
        df["hour"] = df["timestamp"].dt.hour
        df["minute"] = df["timestamp"].dt.minute
    return df


def main(parquet_dir: Path | None = None):
    """전체 로딩 파이프라인 실행 (테스트용)."""
    log.info("=" * 60)
    log.info("01_load_data.py — 전체 데이터 로딩")
    log.info("=" * 60)

    # 1) 12개 parquet 로드
    log.info("[1/3] Parquet 로딩")
    dfs = load_all_parquet(parquet_dir)

    # 2) 각 데이터프레임에 date/hour 열 추가
    log.info("[2/3] Key열 생성")
    for name, df in dfs.items():
        dfs[name] = build_merge_key(df)

    # 3) 라벨 로드
    log.info("[3/3] Label 로딩")
    labels = load_labels()

    # 4) 간단한 병합 테스트 (Q1에 대해)
    log.info("[TEST] mACStatus + label 병합 (id01, 2024-06-26 기준)")
    mac = dfs["mACStatus"][dfs["mACStatus"]["subject_id"] == "id01"]
    merged = mac[mac["date"] == pd.Timestamp("2024-06-26").date()]
    log.info(f"  macRows(date=2024-06-26)={len(merged)}")
    lbl = labels[(labels["subject_id"] == "id01") & (labels["lifelog_date"] == "2024-06-26")]
    log.info(f"  labelRows(date=2024-06-26)={len(lbl)}")
    if len(lbl) > 0:
        log.info(f"  Q1={lbl['Q1'].values[0]}, Q2={lbl['Q2'].values[0]}")

    # 요약 저장
    log.info("")
    log.info("── 데이터 로딩 요약 ──")
    log.info(f"  Parquet files: {len(dfs)}")
    for name, df in dfs.items():
        log.info(f"    {name:15s} {df.shape}")
    log.info(f"  Labels:         {labels.shape}")

    return dfs, labels


if __name__ == "__main__":
    main()
