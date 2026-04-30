"""
config.py — Dacon2 베이스라인 설정

대회: 제 5회 ETRI 휴먼이해 인공지능 논문경진대회 (dacon2)
"""

import os
from pathlib import Path

# ── 경로 설정 ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data_raw"
DATA_DIR = DATA_RAW / "ch2025_data_items"
DATA_PROCESSED = PROJECT_ROOT / "data_processed"
MODEL_DIR = PROJECT_ROOT / "models"
SUBMIT_DIR = PROJECT_ROOT / "submissions"

# 자동 생성
DATA_PROCESSED.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
SUBMIT_DIR.mkdir(exist_ok=True)

# ── 파일 이름 매핑 ────────────────────────────────────────
PARQUET_FILES = {
    "mACStatus":    "ch2025_mACStatus.parquet",
    "mActivity":    "ch2025_mActivity.parquet",
    "mAmbience":    "ch2025_mAmbience.parquet",
    "mBle":         "ch2025_mBle.parquet",
    "mGps":         "ch2025_mGps.parquet",
    "mLight":       "ch2025_mLight.parquet",
    "mScreenStatus": "ch2025_mScreenStatus.parquet",
    "mUsageStats":  "ch2025_mUsageStats.parquet",
    "mWifi":        "ch2025_mWifi.parquet",
    "wHr":          "ch2025_wHr.parquet",
    "wLight":       "ch2025_wLight.parquet",
    "wPedo":        "ch2025_wPedo.parquet",
}

LABEL_CSV = DATA_RAW / "ch2026_metrics_train.csv"
SAMPLE_CSV = DATA_RAW / "ch2026_submission_sample.csv"

# ── 타겟 변수 ─────────────────────────────────────────────
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]

# ── 시드 ──────────────────────────────────────────────────
RANDOM_SEED = 42

# ── LightGBM 기본 하이퍼파라미터 ──────────────────────────
LGBM_DEFAULTS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "verbose": -1,
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
    "force_row_wise": True,
}

# 기본 학습 파라미터 (각 타겟별 최적화 가능)
LGBM_PARAMS = {
    "num_leaves": 63,
    "max_depth": -1,
    "learning_rate": 0.05,
    "n_estimators": 1000,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 1.0,  # 타겟별 동적 조정
    "early_stopping_round": 50,
}

# ── 교차 검증 설정 ────────────────────────────────────────
# 시간 기반 split: 각 subject의 마지막 N일을 validation으로
CV_CONFIG = {
    "n_splits": 5,       # GroupKFold 스타일 (subject 기반)
    "test_subjects": 0,   # 모든 subject 포함 (내부 validation만)
    "val_days": 7,       # 각 subject의 마지막 7일을 validation
}

# ── 특징 공학 ─────────────────────────────────────────────
# aggregation window (시간 단위)
AGG_WINDOWS = [1, 3, 6, 12, 24]  # 시간 단위

# 시간대 binning
HOUR_BINS = {
    "morning": (6, 12),
    "afternoon": (12, 18),
    "evening": (18, 24),
    "night": (0, 6),
}
