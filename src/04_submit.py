"""
04_submit.py — 제출 파일 생성 및 검증

대회 제출 형식 (ch2026_submission_sample.csv)과 완전히 동일한 형식으로
7개 타겟에 대한 확률 예측값을 생성.

검증 항목:
  1. 컬럼명/순서 일치 (subject_id, sleep_date, lifelog_date, Q1~Q4, S1~S4)
  2. 행 수 일치
  3. 확률값 범위 [0, 1] 검증
  4. 누락값 없음 검증
"""

import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from config import (
    MODEL_DIR,
    SAMPLE_CSV,
    SUBMIT_DIR,
    TARGETS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def load_models() -> dict[str, lgb.Booster]:
    """저장된 LightGBM 모델을 로드."""
    models = {}
    for target in TARGETS:
        model_path = MODEL_DIR / f"lgbm_{target}.txt"
        if not model_path.exists():
            log.warning(f"Model not found: {model_path}")
            continue
        # lightgbm booster를 JSON에서 복구
        # booster.save_model()으로 저장한 경우:
        model = lgb.Booster(model_file=str(model_path))
        models[target] = model
        log.info(f"  Loaded {target}: {model_path}")
    return models


def load_submission_template(path: Path | None = None) -> pd.DataFrame:
    """제출 샘플 템플릿 로드."""
    if path is None:
        path = SAMPLE_CSV
    return pd.read_csv(path, parse_dates=["sleep_date", "lifelog_date"])


def verify_submission_format(submit_df: pd.DataFrame, sample_df: pd.DataFrame) -> list[str]:
    """제출 파일 형식 검증."""
    errors = []

    # 1) 컬럼명/순서 비교
    expected_cols = sample_df.columns.tolist()
    actual_cols = submit_df.columns.tolist()
    if expected_cols != actual_cols:
        errors.append(f"Column mismatch:\n  Expected: {expected_cols}\n  Actual:   {actual_cols}")

    # 2) 행 수 비교
    if len(submit_df) != len(sample_df):
        errors.append(f"Row count mismatch: submit={len(submit_df)}, sample={len(sample_df)}")

    # 3) 확률값 범위
    for t in TARGETS:
        if t in submit_df.columns:
            vals = submit_df[t]
            if (vals < 0).any():
                errors.append(f"{t}: negative values found (min={vals.min()})")
            if (vals > 1).any():
                errors.append(f"{t}: values > 1 found (max={vals.max()})")
            if vals.isnull().any():
                errors.append(f"{t}: NaN values found ({vals.isnull().sum()})")

    # 4) subject_id / date 순서
    if "subject_id" in submit_df.columns and "subject_id" in sample_df.columns:
        if not (submit_df["subject_id"] == sample_df["subject_id"]).all():
            errors.append("subject_id order differs from sample")

    return errors


def create_submission(
    test_df: pd.DataFrame,
    models: dict[str, lgb.Booster],
    features_meta: dict,
) -> pd.DataFrame:
    """
    테스트 데이터로부터 7개 타겟 예측값 생성.

    Parameters
    ----------
    test_df : pd.DataFrame
        테스트용 라이프로그 데이터 (features와 동일한 구조)
    models : dict
        {target: Booster}
    features_meta : dict
        feature_cols 등 메타정보

    Returns
    -------
    pd.DataFrame
        제출용 예측 결과
    """
    log.info("── 예측값 생성 ──")
    meta_cols = ["subject_id", "lifelog_date", "sleep_date", "date"]
    # target 열이 있으면 제거
    predict_cols = [c for c in test_df.columns if c not in meta_cols + TARGETS]

    X_test = test_df[predict_cols].fillna(0).values
    log.info(f"  Test shape: {X_test.shape}, features: {len(predict_cols)}")

    predictions = test_df[["subject_id", "sleep_date", "lifelog_date"]].copy()

    for target, model in models.items():
        pred = model.predict(X_test)
        predictions[target] = pred
        log.info(f"  {target}: pred_range=[{pred.min():.4f}, {pred.max():.4f}], mean={pred.mean():.4f}")

    return predictions


def save_submission(df: pd.DataFrame, path: Path | None = None) -> Path:
    """제출 파일을 CSV로 저장."""
    if path is None:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        path = SUBMIT_DIR / f"submission_{timestamp}.csv"
    else:
        path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info(f"  Saved: {path} ({len(df)} rows)")
    return path


def main(test_features: pd.DataFrame | None = None) -> pd.DataFrame:
    """전체 제출 생성 파이프라인 실행."""
    # 1) 모델 로드
    log.info("=" * 60)
    log.info("04_submit.py — 제출 파일 생성")
    log.info("=" * 60)

    models = load_models()
    if not models:
        log.error("No models found. Run 03_model_training.py first.")
        sys.exit(1)

    # 2) 테스트 데이터
    if test_features is None:
        # 제출 샘플을 테스트 데이터로 사용 (라벨은 0으로 채움)
        sample = load_submission_template()
        for t in TARGETS:
            sample[t] = 0.0
        test_features = sample

    # 3) 예측
    predictions = create_submission(
        test_features,
        models,
        {"feature_cols": list(test_features.columns)},
    )

    # 4) 검증
    sample = load_submission_template()
    errors = verify_submission_format(predictions, sample)

    log.info("\n── 형식 검증 ──")
    if errors:
        for e in errors:
            log.error(f"  ❌ {e}")
    else:
        log.info("  ✅ All checks passed!")

    # 5) 저장
    path = save_submission(predictions)

    # 6) 샘플 출력
    log.info("\n── 제출 파일 샘플 ──")
    log.info(predictions.head(5).to_string(index=False))

    return predictions


if __name__ == "__main__":
    main()
