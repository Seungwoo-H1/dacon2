"""
04_submit.py — 제출 파일 생성 및 검증

대회 제출 형식 (ch2026_submission_sample.csv)과 완전히 동일한 형식으로
7개 타겟에 대한 확률 예측값을 생성.

검증 항목:
  1. 컬럼명/순서 일치 (subject_id, sleep_date, lifelog_date, Q1~Q4, S1~S4)
  2. 행 수 일치
  3. 확률값 범위 [0, 1] 검증
  4. 누락값 없음 검증

전략:
  - 02_feature_engineering.py를 직접 호출하여 테스트 데이터의 features 생성
  - features.parquet의 컬럼 매핑을 통해 예측
"""

import csv
import json
import logging
import re
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
    expected_cols = sample_df.columns.tolist()
    actual_cols = submit_df.columns.tolist()
    if expected_cols != actual_cols:
        errors.append(f"Column mismatch:\n  Expected: {expected_cols}\n  Actual:   {actual_cols}")
    if len(submit_df) != len(sample_df):
        errors.append(f"Row count mismatch: submit={len(submit_df)}, sample={len(sample_df)}")
    for t in TARGETS:
        if t in submit_df.columns:
            vals = submit_df[t]
            if (vals < 0).any():
                errors.append(f"{t}: negative values found (min={vals.min()})")
            if (vals > 1).any():
                errors.append(f"{t}: values > 1 found (max={vals.max()})")
            if vals.isnull().any():
                errors.append(f"{t}: NaN values found ({vals.isnull().sum()})")
    if "subject_id" in submit_df.columns and "subject_id" in sample_df.columns:
        if not (submit_df["subject_id"] == sample_df["subject_id"]).all():
            errors.append("subject_id order differs from sample")
    return errors


def load_feature_columns() -> list[str]:
    """
    Training 때 사용된 feature 열 목록을 로드.
    features.parquet에서 target 열 제외한 numeric 열을 추출.
    """
    from config import DATA_PROCESSED
    feat_path = DATA_PROCESSED / "features.parquet"
    if not feat_path.exists():
        log.error(f"Features not found: {feat_path}")
        log.error("Run 02_feature_engineering.py first.")
        sys.exit(1)
    
    df = pd.read_parquet(feat_path)
    meta_cols = ["subject_id", "lifelog_date", "sleep_date", "date"]
    # 03_model_training.py와 동일한 로직: 각 target별로 다른 feature set
    # 하지만 submit 때는 모든 target의 features를 포함해야 함
    feat_cols = [c for c in df.columns if c not in meta_cols + TARGETS]
    feat_cols = [c for c in feat_cols if df[c].dtype in [np.float64, np.int64, float, int, bool]]
    return feat_cols


def get_train_feature_cols_for_target(target: str) -> list[str]:
    """
    03_model_training.py와 동일한 로직으로 feature 열 추출.
    training: target 열을 제외하고 나머지 target열을 feature로 포함
    """
    from config import DATA_PROCESSED
    feat_path = DATA_PROCESSED / "features.parquet"
    df = pd.read_parquet(feat_path)
    meta_cols = ["subject_id", "lifelog_date", "sleep_date", "date"]
    feature_cols = [c for c in df.columns if c not in meta_cols + [target]]
    feature_cols = [c for c in feature_cols if df[c].dtype in [np.float64, np.int64, float, int, bool]]
    return feature_cols


def create_submission(
    test_features: pd.DataFrame,
    models: dict[str, lgb.Booster],
) -> pd.DataFrame:
    """테스트 데이터로부터 7개 타겟 예측값 생성."""
    log.info("── 예측값 생성 ──")

    predictions = test_features[["subject_id", "sleep_date", "lifelog_date"]].copy()

    for target, model in models.items():
        predict_cols = get_train_feature_cols_for_target(target)
        
        # Ensure all columns exist in test_features
        missing_cols = [c for c in predict_cols if c not in test_features.columns]
        if missing_cols:
            log.warning(f"  {target}: missing columns: {missing_cols[:5]}...")
        
        # Add missing columns with 0
        for c in missing_cols:
            test_features[c] = 0.0
        
        predict_cols_in_df = [c for c in predict_cols if c in test_features.columns]
        X_test = test_features[predict_cols_in_df].fillna(0).values
        log.info(f"  {target}: {len(predict_cols_in_df)} features")
        pred = model.predict(X_test)
        predictions[target] = pred
        log.info(f"    pred_range=[{pred.min():.4f}, {pred.max():.4f}], mean={pred.mean():.4f}")

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
    log.info("=" * 60)
    log.info("04_submit.py — 제출 파일 생성")
    log.info("=" * 60)

    models = load_models()
    if not models:
        log.error("No models found. Run 03_model_training.py first.")
        sys.exit(1)

    if test_features is None:
        log.info("테스트 데이터 파이프라인 실행 중...")

        # 02_feature_engineering 동적 임포트
        sys.path.insert(0, str(Path(__file__).parent))
        import importlib
        from pathlib import Path as P
        spec = importlib.util.spec_from_file_location(
            "02_feature_engineering", 
            P(__file__).parent / "02_feature_engineering.py"
        )
        feat_eng = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(feat_eng)
        
        # 01_load_data 동적 임포트
        spec2 = importlib.util.spec_from_file_location(
            "01_load_data", 
            P(__file__).parent / "01_load_data.py"
        )
        load_data = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(load_data)
        
        # parquet 로드
        parquet_dfs = {}
        for name in load_data.PARQUET_FILES:
            path = load_data.DATA_DIR / load_data.PARQUET_FILES[name]
            log.info(f"  loading {name}: {path.name} ...")
            df = pd.read_parquet(path)
            parquet_dfs[name] = load_data.build_merge_key(df)

        # 제출 템플릿 로드
        sample = load_submission_template()
        sample["lifelog_date"] = pd.to_datetime(sample["lifelog_date"]).dt.date

        # feature engineering 실행
        log.info("  Feature engineering for test data...")
        test_features = feat_eng.create_day_features(parquet_dfs, sample)
        log.info(f"  Test features shape: {test_features.shape}")

    # 예측
    predictions = create_submission(test_features, models)

    # 검증
    sample = load_submission_template()
    errors = verify_submission_format(predictions, sample)
    log.info("\n── 형식 검증 ──")
    if errors:
        for e in errors:
            log.error(f"  ❌ {e}")
    else:
        log.info("  ✅ All checks passed!")

    # 저장
    path = save_submission(predictions)

    # 샘플 출력
    log.info("\n── 제출 파일 샘플 ──")
    log.info(predictions.head(5).to_string(index=False))

    return predictions


if __name__ == "__main__":
    main()
