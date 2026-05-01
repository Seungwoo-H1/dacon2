"""
04_submit_clean.py — Clean submission generation (no target leakage)

Target leakage 제거된 clean LightGBM 모델로 테스트 데이터 예측.
02_feature_engineering.py의 feature engineering 파이프라인을 그대로 사용.
"""

import importlib.util
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent))
from config import MODEL_DIR, SAMPLE_CSV, SUBMIT_DIR, TARGETS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def sanitize(name):
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)


def get_clean_feature_cols(features_df: pd.DataFrame, target: str) -> list[str]:
    """target leakage 없는 feature 열 추출."""
    meta_cols = {"subject_id", "lifelog_date", "sleep_date", "date"}
    leakage_cols = {t for t in TARGETS if t != target}
    exclude = meta_cols | leakage_cols

    return [
        c for c in features_df.columns
        if c not in exclude
        and features_df[c].dtype in [np.float64, np.int64, float, int, bool]
    ]


def load_clean_models() -> dict[str, lgb.Booster]:
    """Clean LightGBM 모델 로드."""
    models = {}
    for target in TARGETS:
        model_path = MODEL_DIR / f"clean_lgbm_{target}.txt"
        if not model_path.exists():
            log.error(f"Clean model not found: {model_path}")
            continue
        models[target] = lgb.Booster(model_file=str(model_path))
        log.info(f"  Loaded clean {target}: {model_path}")
    return models


def load_sample() -> pd.DataFrame:
    """제출 샘플 로드."""
    df = pd.read_csv(SAMPLE_CSV)
    df["lifelog_date"] = pd.to_datetime(df["lifelog_date"]).dt.date
    df["sleep_date"] = pd.to_datetime(df["sleep_date"]).dt.date
    return df


def create_test_features(sample: pd.DataFrame) -> pd.DataFrame:
    """
    02_feature_engineering.py를 호출하여 테스트 데이터의 features 생성.
    JSON 파싱이 크면 메모리 부족으로 실패할 수 있으므로,
    가능한 subset만 로드하도록 함.
    """
    log.info("Feature engineering for test data...")

    # 동적 임포트
    spec = importlib.util.spec_from_file_location(
        "02_feature_engineering", Path(__file__).parent / "02_feature_engineering.py"
    )
    feat_eng = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(feat_eng)

    spec2 = importlib.util.spec_from_file_location(
        "01_load_data", Path(__file__).parent / "01_load_data.py"
    )
    load_data = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(load_data)

    # parquet 로드 (data_raw가 아닌 ch2025_data_items 기준)
    parquet_dfs = {}
    data_dir = Path(__file__).parent.parent / "data_raw" / "ch2025_data_items"
    parquet_names = {
        "mACStatus": "ch2025_mACStatus.parquet",
        "mActivity": "ch2025_mActivity.parquet",
        "mAmbience": "ch2025_mAmbience.parquet",
        "mBle": "ch2025_mBle.parquet",
        "mGps": "ch2025_mGps.parquet",
        "mLight": "ch2025_mLight.parquet",
        "mScreenStatus": "ch2025_mScreenStatus.parquet",
        "mUsageStats": "ch2025_mUsageStats.parquet",
        "mWifi": "ch2025_mWifi.parquet",
        "wHr": "ch2025_wHr.parquet",
        "wLight": "ch2025_wLight.parquet",
        "wPedo": "ch2025_wPedo.parquet",
    }

    # date 열을 문자열로 추출
    test_dates = set(sample["sleep_date"].astype(str).tolist())
    test_lifelog_dates = set(sample["lifelog_date"].astype(str).tolist())
    all_dates = test_dates | test_lifelog_dates

    for name, fname in parquet_names.items():
        path = data_dir / fname
        log.info(f"  loading {name} ({fname}) ...")
        df = pd.read_parquet(path)
        # build_merge_key로 date 열 생성
        _spec = importlib.util.spec_from_file_location("01_load_data", Path(__file__).parent / "01_load_data.py")
        _ld_mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_ld_mod)
        df = _ld_mod.build_merge_key(df)
        # date 기준 필터링 (테스트 날짜만)
        if "date" in df.columns:
            df = df[df["date"].astype(str).isin(all_dates)]
        parquet_dfs[name] = df
        log.info(f"    {name}: {len(df)} rows after date filter")

    log.info("  Running feature engineering...")
    test_features = feat_eng.create_day_features(parquet_dfs, sample)
    log.info(f"  Test features shape: {test_features.shape}")
    return test_features


def create_submission(test_features: pd.DataFrame, models: dict[str, lgb.Booster]) -> pd.DataFrame:
    """테스트 데이터로 7개 타겟 예측."""
    log.info("── 예측값 생성 ──")

    # 학습 시 사용한 feature 구조 참조
    feat_ref = pd.read_parquet(Path(__file__).parent.parent / "data_processed" / "features.parquet")

    predictions = test_features[["subject_id", "sleep_date", "lifelog_date"]].copy()

    for target, model in models.items():
        clean_cols = get_clean_feature_cols(feat_ref, target)

        # 테스트 데이터에서 matching하는 열만 추출
        matching = [c for c in clean_cols if c in test_features.columns]
        missing = [c for c in clean_cols if c not in test_features.columns]

        if missing:
            log.info(f"  {target}: {len(matching)}/{len(clean_cols)} present, {len(missing)} missing → fill 0")
            for c in missing:
                test_features[c] = 0.0

        X_test = test_features[matching].fillna(0).values
        pred = model.predict(X_test)
        predictions[target] = pred

        log.info(f"    {target}: range=[{pred.min():.4f}, {pred.max():.4f}], mean={pred.mean():.4f}")

    return predictions


def main():
    log.info("=" * 60)
    log.info("04_submit_clean.py — Clean Submission (no target leakage)")
    log.info("=" * 60)

    # 모델 로드
    models = load_clean_models()
    if not models:
        log.error("No models found. Run 03 clean model training first.")
        sys.exit(1)

    # 샘플 로드
    sample = load_sample()
    log.info(f"Sample shape: {sample.shape}")

    # feature 생성
    test_features = create_test_features(sample)

    # 예측
    predictions = create_submission(test_features, models)

    # 저장
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    path = SUBMIT_DIR / f"submission_clean_{timestamp}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(path, index=False)
    log.info(f"\n✅ Saved: {path} ({len(predictions)} rows)")

    # 검증
    expected_cols = list(sample.columns)
    if list(predictions.columns) == expected_cols:
        log.info("✅ Column order matches sample")
    else:
        log.warning(f"Column mismatch:\n  Expected: {expected_cols}\n  Actual:   {list(predictions.columns)}")

    log.info(f"\nPrediction summary:")
    for t in TARGETS:
        vals = predictions[t]
        log.info(f"  {t}: [{vals.min():.4f}, {vals.max():.4f}], mean={vals.mean():.4f}")

    log.info(f"\nFirst 5 rows:")
    log.info(predictions.head().to_string(index=False))

    # training distribution 비교
    train = pd.read_csv(Path(__file__).parent.parent / "data_raw" / "ch2026_metrics_train.csv")
    log.info(f"\nTrain vs Pred comparison:")
    for t in TARGETS:
        train_rate = train[t].mean()
        pred_mean = predictions[t].mean()
        log.info(f"  {t}: train_pos_rate={train_rate:.3f} → pred_mean={pred_mean:.3f}")

    return predictions


if __name__ == "__main__":
    main()
