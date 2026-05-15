"""
04_submit_improved.py — 개선된 모델로 제출 파일 생성

새로운 모델 형식:
- LightGBM: clean_lgbm_{target}.txt
- XGBoost:  clean_xgb_{target}.json  
- CatBoost: clean_cb_{target}.cbm

feature map: feature_cols_clean.txt 또는 metrics에서 추출
"""

import csv
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression  # calibrator

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool as CBPool

from config import MODEL_DIR, SAMPLE_CSV, SUBMIT_DIR, TARGETS, DATA_PROCESSED

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def load_models() -> dict[str, dict]:
    """
    개선된 모델들 로드.
    
    Returns: {target: {"model": <model_obj>, "feature_names": list, "calibrator": calibrator, "model_type": str}}
    """
    log.info("=" * 70)
    log.info("개선된 모델 로드 중...")
    log.info("=" * 70)
    
    models = {}
    feat_path = DATA_PROCESSED / "features.parquet"
    df = pd.read_parquet(feat_path)
    
    # Load all feature names from training data (consistent across targets)
    all_clean_cols = get_clean_feature_cols_for_target("Q1", df)
    
    for target in TARGETS:
        metrics_path = MODEL_DIR / f"clean_metrics_{target}.json"
        cb_path = MODEL_DIR / f"clean_cb_{target}.cbm"
        lgb_path = MODEL_DIR / f"clean_lgbm_{target}.txt"
        xgb_path = MODEL_DIR / f"clean_xgb_{target}.json"
        
        # metrics file is the source of truth
        if not metrics_path.exists():
            log.error(f"  ❌ No metrics for {target}")
            continue
        
        with open(metrics_path) as f:
            meta = json.load(f)
        model_type = meta.get("best_model", "CatBoost")
        calib_data = meta.get("calibration")
        
        # Use all training feature names (not just top 20 from metrics)
        feature_names = all_clean_cols
        
        # Load model based on metrics-confirmed type
        model = None
        if model_type == "CatBoost":
            if cb_path.exists():
                model = CatBoostClassifier()
                model.load_model(str(cb_path))
                log.info(f"  Loaded {target}: CatBoost ({cb_path})")
            else:
                log.error(f"  ❌ {model_type} model not found for {target}")
                continue
        elif model_type == "LightGBM":
            if lgb_path.exists():
                model = lgb.Booster(model_file=str(lgb_path))
                log.info(f"  Loaded {target}: LightGBM ({lgb_path})")
            else:
                log.error(f"  ❌ {model_type} model not found for {target}")
                continue
        elif model_type == "XGBoost":
            if xgb_path.exists():
                model = xgb.Booster()
                model.load_model(str(xgb_path))
                log.info(f"  Loaded {target}: XGBoost ({xgb_path})")
            else:
                log.error(f"  ❌ {model_type} model not found for {target}")
                continue
        else:
            log.error(f"  ❌ Unknown model type: {model_type}")
            continue
        
        models[target] = {
            "model": model,
            "feature_names": feature_names,
            "model_type": model_type,
            "calibration": calib_data,
        }
    
    return models


def get_clean_feature_cols_for_target(target: str, df: pd.DataFrame) -> list[str]:
    """
    target leakage 없는 feature 열 추출 (training과 동일하게).
    """
    meta_cols = {"subject_id", "lifelog_date", "sleep_date", "date"}
    leakage_cols = {t for t in TARGETS if t != target}
    exclude = meta_cols | leakage_cols
    
    return [c for c in df.columns 
            if c not in exclude 
            and df[c].dtype in [np.float64, np.int64, float, int, bool]]


def create_submission(
    test_features: pd.DataFrame,
    models: dict[str, dict],
) -> pd.DataFrame:
    """테스트 데이터로부터 예측값 생성 (calibration 적용)."""
    log.info("\n── 예측값 생성 ──")
    
    # Get full feature set for test
    full_feat_cols = get_clean_feature_cols_for_target("Q1", test_features)
    
    predictions = test_features[["subject_id", "sleep_date", "lifelog_date"]].copy()
    
    for target, m in models.items():
        # Use target-specific feature columns
        feature_names = m["feature_names"]
        if not feature_names:
            feature_names = get_clean_feature_cols_for_target(target, test_features)
        
        # Ensure all columns exist
        missing_cols = [c for c in feature_names if c not in test_features.columns]
        if missing_cols:
            log.warning(f"  {target}: adding {len(missing_cols)} missing columns")
            for c in missing_cols:
                test_features[c] = 0.0
        
        predict_cols = [c for c in feature_names if c in test_features.columns]
        X_test = test_features[predict_cols].fillna(0).values
        
        # Predict
        raw_pred = predict_model(m["model"], X_test, m["model_type"], predict_cols)
        
        # Apply calibration if available
        calib_data = m["calibration"]
        if calib_data:
            cal_preds = apply_calibration(raw_pred, calib_data)
        else:
            cal_preds = raw_pred
        
        # Clip
        cal_preds = np.clip(cal_preds, 1e-7, 1 - 1e-7)
        
        predictions[target] = cal_preds
        log.info(f"  {target}: {len(predict_cols)} features, "
                 f"pred=[{cal_preds.min():.4f}, {cal_preds.max():.4f}], mean={cal_preds.mean():.4f}")
    
    return predictions


def predict_model(model, X, model_type, feature_names=None):
    """모델 유형별 예측."""
    if "LightGBM" in model_type:
        return model.predict(X)
    elif "XGBoost" in model_type:
        dmat = xgb.DMatrix(X)
        return model.predict(dmat)
    elif "CatBoost" in model_type:
        if feature_names:
            pool = CBPool(X, feature_names=feature_names)
            return model.predict_proba(pool)[:, 1]
        else:
            return model.predict_proba(X)[:, 1]
    else:
        return model.predict(X)


def apply_calibration(raw_pred, calib_data):
    """캘리브레이션 파라미터로 확률 보정."""
    coeff = calib_data.get("coeff", 1.0)
    intercept = calib_data.get("intercept", 0.0)
    
    # Logistic regression calibration
    logits = coeff * raw_pred + intercept
    prob = 1.0 / (1.0 + np.exp(-logits))
    return prob


def save_submission(df: pd.DataFrame, path: Path | None = None) -> Path:
    if path is None:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        path = SUBMIT_DIR / f"submission_improved_{timestamp}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info(f"\n  💾 Saved: {path} ({len(df)} rows)")
    return path


def main(test_features: pd.DataFrame | None = None) -> pd.DataFrame:
    """전체 제출 생성 파이프라인."""
    log.info("=" * 70)
    log.info("04_submit_improved.py — 개선된 모델로 제출")
    log.info("=" * 70)
    
    models = load_models()
    if not models:
        log.error("No models found. Run 03_model_training_improved.py first.")
        sys.exit(1)
    
    if test_features is None:
        log.info("테스트 데이터 파이프라인 실행 중...")
        sys.path.insert(0, str(Path(__file__).parent))
        import importlib
        from pathlib import Path as P
        
        spec = importlib.util.spec_from_file_location(
            "02_feature_engineering", P(__file__).parent / "02_feature_engineering.py"
        )
        feat_eng = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(feat_eng)
        
        spec2 = importlib.util.spec_from_file_location(
            "01_load_data", P(__file__).parent / "01_load_data.py"
        )
        load_data = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(load_data)
        
        parquet_dfs = {}
        for name in load_data.PARQUET_FILES:
            path = load_data.DATA_DIR / load_data.PARQUET_FILES[name]
            log.info(f"  loading {name}: {path.name} ...")
            df = pd.read_parquet(path)
            parquet_dfs[name] = load_data.build_merge_key(df)
        
        sample = load_submission_template()
        sample["lifelog_date"] = pd.to_datetime(sample["lifelog_date"]).dt.date
        
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


def load_submission_template(path=None):
    if path is None:
        path = SAMPLE_CSV
    return pd.read_csv(path, parse_dates=["sleep_date", "lifelog_date"])


def verify_submission_format(submit_df, sample_df):
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
                errors.append(f"{t}: negative values found")
            if (vals > 1).any():
                errors.append(f"{t}: values > 1 found")
            if vals.isnull().any():
                errors.append(f"{t}: NaN values found ({vals.isnull().sum()})")
    return errors


if __name__ == "__main__":
    main()
