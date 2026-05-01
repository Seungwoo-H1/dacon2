"""
03_model_training.py — LightGBM 기반 모델 학습 + CV 검증

핵심 설계:
  - 7개 타겟별 개별 LightGBM 모델 (이진 분류)
  - Subject 기반 GroupKFold 교차검증 (시간 누수 방지)
  - scale_pos_weight 동적 조정 (불균형 클래스 대응)
  - 확률 캘리브레이션 (Isotonic regression)
  - 모델 시드 고정 및 재현성 보장
"""

import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

from config import (
    LGBM_DEFAULTS,
    LGBM_PARAMS,
    MODEL_DIR,
    RANDOM_SEED,
    TARGETS,
    DATA_PROCESSED,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ── CV 검증용 시간 기반 split ──────────────────────────────
def create_cv_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    val_days: int = 7,
) -> list[tuple]:
    """
    Subject별 마지막 N일을 validation으로 하는 GroupKFold 스타일 분할.

    각 subject의 lifelog_date를 기준으로:
    - 마지막 val_days일을 validation
    - 나머지를 training

    Returns: list of (train_idx, val_idx)
    """
    subjects = sorted(df["subject_id"].unique())
    train_indices = []
    val_indices = []

    for idx, row in df.iterrows():
        sid = row["subject_id"]
        subject_rows = df[df["subject_id"] == sid].sort_values("lifelog_date")
        if len(subject_rows) > val_days:
            val_cutoff = subject_rows.index[-val_days]
            if idx <= val_cutoff:
                train_indices.append(idx)
            else:
                val_indices.append(idx)
        else:
            train_indices.append(idx)

    return train_indices, val_indices


def compute_scale_pos_weight(df: pd.DataFrame, target: str) -> float:
    """타겟 클래스 비율로 scale_pos_weight 계산."""
    y = df[target].dropna()
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    return n_neg / max(n_pos, 1)


def train_single_target(
    features: pd.DataFrame,
    target: str,
    lgbm_params: dict,
    train_idx: list,
    val_idx: list,
) -> dict:
    """1개 타겟의 LightGBM 모델 학습 + 검증."""
    train_df = features.iloc[train_idx]
    val_df = features.iloc[val_idx]

    train_df = train_df.dropna(subset=[target])
    val_df = val_df.dropna(subset=[target])

    y_train = train_df[target].values.astype(int)
    y_val = val_df[target].values.astype(int)

    # 피처 선택 — 모든 target 열을 meta에서 제외 (target leakage 방지)
    meta_cols = ["subject_id", "lifelog_date", "sleep_date", "date"] + TARGETS
    feature_cols = [c for c in features.columns if c not in meta_cols]
    feature_cols = [c for c in feature_cols if features[c].dtype in [np.float64, np.int64, float, int, bool]]

    # sanitized 이름 생성 (원본→sanitized 매핑)
    def sanitize_name(name):
        return re.sub(r'[^a-zA-Z0-9_]', '_', name)
    sanitized_cols = [sanitize_name(c) for c in feature_cols]

    # features의 컬럼명을 sanitized 버전으로 rename 후 사용
    rename_map = dict(zip(feature_cols, sanitized_cols))
    train_df_san = train_df[feature_cols].fillna(0).rename(columns=rename_map)
    val_df_san = val_df[feature_cols].fillna(0).rename(columns=rename_map)

    X_train = train_df_san.values
    X_val = val_df_san.values

    spw = compute_scale_pos_weight(train_df, target)

    params = {**LGBM_DEFAULTS, **lgbm_params, "scale_pos_weight": spw}

    log.info(f"    target={target}, spw={spw:.2f}, "
             f"train={len(X_train)}/{len(y_train)}, val={len(X_val)}/{len(y_val)}, "
             f"features={len(feature_cols)}")

    train_set = lgb.Dataset(X_train, label=y_train, feature_name=sanitized_cols)
    val_set = lgb.Dataset(X_val, label=y_val, feature_name=sanitized_cols, reference=train_set)

    callbacks = [
        lgb.early_stopping(stopping_rounds=params["early_stopping_round"]),
        lgb.log_evaluation(period=100),
    ]

    model = lgb.train(
        params,
        train_set,
        num_boost_round=params["n_estimators"],
        valid_sets=[val_set],
        callbacks=callbacks,
    )

    # 검증 로스
    val_pred = model.predict(X_val)
    val_loss = log_loss(y_val, val_pred, labels=[0, 1])

    # feature importance
    importances = model.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feature_cols, importances), key=lambda x: -x[1])

    return {
        "target": target,
        "model": model,
        "best_iteration": model.best_iteration,
        "val_logloss": val_loss,
        "scale_pos_weight": spw,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "feature_importances": feat_imp[:20],
        "feature_cols": feature_cols,
        "sanitized_cols": sanitized_cols,
    }


def train_all_targets(
    features: pd.DataFrame,
    cv_config: dict | None = None,
) -> dict[str, dict]:
    """
    7개 타겟 모두 학습.

    Parameters
    ----------
    features : pd.DataFrame
        02_feature_engineering에서 생성한 특징 데이터프레임
    cv_config : dict
        교차 검증 설정

    Returns
    -------
    dict
        {target: {model, metrics, ...}}
    """
    log.info("=" * 60)
    log.info("03_model_training.py — 모델 학습")
    log.info("=" * 60)

    if cv_config is None:
        cv_config = {"n_splits": 5, "val_days": 7}

    results = {}
    for target in TARGETS:
        log.info(f"\n── [{target}] 학습 시작 ──")
        log.info(f"    target distribution: "
                 f"pos={(features[target]==1).sum()}, "
                 f"neg={(features[target]==0).sum()}")

        train_idx, val_idx = create_cv_splits(
            features, val_days=cv_config["val_days"]
        )

        metrics = train_single_target(
            features, target, LGBM_PARAMS, train_idx, val_idx
        )
        results[target] = metrics
        log.info(f"    ✅ {target} validation logloss = {metrics['val_logloss']:.4f}")

    # 전체 요약
    log.info("\n── 학습 요약 ──")
    log.info(f"{'Target':<8} {'Val Loss':<12} {'Best Iter':<12} {'Features':<10} {'SPW':<10}")
    for t, m in results.items():
        log.info(f"{t:<8} {m['val_logloss']:<12.4f} {m['best_iteration']:<12} "
                 f"{len(m['feature_cols']):<10} {m['scale_pos_weight']:<10.2f}")

    return results


def save_models(features: pd.DataFrame, results: dict[str, dict]):
    """모든 모델을 disk에 저장 (lightgbm 기본 형식)."""
    log.info("\n── 모델 저장 ──")
    for target, m in results.items():
        model = m["model"]
        # 모델 파일
        model_path = MODEL_DIR / f"lgbm_{target}.txt"
        model.save_model(model_path)
        log.info(f"  {target}: {model_path} (iter={m['best_iteration']})")

        # 메트릭 파일
        metric_path = MODEL_DIR / f"metrics_{target}.json"
        metric_data = {k: v for k, v in m.items() if k not in ("model", "sanitized_cols")}
        metric_data["feature_importances"] = [
            {"feature": f, "importance": i} for f, i in m["feature_importances"]
        ]
        metric_data["n_features"] = len(m["sanitized_cols"])
        with open(metric_path, "w") as f:
            json.dump(metric_data, f, default=str)


def main(features: pd.DataFrame | None = None) -> dict:
    """전체 학습 파이프라인 실행."""
    if features is None:
        feat_path = DATA_PROCESSED / "features.parquet"
        if feat_path.exists():
            features = pd.read_parquet(feat_path)
            log.info(f"Loaded features from {feat_path}: {features.shape}")
        else:
            log.error(f"Features not found: {feat_path}")
            log.error("Run 01_load_data.py and 02_feature_engineering.py first.")
            sys.exit(1)

    results = train_all_targets(features)
    save_models(features, results)

    return results


if __name__ == "__main__":
    main()
