"""
03_model_training_improved.py — Clean model training (memory-efficient)

개선:
1. Target leakage 제거: 각 타겟 학습 시 다른 target열(feature)에서 제외
2. Model comparison: LightGBM + XGBoost + CatBoost (순차적 훈련, 메모리 절약)
3. Calibration: Platt scaling
4. Subject-based split (GroupKFold, n_splits=5)
5. 메모리 절감: fold별 모델 즉시 삭제, top-n feature 선택

전략: 105 models (7 targets × 5 folds × 3 models)를 한 번에 만드는 대신
각 target을 순차적으로 학습하고 fold별 결과를 저장만 한다.
"""

import json
import gc
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool as CBPool

from config import MODEL_DIR, RANDOM_SEED, TARGETS, DATA_PROCESSED

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def get_clean_feature_cols(features_df, target):
    meta_cols = {"subject_id", "lifelog_date", "sleep_date", "date"}
    leakage_cols = {t for t in TARGETS if t != target}
    exclude = meta_cols | leakage_cols
    return [c for c in features_df.columns
            if c not in exclude
            and features_df[c].dtype in [np.float64, np.int64, float, int, bool]]


def sanitize_name(name):
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)


def make_lgb_params(y_train, n_iters=300):
    n_pos = max((y_train == 1).sum(), 1)
    n_neg = (y_train == 0).sum()
    return {
        "objective": "binary", "metric": "binary_logloss", "verbose": -1,
        "n_jobs": -1, "random_state": RANDOM_SEED, "force_row_wise": True,
        "num_leaves": 31, "max_depth": 5, "learning_rate": 0.05,
        "n_estimators": n_iters, "subsample": 0.7, "colsample_bytree": 0.7,
        "min_child_samples": 5, "reg_alpha": 1.0, "reg_lambda": 5.0,
        "scale_pos_weight": n_neg / n_pos, "early_stopping_round": 50,
    }


def make_xgb_params(y_train, n_iters=300):
    n_pos = max((y_train == 1).sum(), 1)
    n_neg = (y_train == 0).sum()
    return {
        "objective": "binary:logistic", "eval_metric": "logloss",
        "random_state": RANDOM_SEED, "n_estimators": n_iters,
        "learning_rate": 0.05, "max_depth": 5, "colsample_bytree": 0.7,
        "subsample": 0.7, "reg_alpha": 1.0, "reg_lambda": 5.0,
        "min_child_weight": 3, "scale_pos_weight": n_neg / n_pos,
        "early_stopping_rounds": 50,
    }


def make_cb_params(y_train, n_iters=300):
    n_pos = max((y_train == 1).sum(), 1)
    n_neg = (y_train == 0).sum()
    return {
        "objective": "Logloss", "eval_metric": "Logloss",
        "random_seed": RANDOM_SEED, "n_estimators": n_iters,
        "learning_rate": 0.05, "max_depth": 5, "colsample_bylevel": 0.7,
        "subsample": 0.7, "l2_leaf_reg": 5.0, "min_child_samples": 5,
        "scale_pos_weight": n_neg / n_pos, "early_stopping_rounds": 50,
        "verbose": 100,  # CatBoost: show every 100 iters
    }


def gc_clean():
    gc.collect()


def train_lgb(X_tr, y_tr, X_va, y_va, feat_names, params):
    train_set = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_names)
    val_set = lgb.Dataset(X_va, label=y_va, feature_name=feat_names, reference=train_set)
    model = lgb.train(params, train_set, num_boost_round=params["n_estimators"],
                      valid_sets=[val_set],
                      callbacks=[lgb.early_stopping(stopping_rounds=params["early_stopping_round"]),
                                 lgb.log_evaluation(period=0)])
    pred = model.predict(X_va)
    loss = log_loss(y_va, pred, labels=[0, 1])
    imp = model.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feat_names, imp), key=lambda x: -x[1])
    return model, loss, feat_imp, model.best_iteration


def train_xgb(X_tr, y_tr, X_va, y_va, feat_names, params):
    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feat_names)
    dval = xgb.DMatrix(X_va, label=y_va, feature_names=feat_names)
    watchlist = [(dtrain, "train"), (dval, "eval")]
    xgb_p = {k: v for k, v in params.items() if k not in ("n_estimators", "early_stopping_rounds")}
    model = xgb.train(xgb_p, dtrain, num_boost_round=params["n_estimators"],
                      evals=watchlist,
                      callbacks=[xgb.callback.EarlyStopping(rounds=params["early_stopping_rounds"])])
    pred = model.predict(dval)
    loss = log_loss(y_va, pred, labels=[0, 1])
    imp_map = model.get_score(importance_type="gain")
    feat_imp = sorted([(f, imp_map.get(f, 0.0)) for f in feat_names], key=lambda x: -x[1])
    return model, loss, feat_imp, None


def train_cb(X_tr, y_tr, X_va, y_va, feat_names, params):
    train_pool = CBPool(X_tr, label=y_tr, feature_names=feat_names)
    val_pool = CBPool(X_va, label=y_va, feature_names=feat_names)
    cb = CatBoostClassifier(**params)
    cb.fit(X_tr, y_tr, eval_set=val_pool, use_best_model=True)
    pred = cb.predict_proba(X_va)[:, 1]
    loss = log_loss(y_va, pred, labels=[0, 1])
    importances = cb.get_feature_importance()
    feat_imp = sorted(zip(feat_names, importances), key=lambda x: -x[1])
    return cb, loss, feat_imp, getattr(cb, 'best_iteration_', None)


def calibrate_with_lr(raw_preds, y_true):
    from sklearn.linear_model import LogisticRegression
    cal = LogisticRegression(C=1.0, random_state=RANDOM_SEED)
    cal.fit(raw_preds.reshape(-1, 1), y_true)
    logits = cal.coef_[0, 0] * raw_preds + cal.intercept_[0]
    return 1.0 / (1.0 + np.exp(-logits)), cal


def train_all_targets(features):
    log.info("=" * 70)
    log.info("03_model_training_improved.py — Clean Model Training")
    log.info("=" * 70)

    gkf = GroupKFold(n_splits=5)
    all_results = {}

    for target in TARGETS:
        log.info(f"\n{'='*70}")
        log.info(f"── [{target}] 학습 시작 ──")

        y = features[target].dropna()
        pos = int((y == 1).sum())
        neg = int((y == 0).sum())
        log.info(f"    dist: pos={pos}, neg={neg}")

        clean_cols = get_clean_feature_cols(features, target)
        log.info(f"    Clean features: {len(clean_cols)} (removed: Q2,Q3,S1-S4)")

        valid_mask = features[target].notna()
        X = features[clean_cols].fillna(0).values
        y_arr = features[target].values.astype(int)
        sanitized_cols = [sanitize_name(c) for c in clean_cols]

        # ── CV Comparison (순차적, 모델 즉시 해제) ──
        fold_scores = {"LightGBM": [], "XGBoost": [], "CatBoost": []}

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_arr, features["subject_id"])):
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = y_arr[train_idx], y_arr[val_idx]

            # LightGBM
            lgb_p = make_lgb_params(y_tr)
            _, lgb_l, _, _ = train_lgb(X_tr, y_tr, X_va, y_va, sanitized_cols, lgb_p)
            fold_scores["LightGBM"].append(lgb_l)
            gc_clean()

            # XGBoost
            xgb_p = make_xgb_params(y_tr)
            _, xgb_l, _, _ = train_xgb(X_tr, y_tr, X_va, y_va, clean_cols, xgb_p)
            fold_scores["XGBoost"].append(xgb_l)
            gc_clean()

            # CatBoost
            cb_p = make_cb_params(y_tr)
            _, cb_l, _, _ = train_cb(X_tr, y_tr, X_va, y_va, clean_cols, cb_p)
            fold_scores["CatBoost"].append(cb_l)
            gc_clean()

        # Summary
        for m in fold_scores:
            fold_scores[m] = np.mean(fold_scores[m])

        mean_scores = {m: fold_scores[m] for m in fold_scores}
        best_model = min(mean_scores, key=mean_scores.get)

        log.info(f"    CV Summary:")
        for m in ["LightGBM", "XGBoost", "CatBoost"]:
            log.info(f"      {m:<12} {fold_scores[m]:.4f}")
        log.info(f"    🏆 Best: {best_model} ({mean_scores[best_model]:.4f})")

        # ── Final: train best model on all data ──
        log.info(f"\n    Final: {best_model} on all {len(X)} rows...")

        if best_model == "LightGBM":
            f_p = make_lgb_params(y_arr, n_iters=500)
            final_model, final_loss, final_imp, _ = train_lgb(X, y_arr, X, y_arr, sanitized_cols, f_p)
        elif best_model == "XGBoost":
            f_p = make_xgb_params(y_arr, n_iters=500)
            final_model, final_loss, final_imp, _ = train_xgb(X, y_arr, X, y_arr, clean_cols, f_p)
        else:
            f_p = make_cb_params(y_arr, n_iters=500)
            final_model, final_loss, final_imp, _ = train_cb(X, y_arr, X, y_arr, clean_cols, f_p)

        log.info(f"    Train logloss: {final_loss:.4f}")

        # ── Calibration (LSO) ──
        log.info(f"    Calibration (LSO)...")
        cal_raw = []
        cal_true = []

        for train_idx, val_idx in gkf.split(X, y_arr, features["subject_id"]):
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = y_arr[train_idx], y_arr[val_idx]

            if best_model == "LightGBM":
                fp = make_lgb_params(y_tr, n_iters=300)
                fm, _, _, _ = train_lgb(X_tr, y_tr, X_va, y_va, sanitized_cols, fp)
                raw = fm.predict(X_va)
            elif best_model == "XGBoost":
                fp = make_xgb_params(y_tr, n_iters=300)
                fm, _, _, _ = train_xgb(X_tr, y_tr, X_va, y_va, clean_cols, fp)
                raw = fm.predict(xgb.DMatrix(X_va, feature_names=clean_cols))
            else:
                fp = make_cb_params(y_tr, n_iters=300)
                fm, _, _, _ = train_cb(X_tr, y_tr, X_va, y_va, clean_cols, fp)
                raw = fm.predict_proba(X_va)[:, 1]
            gc_clean()

            cal_raw.extend(raw)
            cal_true.extend(y_va)

        cal_raw = np.array(cal_raw)
        cal_true = np.array(cal_true)

        calibrated_preds, calibrator = calibrate_with_lr(cal_raw, cal_true)
        cal_logloss = log_loss(cal_true, calibrated_preds, labels=[0, 1])

        # Full-data calibration
        if best_model == "LightGBM":
            all_raw = final_model.predict(X)
        elif best_model == "XGBoost":
            all_raw = final_model.predict(xgb.DMatrix(X, feature_names=clean_cols))
        else:
            all_raw = final_model.predict_proba(X)[:, 1]

        all_calibrated, cal_all = calibrate_with_lr(all_raw, y_arr)
        all_cal_loss = log_loss(y_arr, all_calibrated, labels=[0, 1])

        log.info(f"    Calibration: val_logloss={cal_logloss:.4f}, full_logloss={all_cal_loss:.4f}")

        # Top features (gain > 0)
        feat_importance_dict = {f: i for f, i in final_imp}
        top_feats = [(f, i) for f, i in final_imp if i > 0]

        all_results[target] = {
            "best_model": best_model,
            "cv_scores": {m: float(fold_scores[m]) for m in ["LightGBM", "XGBoost", "CatBoost"]},
            "final_model": final_model,
            "final_model_type": best_model,
            "train_loss": float(final_loss),
            "calibrated_val_loss": float(cal_logloss),
            "calibrated_full_loss": float(all_cal_loss),
            "calibrator": cal_all,
            "feature_names": clean_cols,
            "sanitized_cols": sanitized_cols,
            "feature_importances": final_imp,
            "top_features": [f for f, i in top_feats],
            "n_features": len(clean_cols),
            "n_top_features": len(top_feats),
        }

        # Delete non-final models, gc
        del final_imp
        gc_clean()

        log.info(f"    ✅ {target} complete!")

    return all_results


def save_improved_models(results):
    log.info("\n" + "=" * 70)
    log.info("Save models")
    log.info("=" * 70)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    feat_cols = results["Q1"]["feature_names"]
    with open(MODEL_DIR / "feature_cols_clean.txt", "w") as f:
        for c in feat_cols:
            f.write(c + "\n")
    log.info(f"  feature_cols_clean.txt: {len(feat_cols)} features")

    for target in TARGETS:
        r = results[target]
        model = r["final_model"]
        mtype = r["final_model_type"]

        if mtype == "LightGBM":
            path = MODEL_DIR / f"clean_lgbm_{target}.txt"
            model.save_model(str(path))
        elif mtype == "XGBoost":
            path = MODEL_DIR / f"clean_xgb_{target}.json"
            model.save(str(path))
        else:
            path = MODEL_DIR / f"clean_cb_{target}.cbm"
            model.save_model(str(path))
        log.info(f"  {target}: {path.name} ({mtype})")

        metric_data = {
            "target": target,
            "best_model": mtype,
            "n_features": r["n_features"],
            "n_top_features": r["n_top_features"],
            "train_loss": r["train_loss"],
            "calibrated_val_loss": r["calibrated_val_loss"],
            "calibrated_full_loss": r["calibrated_full_loss"],
            "calibration": {
                "coeff": float(r["calibrator"].coef_[0, 0]),
                "intercept": float(r["calibrator"].intercept_[0]),
            },
            "cv_scores": r["cv_scores"],
            "feature_importances": [
                {"feature": f, "importance": float(i)}
                for f, i in r["feature_importances"][:20]
            ],
            "top_features": r["top_features"][:15],
        }

        path = MODEL_DIR / f"clean_metrics_{target}.json"
        with open(path, "w") as f:
            json.dump(metric_data, f, indent=2, default=str)
        log.info(f"  {target}: metrics -> {path.name}")

    # Summary
    log.info("\n" + "=" * 70)
    log.info("Summary")
    log.info("=" * 70)
    log.info(f"{'Target':<8} {'Model':<12} {'Train':<10} {'Cal-Val':<10} {'Cal-Full':<10} {'Feat':<6}")
    for t in TARGETS:
        r = results[t]
        log.info(f"{t:<8} {r['final_model_type']:<12} {r['train_loss']:<10.4f} "
                 f"{r['calibrated_val_loss']:<10.4f} {r['calibrated_full_loss']:<10.4f} "
                 f"{r['n_features']:<6}")


def main(features=None):
    if features is None:
        feat_path = DATA_PROCESSED / "features.parquet"
        if feat_path.exists():
            features = pd.read_parquet(feat_path)
            log.info(f"Loaded: {feat_path} ({features.shape})")
        else:
            log.error(f"Not found: {feat_path}")
            sys.exit(1)

    results = train_all_targets(features)
    save_improved_models(results)
    return results


if __name__ == "__main__":
    main()
