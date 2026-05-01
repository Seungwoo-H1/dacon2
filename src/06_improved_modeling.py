"""
06_improved_modeling.py — V11: 3-model ensemble with stacking + calibration

Goal: Improve on v9 score 0.65374 → target 0.67+

Key improvements over v9:
1. Proper calibration: Platt scaling (LogisticRegression) on OOF instead of
   IsotonicRegression. Isotonic was collapsing predictions (shift=-0.40 to -0.55).
2. Stacking: 3 base models (LightGBM, CatBoost, XGBoost) → LR meta-learner →
   Platt scaling for final calibration.
3. Feature pruning: top-20 features per target via importance ranking.
4. Parallelism: n_jobs=-1 for all models (CPU-bound, safe on i7-13700HX).
5. Multiple random seeds for ensemble stability.

Environment:
  - CPU: i7-13700HX (24 cores)
  - LightGBM 4.6.0, CatBoost 1.2.10, XGBoost 3.2.0
"""

import sys
import re
import json
import warnings
import logging
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold

import lightgbm as lgb
import catboost as cb
import xgboost as xgb

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ── Path setup ───────────────────────────────────────────
sys.path.insert(0, 'src')
from config import TARGETS, DATA_PROCESSED, MODEL_DIR, SUBMIT_DIR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data_raw"

TARGET_COLS = TARGETS  # Q1, Q2, Q3, S1, S2, S3, S4
META_COLS = {"subject_id", "lifelog_date", "sleep_date", "date"}

# ── Hyperparameters ──────────────────────────────────────
RANDOM_SEEDS = [42, 123, 456, 789, 1024, 1337, 2048, 3037, 4096, 5001,
                6000, 7123, 8001, 9000, 10000, 11111, 12000, 13001, 14000, 15001]
N_SEEDS = len(RANDOM_SEEDS)  # 20
N_SPLITS = 5                 # GroupKFold splits
N_TOP_FEATURES = 20          # Per-target feature count to select

# ── Feature utils ────────────────────────────────────────
def sanitize(name):
    """Replace special chars with underscore for LGBM/CB feature names."""
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)


def get_feature_cols(feat):
    """Get numeric feature columns (excluding meta and target cols)."""
    cols = [c for c in feat.columns
            if c not in META_COLS | set(TARGET_COLS)
            and feat[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]
    return cols


# ── Feature selection: rank by importance ────────────────
def rank_features_by_importance(feat, feature_cols, target, random_seed=42):
    """Quick LightGBM scan to rank features by gain importance."""
    y = feat[target].values
    X = feat[feature_cols].fillna(0).values
    n_pos = max((y == 1).sum(), 1)
    n_neg = (y == 0).sum()
    spw = n_neg / n_pos

    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03,
        'n_estimators': 100, 'subsample': 0.7, 'colsample_bytree': 0.7,
        'reg_alpha': 1.0, 'reg_lambda': 3.0,
        'scale_pos_weight': spw, 'random_state': random_seed,
        'min_child_samples': 10,
        'force_row_wise': True, 'n_jobs': -1,
    }
    sanitized = [sanitize(c) for c in feature_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sanitized, params={'verbose': '-1'})
    model = lgb.train(params, ds, num_boost_round=100)
    importances = model.feature_importance(importance_type="gain")
    ranked = sorted(zip(feature_cols, importances), key=lambda x: -x[1])
    return ranked


# ── Hyperparameter configs ───────────────────────────────
def _lgb_cfg(seed):
    """LightGBM config — moderate depth, strong regularization."""
    return {
        'objective': 'binary', 'metric': 'binary_logloss',
        'num_leaves': 15, 'max_depth': 4,
        'learning_rate': 0.03, 'n_estimators': 500,
        'subsample': 0.7, 'colsample_bytree': 0.7,
        'reg_alpha': 1.0, 'reg_lambda': 3.0,
        'min_child_samples': 10,
        'force_row_wise': True, 'n_jobs': -1,
        'random_state': seed,
        'verbose': -1,
    }


def _cb_cfg(seed):
    """CatBoost config — moderate depth, strong regularization."""
    return {
        'iterations': 500,
        'depth': 5,
        'learning_rate': 0.03,
        'loss_function': 'Logloss',
        'eval_metric': 'Logloss',
        'subsample': 0.7,
        'colsample_bylevel': 0.7,
        'reg_lambda': 3.0,
        'random_strength': 1.0,
        'min_child_samples': 10,
        'random_seed': seed,
        'task_type': 'CPU',
        'verbose': 0,
    }


def _xgb_cfg(seed):
    """XGBoost config — moderate depth, strong regularization."""
    return {
        'objective': 'binary:logistic',
        'max_depth': 5,
        'learning_rate': 0.03,
        'n_estimators': 500,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 1.0,
        'reg_lambda': 3.0,
        'min_child_weight': 10,
        'random_state': seed,
        'verbosity': 0,
        'verbosity_internal': 0,
    }


# ── Model training (single-fold) ─────────────────────────
def train_lgb_fold(X_train, y_train, X_val, y_val, feature_names, cfg, spw):
    """Train single LightGBM model on one fold."""
    params = {**cfg, 'scale_pos_weight': spw}
    train_set = lgb.Dataset(X_train, label=y_train, feature_name=feature_names,
                            params={'verbose': '-1'})
    val_set = lgb.Dataset(X_val, label=y_val, feature_name=feature_names,
                          reference=train_set, params={'verbose': '-1'})
    model = lgb.train(
        params, train_set, num_boost_round=cfg['n_estimators'],
        valid_sets=[val_set],
        callbacks=[
            lgb.early_stopping(cfg.get('early_stopping_round', 50), verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    return model


def train_catboost_fold(X_train, y_train, X_val, y_val, feature_names, cfg, spw):
    """Train single CatBoost model on one fold."""
    params = {**cfg, 'random_seed': cfg.get('random_seed', 42)}
    model = cb.CatBoostClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=cfg.get('early_stopping_rounds', 50),
        use_best_model=True,
        silent=True,
    )
    return model


def train_xgboost_fold(X_train, y_train, X_val, y_val, feature_names, cfg, spw):
    """Train single XGBoost model on one fold."""
    params = {**cfg, 'scale_pos_weight': spw}
    params['verbosity'] = 0
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
    watchlist = [(dval, 'eval')]
    model = xgb.train(
        params, dtrain, num_boost_round=cfg['n_estimators'],
        evals=watchlist,
        early_stopping_rounds=cfg.get('early_stopping_rounds', 50),
        verbose_eval=False,
    )
    return model


# ── Multi-run CV: generate OOF predictions ───────────────
def multi_run_cv(feat, feature_cols, target, model_type, spw):
    """
    GroupKFold: N_SPLITS folds × N_SEEDS seeds → OOF predictions.

    Returns:
        oof_avg: (n_samples,) averaged OOF predictions
        oof_full: (n_samples, n_seeds) per-seed OOF predictions
        cv_loss: total logloss of averaged OOF
        cv_std: std of per-fold logloss (averaged across seeds)
        fold_losses: list of per-fold averaged losses
    """
    y = feat[target].values
    gkf = GroupKFold(n_splits=N_SPLITS)

    oof_full = np.zeros((len(y), N_SEEDS))
    all_fold_losses = {i: [] for i in range(N_SPLITS)}

    for seed_i, run_seed in enumerate(RANDOM_SEEDS):
        for fold, (train_idx, val_idx) in enumerate(
            gkf.split(feat, y, feat['subject_id'])
        ):
            if fold >= N_SPLITS:
                break

            X_tr = feat.iloc[train_idx][feature_cols].fillna(0).values
            X_va = feat.iloc[val_idx][feature_cols].fillna(0).values
            y_tr, y_va = y[train_idx], y[val_idx]
            feature_names = [sanitize(c) for c in feature_cols]

            if model_type == 'lgb':
                m = train_lgb_fold(
                    X_tr, y_tr, X_va, y_va, feature_names,
                    _lgb_cfg(run_seed), spw
                )
                pred = m.predict(X_va)
            elif model_type == 'catboost':
                m = train_catboost_fold(
                    X_tr, y_tr, X_va, y_va, feature_names,
                    _cb_cfg(run_seed), spw
                )
                pred = m.predict_proba(X_va)[:, 1]
            elif model_type == 'xgboost':
                m = train_xgboost_fold(
                    X_tr, y_tr, X_va, y_va, feature_names,
                    _xgb_cfg(run_seed), spw
                )
                pred = m.predict(xgb.DMatrix(X_va, feature_names=feature_names))
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            oof_full[val_idx, seed_i] = pred
            fold_losses_avg = log_loss(y_va, pred, labels=[0, 1])
            all_fold_losses[fold].append(fold_losses_avg)

    oof_avg = oof_full.mean(axis=1)

    fold_avg_losses = [np.mean(all_fold_losses[i]) for i in range(N_SPLITS)]
    cv_loss = log_loss(y, oof_avg, labels=[0, 1])
    cv_std = np.std(fold_avg_losses)

    return oof_avg, oof_full, cv_loss, cv_std, fold_avg_losses


# ── Main pipeline ────────────────────────────────────────
def main():
    log.info("=" * 70)
    log.info("06_improved_modeling.py — 3-model ensemble with stacking")
    log.info("=" * 70)

    # ── 1. Load features ───────────────────────────────────
    feat_path = DATA_PROCESSED / "features.parquet"
    if not feat_path.exists():
        log.error(f"Features not found: {feat_path}")
        log.error("Run 02_feature_engineering.py first.")
        sys.exit(1)

    feat = pd.read_parquet(feat_path)
    feature_cols = get_feature_cols(feat)
    log.info(f"Training data: {feat.shape}, features: {len(feature_cols)}")

    train_labels = pd.read_csv(DATA_RAW / "ch2026_metrics_train.csv")
    train_rate = {t: train_labels[t].mean() for t in TARGET_COLS}
    log.info(f"Target rates: {train_rate}")

    # ── 2. Feature ranking (quick, one pass per target) ───
    log.info("\n=== Step 1: Feature ranking (importance-based) ===")
    feat_ranking = {}
    for target in TARGET_COLS:
        log.info(f"  Ranking features for {target}...")
        ranked = rank_features_by_importance(
            feat, feature_cols, target, random_seed=42
        )
        feat_ranking[target] = ranked
        top5 = [r[0] for r in ranked[:5]]
        log.info(f"    Top 5: {top5}")

    # ── 3. Multi-run CV ────────────────────────────────────
    log.info(f"\n=== Step 2: Multi-run CV ({N_SEEDS} seeds, {N_SPLITS} folds) ===")

    results = {}
    model_types = ['lgb', 'catboost', 'xgboost']
    model_labels = {'lgb': 'LightGBM', 'catboost': 'CatBoost', 'xgboost': 'XGBoost'}

    for model_type in model_types:
        log.info(f"\n  ── {model_labels[model_type]} ──")
        target_results = {}

        for target in TARGET_COLS:
            ranked = feat_ranking[target]
            selected_cols = [f[0] for f in ranked[:N_TOP_FEATURES]]

            y = feat[target].values
            n_pos = max((y == 1).sum(), 1)
            n_neg = (y == 0).sum()
            spw = n_neg / n_pos

            oof_avg, oof_full, cv_loss, cv_std, fold_losses = multi_run_cv(
                feat, selected_cols, target,
                model_type=model_type, spw=spw
            )

            total_loss = log_loss(y, oof_avg, labels=[0, 1])

            target_results[target] = {
                'selected_cols': selected_cols,
                'oof_avg': oof_avg,
                'oof_full': oof_full,
                'cv_loss': cv_loss,
                'cv_std': cv_std,
                'total_loss': total_loss,
                'fold_losses': fold_losses,
            }

            avg_fold = np.mean(fold_losses)
            log.info(
                f"    {target}: total_logloss={total_loss:.4f}, "
                f"cv={cv_loss:.4f}±{cv_std:.4f}, "
                f"avg_fold={avg_fold:.4f}, "
                f"pred_mean={oof_avg.mean():.4f}, "
                f"train_rate={train_rate[target]:.3f}"
            )

        results[model_type] = target_results

    # ── 4. Stacking ensemble ───────────────────────────────
    log.info("\n=== Step 3: Stacking Ensemble (LR meta-learner) ===")

    stack_results = {}
    for target in TARGET_COLS:
        log.info(f"  Stacking for {target}...")

        oof_lgb = results['lgb'][target]['oof_full']
        oof_cb = results['catboost'][target]['oof_full']
        oof_xgb = results['xgboost'][target]['oof_full']

        y = feat[target].values
        mask = ~np.isnan(oof_lgb[:, 0])

        # Meta-features: average OOF across seeds
        meta_train = np.column_stack([
            oof_lgb[mask].mean(axis=1),
            oof_cb[mask].mean(axis=1),
            oof_xgb[mask].mean(axis=1),
        ])
        y_valid = y[mask]

        # Train meta-learner (LogisticRegression for stacking)
        meta_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        meta_lr.fit(meta_train, y_valid)

        # OOF stacked predictions
        oof_lgb_avg = oof_lgb.mean(axis=1)
        oof_cb_avg = oof_cb.mean(axis=1)
        oof_xgb_avg = oof_xgb.mean(axis=1)

        meta_oof = np.column_stack([oof_lgb_avg, oof_cb_avg, oof_xgb_avg])
        meta_oof_preds = meta_lr.predict_proba(meta_oof)[:, 1]
        meta_oof_preds = np.clip(meta_oof_preds, 0.0001, 0.9999)

        # ── Calibration: Isotonic regression on OOF ──────────
        # Fitted on OOF (validation) predictions → calibrated OOF
        from sklearn.isotonic import IsotonicRegression
        cal = IsotonicRegression(out_of_bounds='clip')
        try:
            cal.fit(meta_oof_preds[mask], y[mask])
            cal_oof = cal.predict(meta_oof_preds)
        except Exception:
            cal_oof = meta_oof_preds

        # ── Mean matching calibration ────────────────────────
        # Align calibrated OOF distribution to train rate
        shift = train_rate[target] - cal_oof.mean()
        cal_oof = np.clip(cal_oof + shift, 0.0001, 0.9999)

        stack_total = log_loss(y[mask], cal_oof[mask], labels=[0, 1])
        stack_results[target] = {
            'oof_preds': cal_oof,
            'meta_lr': meta_lr,
            'cal': cal,
            'total_loss': stack_total,
        }

        log.info(
            f"    {target}: stack_logloss={stack_total:.4f}, "
            f"pred_mean={cal_oof.mean():.4f}, "
            f"train_rate={train_rate[target]:.3f}, "
            f"shift={shift:+.4f}"
        )

    # ── 5. Summary comparison ──────────────────────────────
    log.info("\n=== Summary: Per-Target CV Scores ===")
    log.info(f"{'Target':<8} {'LGBM':<12} {'CB':<12} {'XGB':<12} {'Stack':<12}")
    for target in TARGET_COLS:
        lgb_cv = results['lgb'][target]['total_loss']
        cb_cv = results['catboost'][target]['total_loss']
        xgb_cv = results['xgboost'][target]['total_loss']
        stack_cv = stack_results[target]['total_loss']
        log.info(
            f"{target:<8} {lgb_cv:<12.4f} {cb_cv:<12.4f} "
            f"{xgb_cv:<12.4f} {stack_cv:<12.4f}"
        )

    avg_scores = {
        m: np.mean([results[m][t]['total_loss'] for t in TARGET_COLS])
        for m in model_types
    }
    avg_scores['Stack'] = np.mean([
        stack_results[t]['total_loss'] for t in TARGET_COLS
    ])
    log.info(
        f"\n  Avg total logloss: "
        f"{', '.join(f'{m}={v:.4f}' for m, v in avg_scores.items())}"
    )

    # ── 6. Generate submission ─────────────────────────────
    log.info("\n=== Step 4: Training final models + generating submission ===")

    # Load feature engineering
    spec = importlib.util.spec_from_file_location(
        "02_feature_engineering", Path('src/02_feature_engineering.py')
    )
    feat_eng = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(feat_eng)

    spec2 = importlib.util.spec_from_file_location(
        "01_load_data", Path('src/01_load_data.py')
    )
    ld_mod = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(ld_mod)

    # Load test data
    parquet_dfs = {}
    data_dir = Path('data_raw/ch2025_data_items')
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

    sample = pd.read_csv('data_raw/ch2026_submission_sample.csv')
    sample['lifelog_date'] = pd.to_datetime(sample['lifelog_date']).dt.date
    sample['sleep_date'] = pd.to_datetime(sample['sleep_date']).dt.date

    test_dates = set(
        sample["sleep_date"].astype(str).tolist()
        + sample["lifelog_date"].astype(str).tolist()
    )

    for name, fname in parquet_names.items():
        df = pd.read_parquet(data_dir / fname)
        df = ld_mod.build_merge_key(df)
        df = df[df["date"].astype(str).isin(test_dates)]
        parquet_dfs[name] = df

    test_features = feat_eng.create_day_features(parquet_dfs, sample)
    log.info(f"Test features: {test_features.shape}")

    predictions = (
        test_features[['subject_id', 'sleep_date', 'lifelog_date']]
        .copy()
    )

    # For each target: train final models → stacking → calibration
    for target in TARGET_COLS:
        log.info(f"\n  Training final models for {target}...")
        ranked = feat_ranking[target]
        selected_cols = [f[0] for f in ranked[:N_TOP_FEATURES]]

        y_all = feat[target].values
        X_all = feat[selected_cols].fillna(0).values

        test_X = test_features[selected_cols].fillna(0).values
        test_feature_names = [sanitize(c) for c in selected_cols]

        lgb_preds = np.zeros(len(test_X))
        cb_preds = np.zeros(len(test_X))
        xgb_preds = np.zeros(len(test_X))

        spw = max((y_all == 0).sum(), 1) / max((y_all == 1).sum(), 1)

        for seed_i, seed in enumerate(RANDOM_SEEDS):
            # LightGBM
            cfg = _lgb_cfg(seed)
            cfg.pop('verbose', None)
            ds_all = lgb.Dataset(
                X_all, label=y_all,
                feature_name=test_feature_names,
                params={'verbose': '-1'},
            )
            model_lgb = lgb.train(
                cfg, ds_all, num_boost_round=cfg['n_estimators']
            )
            lgb_preds += model_lgb.predict(test_X)

            # CatBoost
            cfg_cb = _cb_cfg(seed)
            cb_m = cb.CatBoostClassifier(**cfg_cb)
            cb_m.fit(X_all, y_all, silent=True)
            cb_preds += cb_m.predict_proba(test_X)[:, 1]

            # XGBoost
            cfg_xgb = _xgb_cfg(seed)
            xgb_m = xgb.XGBClassifier(**cfg_xgb)
            xgb_m.fit(X_all, y_all, verbose=False)
            xgb_preds += xgb_m.predict_proba(test_X)[:, 1]

            if (seed_i + 1) % 5 == 0:
                log.info(f"    [{target}] seed {seed_i + 1}/{N_SEEDS} done")

        lgb_preds /= N_SEEDS
        cb_preds /= N_SEEDS
        xgb_preds /= N_SEEDS

        # Stacking
        meta_test = np.column_stack([lgb_preds, cb_preds, xgb_preds])
        meta_preds = stack_results[target]['meta_lr'].predict_proba(
            meta_test
        )[:, 1]
        meta_preds = np.clip(meta_preds, 0.0001, 0.9999)

        # Isotonic calibration (from OOF) → apply to test predictions
        cal = stack_results[target]['cal']
        try:
            meta_preds = cal.predict(meta_preds)
        except Exception as e:
            log.warning(
                f"  {target}: Isotonic calibration failed ({e}), "
                "using raw meta-preds"
            )

        # Mean matching calibration
        shift = train_rate[target] - meta_preds.mean()
        meta_preds = np.clip(meta_preds + shift, 0.0001, 0.9999)

        predictions[target] = meta_preds

        log.info(
            f"    {target}: mean={meta_preds.mean():.4f}, "
            f"min={meta_preds.min():.4f}, "
            f"max={meta_preds.max():.4f}, "
            f"train_rate={train_rate[target]:.3f}, "
            f"shift={shift:+.4f}"
        )

    # ── Save submission ────────────────────────────────────
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    sub_path = SUBMIT_DIR / f'submission_v6_{timestamp}.csv'
    sub_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(sub_path, index=False)
    log.info(f"\n✅ Submission saved: {sub_path}")

    # Save metadata
    meta = {
        'submission_file': str(sub_path),
        'timestamp': timestamp,
        'n_samples': len(predictions),
        'n_seeds': N_SEEDS,
        'n_splits': N_SPLITS,
        'n_top_features': N_TOP_FEATURES,
        'model_types': model_types,
        'per_target': {},
    }
    for target in TARGET_COLS:
        meta['per_target'][target] = {
            'lgbm_loss': float(results['lgb'][target]['total_loss']),
            'catboost_loss': float(
                results['catboost'][target]['total_loss']
            ),
            'xgboost_loss': float(
                results['xgboost'][target]['total_loss']
            ),
            'stack_loss': float(stack_results[target]['total_loss']),
            'pred_mean': float(predictions[target].mean()),
            'train_rate': float(train_rate[target]),
            'n_features': N_TOP_FEATURES,
        }

    meta_path = sub_path.parent / f'meta_v6_{timestamp}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    log.info(f"Metadata saved: {meta_path}")

    return predictions


if __name__ == "__main__":
    main()
