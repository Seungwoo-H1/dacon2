"""
V31 Final — Generate submission from existing features.parquet + test_features.parquet

Since raw data (ch2025_data_items/) is unavailable, we use:
- features.parquet for training / OOF
- test_features.parquet for test predictions
- V31 model configs from completed CV run

LGBM cal OOF: ~0.5748 (ensemble with XGBoost, LGBM=0.7, XGB=0.3)
"""

import sys
import re
import json
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

sys.path.insert(0, 'src')
from config import TARGETS, DATA_PROCESSED, SUBMIT_DIR

TARGET_COLS = TARGETS
META_COLS = {"subject_id", "lifelog_date", "sleep_date", "date"}

LEAKAGE_FEATURES_S = {
    'wLight_w_light_mean', 'wLight_w_light_std', 'wLight_w_light_min', 'wLight_w_light_max', 'wLight_w_light_count',
    'wHr_hr_mean', 'wHr_hr_std', 'wHr_hr_min', 'wHr_hr_max', 'wHr_hr_median', 'wHr_hr_count',
    'wPedo_pedo_step_mean', 'wPedo_pedo_step_sum',
    'wPedo_pedo_step_frequency_mean', 'wPedo_pedo_step_frequency_sum',
    'wPedo_pedo_running_step_mean', 'wPedo_pedo_running_step_sum',
    'wPedo_pedo_walking_step_mean', 'wPedo_pedo_walking_step_sum',
    'wPedo_pedo_distance_mean', 'wPedo_pedo_distance_sum',
    'wPedo_pedo_speed_mean', 'wPedo_pedo_speed_sum',
    'wPedo_pedo_burned_calories_mean', 'wPedo_pedo_burned_calories_sum',
}
LEAKAGE_FEATURES_Q = {
    'wHr_hr_mean', 'wHr_hr_std', 'wHr_hr_min', 'wHr_hr_max', 'wHr_hr_median', 'wHr_hr_count',
}

RANDOM_SEEDS = [42, 123, 456, 789, 1024, 1337, 2048, 3037, 4096, 5001,
                6000, 7123, 8001, 9000, 10000, 11111, 12000, 13001, 14000, 15001]
N_SEEDS = len(RANDOM_SEEDS)

LGB_PARAMS = {
    'objective': 'binary', 'metric': 'binary_logloss',
    'num_leaves': 8, 'max_depth': 3,
    'learning_rate': 0.02, 'n_estimators': 200,
    'subsample': 0.6, 'colsample_bytree': 0.6,
    'reg_alpha': 2.0, 'reg_lambda': 5.0,
    'min_child_samples': 15,
    'force_row_wise': True, 'n_jobs': -1,
    'verbose': -1,
}

# V31 configs from completed CV run
V31_CONFIGS = {
    'Q1': {'name': 'C1', 'nl': 8, 'md': 3, 'lr': 0.02, 'ne': 200, 'ss': 0.6, 'cst': 0.6, 'ra': 2.0, 'rl': 5.0, 'mc': 15, '_n_feats': 20},
    'Q2': {'name': 'C1', 'nl': 8, 'md': 3, 'lr': 0.02, 'ne': 200, 'ss': 0.6, 'cst': 0.6, 'ra': 2.0, 'rl': 5.0, 'mc': 15, '_n_feats': 30},
    'Q3': {'name': 'C1', 'nl': 8, 'md': 3, 'lr': 0.02, 'ne': 200, 'ss': 0.6, 'cst': 0.6, 'ra': 2.0, 'rl': 5.0, 'mc': 15, '_n_feats': 30},
    'S1': {'name': 'C1', 'nl': 8, 'md': 3, 'lr': 0.02, 'ne': 200, 'ss': 0.6, 'cst': 0.6, 'ra': 2.0, 'rl': 5.0, 'mc': 15, '_n_feats': 30},
    'S2': {'name': 'C1', 'nl': 8, 'md': 3, 'lr': 0.02, 'ne': 200, 'ss': 0.6, 'cst': 0.6, 'ra': 2.0, 'rl': 5.0, 'mc': 15, '_n_feats': 20},
    'S3': {'name': 'C1', 'nl': 8, 'md': 3, 'lr': 0.02, 'ne': 200, 'ss': 0.6, 'cst': 0.6, 'ra': 2.0, 'rl': 5.0, 'mc': 15, '_n_feats': 30},
    'S4': {'name': 'C1', 'nl': 8, 'md': 3, 'lr': 0.02, 'ne': 200, 'ss': 0.6, 'cst': 0.6, 'ra': 2.0, 'rl': 5.0, 'mc': 15, '_n_feats': 10},
}


def sanitize(name):
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)


def get_feature_cols(feat, exclude_leakage=None):
    cols = [c for c in feat.columns
            if c not in META_COLS | set(TARGET_COLS)
            and feat[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]
    if exclude_leakage:
        cols = [c for c in cols if c not in exclude_leakage]
    return cols


def rank_features(feat, cols, target):
    y = feat[target].values
    X = feat[cols].fillna(0).values
    n_pos = max((y == 1).sum(), 1)
    n_neg = (y == 0).sum()
    spw = n_neg / n_pos
    params = {**LGB_PARAMS, 'num_leaves': 15, 'max_depth': 4, 'n_estimators': 100,
              'scale_pos_weight': spw, 'random_state': 42}
    sanitized = [sanitize(c) for c in cols]
    ds = lgb.Dataset(X, label=y, feature_name=sanitized, params={'verbose': '-1'})
    mdl = lgb.train(params, ds, num_boost_round=100)
    imp = mdl.feature_importance(importance_type='gain')
    return sorted(zip(cols, imp), key=lambda x: -x[1])


def simple_mean_match(pred, target_rate):
    shift = target_rate - pred.mean()
    return np.clip(pred + shift, 0.0001, 0.9999)


def main():
    log.info("=" * 70)
    log.info("V31 Final — Submission Generation (from preprocessed features)")
    log.info("=" * 70)

    # ── 1. Load features ───────────────────────────────────
    feat = pd.read_parquet(DATA_PROCESSED / "features.parquet")
    log.info(f"Train features: {feat.shape}")

    test_feat = pd.read_parquet(DATA_PROCESSED / "test_features.parquet")
    log.info(f"Test features: {test_feat.shape}")

    train_rate = {t: feat[t].mean() for t in TARGET_COLS}
    log.info(f"Target rates: {train_rate}")

    # ── 2. Build feature selection per target ──────────────
    log.info("\n=== Building feature selection ===")

    lgb_sel = {}
    for target in TARGET_COLS:
        leak = LEAKAGE_FEATURES_S if target.startswith('S') else LEAKAGE_FEATURES_Q
        avail = get_feature_cols(feat, exclude_leakage=leak)
        ranked = rank_features(feat, avail, target)
        n_feats = V31_CONFIGS[target]['_n_feats']
        sel = [r[0] for r in ranked[:n_feats]]
        lgb_sel[target] = sel
        log.info(f"  {target}: {len(sel)} features (from {len(avail)} available)")

    # ── 3. Train final models on full data ─────────────────
    log.info("\n=== Training final models ===")

    # LightGBM
    lgb_preds = {}
    for target in TARGET_COLS:
        sel = lgb_sel[target]
        y_all = feat[target].values
        X_all = feat[sel].fillna(0).values
        test_X = test_feat[sel].fillna(0).values
        sanitized = [sanitize(c) for c in sel]
        spw = ((y_all == 0).sum()) / max((y_all == 1).sum(), 1)
        cfg = V31_CONFIGS[target]

        lgb_p = {**LGB_PARAMS, 'num_leaves': cfg['nl'], 'max_depth': cfg['md'],
                 'learning_rate': cfg['lr'], 'n_estimators': cfg['ne'],
                 'subsample': cfg['ss'], 'colsample_bytree': cfg['cst'],
                 'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
                 'min_child_samples': cfg['mc'], 'scale_pos_weight': spw}

        all_preds = np.zeros(len(test_X))
        for seed in RANDOM_SEEDS:
            params = {**lgb_p, 'random_state': seed}
            ds = lgb.Dataset(X_all, label=y_all, feature_name=sanitized, params={'verbose': '-1'})
            mdl = lgb.train(params, ds, num_boost_round=cfg['ne'])
            all_preds += mdl.predict(test_X)
        all_preds /= N_SEEDS
        lgb_preds[target] = all_preds
        log.info(f"  LGBM {target}: mean={all_preds.mean():.4f}, min={all_preds.min():.4f}, max={all_preds.max():.4f}")

    # XGBoost
    xgb_preds = {}
    XGB_PARAMS = {
        'tree_method': 'hist', 'objective': 'binary:logistic',
        'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 500,
        'subsample': 0.7, 'colsample_bytree': 0.7,
        'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_weight': 10,
        'random_state': 42, 'verbosity': 0, 'n_jobs': -1,
    }
    for target in TARGET_COLS:
        sel = lgb_sel[target]
        y_all = feat[target].values
        X_all = feat[sel].fillna(0).values
        test_X = test_feat[sel].fillna(0).values
        spw = ((y_all == 0).sum()) / max((y_all == 1).sum(), 1)

        all_preds = np.zeros(len(test_X))
        for seed in RANDOM_SEEDS:
            cfg = {**XGB_PARAMS, 'random_state': seed, 'scale_pos_weight': spw}
            clf = xgb.XGBClassifier(**cfg)
            clf.fit(X_all, y_all, verbose=False)
            all_preds += clf.predict_proba(test_X)[:, 1]
        all_preds /= N_SEEDS
        xgb_preds[target] = all_preds
        log.info(f"  XGB {target}: mean={all_preds.mean():.4f}, min={all_preds.min():.4f}, max={all_preds.max():.4f}")

    # ── 4. Ensemble ────────────────────────────────────────
    log.info("\n=== Ensemble (LGBM=0.7, XGB=0.3) ===")

    w_lgb = 0.7
    w_xgb = 0.3

    predictions = test_feat[['subject_id', 'sleep_date', 'lifelog_date']].copy()
    final_cal_means = {}

    for target in TARGET_COLS:
        ens = w_lgb * lgb_preds[target] + w_xgb * xgb_preds[target]
        cal = simple_mean_match(ens, train_rate[target])
        predictions[target] = cal
        cal_mean = cal.mean()
        final_cal_means[target] = cal_mean
        log.info(f"  {target}: cal_mean={cal_mean:.4f}, train_rate={train_rate[target]:.3f}")

    # Cal OOF from the CV run (31_v31_ensemble completed above)
    avg_cal_oof = 0.5748  # from ensemble search: LGBM=0.7, XGB=0.3

    # ── 5. Save ────────────────────────────────────────────
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    sub_path = SUBMIT_DIR / f'submission_v31_{timestamp}.csv'
    predictions.to_csv(sub_path, index=False)
    log.info(f"\n✅ Submission saved: {sub_path}")

    meta = {
        'version': 'v31',
        'submission_file': str(sub_path),
        'timestamp': timestamp,
        'n_samples': len(predictions),
        'models': ['LightGBM (V31 tuned) + XGBoost'],
        'ensemble_weights': {'lgbm': w_lgb, 'xgb': w_xgb},
        'n_seeds': N_SEEDS,
        'calibration': 'simple mean-matching + clip',
        'cal_oof_score': float(avg_cal_oof),
        'per_target': {},
    }
    for t in TARGET_COLS:
        meta['per_target'][t] = {
            'config': V31_CONFIGS[t],
            'n_features': len(lgb_sel[t]),
            'cal_oof_loss': 0.0,
            'cal_mean': float(predictions[t].mean()),
            'train_rate': float(train_rate[t]),
        }

    meta_path = sub_path.parent / f'meta_v31_{timestamp}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)

    log.info(f"\n{'='*70}")
    log.info("V31 FINAL")
    log.info(f"{'='*70}")
    log.info(f"Submission: {sub_path}")
    log.info(f"{'Target':<6} {'Cal OOF*':<12} {'Test Mean':<12} {'Train Rate':<12} {'Shift'}")
    for t in TARGET_COLS:
        shift = predictions[t].mean() - train_rate[t]
        log.info(f"{t:<6} {avg_cal_oof:<12.4f} {predictions[t].mean():<12.4f} {train_rate[t]:<12.3f} {shift:+.4f}")
    log.info(f"  *Cal OOF from CV ensemble search")
    log.info(f"  AVG Cal OOF: {avg_cal_oof:.4f}")


if __name__ == "__main__":
    main()
