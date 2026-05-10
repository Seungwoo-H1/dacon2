"""
V73 — Aggressive push toward 0.5000 CV

Strategy (layered improvements):
1. Multi-model ensemble: LGBM + CatBoost + XGBoost (each with many seeds)
2. Per-target feature count sweep (10-40) during CV to find optimal
3. GroupKFold 5-fold (subject-level)
4. Leakage-clean features (V61 pipeline)
5. Multiple configs per model per target (conservative/deep/wide)
6. Simple average of all 3 models → final predictions
7. Post-processing: clip + mean-match calibration

Architecture:
  Level 0: 
    - LGBM: 3 configs × 20 seeds = 60 models per target
    - CatBoost: 3 configs × 20 seeds = 60 models per target
    - XGBoost: 3 configs × 20 seeds = 60 models per target
  Level 1: Simple average of all 180 predictions
  Feature selection: top-K per target via LGBM gain importance (sweep K=10..40)
"""

import sys, gc, logging, json, re, time, warnings, itertools
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_processed"
SUBMIT = ROOT / "submissions"
SUBMIT.mkdir(exist_ok=True)
EXPERIMENTS = ROOT / "experiments"
EXPERIMENTS.mkdir(exist_ok=True)

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
META = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}

# ── Leakage columns ──
LEAK_S = {'wLight_w_light_mean','wLight_w_light_std','wLight_w_light_min',
    'wLight_w_light_max','wLight_w_light_count',
    'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max',
    'wHr_hr_median','wHr_hr_count',
    'wPedo_pedo_step_mean','wPedo_pedo_step_sum',
    'wPedo_pedo_step_frequency_mean','wPedo_pedo_step_frequency_sum',
    'wPedo_pedo_running_step_mean','wPedo_pedo_step_frequency_sum',
    'wPedo_pedo_walking_step_mean','wPedo_pedo_walking_step_sum',
    'wPedo_pedo_distance_mean','wPedo_pedo_distance_sum',
    'wPedo_pedo_speed_mean','wPedo_pedo_speed_sum',
    'wPedo_pedo_burned_calories_mean','wPedo_pedo_burned_calories_sum',}
LEAK_Q = {'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max',
    'wHr_hr_median','wHr_hr_count'}
NIGHTTIME_LEAK = {
    'mScreenStatus_hour_night', 'mACStatus_hour_night',
    'mScreenStatus_hour_morning', 'wLight_w_light_sum',
    'mACStatus_charging_sum', 'mACStatus_charging_max',
}
SLEEP_DIRECT_LEAK = {
    'mGps_gps_avg_speed_max', 'mGps_gps_count_mean',
    'mActivity_m_activity_sum', 'mActivity_m_activity_max',
    'mActivity_m_activity_min',
}

def sanitize(n):
    return re.sub(r'[^a-zA-Z0-9_]','_',n)

def get_feature_cols(df):
    return [c for c in df.columns
            if c not in META | set(TARGETS)
            and np.issubdtype(df[c].dtype, np.number)]

def remove_leak(cols, target):
    leak = set()
    if target.startswith('S'):
        leak = LEAK_S | NIGHTTIME_LEAK | SLEEP_DIRECT_LEAK
    elif target.startswith('Q'):
        leak = LEAK_Q | NIGHTTIME_LEAK
    return [c for c in cols if c not in leak]

def logloss(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# ── Model configs ──
# 3 configs per model type for diversity
LGBM_CONFIGS = [
    {'name': 'lgb_conservative', 'nl': 10, 'md': 3, 'lr': 0.02, 'ne': 800, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
    {'name': 'lgb_deep', 'nl': 20, 'md': 5, 'lr': 0.015, 'ne': 1500, 'ss': 0.7, 'cb': 0.6, 'ra': 0.5, 'rl': 2.0, 'mc': 15},
    {'name': 'lgb_wide', 'nl': 30, 'md': 3, 'lr': 0.03, 'ne': 500, 'ss': 0.8, 'cb': 0.8, 'ra': 2.0, 'rl': 5.0, 'mc': 5},
]

CB_CONFIGS = [
    {'name': 'cb_conservative', 'iter': 800, 'lr': 0.02, 'depth': 5, 'l2': 5.0, 'bagging': 0.5, 'rs': 1.0},
    {'name': 'cb_deep', 'iter': 1500, 'lr': 0.015, 'depth': 6, 'l2': 3.0, 'bagging': 0.5, 'rs': 1.0},
    {'name': 'cb_wide', 'iter': 500, 'lr': 0.03, 'depth': 4, 'l2': 7.0, 'bagging': 0.6, 'rs': 2.0},
]

XGB_CONFIGS = [
    {'name': 'xgb_conservative', 'md': 3, 'lr': 0.02, 'ne': 800, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
    {'name': 'xgb_deep', 'md': 5, 'lr': 0.015, 'ne': 1500, 'ss': 0.7, 'cb': 0.6, 'ra': 0.5, 'rl': 2.0, 'mc': 15},
    {'name': 'xgb_wide', 'md': 3, 'lr': 0.03, 'ne': 500, 'ss': 0.8, 'cb': 0.8, 'ra': 2.0, 'rl': 5.0, 'mc': 5},
]

SEEDS = list(range(1, 21))  # 20 seeds per model per config

# n_feat sweep range
N_FEAT_RANGE = range(10, 41, 5)  # 10, 15, 20, 25, 30, 35, 40


def train_lgbm(X_train, y_train, X_val, sel_sn, cfg, seed):
    spw = max(((y_train == 0).sum()) / max((y_train == 1).sum(), 1), 0.1)
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'num_leaves': cfg['nl'], 'max_depth': cfg['md'], 'learning_rate': cfg['lr'],
        'n_estimators': cfg['ne'], 'subsample': cfg['ss'], 'colsample_bytree': cfg['cb'],
        'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
        'min_child_samples': cfg['mc'], 'random_state': seed,
        'scale_pos_weight': spw, 'force_row_wise': True, 'n_jobs': 1,
    }
    ds = lgb.Dataset(X_train, label=y_train, feature_name=sel_sn, params={'verbose': '-1'})
    model = lgb.train(params, ds, num_boost_round=cfg['ne'])
    return model.predict(X_val)


def train_catboost(X_train, y_train, X_val, cfg, seed):
    spw = max(((y_train == 0).sum()) / max((y_train == 1).sum(), 1), 0.1)
    cb_model = cb.CatBoostClassifier(
        iterations=cfg['iter'], learning_rate=cfg['lr'],
        depth=cfg['depth'], loss_function='Logloss', eval_metric='Logloss',
        random_seed=seed, verbose=0, task_type='CPU',
        bagging_temperature=cfg['bagging'], l2_leaf_reg=cfg['l2'], random_strength=cfg['rs'],
        scale_pos_weight=spw,
    )
    cb_model.fit(X_train, y_train, verbose=0)
    return np.clip(cb_model.predict_proba(np.where(np.isnan(X_val), 0, X_val))[:, 1], 0.0001, 0.9999)


def train_xgboost(X_train, y_train, X_val, cfg, seed):
    spw = max(((y_train == 0).sum()) / max((y_train == 1).sum(), 1), 0.1)
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic', eval_metric='logloss',
        max_depth=cfg['md'], learning_rate=cfg['lr'],
        n_estimators=cfg['ne'], subsample=cfg['ss'],
        colsample_bytree=cfg['cb'],
        reg_alpha=cfg['ra'], reg_lambda=cfg['rl'],
        min_child_weight=cfg['mc'], random_state=seed,
        scale_pos_weight=spw, tree_method='hist',
        verbosity=0, n_jobs=1,
    )
    xgb_model.fit(X_train, y_train, verbose=False)
    return np.clip(xgb_model.predict_proba(np.where(np.isnan(X_val), 0, X_val))[:, 1], 0.0001, 0.9999)


def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("V73 — Aggressive push toward 0.5000 CV")
    log.info("=" * 70)

    # ── 1. Load data ──
    log.info("\n--- 1. Load data ---")
    train = pd.read_parquet(DATA / "features_clean_v60.parquet")
    test = pd.read_parquet(DATA / "test_features_clean_v60.parquet")
    test = test[list(train.columns)]
    log.info(f"  Train: {train.shape}, Test: {test.shape}")

    feat_cols = get_feature_cols(train)
    log.info(f"  Total features: {len(feat_cols)}")
    
    groups = train['subject_id'].values
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=5)

    # ── 2. For each target: feature ranking + n_feat sweep + multi-model ensemble ──
    target_results = {}

    for target in TARGETS:
        log.info(f"\n{'='*60}")
        log.info(f"--- {target} ---")
        log.info(f"  Target rate: {train[target].mean():.3f}")

        y = train[target].values.astype(np.float64)
        leak_cols = remove_leak(feat_cols, target)
        log.info(f"  Leakage-clean features: {len(leak_cols)}")

        X = train[leak_cols].fillna(0).values.astype(np.float64)
        X_test = test[leak_cols].fillna(0).values.astype(np.float64)
        sn = [sanitize(c) for c in leak_cols]

        # ── 2a. Feature ranking (LGBM, single seed, 100 trees) ──
        log.info("  Ranking features...")
        from lightgbm import Dataset as LGBDataset
        spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
        params_rank = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
            'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.02,
            'n_estimators': 100, 'subsample': 0.7, 'colsample_bytree': 0.6,
            'reg_alpha': 0.5, 'reg_lambda': 2.0,
            'scale_pos_weight': spw, 'random_state': 42,
            'min_child_samples': 15, 'force_row_wise': True, 'n_jobs': 1,
        }
        ds_rank = lgb.Dataset(X, label=y, feature_name=sn, params={'verbose': '-1'})
        model_rank = lgb.train(params_rank, ds_rank, num_boost_round=100)
        imp = model_rank.feature_importance(importance_type='gain')
        ranked = sorted(zip(leak_cols, imp), key=lambda x: -x[1])
        del model_rank, ds_rank, X
        gc.collect()

        # ── 2b. Try different n_feat values ──
        best_n_feat = 20
        best_cv = float('inf')
        best_oof = None
        best_test_preds = None

        for n_feat in N_FEAT_RANGE:
            sel_cols = [r[0] for r in ranked[:n_feat]]
            sel_sn = [sanitize(r[0]) for r in ranked[:n_feat]]
            sel_idx = [leak_cols.index(c) for c in sel_cols]

            # Rebuild X with selected cols for CV
            X_sel = train[leak_cols].fillna(0).values.astype(np.float64)[:, sel_idx]
            X_test_sel = test[leak_cols].fillna(0).values.astype(np.float64)[:, sel_idx]

            # ── Multi-model OOF ensemble ──
            oof_lgb = np.zeros(len(y))
            oof_cb = np.zeros(len(y))
            oof_xgb = np.zeros(len(y))

            n_models_lgb = len(LGBM_CONFIGS) * len(SEEDS)
            n_models_cb = len(CB_CONFIGS) * len(SEEDS)
            n_models_xgb = len(XGB_CONFIGS) * len(SEEDS)

            log.info(f"    n_feat={n_feat}: training {n_models_lgb+ n_models_cb + n_models_xgb} models...")
            
            t_train = time.time()
            
            # LightGBM
            for cfg in LGBM_CONFIGS:
                for s in SEEDS:
                    for fold, (tr, va) in enumerate(gkf.split(X_sel, y, groups)):
                        X_tr = X_sel[tr]
                        y_tr = y[tr]
                        X_va = X_sel[va]
                        model = train_lgbm(X_tr, y_tr, X_va, sel_sn, cfg, s)
                        oof_lgb[va] += model
            oof_lgb /= len(LGBM_CONFIGS) * len(SEEDS)
            log.info(f"    LGBM done ({time.time()-t_train:.0f}s)")
            del oof_lgb
            gc.collect()

            # CatBoost
            t_train = time.time()
            for cfg in CB_CONFIGS:
                for s in SEEDS:
                    for fold, (tr, va) in enumerate(gkf.split(X_sel, y, groups)):
                        X_tr = X_sel[tr]
                        y_tr = y[tr]
                        X_va = X_sel[va]
                        pred = train_catboost(X_tr, y_tr, X_va, cfg, s)
                        oof_cb[va] += pred
            oof_cb /= len(CB_CONFIGS) * len(SEEDS)
            log.info(f"    CatBoost done ({time.time()-t_train:.0f}s)")
            del oof_cb
            gc.collect()

            # XGBoost
            t_train = time.time()
            for cfg in XGB_CONFIGS:
                for s in SEEDS:
                    for fold, (tr, va) in enumerate(gkf.split(X_sel, y, groups)):
                        X_tr = X_sel[tr]
                        y_tr = y[tr]
                        X_va = X_sel[va]
                        pred = train_xgboost(X_tr, y_tr, X_va, cfg, s)
                        oof_xgb[va] += pred
            oof_xgb /= len(XGB_CONFIGS) * len(SEEDS)
            log.info(f"    XGBoost done ({time.time()-t_train:.0f}s)")
            del oof_xgb

            # Average of 3 models
            oof_avg = (oof_lgb + oof_cb + oof_xgb) / 3.0
            cv = logloss(y, oof_avg)
            log.info(f"    n_feat={n_feat}: avg_cv={cv:.4f}")

            if cv < best_cv:
                best_cv = cv
                best_n_feat = n_feat
                best_oof = oof_avg.copy()
                best_test_preds = None  # Will compute later

            del X_sel, X_test_sel
            gc.collect()

        log.info(f"\n  Best n_feat={best_n_feat}, best_cv={best_cv:.4f}")

        # ── 2c. Final prediction with best n_feat (train on ALL data) ──
        log.info(f"  Training final model with n_feat={best_n_feat} on all data...")
        sel_cols = [r[0] for r in ranked[:best_n_feat]]
        sel_sn = [sanitize(r[0]) for r in ranked[:best_n_feat]]
        sel_idx = [leak_cols.index(c) for c in sel_cols]
        X_all = train[leak_cols].fillna(0).values.astype(np.float64)[:, sel_idx]
        X_all_test = test[leak_cols].fillna(0).values.astype(np.float64)[:, sel_idx]

        test_lgb = np.zeros(len(X_all_test))
        for cfg in LGBM_CONFIGS:
            for s in SEEDS:
                model = train_lgbm(X_all, y, X_all_test, sel_sn, cfg, s)
                test_lgb += model
        test_lgb /= len(LGBM_CONFIGS) * len(SEEDS)

        test_cb = np.zeros(len(X_all_test))
        for cfg in CB_CONFIGS:
            for s in SEEDS:
                pred = train_catboost(X_all, y, X_all_test, cfg, s)
                test_cb += pred
        test_cb /= len(CB_CONFIGS) * len(SEEDS)

        test_xgb = np.zeros(len(X_all_test))
        for cfg in XGB_CONFIGS:
            for s in SEEDS:
                pred = train_xgboost(X_all, y, X_all_test, cfg, s)
                test_xgb += pred
        test_xgb /= len(XGB_CONFIGS) * len(SEEDS)

        test_avg = (test_lgb + test_cb + test_xgb) / 3.0

        target_results[target] = {
            'best_n_feat': best_n_feat,
            'best_cv': float(best_cv),
            'per_target_rate': float(train[target].mean()),
            'test_mean': float(test_avg.mean()),
        }
        predictions[target] = np.clip(test_avg, 0.0001, 0.9999)

        log.info(f"  {target}: n_feat={best_n_feat}, cv={best_cv:.4f}, test_mean={test_avg.mean():.4f}")
        del sel_cols, X_all, X_all_test
        gc.collect()

    # ── 3. Summary ──
    avg_cv = np.mean([v['best_cv'] for v in target_results.values()])
    log.info(f"\n{'='*70}")
    log.info(f"V73 RESULTS")
    log.info(f"{'='*70}")
    for t in TARGETS:
        r = target_results[t]
        log.info(f"  {t}: cv={r['best_cv']:.4f} (n_feat={r['best_n_feat']})")
    log.info(f"  AVG CV: {avg_cv:.4f}")
    log.info(f"  Total time: {time.time()-t_start:.0f}s")

    # ── 4. Save submission ──
    sample = pd.read_csv(ROOT / "data_raw" / "ch2026_submission_sample.csv")
    sub = sample.copy()
    for t in TARGETS:
        sub[t] = predictions[t]
    sub_path = SUBMIT / f"submission_v73_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"  Submission: {sub_path}")

    # ── 5. Save meta ──
    meta = {
        'version': 'V73_break_050',
        'name': 'Aggressive 3-model ensemble (LGBM+CatBoost+XGB) × 20 seeds × 3 configs each',
        'cv_method': 'GroupKFold_5fold',
        'leakage_removal': 'wrist nighttime + sleep-direct + night-time screen/charging removed',
        'n_models_per_target': 180,  # 3 models × 3 configs × 20 seeds
        'n_feat_sweep': list(N_FEAT_RANGE),
        'target_results': target_results,
        'avg_cv': float(avg_cv),
        'submission_file': str(sub_path),
        'timestamp': datetime.now().isoformat(),
        'total_time': f"{time.time()-t_start:.0f}s",
    }
    meta_path = SUBMIT / f'meta_v73_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    log.info(f"  Meta: {meta_path}")


if __name__ == "__main__":
    import lightgbm as lgb
    import catboost as cb
    import xgboost as xgb
    
    predictions = {}
    main()
