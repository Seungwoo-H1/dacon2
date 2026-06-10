"""
V492 — Per-Target Ensemble Weight Optimization + Calibration

Hypothesis: Simple average of 3 models is suboptimal. 
Optimizing per-model weights per target can improve LB.

Key idea: Find optimal weights (w_lgb, w_cb, w_xgb) that minimize CV logloss.
"""

import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize

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
META_COLS = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}

def sanitize(n):
    return re.sub(r'[^a-zA-Z0-9_]','_',n)

def get_feature_cols(df):
    return [c for c in df.columns
            if c not in META_COLS | set(TARGETS)
            and np.issubdtype(df[c].dtype, np.number)]

def logloss(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def train_lgbm(X_train, y_train, X_val, sel_sn, cfg, seed):
    spw = max(((y_train == 0).sum()) / max((y_train == 1).sum(), 1), 0.1)
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'num_leaves': cfg['nl'], 'max_depth': cfg['md'], 'learning_rate': cfg['lr'],
        'n_estimators': cfg['ne'], 'subsample': cfg['ss'], 'colsample_bytree': cfg['cb'],
        'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
        'min_child_samples': cfg['mc'], 'random_state': seed,
        'scale_pos_weight': spw, 'force_row_wise': True, 'n_jobs': -1,
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
        verbosity=0, n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train, verbose=False)
    return np.clip(xgb_model.predict_proba(np.where(np.isnan(X_val), 0, X_val))[:, 1], 0.0001, 0.9999)

def find_optimal_weights(oof_lgb, oof_cb, oof_xgb, y_true):
    """Find optimal weights for ensemble averaging."""
    def obj(w):
        w = np.maximum(w, 0)
        w = w / w.sum()
        pred = w[0] * oof_lgb + w[1] * oof_cb + w[2] * oof_xgb
        return logloss(y_true, pred)
    
    result = minimize(obj, [1/3, 1/3, 1/3], method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
    weights = np.maximum(result.x, 0)
    weights = weights / weights.sum()
    return weights, result.fun

def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("V492 — Optimal Ensemble Weights + Calibration")
    log.info("=" * 70)

    log.info("\n--- 1. Load data ---")
    train = pd.read_parquet(DATA / "features.parquet")
    test = pd.read_parquet(DATA / "test_features.parquet")
    log.info(f"  Train: {train.shape}, Test: {test.shape}")

    # Ensure train/test have same columns (only common numeric features)
    # Keep target columns in train (not in test)
    target_cols = set(TARGETS)
    feature_cols_all = set(train.columns) - target_cols
    common_cols = sorted(feature_cols_all & set(test.columns))
    train_cols_list = common_cols + [c for c in train.columns if c in target_cols]
    train = train[train_cols_list]
    test = test[common_cols]
    log.info(f"  Common feature columns: {len(common_cols)}")

    feat_cols = get_feature_cols(train)
    log.info(f"  Total features: {len(feat_cols)}")

    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=5)

    LGBM_CONFIGS = [
        {'name': 'lgb_conservative', 'nl': 20, 'md': 4, 'lr': 0.02, 'ne': 800, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 15},
        {'name': 'lgb_deep', 'nl': 30, 'md': 5, 'lr': 0.015, 'ne': 1200, 'ss': 0.7, 'cb': 0.6, 'ra': 0.5, 'rl': 2.0, 'mc': 10},
        {'name': 'lgb_wide', 'nl': 15, 'md': 3, 'lr': 0.03, 'ne': 600, 'ss': 0.8, 'cb': 0.8, 'ra': 2.0, 'rl': 5.0, 'mc': 20},
    ]
    CB_CONFIGS = [
        {'name': 'cb_conservative', 'iter': 800, 'lr': 0.02, 'depth': 5, 'l2': 5.0, 'bagging': 0.5, 'rs': 1.0},
        {'name': 'cb_deep', 'iter': 1200, 'lr': 0.015, 'depth': 6, 'l2': 3.0, 'bagging': 0.5, 'rs': 1.0},
        {'name': 'cb_wide', 'iter': 500, 'lr': 0.03, 'depth': 4, 'l2': 7.0, 'bagging': 0.6, 'rs': 2.0},
    ]
    XGB_CONFIGS = [
        {'name': 'xgb_conservative', 'md': 4, 'lr': 0.02, 'ne': 800, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 15},
        {'name': 'xgb_deep', 'md': 5, 'lr': 0.015, 'ne': 1200, 'ss': 0.7, 'cb': 0.6, 'ra': 0.5, 'rl': 2.0, 'mc': 10},
        {'name': 'xgb_wide', 'md': 3, 'lr': 0.03, 'ne': 600, 'ss': 0.8, 'cb': 0.8, 'ra': 2.0, 'rl': 5.0, 'mc': 20},
    ]
    SEEDS = [1, 2, 3, 4, 5]

    predictions = {}
    target_results = {}

    for target in TARGETS:
        log.info(f"\n{'='*60}")
        log.info(f"--- {target} ---")
        log.info(f"  Target rate: {train[target].mean():.3f}")

        y = train[target].values.astype(np.float64)
        
        # Feature selection
        spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
        signal_cols = [c for c in feat_cols if c not in META_COLS | set(TARGETS)]
        X_signal = train[signal_cols].fillna(0).values.astype(np.float64)
        sn = [sanitize(c) for c in signal_cols]
        
        params_rank = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
            'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.02,
            'n_estimators': 100, 'subsample': 0.7, 'colsample_bytree': 0.6,
            'reg_alpha': 0.5, 'reg_lambda': 2.0,
            'scale_pos_weight': spw, 'random_state': 42,
            'min_child_samples': 15, 'force_row_wise': True, 'n_jobs': -1,
        }
        ds_rank = lgb.Dataset(X_signal, label=y, feature_name=sn, params={'verbose': '-1'})
        model_rank = lgb.train(params_rank, ds_rank, num_boost_round=100)
        imp_signal = model_rank.feature_importance(importance_type='gain')
        signal_ranked = sorted(zip(signal_cols, imp_signal), key=lambda x: -x[1])

        best_cv = float('inf')
        best_n_feat = 30
        best_oof_lgb = None
        best_oof_cb = None
        best_oof_xgb = None

        for n_feat in [30, 40, 50]:
            sel_cols = [r[0] for r in signal_ranked[:n_feat]]
            sel_sn = [sanitize(r[0]) for r in signal_ranked[:n_feat]]
            X_sel = train[sel_cols].fillna(0).values.astype(np.float64)

            oof_lgb = np.zeros(len(y))
            oof_cb = np.zeros(len(y))
            oof_xgb = np.zeros(len(y))

            log.info(f"    n_feat={n_feat}...")
            t_train = time.time()
            
            for cfg in LGBM_CONFIGS:
                for s in SEEDS:
                    for fold, (tr, va) in enumerate(gkf.split(X_sel, y, train['subject_id'].values)):
                        model = train_lgbm(X_sel[tr], y[tr], X_sel[va], sel_sn, cfg, s)
                        oof_lgb[va] += model
            oof_lgb /= len(LGBM_CONFIGS) * len(SEEDS)

            t_train = time.time()
            for cfg in CB_CONFIGS:
                for s in SEEDS:
                    for fold, (tr, va) in enumerate(gkf.split(X_sel, y, train['subject_id'].values)):
                        pred = train_catboost(X_sel[tr], y[tr], X_sel[va], cfg, s)
                        oof_cb[va] += pred
            oof_cb /= len(CB_CONFIGS) * len(SEEDS)

            t_train = time.time()
            for cfg in XGB_CONFIGS:
                for s in SEEDS:
                    for fold, (tr, va) in enumerate(gkf.split(X_sel, y, train['subject_id'].values)):
                        pred = train_xgboost(X_sel[tr], y[tr], X_sel[va], cfg, s)
                        oof_xgb[va] += pred
            oof_xgb /= len(XGB_CONFIGS) * len(SEEDS)

            log.info(f"    Training done ({time.time()-t_train:.0f}s)")
            log.info(f"    LGBM CV={logloss(y, oof_lgb):.4f}, CB={logloss(y, oof_cb):.4f}, XGB={logloss(y, oof_xgb):.4f}")

            # Find optimal weights
            weights, opt_cv = find_optimal_weights(oof_lgb, oof_cb, oof_xgb, y)
            log.info(f"    Optimal weights: LGBM={weights[0]:.3f}, CB={weights[1]:.3f}, XGB={weights[2]:.3f}, CV={opt_cv:.4f}")

            # Simple average for comparison
            avg_cv = logloss(y, (oof_lgb + oof_cb + oof_xgb) / 3.0)
            log.info(f"    Avg CV={avg_cv:.4f}")

            if opt_cv < best_cv:
                best_cv = opt_cv
                best_n_feat = n_feat
                best_oof_lgb = oof_lgb.copy()
                best_oof_cb = oof_cb.copy()
                best_oof_xgb = oof_xgb.copy()
                best_weights = weights

        # Final prediction with optimal weights
        log.info(f"\n  Best n_feat={best_n_feat}, best_cv={best_cv:.4f}")
        sel_cols = [r[0] for r in signal_ranked[:best_n_feat]]
        sel_sn = [sanitize(r[0]) for r in signal_ranked[:best_n_feat]]
        X_all = train[sel_cols].fillna(0).values.astype(np.float64)
        X_all_test = test[sel_cols].fillna(0).values.astype(np.float64)

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

        # Apply optimal weights
        w = best_weights
        test_avg = w[0] * test_lgb + w[1] * test_cb + w[2] * test_xgb
        oof_avg = w[0] * best_oof_lgb + w[1] * best_oof_cb + w[2] * best_oof_xgb

        predictions[target] = np.clip(test_avg, 0.0001, 0.9999)

        meta_oof = logloss(y, oof_avg)
        # student OOF proxy: avg of per-model OOF with optimal weights
        lgb_oof = logloss(y, best_oof_lgb)
        cb_oof = logloss(y, best_oof_cb)
        xgb_oof = logloss(y, best_oof_xgb)
        student_oof = w[0] * lgb_oof + w[1] * cb_oof + w[2] * xgb_oof
        gap = abs(meta_oof - student_oof)

        target_results[target] = {
            'best_n_feat': best_n_feat,
            'best_cv': float(best_cv),
            'per_target_rate': float(train[target].mean()),
            'test_mean': float(test_avg.mean()),
            'meta_oof': float(meta_oof),
            'student_oof': float(student_oof),
            'gap': float(gap),
            'weights': [float(x) for x in w],
        }
        log.info(f"  {target}: meta={meta_oof:.4f}, student={student_oof:.4f}, gap={gap:.4f}")
        log.info(f"  Weights: LGBM={w[0]:.3f}, CB={w[1]:.3f}, XGB={w[2]:.3f}")

        gc.collect()

    avg_meta = np.mean([v['meta_oof'] for v in target_results.values()])
    avg_student = np.mean([v['student_oof'] for v in target_results.values()])
    
    log.info(f"\n{'='*70}")
    log.info(f"V492 RESULTS (Optimal Weights)")
    log.info(f"{'='*70}")
    for t in TARGETS:
        r = target_results[t]
        log.info(f"  {t}: meta={r['meta_oof']:.4f}, student={r['student_oof']:.4f}, gap={r['gap']:.4f}, w={r['weights']}")
    log.info(f"  AVG Meta OOF: {avg_meta:.4f}")
    log.info(f"  AVG Student OOF: {avg_student:.4f}")
    log.info(f"  Total time: {time.time()-t_start:.0f}s")

    # Save submission
    sample = pd.read_csv(ROOT / "data_raw" / "ch2026_submission_sample.csv")
    sub = sample.copy()
    for t in TARGETS:
        sub[t] = predictions[t]
    sub_path = SUBMIT / f"submission_v492_opt_weights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"  Submission: {sub_path}")

    meta = {
        'version': 'V492_opt_weights',
        'name': 'Optimal ensemble weights + 3-model ensemble',
        'cv_method': 'GroupKFold_5fold',
        'n_models_per_target': 45,
        'target_results': target_results,
        'avg_meta_oof': float(avg_meta),
        'avg_student_oof': float(avg_student),
        'submission_file': str(sub_path),
        'timestamp': datetime.now().isoformat(),
        'total_time': f"{time.time()-t_start:.0f}s",
    }
    meta_path = EXPERIMENTS / f'v492_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    log.info(f"  Meta: {meta_path}")


if __name__ == "__main__":
    import lightgbm as lgb
    import catboost as cb
    import xgboost as xgb
    
    predictions = {}
    main()
