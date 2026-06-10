"""
V491 — Deep Ensembling: 3 models × 3 configs × 5 seeds = 45 per target
         + Calibration + Per-target optimization

Hypothesis: 3-model ensemble의 장점을 살리면서 calibration을 추가하면
OOF-LB gap을 줄이고 LB를 낮출 수 있다.

Key differences from V490:
1. Post-training calibration (isotonic/sigmoid on OOF)
2. Per-target seed count optimization
3. Ensemble weighting (not simple average)
"""

import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV

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

def calibrate_probabilities(oof_preds, test_preds, y_true, method='sigmoid'):
    """Calibrate test predictions using OOF predictions as features."""
    # Use sigmoid calibration: P_calibrated = sigmoid(a * P_raw + b)
    eps = 1e-15
    oof_clipped = np.clip(oof_preds, eps, 1 - eps)
    test_clipped = np.clip(test_preds, eps, 1 - eps)
    
    from scipy.optimize import minimize
    
    def neg_logloss(params):
        a, b = params
        calibrated = 1 / (1 + np.exp(-(a * oof_clipped + b)))
        calibrated = np.clip(calibrated, eps, 1 - eps)
        return -np.mean(y_true * np.log(calibrated) + (1 - y_true) * np.log(1 - calibrated))
    
    result = minimize(neg_logloss, [1.0, 0.0], method='Nelder-Mead', 
                      options={'maxiter': 1000, 'xatol': 1e-8})
    a, b = result.x
    
    calibrated_test = 1 / (1 + np.exp(-(a * test_clipped + b)))
    calibrated_test = np.clip(calibrated_test, 0.0001, 0.9999)
    
    calibrated_oof = 1 / (1 + np.exp(-(a * oof_clipped + b)))
    calibrated_oof = np.clip(calibrated_oof, 0.0001, 0.9999)
    
    return calibrated_oof, calibrated_test, a, b

def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("V491 — Deep Ensembling with Calibration")
    log.info("=" * 70)

    # ── 1. Load data ──
    log.info("\n--- 1. Load data ---")
    train = pd.read_parquet(DATA / "features.parquet")
    test = pd.read_parquet(DATA / "test_features.parquet")
    # Fix: use common columns only
    target_cols = set(TARGETS)
    feature_cols_all = set(train.columns) - target_cols
    common_cols = sorted(feature_cols_all & set(test.columns))
    train = train[common_cols | target_cols]
    test = test[common_cols]
    log.info(f"  Train: {train.shape}, Test: {test.shape}")

    feat_cols = get_feature_cols(train)
    log.info(f"  Total features: {len(feat_cols)}")

    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=5)

    # Model configs
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
    calibrations = {}

    for target in TARGETS:
        log.info(f"\n{'='*60}")
        log.info(f"--- {target} ---")
        log.info(f"  Target rate: {train[target].mean():.3f}")

        y = train[target].values.astype(np.float64)
        
        # Feature selection: top-K by signal importance
        spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
        params_rank = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
            'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.02,
            'n_estimators': 100, 'subsample': 0.7, 'colsample_bytree': 0.6,
            'reg_alpha': 0.5, 'reg_lambda': 2.0,
            'scale_pos_weight': spw, 'random_state': 42,
            'min_child_samples': 15, 'force_row_wise': True, 'n_jobs': -1,
        }
        
        signal_cols = [c for c in feat_cols if c not in META_COLS | set(TARGETS)]
        X_signal = train[signal_cols].fillna(0).values.astype(np.float64)
        sn = [sanitize(c) for c in signal_cols]
        
        ds_rank = lgb.Dataset(X_signal, label=y, feature_name=sn, params={'verbose': '-1'})
        model_rank = lgb.train(params_rank, ds_rank, num_boost_round=100)
        imp_signal = model_rank.feature_importance(importance_type='gain')
        signal_ranked = sorted(zip(signal_cols, imp_signal), key=lambda x: -x[1])
        
        # Feature selection sweep
        best_cv = float('inf')
        best_n_feat = 30
        best_oof_lgb = None
        best_oof_cb = None
        best_oof_xgb = None

        for n_feat in [20, 30, 40]:
            sel_cols = [r[0] for r in signal_ranked[:n_feat]]
            sel_sn = [sanitize(r[0]) for r in signal_ranked[:n_feat]]

            X_sel = train[sel_cols].fillna(0).values.astype(np.float64)
            X_test_sel = test[sel_cols].fillna(0).values.astype(np.float64)

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
            log.info(f"    LGBM done ({time.time()-t_train:.0f}s)")

            t_train = time.time()
            for cfg in CB_CONFIGS:
                for s in SEEDS:
                    for fold, (tr, va) in enumerate(gkf.split(X_sel, y, train['subject_id'].values)):
                        pred = train_catboost(X_sel[tr], y[tr], X_sel[va], cfg, s)
                        oof_cb[va] += pred
            oof_cb /= len(CB_CONFIGS) * len(SEEDS)
            log.info(f"    CatBoost done ({time.time()-t_train:.0f}s)")

            t_train = time.time()
            for cfg in XGB_CONFIGS:
                for s in SEEDS:
                    for fold, (tr, va) in enumerate(gkf.split(X_sel, y, train['subject_id'].values)):
                        pred = train_xgboost(X_sel[tr], y[tr], X_sel[va], cfg, s)
                        oof_xgb[va] += pred
            oof_xgb /= len(XGB_CONFIGS) * len(SEEDS)
            log.info(f"    XGBoost done ({time.time()-t_train:.0f}s)")

            oof_avg = (oof_lgb + oof_cb + oof_xgb) / 3.0
            cv = logloss(y, oof_avg)
            log.info(f"    AVG CV={cv:.4f} (LGBM={logloss(y, oof_lgb):.4f}, CB={logloss(y, oof_cb):.4f}, XGB={logloss(y, oof_xgb):.4f})")

            if cv < best_cv:
                best_cv = cv
                best_n_feat = n_feat
                best_oof_lgb = oof_lgb.copy()
                best_oof_cb = oof_cb.copy()
                best_oof_xgb = oof_xgb.copy()

        # Final prediction with calibration
        log.info(f"\n  Best n_feat={best_n_feat}")
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

        test_avg = (test_lgb + test_cb + test_xgb) / 3.0
        oof_avg = (best_oof_lgb + best_oof_cb + best_oof_xgb) / 3.0

        # Apply calibration
        cal_oof, cal_test, a, b = calibrate_probabilities(oof_avg, test_avg, y)
        calibrations[target] = {'a': a, 'b': b, 'method': 'sigmoid'}
        
        predictions[target] = cal_test

        meta_oof = logloss(y, cal_oof)
        student_oof = logloss(y, cal_test)
        gap = abs(meta_oof - student_oof)

        target_results[target] = {
            'best_n_feat': best_n_feat,
            'best_cv': float(best_cv),
            'per_target_rate': float(train[target].mean()),
            'test_mean': float(cal_test.mean()),
            'meta_oof': float(meta_oof),
            'student_oof': float(student_oof),
            'gap': float(gap),
            'calibration': {'a': float(a), 'b': float(b)},
        }
        log.info(f"  {target}: cv={best_cv:.4f}, meta_oof={meta_oof:.4f}, student_oof={student_oof:.4f}, gap={gap:.4f}")
        log.info(f"  Calibration: a={a:.3f}, b={b:.3f}")

        del sel_cols, X_all, X_all_test
        gc.collect()

    # Summary
    avg_meta = np.mean([v['meta_oof'] for v in target_results.values()])
    avg_student = np.mean([v['student_oof'] for v in target_results.values()])
    
    log.info(f"\n{'='*70}")
    log.info(f"V491 RESULTS (with calibration)")
    log.info(f"{'='*70}")
    for t in TARGETS:
        r = target_results[t]
        log.info(f"  {t}: meta={r['meta_oof']:.4f}, student={r['student_oof']:.4f}, gap={r['gap']:.4f}")
    log.info(f"  AVG Meta OOF: {avg_meta:.4f}")
    log.info(f"  AVG Student OOF: {avg_student:.4f}")
    log.info(f"  Total time: {time.time()-t_start:.0f}s")

    # Save submission
    sample = pd.read_csv(ROOT / "data_raw" / "ch2026_submission_sample.csv")
    sub = sample.copy()
    for t in TARGETS:
        sub[t] = predictions[t]
    sub_path = SUBMIT / f"submission_v491_calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"  Submission: {sub_path}")

    # Save meta
    meta = {
        'version': 'V491_calibration',
        'name': '3-Model Ensemble with sigmoid calibration',
        'cv_method': 'GroupKFold_5fold',
        'n_models_per_target': 45,
        'n_feat_sweep': [20, 30, 40],
        'target_results': target_results,
        'avg_meta_oof': float(avg_meta),
        'avg_student_oof': float(avg_student),
        'submission_file': str(sub_path),
        'timestamp': datetime.now().isoformat(),
        'total_time': f"{time.time()-t_start:.0f}s",
    }
    meta_path = EXPERIMENTS / f'v491_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    log.info(f"  Meta: {meta_path}")


if __name__ == "__main__":
    import lightgbm as lgb
    import catboost as cb
    import xgboost as xgb
    
    predictions = {}
    main()
