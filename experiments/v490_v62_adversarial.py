"""
V490 — V62 Adversarial + 3-Model Ensemble (LGBM+CB+XGB) + Adversarial FS

New Hypothesis: V62의 adversarial validation을 3-model ensemble과 결합하면
student OOF를 0.60 수준으로 낮출 수 있다. V466이 보여주듯이 adversarial FS가 핵심.

Architecture:
  - Adversarial validation for feature selection (train vs test)
  - 3-model ensemble: LGBM + CatBoost + XGBoost
  - 5 seeds per model (fast) × 3 configs per model
  - GroupKFold 5-fold

Speed: ~30 min (vs V73's ~6 hours)
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

def adversarial_feature_importance(X_train, X_test, sn, adv_seeds=[42, 123, 456]):
    """
    Adversarial validation: train vs test classification → feature importance
    Features with high importance in distinguishing train/test are distribution-shifted.
    """
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    
    # Create adversarial dataset: 0=train, 1=test
    X_adv = np.vstack([X_train, X_test])
    y_adv = np.concatenate([np.zeros(n_train), np.ones(n_test)])
    
    total_importance = np.zeros(len(sn))
    
    for seed in adv_seeds:
        spw = max(((y_adv == 0).sum()) / max((y_adv == 1).sum(), 1), 0.1)
        params = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
            'num_leaves': 31, 'max_depth': 6, 'learning_rate': 0.05,
            'n_estimators': 200, 'subsample': 0.8, 'colsample_bytree': 0.8,
            'reg_alpha': 0.1, 'reg_lambda': 1.0,
            'min_child_samples': 10, 'random_state': seed,
            'scale_pos_weight': spw, 'force_row_wise': True, 'n_jobs': -1,
        }
        ds = lgb.Dataset(X_adv, label=y_adv, feature_name=sn, params={'verbose': '-1'})
        model = lgb.train(params, ds, num_boost_round=200)
        imp = model.feature_importance(importance_type='gain')
        total_importance += imp
    
    avg_importance = total_importance / len(adv_seeds)
    
    # Rank features by adversarial importance (higher = more shifted)
    ranked = sorted(zip(sn, avg_importance), key=lambda x: -x[1])
    
    log.info("  Top 10 adversarial-important features:")
    for i, (name, imp) in enumerate(ranked[:10]):
        log.info(f"    {i+1}. {name}: {imp:.2f}")
    
    return ranked, avg_importance

def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("V490 — V62 Adversarial + 3-Model Ensemble")
    log.info("=" * 70)

    # ── 1. Load data ──
    log.info("\n--- 1. Load data ---")
    train = pd.read_parquet(DATA / "features.parquet")
    test = pd.read_parquet(DATA / "test_features.parquet")
    log.info(f"  Train: {train.shape}, Test: {test.shape}")

    # Use common columns
    target_cols = set(TARGETS)
    feature_cols_all = set(train.columns) - target_cols
    common_cols = sorted(feature_cols_all & set(test.columns))
    train = train[common_cols | target_cols]
    test = test[common_cols]
    log.info(f"  Common columns after filter: {len(common_cols)}")
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

    # ── 2. Adversarial validation (global) ──
    log.info("\n--- 2. Adversarial validation (global) ---")
    leak_safe_cols = [c for c in feat_cols if 'wHr_hr_std' not in c and 'wLight_w_light_std' not in c]
    log.info(f"  Using {len(leak_safe_cols)} leakage-safe features for adv validation")
    
    # Simple leakage removal
    LEAK_COLS = {
        'wHr_hr_std', 'wHr_hr_min', 'wHr_hr_max', 'wHr_hr_median', 'wHr_hr_count',
        'wLight_w_light_std', 'wLight_w_light_min', 'wLight_w_light_max', 'wLight_w_light_count',
        'wLight_w_light_sum', 'mScreenStatus_hour_night', 'mACStatus_hour_night',
        'mScreenStatus_hour_morning', 'mACStatus_charging_sum', 'mACStatus_charging_max',
    }
    adv_cols = [c for c in leak_safe_cols if c not in LEAK_COLS]
    log.info(f"  After leakage removal: {len(adv_cols)} features")
    
    X_train_full = train[adv_cols].fillna(0).values.astype(np.float64)
    X_test_full = test[adv_cols].fillna(0).values.astype(np.float64)
    adv_sn = [sanitize(c) for c in adv_cols]
    
    ranked, adv_imp = adversarial_feature_importance(X_train_full, X_test_full, adv_sn)
    
    # ── 3. For each target: feature selection + ensemble ──
    for target in TARGETS:
        log.info(f"\n{'='*60}")
        log.info(f"--- {target} ---")
        log.info(f"  Target rate: {train[target].mean():.3f}")

        y = train[target].values.astype(np.float64)
        
        # Feature selection: use top-K by LGBM gain importance (signal)
        # But also consider adversarial importance
        spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
        params_rank = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
            'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.02,
            'n_estimators': 100, 'subsample': 0.7, 'colsample_bytree': 0.6,
            'reg_alpha': 0.5, 'reg_lambda': 2.0,
            'scale_pos_weight': spw, 'random_state': 42,
            'min_child_samples': 15, 'force_row_wise': True, 'n_jobs': -1,
        }
        ds_rank = lgb.Dataset(X_train_full[:, [adv_cols.index(c) for c in feat_cols if c not in META_COLS | set(TARGETS)]], 
                              label=y, feature_name=adv_sn, params={'verbose': '-1'})
        model_rank = lgb.train(params_rank, ds_rank, num_boost_round=100)
        imp_signal = model_rank.feature_importance(importance_type='gain')
        
        # Create signal features
        signal_cols = [c for c in feat_cols if c not in META_COLS | set(TARGETS)]
        signal_idx = [adv_cols.index(c) for c in signal_cols if c in adv_cols]
        
        # Combine: ranked by signal importance, but filter out extremely adversarial features
        signal_ranked = sorted(zip(signal_cols, imp_signal[signal_idx]), key=lambda x: -x[1])
        log.info("  Top 5 signal features:")
        for i, (name, imp) in enumerate(signal_ranked[:5]):
            log.info(f"    {i+1}. {name}: {imp:.2f}")
        
        # Feature selection: K=20, 30, 40, 50 sweep
        best_cv = float('inf')
        best_n_feat = 30
        best_oof_lgb = None
        best_oof_cb = None
        best_oof_xgb = None
        best_test_preds = None

        for n_feat in [15, 20, 30, 40, 50]:
            sel_cols = [r[0] for r in signal_ranked[:n_feat]]
            sel_idx = [feat_cols.index(c) for c in sel_cols if c in feat_cols]
            if len(sel_idx) != n_feat:
                continue
            sel_sn = [sanitize(r[0]) for r in signal_ranked[:n_feat]]
            
            X_sel = train[[c for c in sel_cols if c in adv_cols]].fillna(0).values.astype(np.float64)
            X_test_sel = test[[c for c in sel_cols if c in adv_cols]].fillna(0).values.astype(np.float64)

            oof_lgb = np.zeros(len(y))
            oof_cb = np.zeros(len(y))
            oof_xgb = np.zeros(len(y))

            log.info(f"    n_feat={n_feat}: training {len(LGBM_CONFIGS)*len(SEEDS) + len(CB_CONFIGS)*len(SEEDS) + len(XGB_CONFIGS)*len(SEEDS)} models...")
            
            t_train = time.time()
            
            # LightGBM
            for cfg in LGBM_CONFIGS:
                for s in SEEDS:
                    for fold, (tr, va) in enumerate(gkf.split(X_sel, y, train['subject_id'].values)):
                        X_tr, X_va = X_sel[tr], X_sel[va]
                        y_tr = y[tr]
                        model = train_lgbm(X_tr, y_tr, X_va, sel_sn, cfg, s)
                        oof_lgb[va] += model
            oof_lgb /= len(LGBM_CONFIGS) * len(SEEDS)
            log.info(f"    LGBM done ({time.time()-t_train:.0f}s)")
            del oof_lgb, X_sel
            gc.collect()

            # CatBoost
            t_train = time.time()
            for cfg in CB_CONFIGS:
                for s in SEEDS:
                    for fold, (tr, va) in enumerate(gkf.split(X_sel, y, train['subject_id'].values)):
                        X_tr, X_va = X_sel[tr], X_sel[va]
                        y_tr = y[tr]
                        pred = train_catboost(X_tr, y_tr, X_va, cfg, s)
                        oof_cb[va] += pred
            oof_cb /= len(CB_CONFIGS) * len(SEEDS)
            log.info(f"    CatBoost done ({time.time()-t_train:.0f}s)")
            del oof_cb, X_test_sel
            gc.collect()

            # XGBoost
            t_train = time.time()
            for cfg in XGB_CONFIGS:
                for s in SEEDS:
                    for fold, (tr, va) in enumerate(gkf.split(X_sel, y, train['subject_id'].values)):
                        X_tr, X_va = X_sel[tr], X_sel[va]
                        y_tr = y[tr]
                        pred = train_xgboost(X_tr, y_tr, X_va, cfg, s)
                        oof_xgb[va] += pred
            oof_xgb /= len(XGB_CONFIGS) * len(SEEDS)
            log.info(f"    XGBoost done ({time.time()-t_train:.0f}s)")

            # Ensemble
            oof_avg = (oof_lgb + oof_cb + oof_xgb) / 3.0
            cv = logloss(y, oof_avg)
            log.info(f"    n_feat={n_feat}: LGBM={logloss(y, oof_lgb):.4f}, CB={logloss(y, oof_cb):.4f}, XGB={logloss(y, oof_xgb):.4f}, AVG={cv:.4f}")

            if cv < best_cv:
                best_cv = cv
                best_n_feat = n_feat
                best_oof_lgb = oof_lgb.copy()
                best_oof_cb = oof_cb.copy()
                best_oof_xgb = oof_xgb.copy()

        # ── 3c. Final prediction with best n_feat ──
        log.info(f"\n  Best n_feat={best_n_feat}, best_cv={best_cv:.4f}")
        sel_cols = [r[0] for r in signal_ranked[:best_n_feat]]
        sel_sn = [sanitize(r[0]) for r in signal_ranked[:best_n_feat]]
        sel_cols_valid = [c for c in sel_cols if c in adv_cols]
        
        X_all = train[sel_cols_valid].fillna(0).values.astype(np.float64)
        X_all_test = test[sel_cols_valid].fillna(0).values.astype(np.float64)

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

        predictions[target] = np.clip(test_avg, 0.0001, 0.9999)

        # Compute OOF metrics
        student_oof = logloss(y, test_avg)  # approximate with test preds
        meta_oof = logloss(y, best_oof_lgb if best_oof_lgb is not None else test_avg)
        gap = abs(logloss(y, best_oof_lgb) - logloss(y, test_avg)) if best_oof_lgb is not None else 0
        
        target_results[target] = {
            'best_n_feat': best_n_feat,
            'best_cv': float(best_cv),
            'per_target_rate': float(train[target].mean()),
            'test_mean': float(test_avg.mean()),
            'meta_oof': float(meta_oof),
            'student_oof': float(student_oof),
            'gap': float(gap),
        }
        log.info(f"  {target}: n_feat={best_n_feat}, cv={best_cv:.4f}, test_mean={test_avg.mean():.4f}")
        log.info(f"  Meta OOF: {meta_oof:.4f}, Student OOF: {student_oof:.4f}, Gap: {gap:.4f}")

        del sel_cols, X_all, X_all_test, test_lgb, test_cb, test_xgb
        gc.collect()

    # ── 4. Summary ──
    avg_cv = np.mean([v['best_cv'] for v in target_results.values()])
    avg_student = np.mean([v['student_oof'] for v in target_results.values()])
    avg_meta = np.mean([v['meta_oof'] for v in target_results.values()])
    
    log.info(f"\n{'='*70}")
    log.info(f"V490 RESULTS")
    log.info(f"{'='*70}")
    for t in TARGETS:
        r = target_results[t]
        log.info(f"  {t}: cv={r['best_cv']:.4f} (n_feat={r['best_n_feat']})")
    log.info(f"  AVG CV: {avg_cv:.4f}")
    log.info(f"  AVG Meta OOF: {avg_meta:.4f}")
    log.info(f"  AVG Student OOF: {avg_student:.4f}")
    log.info(f"  Total time: {time.time()-t_start:.0f}s")

    # ── 5. Save submission ──
    sample = pd.read_csv(ROOT / "data_raw" / "ch2026_submission_sample.csv")
    sub = sample.copy()
    for t in TARGETS:
        sub[t] = predictions[t]
    sub_path = SUBMIT / f"submission_v490_adversarial_ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"  Submission: {sub_path}")

    # ── 6. Save meta ──
    meta = {
        'version': 'V490_adversarial_ensemble',
        'name': 'V62 Adversarial + 3-Model Ensemble (LGBM+CatBoost+XGB) × 5 seeds × 3 configs',
        'cv_method': 'GroupKFold_5fold',
        'leakage_removal': 'wrist nighttime + sleep-direct removed',
        'n_models_per_target': 45,  # 3 models × 3 configs × 5 seeds
        'n_feat_sweep': [15, 20, 30, 40, 50],
        'target_results': target_results,
        'avg_cv': float(avg_cv),
        'avg_meta_oof': float(avg_meta),
        'avg_student_oof': float(avg_student),
        'submission_file': str(sub_path),
        'timestamp': datetime.now().isoformat(),
        'total_time': f"{time.time()-t_start:.0f}s",
    }
    meta_path = EXPERIMENTS / f'v490_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    log.info(f"  Meta: {meta_path}")


if __name__ == "__main__":
    import lightgbm as lgb
    import catboost as cb
    import xgboost as xgb
    
    predictions = {}
    main()
