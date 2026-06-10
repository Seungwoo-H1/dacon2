#!/usr/bin/env python3
"""
V501 — V496 pattern, n_estimators=300, seeds=5, CB no use_best_model
Faster version of V500 that won't hang.
"""
import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import GroupKFold
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    import xgboost as xgb
    import catboost as cb
except ImportError:
    print("ERROR: Required packages not installed")
    sys.exit(1)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data_processed'
SUBMIT = ROOT / 'submissions'
SUBMIT.mkdir(exist_ok=True)

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
SEEDS = [42, 123, 456, 789, 101]  # 5 seeds
N_FEAT = 40

def sanitize(n):
    return re.sub(r'[^a-zA-Z0-9_]','_', n)

def logloss(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def per_subject_zscore(df, feature_cols):
    result = df[feature_cols].copy()
    for col in feature_cols:
        for subj, grp in df.groupby('subject_id')[col]:
            mean = grp.mean()
            std = grp.std()
            if std < 1e-8:
                result.loc[grp.index, col] = 0
            else:
                result.loc[grp.index, col] = (grp.values - mean) / std
    return result

def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("V501 — V496 Pattern, n_est=300, 5 seeds, CB no_best_model")
    log.info("=" * 70)

    log.info("\n--- 1. Load data ---")
    train = pd.read_parquet(DATA / "features.parquet")
    test = pd.read_parquet(DATA / "test_features.parquet")

    target_cols_set = set(TARGETS)
    meta_cols = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}
    feature_cols_all = [c for c in train.columns
                        if c not in target_cols_set and c not in meta_cols
                        and np.issubdtype(train[c].dtype, np.number)]
    test_numeric = [c for c in test.columns if np.issubdtype(test[c].dtype, np.number)]
    leak_cols = [c for c in feature_cols_all if c in test_numeric]
    
    LEAK_REMOVE = {'wHr_hr_median', 'wLight_w_light_sum', 'mActivity_m_activity_sum'}
    leak_cols = [c for c in leak_cols if c not in LEAK_REMOVE]
    
    train = train[leak_cols + list(target_cols_set) + ['subject_id']]
    test = test[['subject_id'] + leak_cols] if 'subject_id' in test.columns else test[leak_cols]
    
    log.info(f"  Train: {train.shape}, Test: {test.shape}, Features: {len(leak_cols)}")

    groups = train['subject_id'].values
    gkf = GroupKFold(n_splits=5)

    log.info("\n--- 2. Per-subject z-score normalization ---")
    train_z = per_subject_zscore(train, leak_cols)
    test_z = per_subject_zscore(test, leak_cols)
    
    train_orig = train[leak_cols].fillna(0).values.astype(np.float64)
    test_orig = test[leak_cols].fillna(0).values.astype(np.float64)
    train_z_vals = train_z.fillna(0).values.astype(np.float64)
    test_z_vals = test_z.fillna(0).values.astype(np.float64)

    X_train = np.hstack([train_orig, train_z_vals])
    X_test = np.hstack([test_orig, test_z_vals])
    
    sn = [sanitize(c) for c in (leak_cols + [f'{c}_zscore' for c in leak_cols])]
    log.info(f"  Combined: {X_train.shape[1]} features")

    # Feature ranking
    log.info("\n--- 3. Feature ranking ---")
    predictions = {}
    target_results = {}

    for target in TARGETS:
        t1 = time.time()
        log.info(f"\n--- {target} (rate={train[target].mean():.3f}) ---")
        y = train[target].values.astype(np.float64)
        
        # Quick feature ranking (20 rounds on raw features only, 141 not 282)
        log.info("  Ranking...")
        params_rank = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
            'num_leaves': 15, 'max_depth': 3, 'learning_rate': 0.05,
            'n_estimators': 20, 'subsample': 0.7, 'colsample_bytree': 0.7,
            'reg_alpha': 1.0, 'reg_lambda': 5.0,
            'random_state': 42, 'min_child_samples': 10,
        }
        ds_rank = lgb.Dataset(X_train, label=y, params={'verbose': '-1'})
        model_rank = lgb.train(params_rank, ds_rank, num_boost_round=20)
        imp = model_rank.feature_importance(importance_type='gain')
        ranked_idx = np.argsort(-imp)
        log.info("  Rank done")

        top_idx = ranked_idx[:N_FEAT]
        top_sn = [sn[i] for i in top_idx]
        X_top = X_train[:, top_idx]
        X_test_top = X_test[:, top_idx]

        spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
        
        lgb_cfg = {'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.02,
                    'n_estimators': 300, 'subsample': 0.7, 'colsample_bytree': 0.7,
                    'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_samples': 10,
                    'scale_pos_weight': spw, 'random_state': 42, 'verbose': -1}
        xgb_cfg = {'max_depth': 4, 'learning_rate': 0.02, 'n_estimators': 300,
                    'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 2.0,
                    'reg_lambda': 5.0, 'min_child_weight': 3, 'random_state': 42}
        cb_cfg = {'iterations': 300, 'learning_rate': 0.02, 'depth': 4,
                  'subsample': 0.7, 'colsample_bylevel': 0.7, 'l2_leaf_reg': 5.0,
                  'min_data_in_leaf': 10, 'random_state': 42, 'loss_function': 'Logloss',
                  'verbose': False, 'best_model_use_best_model': False}

        # OOF (seed=42 only for evaluation)
        oof_lgb = np.zeros(len(y))
        oof_xgb = np.zeros(len(y))
        oof_cb = np.zeros(len(y))
        
        for fold, (tr, va) in enumerate(gkf.split(X_top, y, groups)):
            ds_tr = lgb.Dataset(X_top[tr], label=y[tr], params={'verbose': '-1'})
            m_lgb = lgb.train(lgb_cfg, ds_tr, num_boost_round=300)
            oof_lgb[va] = m_lgb.predict(X_top[va])
            
            dtrain = xgb.DMatrix(X_top[tr], label=y[tr])
            dval = xgb.DMatrix(X_top[va], label=y[va])
            m_xgb = xgb.train(xgb_cfg, dtrain, num_boost_round=300,
                              evals=[(dval, 'val')], early_stopping_rounds=50, verbose_eval=False)
            oof_xgb[va] = m_xgb.predict(dval)
            
            m_cb = cb.CatBoostClassifier(**cb_cfg)
            m_cb.fit(X_top[tr], y[tr], eval_set=(X_top[va], y[va]),
                     early_stopping_rounds=50, verbose=False)
            oof_cb[va] = m_cb.predict_proba(X_top[va])[:, 1]

        oof_avg = (oof_lgb + oof_xgb + oof_cb) / 3.0
        avg_oof = logloss(y, oof_avg)
        log.info(f"  OOF: LGB={logloss(y, oof_lgb):.4f}, XGB={logloss(y, oof_xgb):.4f}, CB={logloss(y, oof_cb):.4f}, AVG={avg_oof:.4f}")

        # Final predictions with 5 seeds
        final_lgb = np.zeros(len(X_test_top))
        final_xgb = np.zeros(len(X_test_top))
        final_cb = np.zeros(len(X_test_top))
        n_div = len(SEEDS) * 5

        for seed in SEEDS:
            cfg_l = {**lgb_cfg, 'random_state': seed}
            cfg_x = {**xgb_cfg, 'random_state': seed}
            cfg_c = {**cb_cfg, 'random_state': seed}
            
            for fold, (tr, va) in enumerate(gkf.split(X_top, y, groups)):
                ds_tr = lgb.Dataset(X_top[tr], label=y[tr], params={'verbose': '-1'})
                m = lgb.train(cfg_l, ds_tr, num_boost_round=300)
                final_lgb += m.predict(X_test_top) / n_div
            
            for fold, (tr, va) in enumerate(gkf.split(X_top, y, groups)):
                dtrain = xgb.DMatrix(X_top[tr], label=y[tr])
                dtest = xgb.DMatrix(X_test_top)
                m = xgb.train(cfg_x, dtrain, num_boost_round=300, verbose_eval=False)
                final_xgb += m.predict(dtest) / n_div
            
            for fold, (tr, va) in enumerate(gkf.split(X_top, y, groups)):
                m = cb.CatBoostClassifier(**cfg_c, best_model_use_best_model=False)
                m.fit(X_top[tr], y[tr], verbose=False)
                final_cb += m.predict_proba(X_test_top)[:, 1] / n_div

        test_preds = (final_lgb + final_xgb + final_cb) / 3.0
        predictions[target] = np.clip(test_preds, 0.0001, 0.9999)

        target_results[target] = {
            'n_feat': N_FEAT,
            'avg_oof': float(avg_oof),
            'lgb_oof': float(logloss(y, oof_lgb)),
            'xgb_oof': float(logloss(y, oof_xgb)),
            'cb_oof': float(logloss(y, oof_cb)),
            'time': time.time() - t1,
        }
        log.info(f"  {target}: AVG={avg_oof:.4f}, Time={time.time()-t1:.0f}s")
        gc.collect()

    # Summary
    avg_oof = np.mean([v['avg_oof'] for v in target_results.values()])
    log.info(f"\n{'='*70}")
    log.info("V501 RESULTS")
    log.info(f"{'='*70}")
    for t in TARGETS:
        r = target_results[t]
        log.info(f"  {t}: LL={r['avg_oof']:.4f}, LGB={r['lgb_oof']:.4f}, XGB={r['xgb_oof']:.4f}, CB={r['cb_oof']:.4f}, T={r['time']:.0f}s")
    log.info(f"  AVG OOF: {avg_oof:.4f}")
    log.info(f"  Total: {time.time()-t_start:.0f}s")

    # Save submission
    sample = pd.read_csv(ROOT / "data_raw" / "ch2026_submission_sample.csv")
    sub = sample.copy()
    for t in TARGETS:
        sub[t] = predictions[t]
    sub_path = SUBMIT / f"submission_v501_nest300_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"  Submission: {sub_path}")

if __name__ == "__main__":
    main()
