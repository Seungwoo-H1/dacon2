#!/usr/bin/env python3
"""
V500 — Per-Subject Z-Score + 10 Seeds + Fixed K=40 + 3-Model Ensemble
Hypothesis: V496 pattern is solid. Just add more seeds for diversity.
Fixed n_feat=40 to speed up (skip sweep).
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
SEEDS = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]  # 10 seeds
N_FEAT = 40  # Fixed

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
    log.info("V500 — Per-Subject Z-Score + 10 Seeds + Fixed K=40")
    log.info("=" * 70)

    # ── 1. Load data ──
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
    
    # Remove leak columns
    LEAK_REMOVE = {'wHr_hr_median', 'wLight_w_light_sum', 'mActivity_m_activity_sum'}
    leak_cols = [c for c in leak_cols if c not in LEAK_REMOVE]
    
    train = train[leak_cols + list(target_cols_set) + ['subject_id']]
    test = test[['subject_id'] + leak_cols] if 'subject_id' in test.columns else test[leak_cols]
    
    log.info(f"  Train: {train.shape}, Test: {test.shape}")
    log.info(f"  Features: {len(leak_cols)}")

    groups = train['subject_id'].values
    gkf = GroupKFold(n_splits=5)

    # ── 2. Per-subject z-score normalization ──
    log.info("\n--- 2. Per-subject z-score normalization ---")
    train_z = per_subject_zscore(train, leak_cols)
    test_z = per_subject_zscore(test, leak_cols)
    
    train_orig = train[leak_cols].fillna(0).values.astype(np.float64)
    test_orig = test[leak_cols].fillna(0).values.astype(np.float64)
    train_z_vals = train_z.fillna(0).values.astype(np.float64)
    test_z_vals = test_z.fillna(0).values.astype(np.float64)

    X_train = np.hstack([train_orig, train_z_vals])
    X_test = np.hstack([test_orig, test_z_vals])
    
    feature_names = leak_cols + [f'{c}_zscore' for c in leak_cols]
    sn = [sanitize(c) for c in feature_names]
    
    log.info(f"  Combined features: {X_train.shape[1]}")

    # ── 3. Feature ranking (single pass) ──
    log.info("\n--- 3. Feature ranking ---")
    predictions = {}
    target_results = {}

    for target in TARGETS:
        t1 = time.time()
        log.info(f"\n--- {target} (rate={train[target].mean():.3f}) ---")
        y = train[target].values.astype(np.float64)
        
        # Feature ranking
        params_rank = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
            'num_leaves': 20, 'max_depth': 4, 'learning_rate': 0.03,
            'n_estimators': 50, 'subsample': 0.7, 'colsample_bytree': 0.7,
            'reg_alpha': 1.0, 'reg_lambda': 5.0,
            'random_state': 42, 'min_child_samples': 10,
        }
        ds_rank = lgb.Dataset(X_train, label=y, feature_name=sn, params={'verbose': '-1'})
        model_rank = lgb.train(params_rank, ds_rank, num_boost_round=50)
        imp = model_rank.feature_importance(importance_type='gain')
        ranked_idx = np.argsort(-imp)

        # Use fixed K
        top_idx = ranked_idx[:N_FEAT]
        top_sn = [sn[i] for i in top_idx]
        X_top = X_train[:, top_idx]
        X_test_top = X_test[:, top_idx]
        log.info(f"  Using K={N_FEAT} features")

        # ── 3-model ensemble with 10 seeds ──
        oof_lgb = np.zeros(len(y))
        oof_xgb = np.zeros(len(y))
        oof_cb = np.zeros(len(y))
        
        spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
        
        lgb_cfg = {
            'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.02,
            'n_estimators': 300, 'subsample': 0.7, 'colsample_bytree': 0.7,
            'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_samples': 10,
            'scale_pos_weight': spw, 'random_state': 42,
            'verbose': -1,
        }
        xgb_cfg = {
            'max_depth': 4, 'learning_rate': 0.02,
            'n_estimators': 300, 'subsample': 0.7, 'colsample_bytree': 0.7,
            'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_weight': 3,
            'random_state': 42, 'use_label_encoder': False,
            'eval_metric': 'logloss',
        }
        cb_cfg = {
            'iterations': 300, 'learning_rate': 0.02,
            'depth': 4, 'subsample': 0.7, 'colsample_bylevel': 0.7,
            'l2_leaf_reg': 5.0, 'min_data_in_leaf': 10,
            'random_state': 42, 'loss_function': 'Logloss',
            'verbose': False,
        }

        # OOF with first seed
        for fold, (tr, va) in enumerate(gkf.split(X_top, y, groups)):
            ds_tr = lgb.Dataset(X_top[tr], label=y[tr], feature_name=top_sn, params={'verbose': '-1'})
            m_lgb = lgb.train(lgb_cfg, ds_tr, num_boost_round=lgb_cfg['n_estimators'])
            oof_lgb[va] = m_lgb.predict(X_top[va])
            
            dtrain = xgb.DMatrix(X_top[tr], label=y[tr])
            dval = xgb.DMatrix(X_top[va], label=y[va])
            m_xgb = xgb.train(xgb_cfg, dtrain, num_boost_round=xgb_cfg['n_estimators'],
                              evals=[(dval, 'val')], early_stopping_rounds=50, verbose_eval=False)
            oof_xgb[va] = m_xgb.predict(dval)
            
            m_cb = cb.CatBoostClassifier(**cb_cfg, best_model_use_best_model=False)
            m_cb.fit(X_top[tr], y[tr], eval_set=(X_top[va], y[va]), early_stopping_rounds=50, verbose=False)
            oof_cb[va] = m_cb.predict_proba(X_top[va])[:, 1]

        oof_avg = (oof_lgb + oof_xgb + oof_cb) / 3.0
        avg_oof = logloss(y, oof_avg)
        log.info(f"  OOF: LGB={logloss(y, oof_lgb):.4f}, XGB={logloss(y, oof_xgb):.4f}, CB={logloss(y, oof_cb):.4f}, AVG={avg_oof:.4f}")

        # ── Final predictions with all 10 seeds ──
        final_lgb = np.zeros(len(X_test_top))
        final_xgb = np.zeros(len(X_test_top))
        final_cb = np.zeros(len(X_test_top))

        for seed in SEEDS:
            lgb_cfg_s = {**lgb_cfg, 'random_state': seed}
            xgb_cfg_s = {**xgb_cfg, 'random_state': seed}
            cb_cfg_s = {**cb_cfg, 'random_state': seed}
            
            for fold, (tr, va) in enumerate(gkf.split(X_top, y, groups)):
                ds_tr = lgb.Dataset(X_top[tr], label=y[tr], feature_name=top_sn, params={'verbose': '-1'})
                m = lgb.train(lgb_cfg_s, ds_tr, num_boost_round=lgb_cfg_s['n_estimators'])
                final_lgb += m.predict(X_test_top) / (len(SEEDS) * 5)
            
            for fold, (tr, va) in enumerate(gkf.split(X_top, y, groups)):
                dtrain = xgb.DMatrix(X_top[tr], label=y[tr])
                dtest = xgb.DMatrix(X_test[:, top_idx])
                m_xgb = xgb.train(xgb_cfg_s, dtrain, num_boost_round=xgb_cfg_s['n_estimators'], verbose_eval=False)
                final_xgb += m_xgb.predict(dtest) / (len(SEEDS) * 5)
            
            for fold, (tr, va) in enumerate(gkf.split(X_top, y, groups)):
                m_cb = cb.CatBoostClassifier(**cb_cfg_s, best_model_use_best_model=False)
                m_cb.fit(X_top[tr], y[tr], verbose=False)
                final_cb += m_cb.predict_proba(X_test[:, top_idx])[:, 1] / (len(SEEDS) * 5)

        test_preds = (final_lgb + final_xgb + final_cb) / 3.0
        predictions[target] = np.clip(test_preds, 0.0001, 0.9999)

        target_results[target] = {
            'n_feat': N_FEAT,
            'avg_oof': float(avg_oof),
            'lgb_oof': float(logloss(y, oof_lgb)),
            'xgb_oof': float(logloss(y, oof_xgb)),
            'cb_oof': float(logloss(y, oof_cb)),
            'n_seeds': len(SEEDS),
            'time': time.time() - t1,
        }
        log.info(f"  {target}: AVG_OOF={avg_oof:.4f}, Time={time.time()-t1:.0f}s")
        del X_top, X_test_top, final_lgb, final_xgb, final_cb, oof_lgb, oof_xgb, oof_cb
        gc.collect()

    # ── Summary ──
    avg_oof = np.mean([v['avg_oof'] for v in target_results.values()])
    log.info(f"\n{'='*70}")
    log.info("V500 RESULTS")
    log.info(f"{'='*70}")
    for t in TARGETS:
        r = target_results[t]
        log.info(f"  {t}: K={r['n_feat']}, LL={r['avg_oof']:.4f}, LGB={r['lgb_oof']:.4f}, XGB={r['xgb_oof']:.4f}, CB={r['cb_oof']:.4f}, Seeds={r['n_seeds']}, Time={r['time']:.0f}s")
    log.info(f"  AVG OOF LogLoss: {avg_oof:.4f}")
    log.info(f"  Total time: {time.time()-t_start:.0f}s")

    # ── Save submission ──
    sample = pd.read_csv(ROOT / "data_raw" / "ch2026_submission_sample.csv")
    sub = sample.copy()
    for t in TARGETS:
        sub[t] = predictions[t]
    sub_path = SUBMIT / f"submission_v500_10seeds_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"  Submission: {sub_path}")

    meta = {
        'version': 'V500_10seeds',
        'name': 'Per-Subject Z-Score + 3-Model Ensemble (10 seeds, K=40)',
        'avg_oof': float(avg_oof),
        'n_features_base': len(leak_cols),
        'n_features_combined': X_train.shape[1],
        'target_results': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv for kk, vv in v.items()} for k, v in target_results.items()},
        'submission_file': str(sub_path),
        'timestamp': datetime.now().isoformat(),
        'total_time': f"{time.time()-t_start:.0f}s",
    }
    meta_path = SUBMIT / f'meta_v500_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    log.info(f"  Meta: {meta_path}")
    log.info(f"  DONE.")

if __name__ == "__main__":
    main()
