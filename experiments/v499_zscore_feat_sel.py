#!/usr/bin/env python3
"""
V499 — Per-Subject Z-Score + Feature Selection + Optimized Config
Building on V496 success: use per-subject z-score, add per-target feature selection
and optimized model configs.
"""
import sys, gc, logging, json, re, time, warnings, os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data_processed'
SUBMIT = ROOT / 'submissions'
SUBMIT.mkdir(exist_ok=True)

TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']

def per_subject_zscore(df, feature_cols):
    """Per-subject z-score normalization (same as V496)."""
    result = df[feature_cols].copy()
    for col in feature_cols:
        for subj, grp in df.groupby('subject_id')[col]:
            mask = (df['subject_id'] == subj)
            mean = grp.mean()
            std = grp.std(ddof=0)
            if std < 1e-8:
                result.loc[mask, col] = 0.0
            else:
                result.loc[mask, col] = (grp - mean) / std
    return result

def main():
    t0 = time.time()
    log.info("=== V499: Per-Subject Z-Score + Feature Selection + Optimized ===")
    
    train = pd.read_parquet(DATA / 'features.parquet')
    test = pd.read_parquet(DATA / 'test_features.parquet')
    log.info(f"Loaded train={len(train)}, test={len(test)}")
    
    # Feature columns (same as V496)
    meta_cols = ['subject_id', 'lifelog_date', 'sleep_date']
    feature_cols = [c for c in train.columns 
                    if c not in TARGETS and c not in meta_cols
                    and np.issubdtype(train[c].dtype, np.number)]
    log.info(f"Feature count: {len(feature_cols)}")
    
    # Per-subject z-score
    log.info("Applying per-subject z-score...")
    train_zscore = per_subject_zscore(train, feature_cols)
    test_zscore = per_subject_zscore(test, feature_cols)
    
    # Original features standardized globally
    scaler = StandardScaler()
    X_orig_train = scaler.fit_transform(train[feature_cols].values)
    X_orig_test = scaler.transform(test[feature_cols].values)
    
    # Combined: z-score + original (282 features)
    X_combined_train = np.hstack([train_zscore.values, X_orig_train])
    X_combined_test = np.hstack([test_zscore.values, X_orig_test])
    # Sanitized feature names for LGBM/XGB (no special chars)
    def sanitize(name):
        return name.replace('-', '_').replace('(', '').replace(')', '').replace(' ', '_').replace('/', '_').replace('.', '_')
    all_feat_names = [f"z_{sanitize(c)}" for c in feature_cols] + [f"o_{sanitize(c)}" for c in feature_cols]
    
    gkf = GroupKFold(n_splits=5)
    
    # Optimized configs based on V496 insights
    lgb_configs = [
        {'num_leaves': 15, 'learning_rate': 0.05, 'n_estimators': 500, 'min_child_samples': 10,
         'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
         'boosting_type': 'gbdt', 'random_state': 42},
        {'num_leaves': 10, 'learning_rate': 0.03, 'n_estimators': 800, 'min_child_samples': 15,
         'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 0.5, 'reg_lambda': 0.5,
         'boosting_type': 'dart', 'random_state': 100},
    ]
    
    xgb_configs = [
        {'max_depth': 4, 'learning_rate': 0.05, 'n_estimators': 500, 'subsample': 0.8,
         'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
         'random_state': 300},
    ]
    
    cb_configs = [
        {'iterations': 500, 'learning_rate': 0.05, 'depth': 4, 'l2_leaf_reg': 3,
         'random_strength': 1, 'subsample': 0.8, 'colsample_bylevel': 0.8,
         'random_seed': 500},
    ]
    
    all_preds = {}
    all_results = {}
    
    for target in TARGETS:
        t1 = time.time()
        if target not in train.columns:
            continue
        
        log.info(f"--- {target} (rate={train[target].mean():.3f}) ---")
        y = train[target].values.astype(np.float64)
        groups = train['subject_id'].values
        
        # Feature selection using LGBM importance (quick rank)
        # Use only z-score features first
        X_z = X_combined_train[:, :len(feature_cols)]
        X_z_test = X_combined_test[:, :len(feature_cols)]
        
        # Quick feature importance ranking (no feature_name to avoid special char issues)
        tr_data = lgb.Dataset(X_z, label=y)
        va_idx = list(gkf.split(X_z, y, groups))[0][1]
        va_data = lgb.Dataset(X_z[va_idx], label=y[va_idx], reference=tr_data)
        
        quick_model = lgb.train(
            {'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
             'num_leaves': 15, 'learning_rate': 0.05, 'n_estimators': 50,
             'boosting_type': 'gbdt', 'random_state': 42},
            tr_data, num_boost_round=50, valid_sets=[va_data],
            callbacks=[lgb.log_evaluation(0)]
        )
        importance = quick_model.feature_importance(importance_type='gain')
        rank = np.argsort(-importance)
        
        # Try different K values
        best_oof = 999
        best_k = len(feature_cols)
        best_preds_val = None
        best_preds_test = None
        best_oof_full = np.zeros(len(y))
        best_test_preds_full = np.zeros(len(X_z_test))
        
        for K in [30, 50, 80, 100, 141]:
            top_idx = rank[:K]
            X_sel = X_z[:, top_idx]
            X_sel_test = X_z_test[:, top_idx]
            
            oof_cur = np.zeros(len(y))
            test_cur = np.zeros(len(X_sel_test))
            n_m = 0
            
            for fold_idx, (tr_idx, va_idx2) in enumerate(gkf.split(X_sel, y, groups)):
                X_tr, X_va = X_sel[tr_idx], X_sel[va_idx2]
                y_tr, y_va = y[tr_idx], y[va_idx2]
                fold_test = np.zeros(len(X_sel_test))
                
                for cfg in lgb_configs:
                    tr_d = lgb.Dataset(X_tr, label=y_tr)
                    va_d = lgb.Dataset(X_va, label=y_va, reference=tr_d)
                    params = {'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1, **cfg}
                    mdl = lgb.train(params, tr_d, num_boost_round=cfg['n_estimators'],
                                    valid_sets=[va_d],
                                    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
                    oof_cur[va_idx2] += mdl.predict(X_va)
                    fold_test += mdl.predict(X_sel_test)
                    n_m += 1
                
                for cfg in xgb_configs:
                    dtrain = xgb.DMatrix(X_tr, label=y_tr)
                    dval = xgb.DMatrix(X_va, label=y_va)
                    dtest = xgb.DMatrix(X_sel_test)
                    params = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'use_label_encoder': False, **cfg}
                    mdl = xgb.train(params, dtrain, num_boost_round=cfg['n_estimators'],
                                    evals=[(dval, 'val')], early_stopping_rounds=50, verbose_eval=False)
                    oof_cur[va_idx2] += mdl.predict(dval)
                    fold_test += mdl.predict(dtest)
                    n_m += 1
                
                for cfg in cb_configs:
                    mdl = cb.CatBoostClassifier(**cfg)
                    mdl.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=False)
                    oof_cur[va_idx2] += mdl.predict_proba(X_va)[:, 1]
                    fold_test += mdl.predict_proba(X_sel_test)[:, 1]
                    n_m += 1
                
                oof_cur[va_idx2] /= n_m
                test_cur += fold_test / n_m
            
            test_cur /= 5
            oof_clamped = np.clip(oof_cur, 1e-5, 1-1e-5)
            oof_score = log_loss(y, oof_clamped)
            log.info(f"  K={K:3d}: LogLoss={oof_score:.4f}")
            
            if oof_score < best_oof:
                best_oof = oof_score
                best_k = K
                best_oof_full = oof_cur.copy()
                best_test_preds_full = test_cur.copy()
        
        log.info(f"  Best K={best_k}, LogLoss={best_oof:.4f}")
        
        # Final evaluation with best K
        oof_auc = roc_auc_score(y, best_oof_full)
        all_preds[target] = np.clip(best_test_preds_full, 0.0001, 0.9999)
        all_results[target] = {'oof_auc': oof_auc, 'oof_ll': best_oof, 'best_k': best_k, 'time': time.time()-t1}
        log.info(f"  {target}: AUC={oof_auc:.4f}, LL={best_oof:.4f}, K={best_k}, Time={time.time()-t1:.0f}s")
    
    # Create submission
    sample = pd.read_csv(ROOT / 'data_raw' / 'ch2026_submission_sample.csv')
    sub = pd.DataFrame({'subject_id': sample['subject_id'].values})
    for target in TARGETS:
        sub[target] = all_preds.get(target, 0.5)
    
    fname = f"submission_v499_zscore_feat_sel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fpath = os.path.join(SUBMIT, fname)
    sub.to_csv(fpath, index=False)
    log.info(f"\n✅ Submission: {fpath}")
    log.info(f"Total time: {time.time()-t0:.0f}s")
    
    # Summary
    log.info("\n=== V499 SUMMARY ===")
    avg_ll = np.mean([v['oof_ll'] for v in all_results.values()])
    log.info(f"AVG OOF LogLoss: {avg_ll:.4f}")
    for t in TARGETS:
        if t in all_results:
            r = all_results[t]
            log.info(f"  {t}: AUC={r['oof_auc']:.4f}, LL={r['oof_ll']:.4f}, K={r['best_k']}")

if __name__ == '__main__':
    from datetime import datetime
    main()
