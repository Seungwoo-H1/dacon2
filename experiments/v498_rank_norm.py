#!/usr/bin/env python3
"""
V498 — Per-Subject Rank Normalization + Quantile Transformer + 6-Config Ensemble
Hypothesis: Quantile (rank+gaussian) transform per subject > z-score
"""
import sys, gc, logging, json, re, time, warnings, os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, log_loss
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data_processed'
SUB_DIR = ROOT / 'submissions'
SUB_DIR.mkdir(exist_ok=True)

TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']

def per_subject_quantile_transform(train_df, test_df, feature_cols):
    """Per-subject quantile (rank+gaussian) normalization."""
    train_out = pd.DataFrame(index=train_df.index)
    test_out = pd.DataFrame(index=test_df.index)
    
    for col in feature_cols:
        if train_df[col].dtype == 'object' or not np.issubdtype(train_df[col].dtype, np.number):
            continue
        # Per-subject rank -> gaussian via QuantileTransformer
        q = QuantileTransformer(output_distribution='normal', random_state=42, n_quantiles=min(100, len(train_df)//5))
        
        # Fit per subject
        all_vals = []
        train_idx_list = []
        test_idx_list = []
        
        for subj_id, grp in train_df.groupby('subject_id'):
            subj_mask = train_df['subject_id'] == subj_id
            subj_train = grp[col].values
            
            # Get test subjects that overlap
            if subj_id in test_df['subject_id'].values:
                subj_test_mask = test_df['subject_id'] == subj_id
                
                # Fit on combined to maintain rank consistency
                combined = np.concatenate([subj_train, test_df.loc[subj_test_mask, col].values])
                combined = combined[~np.isnan(combined)]
                if len(combined) < 3:
                    train_out.loc[subj_mask, col] = 0.0
                    test_out.loc[subj_test_mask, col] = 0.0
                    continue
                q.fit(combined.reshape(-1, 1))
                
                train_vals = q.transform(subj_train.reshape(-1, 1)).ravel()
                test_vals = q.transform(test_df.loc[subj_test_mask, col].values.reshape(-1, 1)).ravel()
                
                # Clip extreme quantiles
                train_vals = np.clip(train_vals, -3, 3)
                test_vals = np.clip(test_vals, -3, 3)
                
                train_out.loc[subj_mask, col] = train_vals
                test_out.loc[subj_test_mask, col] = test_vals
            else:
                q.fit(subj_train.reshape(-1, 1))
                train_vals = q.transform(subj_train.reshape(-1, 1)).ravel()
                train_vals = np.clip(train_vals, -3, 3)
                train_out.loc[subj_mask, col] = train_vals
    
    return train_out, test_out

from sklearn.preprocessing import QuantileTransformer

def main():
    t0 = time.time()
    log.info("=== V498: Per-Subject Rank/Quantile Transform ===")
    
    # Load data
    train = pd.read_parquet(DATA / 'features.parquet')
    test = pd.read_parquet(DATA / 'test_features.parquet')
    log.info(f"Loaded train={len(train)}, test={len(test)}")
    log.info(f"Columns: {list(train.columns[:5])}... total={len(train.columns)}")
    
    # Target rate check
    for t in TARGETS:
        if t in train.columns:
            log.info(f"  {t} rate: {train[t].mean():.3f} (n_pos={train[t].sum():.0f})")
    
    # Feature cols
    target_cols_set = set(TARGETS)
    meta_cols = ['subject_id', 'id']
    feature_cols = [c for c in train.columns 
                    if c not in target_cols_set and c not in meta_cols
                    and np.issubdtype(train[c].dtype, np.number)]
    log.info(f"Feature count: {len(feature_cols)}")
    
    # Per-subject quantile transform
    log.info("Applying per-subject quantile transform...")
    train_q, test_q = per_subject_quantile_transform(train, test, feature_cols)
    
    # Fill any NaNs
    train_q = train_q.fillna(0.0)
    test_q = test_q.fillna(0.0)
    
    # Also keep original numeric features as-is (combine)
    log.info("Creating ensemble: quantile-transformed + original features...")
    
    all_preds = {}
    all_results = {}
    
    gkf = GroupKFold(n_splits=5)
    
    # Model configs for diversity
    lgb_configs = [
        {'num_leaves': 15, 'learning_rate': 0.05, 'n_estimators': 500, 'min_child_samples': 10,
         'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
         'boosting_type': 'gbdt', 'random_state': 42},
        {'num_leaves': 10, 'learning_rate': 0.03, 'n_estimators': 800, 'min_child_samples': 15,
         'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 0.5, 'reg_lambda': 0.5,
         'boosting_type': 'dart', 'random_state': 100},
        {'num_leaves': 20, 'learning_rate': 0.08, 'n_estimators': 400, 'min_child_samples': 8,
         'subsample': 0.9, 'colsample_bytree': 0.9, 'reg_alpha': 0.01, 'reg_lambda': 0.01,
         'boosting_type': 'gbdt', 'random_state': 200},
    ]
    
    xgb_configs = [
        {'max_depth': 4, 'learning_rate': 0.05, 'n_estimators': 500, 'subsample': 0.8,
         'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
         'random_state': 300},
        {'max_depth': 3, 'learning_rate': 0.03, 'n_estimators': 700, 'subsample': 0.7,
         'colsample_bytree': 0.6, 'reg_alpha': 0.5, 'reg_lambda': 0.5,
         'random_state': 400},
    ]
    
    cb_configs = [
        {'iterations': 500, 'learning_rate': 0.05, 'depth': 4, 'l2_leaf_reg': 3,
         'random_strength': 1, 'subsample': 0.8, 'colsample_bylevel': 0.8,
         'random_seed': 500},
        {'iterations': 400, 'learning_rate': 0.03, 'depth': 3, 'l2_leaf_reg': 5,
         'random_strength': 2, 'subsample': 0.7, 'colsample_bylevel': 0.6,
         'random_seed': 600},
    ]
    
    for target in TARGETS:
        t1 = time.time()
        if target not in train.columns:
            log.warning(f"  {target} not in train, skipping")
            continue
        
        log.info(f"--- {target} (rate={train[target].mean():.3f}) ---")
        y = train[target].values.astype(np.float64)
        groups = train['subject_id'].values
        
        # Prepare feature matrix (quantile + original combined)
        X_q = train_q[feature_cols].values.astype(np.float32)
        X_test_q = test_q[feature_cols].values.astype(np.float32)
        
        # Also add original features (scaled to similar range)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_orig = scaler.fit_transform(train[feature_cols].values.astype(np.float32))
        X_test_orig = scaler.transform(test[feature_cols].values.astype(np.float32))
        
        # Combined features
        X = np.hstack([X_q, X_orig])
        X_test = np.hstack([X_test_q, X_test_orig])
        log.info(f"  Combined feature dim: {X.shape[1]}")
        
        oof_preds = np.zeros(len(X))
        test_preds = np.zeros(len(X_test))
        n_models = 0
        model_oofs = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]
            
            fold_test = np.zeros(X_test.shape[0])
            
            # LGBM
            for cfg in lgb_configs:
                tr_data = lgb.Dataset(X_tr, label=y_tr, feature_name=[f"f{i}" for i in range(X.shape[1])])
                va_data = lgb.Dataset(X_va, label=y_va, feature_name=[f"f{i}" for i in range(X.shape[1])], reference=tr_data)
                params = {'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1, **cfg}
                model = lgb.train(params, tr_data, num_boost_round=cfg['n_estimators'],
                                  valid_sets=[va_data],
                                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
                oof_preds[val_idx] += model.predict(X_va)
                fold_test += model.predict(X_test)
                n_models += 1
            
            # XGB
            for cfg in xgb_configs:
                dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=[f"f{i}" for i in range(X.shape[1])])
                dval = xgb.DMatrix(X_va, label=y_va, feature_names=[f"f{i}" for i in range(X.shape[1])])
                dtest = xgb.DMatrix(X_test, feature_names=[f"f{i}" for i in range(X.shape[1])])
                params = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'use_label_encoder': False, **cfg}
                model = xgb.train(params, dtrain, num_boost_round=cfg['n_estimators'],
                                  evals=[(dval, 'val')],
                                  early_stopping_rounds=50, verbose_eval=False)
                oof_preds[val_idx] += model.predict(dval)
                fold_test += model.predict(dtest)
                n_models += 1
            
            # CatBoost
            for cfg in cb_configs:
                model = cb.CatBoostClassifier(**cfg)
                model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=False)
                oof_preds[val_idx] += model.predict_proba(X_va)[:, 1]
                fold_test += model.predict_proba(X_test)[:, 1]
                n_models += 1
            
            # Average within fold
            oof_preds[val_idx] /= n_models
            test_preds += fold_test / n_models
        
        # Average across folds
        test_preds /= 5
        oof_preds = np.clip(oof_preds, 1e-5, 1-1e-5)
        
        oof_auc = roc_auc_score(y, oof_preds)
        oof_ll = log_loss(y, oof_preds)
        all_preds[target] = np.clip(test_preds, 0.0001, 0.9999)
        all_results[target] = {'oof_auc': oof_auc, 'oof_ll': oof_ll, 'time': time.time()-t1}
        
        log.info(f"  OOF AUC: {oof_auc:.4f} | LogLoss: {oof_ll:.4f} | Time: {time.time()-t1:.0f}s")
    
    # Create submission
    sample = pd.read_csv(ROOT / 'data_raw' / 'ch2026_submission_sample.csv')
    sub = pd.DataFrame({'subject_id': sample['subject_id'].values})
    for target in TARGETS:
        sub[target] = all_preds.get(target, 0.5)
    
    fname = f"submission_v498_rank_norm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fpath = os.path.join(SUB_DIR, fname)
    sub.to_csv(fpath, index=False)
    log.info(f"✅ Submission: {fpath}")
    log.info(f"Total time: {time.time()-t0:.0f}s")
    
    # Summary
    log.info("\n=== V498 SUMMARY ===")
    avg_auc = np.mean([v['oof_auc'] for v in all_results.values()])
    log.info(f"AVG OOF AUC: {avg_auc:.4f}")
    for t in TARGETS:
        if t in all_results:
            log.info(f"  {t}: AUC={all_results[t]['oof_auc']:.4f}, LL={all_results[t]['oof_ll']:.4f}")

if __name__ == '__main__':
    from datetime import datetime
    main()
