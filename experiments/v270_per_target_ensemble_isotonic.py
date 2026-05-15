"""
V270: Per-Target Feature Selection + Isotonic Calibration + Multi-n_feat Ensemble
Date: 2026-05-15

Key insight: Isotonic calibration dramatically improves OOF.
Ensembling different feature counts + isotonic gives best result.

Results:
  n_feat=30 + Iso: 0.5943 (best single model)
  Ensemble [all n_feat 15-40] + Iso: 0.5872 (best overall)
  Subject baseline (LOO): 0.5936

Method:
  1. Per-target feature ranking via LGBM gain importance
  2. Train per-target LGBM with selected features (n_feat = 15,20,25,30,35,40)
  3. Apply isotonic calibration to each OOF prediction
  4. Average OOF across all n_feat values
  5. Clip predictions to [0.0001, 0.9999]

Gap analysis: LB ≈ OOF + 0.105
  est_LB = 0.5872 + 0.105 = 0.692
  Target LB 0.50 → OOF 0.395 (gap 0.192 from best)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

df = pd.read_parquet('/home/mwoo423/projects/dacon2/data_processed/features_clean_v60.parquet')

META = {'subject_id','lifelog_date','sleep_date','date'}
TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']

feat_cols = [c for c in df.columns if c not in META | set(TARGETS) 
             and df[c].dtype in [np.float64,np.int64,float,int,bool]]

y_dict = {t: df[t].values.astype(np.float64) for t in TARGETS}
gkf = GroupKFold(n_splits=5)

all_zscore = [c for c in df.columns if c.endswith('_zscore') and c not in META | set(TARGETS)]
feat_all_combined = feat_cols + all_zscore
X_all = df[feat_all_combined].fillna(0).values.astype(np.float64)

# Per-target feature ranking
target_feature_scores = {}
for t in TARGETS:
    y = y_dict[t]
    spw = max((y==0).sum() / max((y==1).sum(), 1), 0.1)
    params = {
        'objective':'binary','metric':'binary_logloss','verbose':-1,
        'num_leaves':10,'max_depth':3,'learning_rate':0.05,'n_estimators':100,
        'subsample':0.8,'colsample_bytree':0.8,'reg_alpha':2.0,'reg_lambda':5.0,
        'scale_pos_weight':spw,'random_state':42,'min_child_samples':15,
        'force_row_wise':True,'n_jobs':1
    }
    ds = lgb.Dataset(X_all, label=y, params={'verbose':'-1'})
    m = lgb.train(params, ds, num_boost_round=100)
    imp = m.feature_importance(importance_type='gain')
    ranked = sorted(zip(feat_all_combined, imp), key=lambda x: -x[1])
    target_feature_scores[t] = [r[0] for r in ranked]

# Train with different n_feat values
n_feats_list = [15, 20, 25, 30, 35, 40]
oof_dict = {}
for n_feat in n_feats_list:
    oof_dict[n_feat] = {}
    for t in TARGETS:
        y = y_dict[t]
        sel_cols = target_feature_scores[t][:n_feat]
        X_sel = df[sel_cols].fillna(0).values.astype(np.float64)
        spw = max((y==0).sum() / max((y==1).sum(), 1), 0.1)
        
        oof = np.zeros(len(y))
        for tr_i, va_i in gkf.split(X_sel, y, df['subject_id'].values):
            params = {
                'objective':'binary','metric':'binary_logloss','verbose':-1,
                'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':1000,
                'subsample':0.7,'colsample_bytree':0.6,'reg_alpha':0.5,'reg_lambda':2.0,
                'scale_pos_weight':spw,'random_state':42,'min_child_samples':15,
                'force_row_wise':True,'n_jobs':1
            }
            ds = lgb.Dataset(X_sel[tr_i], label=y[tr_i], params={'verbose':'-1'})
            m = lgb.train(params, ds, num_boost_round=1000)
            oof[va_i] = m.predict(X_sel[va_i])
        
        iso = IsotonicRegression(y_min=0.0001, y_max=0.9999, out_of_bounds='clip')
        iso.fit(oof, y)
        oof_dict[n_feat][t] = iso.predict(oof)

# Results
print('=== Per-Target + Isotonic Results ===')
for n_feat in n_feats_list:
    scores = []
    for t in TARGETS:
        ll = log_loss(y_dict[t], np.clip(oof_dict[n_feat][t], 0.0001, 0.9999))
        scores.append(ll)
    print(f'  n_feat={n_feat}: AVG={np.mean(scores):.4f}')

# Ensemble
print('\n=== Ensemble ===')
for n_feat_combo in [n_feats_list, [15,20,25,30,35], [20,25,30], [25,30,35], [30,35,40]]:
    scores = []
    for t in TARGETS:
        preds = [oof_dict[nf][t] for nf in n_feat_combo]
        avg_pred = np.mean(preds, axis=0)
        ll = log_loss(y_dict[t], np.clip(avg_pred, 0.0001, 0.9999))
        scores.append(ll)
    print(f'  {n_feat_combo}: AVG={np.mean(scores):.4f}')
