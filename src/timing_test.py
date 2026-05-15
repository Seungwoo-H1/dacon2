"""
Quick timing test: one fold of LGBM with 138 features, 500 est.
"""
import time, sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

feat = pd.read_parquet('data_processed/features.parquet')
TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']

def get_cols(f):
    return [c for c in f.columns if c not in {'subject_id','lifelog_date','sleep_date','date'} | set(TARGETS) and f[c].dtype in [np.float64,np.int64,float,int,bool,np.bool_]]

import re
SAN = re.compile(r'[^a-zA-Z0-9_]')
def san(n): return SAN.sub('_', n)

LEAK_Q = {'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max','wHr_hr_median','wHr_hr_count'}
LEAK_S = {"wLight_w_light_mean","wLight_w_light_std","wLight_w_light_min","wLight_w_light_max","wLight_w_light_count",
    "wHr_hr_mean","wHr_hr_std","wHr_hr_min","wHr_hr_max","wHr_hr_median","wHr_hr_count",
    "wPedo_pedo_step_mean","wPedo_pedo_step_sum","wPedo_pedo_step_frequency_mean","wPedo_pedo_step_frequency_sum",
    "wPedo_pedo_running_step_mean","wPedo_pedo_running_step_sum","wPedo_pedo_walking_step_mean","wPedo_pedo_walking_step_sum",
    "wPedo_pedo_distance_mean","wPedo_pedo_distance_sum","wPedo_pedo_speed_mean","wPedo_pedo_speed_sum",
    "wPedo_pedo_burned_calories_mean","wPedo_pedo_burned_calories_sum"}

cfg = {'objective':'binary','metric':'binary_logloss','verbose':-1,
       'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':500,
       'subsample':0.7,'colsample_bytree':0.7,'reg_alpha':1.0,'reg_lambda':3.0,
       'min_child_samples':10,'force_row_wise':True,'n_jobs':-1}

for target in TARGETS:
    leak = get_cols(feat)
    if target.startswith('S'): leak = [c for c in leak if c not in LEAK_S]
    elif target.startswith('Q'): leak = [c for c in leak if c not in LEAK_Q]
    y = feat[target].values
    gkf = GroupKFold(n_splits=5)
    for fold, (ti, vi) in enumerate(gkf.split(feat, y, feat['subject_id'])):
        Xtr = feat.iloc[ti][leak].fillna(0).values.astype(np.float32)
        Xva = feat.iloc[vi][leak].fillna(0).values.astype(np.float32)
        sn = [san(c) for c in leak]
        trd = lgb.Dataset(Xtr, label=y[ti], feature_name=sn)
        vad = lgb.Dataset(Xva, label=y[vi], feature_name=sn, reference=trd)
        t0 = time.time()
        mdl = lgb.train(cfg, trd, num_boost_round=500, valid_sets=[vad],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        elapsed = time.time() - t0
        print(f'{target} fold {fold}: {elapsed:.1f}s (est per-target×20seeds: {elapsed*20:.0f}s, all 10 combos: {elapsed*200:.0f}s)')
        break
