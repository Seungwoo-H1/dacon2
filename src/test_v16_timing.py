"""Fast test: personalization + ranking timing."""
import sys, time, warnings, os
import numpy as np
import pandas as pd
import lightgbm as lgb
import re

os.environ['PYTHONUNBUFFERED'] = '1'
warnings.filterwarnings('ignore')

TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']
META_COLS = {"subject_id", "lifelog_date", "sleep_date", "date"}

def get_cols(f):
    return [c for c in f.columns
            if c not in META_COLS | set(TARGETS)
            and f[c].dtype in [np.float64,np.int64,float,int,bool,np.bool_]]

SAN = re.compile(r'[^a-zA-Z0-9_]')

LEAK_S = {"wLight_w_light_mean","wLight_w_light_std","wLight_w_light_min","wLight_w_light_max","wLight_w_light_count",
    "wHr_hr_mean","wHr_hr_std","wHr_hr_min","wHr_hr_max","wHr_hr_median","wHr_hr_count",
    "wPedo_pedo_step_mean","wPedo_pedo_step_sum","wPedo_pedo_step_frequency_mean","wPedo_pedo_step_frequency_sum",
    "wPedo_pedo_running_step_mean","wPedo_pedo_running_step_sum","wPedo_pedo_walking_step_mean","wPedo_pedo_walking_step_sum",
    "wPedo_pedo_distance_mean","wPedo_pedo_distance_sum","wPedo_pedo_speed_mean","wPedo_pedo_speed_sum",
    "wPedo_pedo_burned_calories_mean","wPedo_pedo_burned_calories_sum"}
LEAK_Q = {"wHr_hr_mean","wHr_hr_std","wHr_hr_min","wHr_hr_max","wHr_hr_median","wHr_hr_count"}

print("Loading...")
feat = pd.read_parquet('data_processed/features.parquet')
feat_cols = get_cols(feat)
print(f"Features: {len(feat_cols)}")

print("Personalization...")
t0 = time.time()
for col in feat_cols:
    filled = feat[col].fillna(0)
    grp = filled.groupby(feat['subject_id'])
    mean_s = grp.transform('mean')
    std_s = grp.transform('std').replace(0, 1)
    zscore = (filled - mean_s) / std_s
    feat = feat.assign(**{f'{col}_z': zscore})
elapsed = time.time() - t0
total_feats = len(get_cols(feat))
print(f"Personalization done: {elapsed:.1f}s → {total_feats} features total")

# Ranking per target
for target in TARGETS:
    leak = [c for c in feat_cols if c not in (LEAK_S if target.startswith('S') else LEAK_Q)]
    y = feat[target].values
    X = feat[leak].fillna(0).values
    n_pos = max((y==1).sum(), 1); n_neg = (y==0).sum()
    spw = n_neg / n_pos
    params = {'objective':'binary','metric':'binary_logloss','verbose':-1,
        'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':100,
        'subsample':0.7,'colsample_bytree':0.7,'reg_alpha':1.0,'reg_lambda':3.0,
        'scale_pos_weight':spw,'random_state':42,'min_child_samples':10,
        'force_row_wise':True,'n_jobs':-1}
    sn = [SAN.sub('_', c) for c in leak]
    ds = lgb.Dataset(X, label=y, feature_name=sn)
    t0 = time.time()
    mdl = lgb.train(params, ds, num_boost_round=100)
    imp = mdl.feature_importance(importance_type="gain")
    ranked = sorted(zip(leak, imp), key=lambda x: -x[1])
    top20 = ranked[:20]
    print(f"{target}: ranked {len(leak)} → top20 = {[r[0] for r in top20][:5]}... [{time.time()-t0:.1f}s]")
