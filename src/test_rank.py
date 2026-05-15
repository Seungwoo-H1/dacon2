import time, numpy as np, pandas as pd, lightgbm as lgb, re, warnings, os
os.environ['PYTHONUNBUFFERED'] = '1'
warnings.filterwarnings('ignore')

TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']
META = {"subject_id","lifelog_date","sleep_date","date"}
SAN = re.compile(r'[^a-zA-Z0-9_]')
LEAK_S = {"wLight_w_light_mean","wLight_w_light_std","wLight_w_light_min","wLight_w_light_max","wLight_w_light_count","wHr_hr_mean","wHr_hr_std","wHr_hr_min","wHr_hr_max","wHr_hr_median","wHr_hr_count","wPedo_pedo_step_mean","wPedo_pedo_step_sum","wPedo_pedo_step_frequency_mean","wPedo_pedo_step_frequency_sum","wPedo_pedo_running_step_mean","wPedo_pedo_running_step_sum","wPedo_pedo_walking_step_mean","wPedo_pedo_walking_step_sum","wPedo_pedo_distance_mean","wPedo_pedo_distance_sum","wPedo_pedo_speed_mean","wPedo_pedo_speed_sum","wPedo_pedo_burned_calories_mean","wPedo_pedo_burned_calories_sum"}
LEAK_Q = {"wHr_hr_mean","wHr_hr_std","wHr_hr_min","wHr_hr_max","wHr_hr_median","wHr_hr_count"}

feat = pd.read_parquet('data_processed/features.parquet')
fc = [c for c in feat.columns if c not in META|set(TARGETS) and feat[c].dtype in [np.float64,np.int64,float,int,bool,np.bool_]]
print(f"Features: {len(fc)}", flush=True)

params = {'objective':'binary','metric':'binary_logloss','verbose':-1,'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':100,'subsample':0.7,'colsample_bytree':0.7,'reg_alpha':1.0,'reg_lambda':3.0,'min_child_samples':10,'force_row_wise':True,'n_jobs':-1}

for t in TARGETS:
    leak = [c for c in fc if c not in (LEAK_S if t.startswith('S') else LEAK_Q)]
    y = feat[t].values
    n_p = max((y==1).sum(),1); n_n = (y==0).sum(); spw = n_n/n_p
    params2 = {**params, 'scale_pos_weight':spw, 'random_state':42}
    X = feat[leak].fillna(0).values.astype(np.float32)
    sn = [SAN.sub('_',c) for c in leak]
    ds = lgb.Dataset(X, label=y, feature_name=sn)
    t0 = time.time()
    mdl = lgb.train(params2, ds, num_boost_round=100)
    imp = mdl.feature_importance(importance_type="gain")
    ranked = sorted(zip(leak, imp), key=lambda x: -x[1])
    elapsed = time.time() - t0
    top5 = [r[0] for r in ranked[:5]]
    print(f"{t}: {len(leak)}f → top5={top5} [{elapsed:.1f}s]", flush=True)
