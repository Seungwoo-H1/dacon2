"""
V25 & V30 submission — direct from features.parquet.
3 seeds, 5 folds, 300 trees, num_leaves=15.
n_jobs=1 for memory safety.
"""
import sys, re, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore')
np.random.seed(42)

TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']
META_COLS = {'subject_id','lifelog_date','sleep_date','date'}

def sanitize(n):
    return re.sub(r'[^a-zA-Z0-9_]','_',n)

def mm(p, r):
    return np.clip(p + (r - p.mean()), 0.0001, 0.9999)

CONSTANT_COLS = [
    'mACStatus_m_charging_min','mACStatus_m_charging_max','mLight_m_light_min',
    'mScreenStatus_m_screen_use_min','mScreenStatus_m_screen_use_max',
    'wPedo_pedo_running_step_mean','wPedo_pedo_running_step_sum',
    'wPedo_pedo_walking_step_mean','wPedo_pedo_walking_step_sum',
    'mGps_gps_has_speed_mean','mGps_gps_has_speed_std',
    'mGps_gps_has_speed_max','mGps_gps_has_speed_min',
    'mUsageStats_usage_major_ratio_min','mUsageStats_usage_game_ratio_min',
]
COLLINEAR_DROP = [
    'wPedo_pedo_step_frequency_mean','wPedo_pedo_step_frequency_sum',
    'mBle_ble_device_count_mean','mBle_ble_device_count_std',
    'mBle_ble_device_count_max',
    'mWifi_wifi_bssid_count_mean','mWifi_wifi_bssid_count_std',
    'mWifi_wifi_bssid_count_max',
]
LEAKAGE_S = {
    'wLight_w_light_mean','wLight_w_light_std','wLight_w_light_min','wLight_w_light_max','wLight_w_light_count',
    'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max','wHr_hr_median','wHr_hr_count',
    'wPedo_pedo_step_mean','wPedo_pedo_step_sum',
    'wPedo_pedo_step_frequency_mean','wPedo_pedo_step_frequency_sum',
    'wPedo_pedo_running_step_mean','wPedo_pedo_running_step_sum',
    'wPedo_pedo_walking_step_mean','wPedo_pedo_walking_step_sum',
    'wPedo_pedo_distance_mean','wPedo_pedo_distance_sum',
    'wPedo_pedo_speed_mean','wPedo_pedo_speed_sum',
    'wPedo_pedo_burned_calories_mean','wPedo_pedo_burned_calories_sum',
}
LEAKAGE_Q = {'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max','wHr_hr_median','wHr_hr_count'}

# 3 seeds for memory safety
SEEDS = [42, 123, 456]
N_SEEDS = len(SEEDS)
N_TOP = 30

LGB_CFG = {
    'objective':'binary','metric':'binary_logloss','num_leaves':15,'max_depth':4,
    'learning_rate':0.03,'n_estimators':300,'subsample':0.7,'colsample_bytree':0.7,
    'reg_alpha':1.0,'reg_lambda':3.0,'min_child_samples':10,'force_row_wise':True,
    'verbose':-1,'n_jobs':1,
}
XGB_CFG = {
    'objective':'binary:logistic','tree_method':'hist','max_depth':4,
    'learning_rate':0.03,'n_estimators':300,'subsample':0.8,'colsample_bytree':0.8,
    'reg_alpha':1.0,'reg_lambda':3.0,'min_child_weight':3,
}

SUBMIT_DIR = Path('submissions')
ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

print("Loading data...", flush=True)
feat = pd.read_parquet('data_processed/features.parquet')
tf = pd.read_parquet('data_processed/test_features.parquet')
print(f"Train: {feat.shape}, Test: {tf.shape}", flush=True)

# Feature selection
NUM_DTYPES = [np.float64, np.float32, np.int64, np.int32, float, int, bool, np.bool_]
all_num = [c for c in feat.columns if c not in META_COLS | set(TARGETS) and feat[c].dtype in NUM_DTYPES]
obj_cols = [c for c in feat.columns if c not in META_COLS | set(TARGETS) and feat[c].dtype == object]
print(f"Numeric: {len(all_num)}, Object(exclude): {obj_cols}", flush=True)

clean = [c for c in all_num if c not in CONSTANT_COLS and c not in COLLINEAR_DROP and c not in obj_cols]
dynamic = [c for c in clean if feat[c].nunique() > 1]
dynamic = [c for c in dynamic if feat[c].dtype not in [object, str]]

# wHr fix
bad = (feat['wHr_hr_mean'] < 20) | (feat['wHr_hr_mean'] > 180)
feat.loc[bad, 'wHr_hr_mean'] = np.nan
feat.loc[bad, 'wHr_hr_std'] = np.nan

feat = feat.fillna(0)
tf = tf.fillna(0)
train_rate = {t: float(feat[t].mean()) for t in TARGETS}
print(f"Train rates: {train_rate}", flush=True)

# =============== V25 (LGBM) ===============
print(f"\n{'='*70}")
print("V25 — LightGBM", flush=True)
print(f"{'='*70}", flush=True)

all_oof = {}
all_sel = {}
v25_predictions = tf[['subject_id','sleep_date','lifelog_date']].copy()

for target in TARGETS:
    print(f"\n--- {target} ---", flush=True)
    leak = LEAKAGE_S if target.startswith('S') else LEAKAGE_Q
    avail = [c for c in dynamic if c not in leak]
    y = feat[target].values.astype(np.float64)
    spw = float((y==0).sum()) / max(float((y==1).sum()), 1)
    print(f"  Avail: {len(avail)}", flush=True)

    # Feature ranking (20 trees)
    sn_avail = [sanitize(c) for c in avail]
    X_avail = feat[avail].fillna(0).values.astype(np.float32)
    ds_rank = lgb.Dataset(X_avail, label=y, feature_name=sn_avail, params={'verbose':'-1'})
    m_rank = lgb.train({**LGB_CFG, 'n_estimators':20, 'scale_pos_weight':spw, 'random_state':42},
                       ds_rank, num_boost_round=20)
    imp = m_rank.feature_importance(importance_type='gain')
    ranked = sorted(zip(avail, imp), key=lambda x: -x[1])
    sel = [r[0] for r in ranked[:N_TOP]]
    sn_sel = [sanitize(c) for c in sel]
    print(f"  Top-30 selected", flush=True)

    # CV: 3 seeds × 5 folds = 15 models
    oof = np.zeros((len(y), N_SEEDS))
    X = feat[sel].fillna(0).values.astype(np.float32)
    gkf = GroupKFold(n_splits=5)
    
    for si, s in enumerate(SEEDS):
        cfg = {**LGB_CFG, 'random_state':s, 'scale_pos_weight':spw}
        for tr_i, va_i in gkf.split(feat, y, feat['subject_id']):
            ds_t = lgb.Dataset(X[tr_i], label=y[tr_i], feature_name=sn_sel, params={'verbose':'-1'})
            ds_v = lgb.Dataset(X[va_i], label=y[va_i], feature_name=sn_sel, reference=ds_t, params={'verbose':'-1'})
            m = lgb.train(cfg, ds_t, num_boost_round=300, valid_sets=[ds_v],
                         callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            oof[va_i, si] = m.predict(X[va_i])
        print(f"    seed {s} done ({si+1}/{N_SEEDS})", flush=True)

    oof_avg = oof.mean(axis=1)
    cal = mm(oof_avg, train_rate[target])
    loss = log_loss(y, cal, labels=[0,1])
    all_oof[target] = oof_avg
    all_sel[target] = sel
    print(f"  Cal OOF={loss:.4f}, shift={cal.mean()-train_rate[target]:+.4f}", flush=True)

    # Test predictions: train full data
    Xt = tf[sel].fillna(0).values.astype(np.float32)
    X_all = X
    ap = np.zeros(len(Xt))
    for s in SEEDS:
        cfg = {**LGB_CFG, 'random_state':s, 'scale_pos_weight':spw}
        ds = lgb.Dataset(X_all, label=y, feature_name=sn_sel, params={'verbose':'-1'})
        m = lgb.train(cfg, ds, num_boost_round=300)
        ap += m.predict(Xt)
    ap /= N_SEEDS
    cal = mm(ap, train_rate[target])
    v25_predictions[target] = cal
    print(f"  test mean={cal.mean():.4f}, shift={cal.mean()-train_rate[target]:+.4f}", flush=True)
    # Free memory
    del oof, X_all, ap, m, ds

avg_v25 = np.mean([log_loss(feat[t], mm(all_oof[t], train_rate[t]), labels=[0,1]) for t in TARGETS])
print(f"\nV25 Cal OOF Avg: {avg_v25:.4f} (V10: 0.6038, delta: {avg_v25-0.6038:+.4f})", flush=True)

sp_v25 = SUBMIT_DIR / f'submission_v25_{ts}.csv'
v25_predictions.to_csv(sp_v25, index=False)
print(f"Saved: {sp_v25}  shape={v25_predictions.shape}", flush=True)

meta_v25 = {
    'version':'v25','submission_file':str(sp_v25),'timestamp':ts,
    'n_samples':len(v25_predictions),'n_seeds':N_SEEDS,'n_splits':5,'n_top':N_TOP,
    'calibration':'mean-matching+clip',
    'features':{'base_cleaned':len(dynamic),'selected_per_target':N_TOP},
    'cv_avg':float(avg_v25),
    'per_target':{}
}
for t in TARGETS:
    co = log_loss(feat[t], mm(all_oof[t], train_rate[t]), labels=[0,1])
    meta_v25['per_target'][t] = {'cal_oof_loss':float(co),'cal_mean':float(v25_predictions[t].mean()),
                                  'train_rate':float(train_rate[t]),
                                  'pred_min':float(v25_predictions[t].min()),'pred_max':float(v25_predictions[t].max())}
mp_v25 = SUBMIT_DIR / f'meta_v25_{ts}.json'
with open(mp_v25,'w') as f:
    json.dump(meta_v25, f, indent=2, default=str)
print(f"Metadata: {mp_v25}", flush=True)

# =============== V30 (XGB Histogram) ===============
print(f"\n{'='*70}")
print("V30 — XGB Histogram", flush=True)
print(f"{'='*70}", flush=True)

all_oof_xgb = {}
all_sel_xgb = {}
v30_predictions = tf[['subject_id','sleep_date','lifelog_date']].copy()

for target in TARGETS:
    print(f"\n--- {target} ---", flush=True)
    leak = LEAKAGE_S if target.startswith('S') else LEAKAGE_Q
    avail = [c for c in dynamic if c not in leak]
    y = feat[target].values.astype(np.float64)
    spw = float((y==0).sum()) / max(float((y==1).sum()), 1)

    # Feature ranking (XGB 3.x API: trees_to_dataframe)
    X_avail = feat[avail].fillna(0).values.astype(np.float32)
    sn_avail = [sanitize(c) for c in avail]
    dm = xgb.DMatrix(X_avail, label=y, feature_names=sn_avail)
    m_rank = xgb.train({**XGB_CFG, 'n_estimators':20, 'scale_pos_weight':spw, 'random_state':42}, dm, num_boost_round=20)
    df_gain = m_rank.trees_to_dataframe()
    gain_by_feat = df_gain.groupby('Feature')['Gain'].sum().to_dict()
    ranked = sorted([(sn_avail[i], gain_by_feat.get(sn_avail[i], 0)) for i in range(len(sn_avail))], key=lambda x: -x[1])
    ranked = [(avail[i], gain_by_feat.get(sn_avail[i], 0)) for i in range(len(sn_avail))]
    ranked = sorted(ranked, key=lambda x: -x[1])
    sel = [r[0] for r in ranked[:N_TOP]]
    all_sel_xgb[target] = sel
    sn_sel = [sanitize(c) for c in sel]

    # CV: 3 seeds × 5 folds
    oof = np.zeros((len(y), N_SEEDS))
    X = feat[sel].fillna(0).values.astype(np.float32)
    gkf = GroupKFold(n_splits=5)
    
    for si, s in enumerate(SEEDS):
        for tr_i, va_i in gkf.split(feat, y, feat['subject_id']):
            dm_t = xgb.DMatrix(X[tr_i], label=y[tr_i], feature_names=sn_sel)
            dm_v = xgb.DMatrix(X[va_i], label=y[va_i], feature_names=sn_sel)
            m = xgb.train({**XGB_CFG, 'random_state':s, 'scale_pos_weight':spw}, dm_t, num_boost_round=300,
                         evals=[(dm_v,'val')])
            oof[va_i, si] = m.predict(dm_v)
        print(f"    seed {s} done ({si+1}/{N_SEEDS})", flush=True)

    oof_avg = oof.mean(axis=1)
    cal = mm(oof_avg, train_rate[target])
    loss = log_loss(y, cal, labels=[0,1])
    all_oof_xgb[target] = oof_avg
    print(f"  Cal OOF={loss:.4f}", flush=True)

    # Test
    Xt = tf[sel].fillna(0).values.astype(np.float32)
    dm_test = xgb.DMatrix(Xt, feature_names=sn_sel)
    ap = np.zeros(len(Xt))
    for s in SEEDS:
        dm_train = xgb.DMatrix(X, label=y, feature_names=sn_sel)
        m = xgb.train({**XGB_CFG, 'random_state':s, 'scale_pos_weight':spw}, dm_train, num_boost_round=300)
        ap += m.predict(dm_test)
    ap /= N_SEEDS
    cal = mm(ap, train_rate[target])
    v30_predictions[target] = cal
    print(f"  test mean={cal.mean():.4f}", flush=True)
    del oof, ap

avg_v30 = np.mean([log_loss(feat[t], mm(all_oof_xgb[t], train_rate[t]), labels=[0,1]) for t in TARGETS])
print(f"\nV30 Cal OOF Avg: {avg_v30:.4f} (V10: 0.6038, delta: {avg_v30-0.6038:+.4f})", flush=True)

sp_v30 = SUBMIT_DIR / f'submission_v30_xgb_{ts}.csv'
v30_predictions.to_csv(sp_v30, index=False)
print(f"Saved: {sp_v30}", flush=True)

meta_v30 = {
    'version':'v30','submission_file':str(sp_v30),'timestamp':ts,
    'n_samples':len(v30_predictions),'n_seeds':N_SEEDS,'n_splits':5,'n_top':N_TOP,
    'calibration':'mean-matching+clip','tree_method':'hist',
    'features':{'base_cleaned':len(dynamic),'selected_per_target':N_TOP},
    'cv_avg':float(avg_v30),
    'per_target':{}
}
for t in TARGETS:
    co = log_loss(feat[t], mm(all_oof_xgb[t], train_rate[t]), labels=[0,1])
    meta_v30['per_target'][t] = {'cal_oof_loss':float(co),'cal_mean':float(v30_predictions[t].mean()),
                                  'train_rate':float(train_rate[t]),
                                  'pred_min':float(v30_predictions[t].min()),'pred_max':float(v30_predictions[t].max())}
mp_v30 = SUBMIT_DIR / f'meta_v30_{ts}.json'
with open(mp_v30,'w') as f:
    json.dump(meta_v30, f, indent=2, default=str)

print(f"\n{'='*70}")
print("FINAL SUMMARY")
print(f"{'Model':<10} {'Cal OOF':<12} {'vs V10':<10}")
print(f"{'V10':<10} {'0.6038':<12} {'—':<10}")
print(f"{'V25-LGBM':<10} {avg_v25:<12.4f} {avg_v25-0.6038:<+10.4f}")
print(f"{'V30-XGB':<10} {avg_v30:<12.4f} {avg_v30-0.6038:<+10.4f}")
print(f"{'='*70}", flush=True)
