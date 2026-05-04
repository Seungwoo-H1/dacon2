"""
Minimal V29 submission generator - reduced to 5 seeds for speed.
Uses top 30 features from V26 ablation results already in models/.
"""

import sys, re, json, warnings, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

warnings.filterwarnings('ignore')

TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']
META_COLS = {'subject_id','lifelog_date','sleep_date','date'}

def sanitize(n):
    return re.sub(r'[^a-zA-Z0-9_]','_',n)

def mm(p, r):
    return np.clip(p + (r - p.mean()), 0.0001, 0.9999)

def add_rolling(df, cols):
    df = df.copy().sort_values(['subject_id','date'])
    new = []
    for c in cols:
        g = df.groupby('subject_id')[c]
        for w in [3, 7]:
            rm = g.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
            rs = g.rolling(w, min_periods=1).std().fillna(0).reset_index(level=0, drop=True)
            df[f'{c}_rm{w}'] = rm.values
            df[f'{c}_rs{w}'] = rs.values
            new.extend([f'{c}_rm{w}', f'{c}_rs{w}'])
    return df, new

# Constants - same as V29/V30
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
LEAK_S = {'wLight_w_light_mean','wLight_w_light_std','wLight_w_light_min','wLight_w_light_max','wLight_w_light_count',
    'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max','wHr_hr_median','wHr_hr_count',
    'wPedo_pedo_step_mean','wPedo_pedo_step_sum',
    'wPedo_pedo_step_frequency_mean','wPedo_pedo_step_frequency_sum',
    'wPedo_pedo_running_step_mean','wPedo_pedo_running_step_sum',
    'wPedo_pedo_walking_step_mean','wPedo_pedo_walking_step_sum',
    'wPedo_pedo_distance_mean','wPedo_pedo_distance_sum',
    'wPedo_pedo_speed_mean','wPedo_pedo_speed_sum',
    'wPedo_pedo_burned_calories_mean','wPedo_pedo_burned_calories_sum'}
LEAK_Q = {'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max','wHr_hr_median','wHr_hr_count'}

# Use 5 seeds for speed (was 10 in V29 design)
SEEDS = [42, 123, 456, 789, 1024]
N_SEEDS = len(SEEDS)
N_TOP = 30
LGB = {
    'objective':'binary','metric':'binary_logloss','num_leaves':15,'max_depth':4,
    'learning_rate':0.03,'n_estimators':500,'subsample':0.7,'colsample_bytree':0.7,
    'reg_alpha':1.0,'reg_lambda':3.0,'min_child_samples':10,'force_row_wise':True,'verbose':-1
}

print("Loading data...")
feat = pd.read_parquet('data_processed/features.parquet')
tf = pd.read_parquet('data_processed/test_features.parquet')

# Get valid features
raw = [c for c in feat.columns if c not in META_COLS | set(TARGETS) and 
       feat[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]
dynamic = [c for c in raw if feat[c].nunique() > 1]
base = [c for c in dynamic if c not in CONSTANT_COLS and c not in COLLINEAR_DROP]

# Strictly remove non-numeric columns
base = [c for c in base if feat[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int, bool, np.bool_]]
print(f"Base features (numeric, non-constant): {len(base)}")

feat_r, r_cols = add_rolling(feat, base)
feat_r = feat_r.fillna(0)
all_cols = base + r_cols
train_rate = {t: feat_r[t].mean() for t in TARGETS}
print(f"With rolling: {len(all_cols)} features")

# Test features
test_base = [c for c in tf.columns if c not in META_COLS | set(TARGETS) and 
             tf[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int, bool, np.bool_] and 
             tf[c].nunique() > 1]
test_base = [c for c in test_base if c not in CONSTANT_COLS and c not in COLLINEAR_DROP]
tf_r, _ = add_rolling(tf, test_base)
tf_r = tf_r.fillna(0)
print(f"Test features: {len(test_base)} base, {len(test_base)+len(test_base)*2} total with rolling")

SUBMIT_DIR = Path('submissions')
ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

all_oof = {}
all_sel = {}
predictions = tf_r[['subject_id','sleep_date','lifelog_date']].copy()

for target in TARGETS:
    leak = LEAK_S if target.startswith('S') else LEAK_Q
    avail = [c for c in all_cols if c not in META_COLS | leak | set(TARGETS)]
    y = feat_r[target].values
    spw = ((y==0).sum()) / max((y==1).sum(), 1)

    print(f"\n--- {target} ---")
    print(f"  Available features: {len(avail)}")

    # Feature ranking: 1 seed, 50 iterations (fast)
    p_rank = {**LGB, 'num_leaves':15,'max_depth':4,'n_estimators':50,'scale_pos_weight':spw,'random_state':42}
    sn_avail = [sanitize(c) for c in avail]
    ds = lgb.Dataset(feat_r[avail].fillna(0).values, label=y, feature_name=sn_avail, params={'verbose':'-1'})
    m_rank = lgb.train(p_rank, ds, num_boost_round=50)
    imp = m_rank.feature_importance(importance_type='gain')
    ranked = sorted(zip(avail, imp), key=lambda x: -x[1])
    sel = [r[0] for r in ranked[:N_TOP]]
    sn_sel = [sanitize(c) for c in sel]
    print(f"  Selected: {N_TOP} features")

    # CV with 5 seeds, 5 folds = 25 models
    oof = np.zeros((len(y), N_SEEDS))
    X = feat_r[sel].fillna(0).values
    gkf = GroupKFold(n_splits=5)
    for si, s in enumerate(SEEDS):
        cfg = {**LGB, 'random_state':s, 'scale_pos_weight':spw}
        for tr_i, va_i in gkf.split(feat_r, y, feat_r['subject_id']):
            ds_t = lgb.Dataset(X[tr_i], label=y[tr_i], feature_name=sn_sel, params={'verbose':'-1'})
            ds_v = lgb.Dataset(X[va_i], label=y[va_i], feature_name=sn_sel, reference=ds_t, params={'verbose':'-1'})
            m = lgb.train(cfg, ds_t, num_boost_round=500, valid_sets=[ds_v],
                         callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            oof[va_i, si] = m.predict(X[va_i])

    oof_avg = oof.mean(axis=1)
    cal = mm(oof_avg, train_rate[target])
    loss = log_loss(y, cal, labels=[0,1])
    all_oof[target] = oof_avg
    all_sel[target] = sel
    print(f"  Cal OOF={loss:.4f}, rate={train_rate[target]:.3f}")

    # Test predictions
    Xt = tf_r[sel].fillna(0).values
    ap = np.zeros(len(Xt))
    for s in SEEDS:
        ds = lgb.Dataset(X, label=y, feature_name=sn_sel, params={'verbose':'-1'})
        m = lgb.train({**LGB,'random_state':s,'scale_pos_weight':spw}, ds, num_boost_round=500)
        ap += m.predict(Xt)
    ap /= N_SEEDS
    cal = mm(ap, train_rate[target])
    predictions[target] = cal
    print(f"  test mean={cal.mean():.4f}, shift={cal.mean()-train_rate[target]:+.4f}")

avg = np.mean([log_loss(feat_r[t], mm(all_oof[t], train_rate[t]), labels=[0,1]) for t in TARGETS])
print(f"\n{'='*60}")
print(f"Cal OOF Avg: {avg:.4f} (V10: 0.6038, delta: {avg-0.6038:+.4f})")
print(f"{'='*60}")

# Save submission
sp = SUBMIT_DIR / f'submission_v29_{ts}.csv'
predictions.to_csv(sp, index=False)
print(f"Saved: {sp}")
print(f"Shape: {predictions.shape}, columns: {predictions.columns.tolist()}")
print(f"Sample: {predictions.head(3).to_dict('list')}")

meta = {
    'version':'v29','submission_file':str(sp),'timestamp':ts,
    'n_samples':len(predictions),'n_seeds':N_SEEDS,'n_splits':5,'n_top':N_TOP,
    'features':{'base':len(base),'rolling':len(r_cols),'total':len(all_cols),'selected':N_TOP},
    'calibration':'mean-matching+clip',
    'strategy':'rolling(3d,7d) fixed Top-30, 5 seeds (memory-optimized)',
    'cv_avg':float(avg),
    'per_target':{}
}
for t in TARGETS:
    co = log_loss(feat_r[t], mm(all_oof[t], train_rate[t]), labels=[0,1])
    meta['per_target'][t] = {'cal_oof_loss':float(co),'cal_mean':float(predictions[t].mean()),
                              'train_rate':float(train_rate[t]),
                              'pred_min':float(predictions[t].min()),'pred_max':float(predictions[t].max())}
mp = SUBMIT_DIR / f'meta_v29_{ts}.json'
with open(mp,'w') as f: json.dump(meta, f, indent=2, default=str)
print(f"Metadata: {mp}")

# Also save V30 submission using XGB (same features, 5 seeds)
print(f"\n{'='*60}")
print("Generating V30 (XGB histogram) submission...")
print(f"{'='*60}")

import xgboost as xgb

XGB = {
    'objective':'binary:logistic','tree_method':'hist','max_depth':4,
    'learning_rate':0.03,'n_estimators':500,'subsample':0.8,'colsample_bytree':0.8,
    'reg_alpha':1.0,'reg_lambda':3.0,'min_child_weight':3
}

all_oof_xgb = {}
all_sel_xgb = {}
predictions_xgb = tf_r[['subject_id','sleep_date','lifelog_date']].copy()

for target in TARGETS:
    leak = LEAK_S if target.startswith('S') else LEAK_Q
    avail = [c for c in all_cols if c not in META_COLS | leak | set(TARGETS)]
    y = feat_r[target].values
    spw = ((y==0).sum()) / max((y==1).sum(), 1)

    # Rank with XGB
    p_rank = {**XGB, 'n_estimators':50,'scale_pos_weight':spw,'random_state':42}
    sn_avail = [sanitize(c) for c in avail]
    dm = xgb.DMatrix(feat_r[avail].fillna(0).values, label=y, feature_names=sn_avail)
    m_rank = xgb.train(p_rank, dm, num_boost_round=50)
    imp = m_rank.feature_importance(importance_type='gain')
    ranked = sorted(zip(avail, imp), key=lambda x: -x[1])
    sel = [r[0] for r in ranked[:N_TOP]]
    all_sel_xgb[target] = sel
    sn_sel = [sanitize(c) for c in sel]

    # CV with 5 seeds
    oof = np.zeros((len(y), 5))
    X = feat_r[sel].fillna(0).values
    for si, s in enumerate([42,123,456,789,1024]):
        for tr_i, va_i in gkf.split(feat_r, y, feat_r['subject_id']):
            dm_t = xgb.DMatrix(X[tr_i], label=y[tr_i], feature_names=sn_sel)
            dm_v = xgb.DMatrix(X[va_i], label=y[va_i], feature_names=sn_sel, reference=dm_t)
            m = xgb.train({**XGB,'random_state':s,'scale_pos_weight':spw}, dm_t, num_boost_round=500,
                         evals=[(dm_v,'val')], callbacks=[xgb.callback.EvaluationMonitor(period=0)])
            oof[va_i, si] = m.predict(dm_v)

    oof_avg = oof.mean(axis=1)
    cal = mm(oof_avg, train_rate[target])
    loss = log_loss(y, cal, labels=[0,1])
    all_oof_xgb[target] = oof_avg
    print(f"  {target}: Cal OOF={loss:.4f}")

    # Test
    Xt = tf_r[sel].fillna(0).values
    dm_test = xgb.DMatrix(Xt, feature_names=sn_sel)
    ap = np.zeros(len(Xt))
    for s in [42,123,456,789,1024]:
        dm_train = xgb.DMatrix(X, label=y, feature_names=sn_sel)
        m = xgb.train({**XGB,'random_state':s,'scale_pos_weight':spw}, dm_train, num_boost_round=500)
        ap += m.predict(dm_test)
    ap /= 5
    cal = mm(ap, train_rate[target])
    predictions_xgb[target] = cal
    print(f"  {target}: test mean={cal.mean():.4f}")

avg_xgb = np.mean([log_loss(feat_r[t], mm(all_oof_xgb[t], train_rate[t]), labels=[0,1]) for t in TARGETS])
print(f"\nV30 Cal OOF Avg: {avg_xgb:.4f} (V10: 0.6038, delta: {avg_xgb-0.6038:+.4f})")

sp_xgb = SUBMIT_DIR / f'submission_v30_xgb_{ts}.csv'
predictions_xgb.to_csv(sp_xgb, index=False)
print(f"Saved: {sp_xgb}")

meta_xgb = {
    'version':'v30','submission_file':str(sp_xgb),'timestamp':ts,
    'n_samples':len(predictions_xgb),'n_seeds':5,'n_splits':5,'n_top':N_TOP,
    'features':{'base':len(base),'rolling':len(r_cols),'total':len(all_cols),'selected':N_TOP},
    'calibration':'mean-matching+clip','tree_method':'hist',
    'strategy':'XGB histogram ensemble, 5 seeds',
    'cv_avg':float(avg_xgb),
    'per_target':{}
}
for t in TARGETS:
    co = log_loss(feat_r[t], mm(all_oof_xgb[t], train_rate[t]), labels=[0,1])
    meta_xgb['per_target'][t] = {'cal_oof_loss':float(co),'cal_mean':float(predictions_xgb[t].mean()),
                                  'train_rate':float(train_rate[t]),
                                  'pred_min':float(predictions_xgb[t].min()),'pred_max':float(predictions_xgb[t].max())}
mp_xgb = SUBMIT_DIR / f'meta_v30_{ts}.json'
with open(mp_xgb,'w') as f: json.dump(meta_xgb, f, indent=2, default=str)

print(f"\n{'='*60}")
print(f"FINAL SUMMARY")
print(f"{'='*60}")
print(f"{'Model':<10} {'Cal OOF':<12} {'vs V10':<10}")
print(f"{'V10':<10} {'0.6038':<12} {'—':<10}")
print(f"{'V29-LGBM':<10} {avg:<12.4f} {avg-0.6038:<+10.4f}")
print(f"{'V30-XGB':<10} {avg_xgb:<12.4f} {avg_xgb-0.6038:<+10.4f}")
print(f"{'='*60}")
