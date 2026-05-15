#!/usr/bin/env python3
"""
DACon2 v260: External Proxy Features — Circadian, Entropy, Routine, Temporal
All derived from EXISTING data columns. No external API calls.
Fixed: proper baseline/enhanced comparison without target leakage.
"""
import os, sys, gc, re, json, warnings, time
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
from scipy.stats import entropy
warnings.filterwarnings('ignore')

ROOT = Path('/home/mwoo423/projects/dacon2')
DATA = ROOT / 'data_processed'
EXPERIMENTS = ROOT / 'experiments'
TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']
SEEDS = [42, 7, 999, 777]

V53_SWEEP = {
    'Q1': {'cfg': 'deep'}, 'Q2': {'cfg': 'deep'}, 'Q3': {'cfg': 'v48'},
    'S1': {'cfg': 'wide'}, 'S2': {'cfg': 'deep'}, 'S3': {'cfg': 'safety'}, 'S4': {'cfg': 'wide'},
}
CFGS = {
    'wide':   {'num_leaves':30,'max_depth':3,'learning_rate':0.05,'n_estimators':300,
               'subsample':0.8,'colsample_bytree':0.8,'reg_alpha':2.0,'reg_lambda':5.0,
               'min_child_samples':5},
    'deep':   {'num_leaves':20,'max_depth':5,'learning_rate':0.02,'n_estimators':1000,
               'subsample':0.7,'colsample_bytree':0.6,'reg_alpha':0.5,'reg_lambda':2.0,
               'min_child_samples':15},
    'v48':    {'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':500,
               'subsample':0.7,'colsample_bytree':0.7,'reg_alpha':1.0,'reg_lambda':3.0,
               'min_child_samples':10},
    'safety': {'num_leaves':10,'max_depth':3,'learning_rate':0.02,'n_estimators':1000,
               'subsample':0.6,'colsample_bytree':0.6,'reg_alpha':3.0,'reg_lambda':10.0,
               'min_child_samples':20},
}

def sanitize_col(n): return re.sub(r'[^a-zA-Z0-9_]', '_', n)

def mean_match(pred, tm):
    return np.clip(pred + (tm - pred.mean()), 0.0001, 0.9999)

def remove_leak(cols, t):
    LEAK_S = {'wlight_w_light_mean','wlight_w_light_std','wlight_w_light_min','wlight_w_light_max','wlight_w_light_count',
              'whr_hr_mean','whr_hr_std','whr_hr_min','whr_hr_max','whr_hr_median','whr_hr_count',
              'wpedo_pedo_step_mean','wpedo_pedo_step_sum','wpedo_pedo_step_frequency_mean','wpedo_pedo_step_frequency_sum',
              'wpedo_pedo_running_step_mean','wpedo_pedo_running_step_sum','wpedo_pedo_walking_step_mean','wpedo_pedo_walking_step_sum',
              'wpedo_pedo_distance_mean','wpedo_pedo_distance_sum','wpedo_pedo_speed_mean','wpedo_pedo_speed_sum',
              'wpedo_pedo_burned_calories_mean','wpedo_pedo_burned_calories_sum'}
    LEAK_Q = {'whr_hr_mean','whr_hr_std','whr_hr_min','whr_hr_max','whr_hr_median','whr_hr_count'}
    if t.startswith('S'): return [c for c in cols if c not in LEAK_S]
    elif t.startswith('Q'): return [c for c in cols if c not in LEAK_Q]
    return cols

def get_feature_cols(df, target_cols=None):
    META = {'subject_id','lifelog_date','sleep_date','date'}
    if target_cols is None:
        target_cols = set(TARGETS)
    return [c for c in df.columns if c not in META | target_cols and df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]

def cfg_to_params(cfg_s, seed, spw):
    params = dict(cfg_s)
    params['scale_pos_weight'] = spw
    params['random_state'] = seed
    params['force_row_wise'] = True
    params['n_jobs'] = 1
    return params

def train_cv(feat, ftst, cols, y, seeds, cfg, target_col='Q1'):
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros((len(y), len(seeds)))
    tp = np.zeros((len(ftst), len(seeds))) if ftst is not None else None
    sn = [sanitize_col(c) for c in cols]
    Xf = feat[cols].fillna(0).values.astype(np.float64)
    Xt = ftst[cols].fillna(0).values.astype(np.float64) if ftst is not None else None
    spw = max(((y==0).sum()) / max((y==1).sum(), 1), 0.1)
    cols_clean = remove_leak(cols, target_col)

    for si, seed in enumerate(seeds):
        p = cfg_to_params(cfg, seed, spw)
        for tri, vai in gkf.split(feat, y, feat['subject_id']):
            ds = lgb.Dataset(Xf[tri, :], label=y[tri], feature_name=sn)
            if Xt is not None and vai.shape[0] > 0:
                vd = lgb.Dataset(Xf[vai, :], label=y[vai], feature_name=sn, reference=ds)
                m = lgb.train(p, ds, num_boost_round=cfg['n_estimators'],
                             valid_sets=[vd],
                             callbacks=[lgb.early_stopping(50, verbose=False),
                                       lgb.log_evaluation(0)])
                oof[vai, si] = m.predict(Xf[vai, :])
                tp[:, si] = m.predict(Xt)
            else:
                m = lgb.train(p, ds, num_boost_round=cfg['n_estimators'],
                             callbacks=[lgb.log_evaluation(0)])
                oof[vai, si] = m.predict(Xf[vai, :])
            del ds, m; gc.collect()

    if tp is not None:
        tp = np.clip(tp, 0.0001, 0.9999)
    return oof, tp

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 60)
print("V260: External Proxy Features")
print("=" * 60)

feat = pd.read_parquet(DATA / 'features_clean_v60.parquet')
ftst = pd.read_parquet(DATA / 'test_features_clean_v60.parquet')

# Save original base column names (BEFORE adding new features)
ORIGINAL_BASE_COLS = get_feature_cols(feat, set(TARGETS))
print(f"Original base features: {len(ORIGINAL_BASE_COLS)}")

# ============================================================
# GENERATE PROXY FEATURES
# ============================================================
print("\nGenerating proxy features...")
new_features = {}

# --- A. Circadian Rhythm Features ---
hour_cols = [c for c in feat.columns if any(h in c for h in ['hour_afternoon','hour_evening','hour_morning','hour_night']) and 'zscore' not in c]
if hour_cols:
    feat['circadian_concentration'] = feat[hour_cols].max(axis=1) / (feat[hour_cols].sum(axis=1) + 1e-8)
    # Circadian evenness: how evenly distributed is activity across hours?
    vals = feat[hour_cols].values + 1e-8
    probs = vals / vals.sum(axis=1, keepdims=True)
    feat['circadian_evenness'] = entropy(probs, base=2, axis=1) / np.log(len(hour_cols))
    print(f"  Circadian: 2 features from {len(hour_cols)} hour cols")

# --- B. Entropy / Routine Regularity ---
# Hourly pattern entropy (cross-hour)
pattern_cols = [c for c in feat.columns if 'zscore' not in c and 'hour' in c and 'subject' not in c.lower()]
if len(pattern_cols) > 2:
    abs_vals = feat[pattern_cols].abs().values + 1e-8
    row_sums = abs_vals.sum(axis=1, keepdims=True)
    probs = abs_vals / (row_sums + 1e-8)
    feat['hourly_pattern_entropy'] = -np.sum(probs * np.log(probs + 1e-8), axis=1)
    feat['hourly_pattern_entropy_norm'] = feat['hourly_pattern_entropy'] / (np.log(len(pattern_cols)) + 1e-8)
    print(f"  Entropy: 2 features from {len(pattern_cols)} pattern cols")

# Feature-level entropy (cross-feature entropy for each sample)
numeric_cols = get_feature_cols(feat, set(TARGETS))
if len(numeric_cols) > 10:
    sample_cols = numeric_cols[:30]
    abs_vals = feat[sample_cols].abs().values + 1e-8
    row_sums = abs_vals.sum(axis=1, keepdims=True)
    probs = abs_vals / (row_sums + 1e-8)
    feat['cross_feature_entropy'] = -np.sum(probs * np.log(probs + 1e-8), axis=1)
    print(f"  Cross-feature entropy: 1 feature from {len(sample_cols)} cols")

# Per-feature coefficient of variation (routine stability)
non_const = [c for c in numeric_cols if feat[c].std() > 0.001]
if len(non_const) > 0:
    vals = feat[non_const].values
    mean_vals = np.nanmean(vals, axis=1, keepdims=True)
    std_vals = np.nanstd(vals, axis=1, keepdims=True)
    feat['day_variability_cv'] = np.where(mean_vals > 0, std_vals / (mean_vals + 1e-8), 0).squeeze()
    print(f"  Day variability: 1 feature from {len(non_const)} cols")

# --- C. Mobility Priors ---
gps_cols = [c for c in feat.columns if 'mGps' in c and 'zscore' not in c]
if gps_cols:
    gps_means = [c for c in gps_cols if 'mean' in c]
    gps_stds = [c for c in gps_cols if 'std' in c]
    if gps_means:
        feat['mobility_total'] = feat[gps_means].sum(axis=1)
    if gps_stds:
        feat['mobility_spread'] = feat[gps_stds].max(axis=1)
    print(f"  Mobility: 2 features from {len(gps_cols)} GPS cols")

# BLE device diversity
ble_cols = [c for c in feat.columns if 'mBle_ble_count_mean' in c and 'zscore' not in c]
if ble_cols:
    feat['ble_diversity_ratio'] = feat[ble_cols[0]]
    print(f"  BLE diversity: 1 feature")

# WiFi strength stability
wifi_cols = [c for c in feat.columns if 'mWifi_wifi_avg_rssi' in c and 'zscore' not in c]
if len(wifi_cols) >= 2:
    mean_c = [c for c in wifi_cols if 'mean' in c]
    std_c = [c for c in wifi_cols if 'std' in c]
    if mean_c and std_c:
        feat['wifi_signal_stability'] = feat[mean_c[0]] / (feat[std_c[0]].abs() + 1e-8)
        print(f"  WiFi signal: 1 feature")

# --- D. Sleep/Activity Heuristics ---
screen_ev_cols = [c for c in feat.columns if 'mScreenStatus_hour_evening' in c and 'zscore' not in c]
screen_night_cols = [c for c in feat.columns if 'mScreenStatus_hour_night' in c and 'zscore' not in c]
if screen_ev_cols:
    feat['evening_screen_intensity'] = feat[screen_ev_cols].max(axis=1) if len(screen_ev_cols) > 0 else 0
if screen_ev_cols and screen_night_cols:
    feat['night_screen_ratio'] = feat[screen_night_cols].sum(axis=1) / (feat[screen_ev_cols].sum(axis=1) + 1e-8)
    print(f"  Screen heuristic: 2 features")

# --- E. Temporal Features (from sleep_date) ---
feat['sleep_date_dt'] = pd.to_datetime(feat['sleep_date'])
feat['dow'] = feat['sleep_date_dt'].dt.dayofweek
feat['month'] = feat['sleep_date_dt'].dt.month
feat['quarter'] = feat['sleep_date_dt'].dt.quarter
feat['is_weekend'] = (feat['dow'] >= 5).astype(int)
feat['day_of_year'] = feat['sleep_date_dt'].dt.dayofyear
feat['dow_sin'] = np.sin(2 * np.pi * feat['dow'] / 7)
feat['dow_cos'] = np.cos(2 * np.pi * feat['dow'] / 7)
feat['month_sin'] = np.sin(2 * np.pi * feat['month'] / 12)
feat['month_cos'] = np.cos(2 * np.pi * feat['month'] / 12)
feat['doy_sin'] = np.sin(2 * np.pi * feat['day_of_year'] / 365)
feat['doy_cos'] = np.cos(2 * np.pi * feat['day_of_year'] / 365)
print(f"  Temporal: 9 features from sleep_date")

# --- F. Cross-Modal Interactions ---
activity_cols = [c for c in feat.columns if 'mActivity_m_activity_mean' in c and 'zscore' not in c]
screen_mean_cols = [c for c in feat.columns if 'mScreenStatus_m_screen_use_mean' in c and 'zscore' not in c]
if activity_cols and screen_mean_cols:
    feat['activity_screen_interaction'] = feat[activity_cols[0]] * feat[screen_mean_cols[0]]
    feat['activity_screen_ratio'] = feat[activity_cols[0]] / (feat[screen_mean_cols[0]] + 1e-8)
    print(f"  Cross-modal: 2 features")

# --- G. Per-Subject Baseline Deviation ---
sample_for_dev = non_const[:20]
subject_means = feat.groupby('subject_id')[sample_for_dev].transform('mean')
subject_stds = feat.groupby('subject_id')[sample_for_dev].transform('std').replace(0, 1e-8)
feat['subject_deviation'] = feat[sample_for_dev].sub(subject_means[sample_for_dev]).abs().sum(axis=1)
feat['subject_deviation_norm'] = feat['subject_deviation'] / (len(sample_for_dev) + 1e-8)
print(f"  Subject deviation: 2 features")

# ============================================================
# IDENTIFY NEW FEATURE COLUMNS (excluding targets, meta, datetime)
# ============================================================
EXCLUDE = {'sleep_date_dt', 'subject_deviation'}
new_feat_cols = []
for c in feat.columns:
    if c in ORIGINAL_BASE_COLS:
        continue
    if c in TARGETS or c in EXCLUDE:
        continue
    if c not in {'subject_id', 'lifelog_date', 'sleep_date', 'date'}:
        if feat[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]:
            if feat[c].std() > 0.001:
                new_feat_cols.append(c)

print(f"\nNew feature columns: {len(new_feat_cols)}")
print(f"  {new_feat_cols}")

# ============================================================
# TRAIN BASELINE vs ENHANCED
# ============================================================
print("\n" + "=" * 60)
print("Training: Baseline vs Baseline + New Features")
print("=" * 60)

baseline_results = {}
enhanced_results = {}

for target in TARGETS:
    sw = V53_SWEEP[target]
    cfg = CFGS[sw['cfg']]
    y = feat[target].values.astype(np.float64)
    
    # Baseline: original base features only
    base_cols_clean = remove_leak(ORIGINAL_BASE_COLS, target)
    base_cols_clean = [c for c in base_cols_clean if c in feat.columns and feat[c].std() > 0.001]
    
    oof_base, _ = train_cv(feat, None, base_cols_clean, y, SEEDS, cfg, target)
    oof_base_avg = np.clip(oof_base.mean(axis=1), 0.0001, 0.9999)
    cal_base = mean_match(oof_base_avg, y.mean())
    ll_base = log_loss(y, cal_base, labels=[0, 1])
    baseline_results[target] = ll_base
    
    # Enhanced: base + new features
    all_cols = base_cols_clean + new_feat_cols
    all_cols = remove_leak(all_cols, target)
    all_cols = [c for c in all_cols if c in feat.columns and feat[c].std() > 0.001]
    
    oof_enh, _ = train_cv(feat, None, all_cols, y, SEEDS, cfg, target)
    oof_enh_avg = np.clip(oof_enh.mean(axis=1), 0.0001, 0.9999)
    cal_enh = mean_match(oof_enh_avg, y.mean())
    ll_enh = log_loss(y, cal_enh, labels=[0, 1])
    enhanced_results[target] = ll_enh
    
    delta = ll_enh - ll_base
    print(f"  {target}: baseline={ll_base:.5f}, enhanced={ll_enh:.5f}, Δ={delta:+.5f} (base={len(base_cols_clean)}, enhanced={len(all_cols)})")

avg_baseline = np.mean(list(baseline_results.values()))
avg_enhanced = np.mean(list(enhanced_results.values()))
avg_delta = avg_enhanced - avg_baseline

print(f"\n  AVG BASELINE: {avg_baseline:.5f}")
print(f"  AVG ENHANCED: {avg_enhanced:.5f}")
print(f"  AVG DELTA:    {avg_delta:+.5f}")

# ============================================================
# FEATURE IMPORTANCE ANALYSIS (which new features help?)
# ============================================================
print("\n" + "=" * 60)
print("Feature Importance: Top New Features")
print("=" * 60)

# Train a single model to check feature importance
target = 'S1'  # representative
sw = V53_SWEEP[target]
cfg = CFGS[sw['cfg']]
y = feat[target].values.astype(np.float64)
base_cols_clean = remove_leak(ORIGINAL_BASE_COLS, target)
base_cols_clean = [c for c in base_cols_clean if c in feat.columns and feat[c].std() > 0.001]
all_cols_imp = base_cols_clean + new_feat_cols
all_cols_imp = remove_leak(all_cols_imp, target)
all_cols_imp = [c for c in all_cols_imp if c in feat.columns and feat[c].std() > 0.001]

# Single seed, 1 fold for importance
gkf = GroupKFold(n_splits=5)
for tri, vai in gkf.split(feat, y, feat['subject_id']):
    break  # just need one fold
X_tr = feat[all_cols_imp].fillna(0).values.astype(np.float64)[tri]
params = dict(cfg)
params['scale_pos_weight'] = max(((y[tri]==0).sum()) / max((y[tri]==1).sum(), 1), 0.1)
params['random_state'] = 42
params['n_jobs'] = 1
params['force_row_wise'] = True
ds = lgb.Dataset(X_tr, label=y[tri], feature_name=[sanitize_col(c) for c in all_cols_imp])
m = lgb.train(params, ds, num_boost_round=100,
            callbacks=[lgb.log_evaluation(0)])

# Get importance
imp = m.feature_importance(importance_type='gain')
feat_names = all_cols_imp
feat_new_set = set(new_feat_cols)

# Top new features by importance
new_feat_imp = [(feat_names[i], imp[i]) for i in range(len(feat_names)) if feat_names[i] in feat_new_set]
new_feat_imp.sort(key=lambda x: x[1], reverse=True)

print(f"Top new features by importance gain:")
for name, score in new_feat_imp[:15]:
    bar = '█' * int(score / max(score, 1e-10) * 30) if score > 0 else ' '
    print(f"  {name:40s} {score:12.2f}  {bar}")

# ============================================================
# SAVE RESULT
# ============================================================
result = {
    "version": "v260_proxy",
    "name": "External Proxy Features (Fixed)",
    "features_created": len(new_feat_cols),
    "feature_list": new_feat_cols,
    "per_target": {},
    "avg_baseline_oof": float(avg_baseline),
    "avg_enhanced_oof": float(avg_enhanced),
    "avg_delta": float(avg_delta),
    "top_new_features": [{"feature": n, "importance_gain": float(s)} for n, s in new_feat_imp[:15]],
}
for t in TARGETS:
    result["per_target"][t] = {
        "baseline": float(baseline_results[t]),
        "enhanced": float(enhanced_results[t]),
        "delta": float(enhanced_results[t] - baseline_results[t])
    }

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
log_path = EXPERIMENTS / f'v260_external_proxy_{ts}.json'
with open(log_path, 'w') as f:
    json.dump(result, f, indent=2, default=str)
print(f"\nResult saved: {log_path}")
