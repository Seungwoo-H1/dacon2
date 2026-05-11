"""V110: Improved Ensemble - V54-family pipeline reproduction + optimization

Key lessons from V109:
- V109 was worse (0.66) than V53 (0.5479) because personalization + feature selection broke the pipeline
- V54/V53/V83 used features.parquet directly (no personalization added) 
- V108 used ALL features (no top-20 selection) and got 0.53971
- V106 ensemble of V54+V83+V54_re2+V55+V53+V55_re2 gave 0.5250

V110 Strategy:
1. Reproduce V108 pipeline (all features, no personalization, multi-config)
2. Add 4 more seeds (2026, 777, 111, 555) 
3. Add interaction features on top of base features
4. Try per-target feature selection based on V108 importance
5. Optimize ensemble weights with non-uniform weighting
6. Try isotonic regression calibration
"""
import sys, re, gc, time, warnings, logging, json, os
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

ROOT = Path('/home/mwoo423/projects/dacon2')
DATA = ROOT / "data_processed"
SUBMIT = ROOT / "submissions"
EXPERIMENTS = ROOT / "experiments"
TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']
META = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}

LEAK_S = {
    'wLight_w_light_mean','wLight_w_light_std','wLight_w_light_min',
    'wLight_w_light_max','wLight_w_light_count',
    'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max',
    'wHr_hr_median','wHr_hr_count',
    'wPedo_pedo_step_mean','wPedo_pedo_step_sum',
    'wPedo_pedo_step_frequency_mean','wPedo_pedo_step_frequency_sum',
    'wPedo_pedo_running_step_mean','wPedo_pedo_running_step_sum',
    'wPedo_pedo_walking_step_mean','wPedo_pedo_walking_step_sum',
    'wPedo_pedo_distance_mean','wPedo_pedo_distance_sum',
    'wPedo_pedo_speed_mean','wPedo_pedo_speed_sum',
    'wPedo_pedo_burned_calories_mean','wPedo_pedo_burned_calories_sum',
}
LEAK_Q = {'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max','wHr_hr_median','wHr_hr_count'}

# Base configs (same as V108)
CFG_WIDE = {'nl': 30, 'md': 3, 'lr': 0.05, 'ne': 300, 'ss': 0.8, 'cb': 0.8, 'ra': 2.0, 'rl': 5.0, 'mc': 5}
CFG_DEEP = {'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 1000, 'ss': 0.7, 'cb': 0.6, 'ra': 0.5, 'rl': 2.0, 'mc': 15}
CFG_V48 = {'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 500, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10}
CFG_SAFETY = {'nl': 10, 'md': 3, 'lr': 0.02, 'ne': 1000, 'ss': 0.6, 'cb': 0.6, 'ra': 3.0, 'rl': 10.0, 'mc': 20}
CFG_V53WIDE = {'nl': 30, 'md': 3, 'lr': 0.05, 'ne': 300, 'ss': 0.8, 'cb': 0.8, 'ra': 2.0, 'rl': 5.0, 'mc': 5}
CFG_V53DEEP = {'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 1000, 'ss': 0.7, 'cb': 0.6, 'ra': 0.5, 'rl': 2.0, 'mc': 15}
CFG_V110_A = {'nl': 25, 'md': 4, 'lr': 0.04, 'ne': 400, 'ss': 0.75, 'cb': 0.75, 'ra': 1.5, 'rl': 4.0, 'mc': 8}
CFG_V110_B = {'nl': 12, 'md': 3, 'lr': 0.015, 'ne': 1500, 'ss': 0.65, 'cb': 0.65, 'ra': 2.0, 'rl': 8.0, 'mc': 12}
CFGS = {'wide': CFG_WIDE, 'deep': CFG_DEEP, 'v48': CFG_V48,
        'safety': CFG_SAFETY, 'v53wide': CFG_V53WIDE, 'v53deep': CFG_V53DEEP,
        'v110a': CFG_V110_A, 'v110b': CFG_V110_B}

# Multiple seed sets
SEEDS_BASE = [42, 123, 7, 999]
SEEDS_EXT = [42, 123, 7, 999, 777, 2026, 111, 555]

def sanitize(n):
    return re.sub(r'[^a-zA-Z0-9_]','_',n)

def mean_match(pred, target_mean):
    return np.clip(pred + (target_mean - pred.mean()), 0.0001, 0.9999)

# ============================================================
# Load data
# ============================================================
t_start = time.time()
log.info("Loading features...")

train_df = pd.read_parquet(DATA / "features.parquet")
y_train = {t: train_df[t].values for t in TARGETS}

feat = train_df.copy()
feat_test = pd.read_parquet(DATA / "test_features.parquet")

for df in [feat, feat_test]:
    for c in ['sleep_date', 'lifelog_date', 'date']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')

# Base feature columns (141 raw features)
feature_cols = [c for c in feat.columns
                if c not in META | set(TARGETS)
                and feat[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]
log.info(f"Base features: {len(feature_cols)}")

# ============================================================
# Feature engineering: interaction features
# ============================================================
log.info("Building interaction features...")

feat_eng = feat.copy()
feat_test_eng = feat_test.copy()

# Interaction 1: hr × activity (heart rate and activity correlation)
for fn in ['hr_mean', 'hr_std', 'hr_max']:
    for an in ['activity_mean', 'activity_std', 'activity_max']:
        if f'mHr_{fn}' in feature_cols and f'mActivity_{an}' in feature_cols:
            feat_eng[f'inter_hr_act_{fn}_{an}'] = feat['mHr_'+fn] * feat['mActivity_'+an]
            feat_test_eng[f'inter_hr_act_{fn}_{an}'] = feat_test['mHr_'+fn] * feat_test['mActivity_'+an]

# Interaction 2: light × GPS
for ln in ['light_mean', 'light_std', 'light_count']:
    for gn in ['gps_mean', 'gps_max', 'gps_std']:
        if f'mLight_m_light_{ln}' in feature_cols and f'mGps_gps_{gn}' in feature_cols:
            feat_eng[f'inter_light_gps_{ln}_{gn}'] = feat['mLight_m_light_'+ln] * feat['mGps_gps_'+gn]
            feat_test_eng[f'inter_light_gps_{ln}_{gn}'] = feat_test['mLight_m_light_'+ln] * feat_test['mGps_gps_'+gn]

# Interaction 3: wifi × ambience
for wn in ['wifi_mean', 'wifi_std', 'wifi_max', 'wifi_count']:
    for wn2 in ['ambience_mean', 'ambience_std']:
        if f'mWifi_wifi_{wn}' in feature_cols and f'mAmbience_{wn2}' in feature_cols:
            feat_eng[f'inter_wifi_amb_{wn}_{wn2}'] = feat['mWifi_wifi_'+wn] * feat['mAmbience_'+wn2]
            feat_test_eng[f'inter_wifi_amb_{wn}_{wn2}'] = feat_test['mWifi_wifi_'+wn] * feat_test['mAmbience_'+wn2]

# Interaction 4: charging × activity
for cn in ['charging_mean', 'charging_std', 'charging_count']:
    for an in ['activity_mean', 'activity_std']:
        if f'mACStatus_m_{cn}' in feature_cols and f'mActivity_{an}' in feature_cols:
            feat_eng[f'inter_charge_act_{cn}_{an}'] = feat['mACStatus_m_'+cn] * feat['mActivity_'+an]
            feat_test_eng[f'inter_charge_act_{cn}_{an}'] = feat_test['mACStatus_m_'+cn] * feat_test['mActivity_'+an]

# Interaction 5: UsageStats × screen
for un in ['usage_total_time_mean', 'usage_app_count_mean', 'usage_major_ratio_mean']:
    for sn in ['screen_use_mean', 'screen_use_std']:
        if f'mUsageStats_{un}' in feature_cols and f'mScreenStatus_{sn}' in feature_cols:
            feat_eng[f'inter_usage_scr_{un}_{sn}'] = feat['mUsageStats_'+un] * feat['mScreenStatus_'+sn]
            feat_test_eng[f'inter_usage_scr_{un}_{sn}'] = feat_test['mUsageStats_'+un] * feat_test['mScreenStatus_'+sn]

# Interaction 6: GPS speed × activity
for sn in ['gps_max_speed_mean', 'gps_max_speed_std', 'gps_max_speed_max']:
    for an in ['activity_mean', 'activity_std', 'activity_max']:
        if f'mGps_{sn}' in feature_cols and f'mActivity_{an}' in feature_cols:
            feat_eng[f'inter_gps_act_{sn}_{an}'] = feat['mGps_'+sn] * feat['mActivity_'+an]
            feat_test_eng[f'inter_gps_act_{sn}_{an}'] = feat_test['mGps_'+sn] * feat_test['mActivity_'+an]

# Interaction 7: BLE × WiFi (device diversity)
for bn in ['ble_device_count_mean', 'ble_count_mean', 'ble_rssi_std_mean']:
    for wn in ['wifi_max_rssi_mean', 'wifi_count_mean']:
        if f'mBle_{bn}' in feature_cols and f'mWifi_{wn}' in feature_cols:
            feat_eng[f'inter_ble_wifi_{bn}_{wn}'] = feat['mBle_'+bn] * feat['mWifi_'+wn]
            feat_test_eng[f'inter_ble_wifi_{bn}_{wn}'] = feat_test['mBle_'+bn] * feat_test['mWifi_'+wn]

# Interaction 8: Ratio features (activity/light ratio, etc.)
if 'mActivity_m_activity_mean' in feature_cols and 'mLight_m_light_mean' in feature_cols:
    feat_eng['ratio_activity_light'] = feat['mActivity_m_activity_mean'] / (feat['mLight_m_light_mean'] + 1e-8)
    feat_test_eng['ratio_activity_light'] = feat_test['mActivity_m_activity_mean'] / (feat_test['mLight_m_light_mean'] + 1e-8)

if 'wPedo_pedo_step_mean' in feature_cols and 'wLight_w_light_count' in feature_cols:
    feat_eng['ratio_step_light'] = feat['wPedo_pedo_step_mean'] / (feat['wLight_w_light_count'] + 1e-8)
    feat_test_eng['ratio_step_light'] = feat_test['wPedo_pedo_step_mean'] / (feat_test['wLight_w_light_count'] + 1e-8)

inter_cols = [c for c in feat_eng.columns if c.startswith('inter_') or c.startswith('ratio_')]
log.info(f"Interaction features added: {len(inter_cols)}")
log.info(f"Total features: {len(feat_eng.columns) - 4}")  # minus meta + targets

all_features = feature_cols + inter_cols
log.info(f"All features: {len(all_features)}")

# ============================================================
# Run multiple experiment directions
# ============================================================
log.info("\n" + "=" * 60)
log.info("V110: Running experiment directions")
log.info("=" * 60)

results = {}

# Exp directions:
# A: V54_family (wide + deep + v53wide + v53deep) × 8 seeds
# B: V48_family (v48 + safety) × 8 seeds
# C: V110_new (v110a + v110b + wide + deep) × 8 seeds
# D: Cross-validation weight optimization

for exp_name, configs, seeds, use_interactions, desc in [
    ('A_V54family', ['wide', 'deep', 'v53wide', 'v53deep'], SEEDS_EXT, True, 'V54 family + 8 seeds + interactions'),
    ('B_V48family', ['v48', 'safety'], SEEDS_EXT, True, 'V48 family + 8 seeds + interactions'),
    ('C_V110new', ['v110a', 'v110b', 'wide', 'deep', 'v48', 'safety'], SEEDS_EXT, True, 'V110 new configs + 8 seeds + interactions'),
]:
    t_exp = time.time()
    log.info(f"\n{'='*60}")
    log.info(f"EXP {exp_name}: {desc}")
    log.info(f"{'='*60}")
    
    oof_all = {t: [] for t in TARGETS}
    test_all = {t: [] for t in TARGETS}
    
    feat_use = feat_eng if use_interactions else feat
    feat_test_use = feat_test_eng if use_interactions else feat_test
    
    for target in TARGETS:
        y = feat_use[target].values.astype(np.float64)
        train_rate = y.mean()
        spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
        
        leak = LEAK_S if target.startswith('S') else LEAK_Q
        safe_cols = [c for c in all_features if c not in leak]
        
        sn = [sanitize(c) for c in safe_cols]
        gkf = GroupKFold(n_splits=5)
        
        for cfg_name in configs:
            cfg = CFGS[cfg_name]
            
            for seed in seeds:
                oof_fold = np.zeros(450)
                
                for tr_i, va_i in gkf.split(feat_use, y, feat_use['subject_id']):
                    X_tr = feat_use.iloc[tr_i][safe_cols].fillna(0).values.astype(np.float64)
                    X_va = feat_use.iloc[va_i][safe_cols].fillna(0).values.astype(np.float64)
                    
                    ds_train = lgb.Dataset(X_tr, label=y[tr_i], feature_name=sn, params={'verbose': '-1'})
                    ds_val = lgb.Dataset(X_va, label=y[va_i], feature_name=sn, reference=ds_train, params={'verbose': '-1'})
                    
                    cfg_full = {
                        'objective': 'binary', 'metric': 'binary_logloss',
                        'verbose': -1, 'force_row_wise': True, 'n_jobs': 1,
                        'num_leaves': cfg['nl'], 'max_depth': cfg['md'],
                        'learning_rate': cfg['lr'], 'n_estimators': cfg['ne'],
                        'subsample': cfg['ss'], 'colsample_bytree': cfg['cb'],
                        'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
                        'min_child_samples': cfg['mc'],
                        'random_state': seed, 'scale_pos_weight': spw,
                    }
                    
                    model = lgb.train(cfg_full, ds_train, num_boost_round=cfg['ne'],
                                     valid_sets=[ds_val],
                                     callbacks=[lgb.early_stopping(30, verbose=False),
                                               lgb.log_evaluation(0)])
                    
                    oof_fold[va_i] = model.predict(X_va)
                    
                    del ds_train, ds_val, model
                    gc.collect()
                
                oof_all[target].append(oof_fold)
                
                # Test prediction
                X_all_feat = feat_use[safe_cols].fillna(0).values.astype(np.float64)
                X_test_feat = feat_test_use[safe_cols].fillna(0).values.astype(np.float64)
                
                ds_all = lgb.Dataset(X_all_feat, label=y, feature_name=sn, params={'verbose': '-1'})
                cfg_all = {**cfg_full, 'n_estimators': cfg['ne']}
                model_test = lgb.train(cfg_all, ds_all, num_boost_round=cfg['ne'])
                test_pred = model_test.predict(X_test_feat)
                test_all[target].append(test_pred)
                
                del ds_all, model_test
                gc.collect()
        
        # Average
        oof_avg = np.array(oof_all[target]).mean(axis=0)
        test_avg = np.array(test_all[target]).mean(axis=0)
        oof_all[target] = [oof_avg]
        test_all[target] = [test_avg]
    
    # Evaluate
    avg_oof = 0
    target_metrics = {}
    for t in TARGETS:
        oof_cal = mean_match(oof_all[t][0], train_rate)
        test_cal = mean_match(test_all[t][0], train_rate)
        ll = log_loss(y_train[t], oof_cal, labels=[0,1])
        target_metrics[t] = {
            'oof_ll': round(ll, 5),
            'oof_mean': round(float(oof_cal.mean()), 4),
            'test_mean': round(float(test_cal.mean()), 4),
            'oof_std': round(float(np.std(oof_cal)), 4),
            'n_models': len(oof_all[t]),
        }
        avg_oof += ll
    avg_oof /= 7
    
    results[exp_name] = {
        'oof': {t: oof_all[t][0] for t in TARGETS},
        'test': {t: test_all[t][0] for t in TARGETS},
        'metrics': target_metrics,
        'avg_oof': avg_oof,
        'time': time.time() - t_exp,
        'configs': configs,
        'seeds': seeds,
        'interactions': use_interactions,
    }
    
    log.info(f"\n{'Target':<10} {'OOF LL':>8} {'Models':>8} {'OOF mean':>10} {'Test mean':>10}")
    for t in TARGETS:
        m = target_metrics[t]
        log.info(f"{t:<10} {m['oof_ll']:>8.5f} {m['n_models']:>8} {m['oof_mean']:>10.4f} {m['test_mean']:>10.4f}")
    log.info(f"{'AVG':<10} {avg_oof:>8.5f} (time: {time.time()-t_exp:.0f}s)")

# ============================================================
# Ensemble optimization: find best weights across experiments
# ============================================================
log.info("\n" + "=" * 60)
log.info("ENSEMBLE OPTIMIZATION")
log.info("=" * 60)

exp_names = list(results.keys())

# For each experiment, get per-target OOF arrays
exp_oof_arrays = {en: np.array([results[en]['oof'][t] for t in TARGETS]) for en in exp_names}

# Try weighted ensemble: minimize OOF LL
# Use simple grid search over weights
from itertools import product as iprod

best_oof = float('inf')
best_weights = None
best_combo = None

# Equal weight for all experiments
n_exp = len(exp_names)
combined = np.zeros((7, 450))
for i, en in enumerate(exp_names):
    combined[i] = results[en]['avg_oof']
    
log.info(f"Individual experiment OOFs:")
for en in exp_names:
    log.info(f"  {en}: {results[en]['avg_oof']:.5f} (configs={results[en]['configs']}, seeds={len(results[en]['seeds'])}, inter={results[en]['interactions']})")

# Try all subsets with equal weights
from itertools import combinations as comb

for r in range(1, len(exp_names) + 1):
    for combo in comb(exp_names, r):
        combined_oof = np.zeros(450)
        combined_test = np.zeros(250)
        for en in combo:
            for t in TARGETS:
                combined_oof += results[en]['oof'][t]
                combined_test += results[en]['test'][t]
        n = len(combo)
        combined_oof /= n
        combined_test /= n
        
        # Mean match per target
        combined_oof_cal = np.column_stack([
            mean_match(combined_oof[j], feat[target].mean())
            for j, target in enumerate(TARGETS)
        ]).T  # Actually each target has its own array
        
        # Per-target mean match
        combined_oof_cal = np.zeros((7, 450))
        combined_test_cal = np.zeros((7, 250))
        for j, t in enumerate(TARGETS):
            combined_oof_cal[j] = mean_match(combined_oof[j], feat[t].mean())
            combined_test_cal[j] = mean_match(combined_test[j], feat[t].mean())
        
        y_flat = np.array([y_train[t] for t in TARGETS])
        avg_ll = np.mean([log_loss(y_train[t], combined_oof_cal[j], labels=[0,1]) for j, t in enumerate(TARGETS)])
        
        if avg_ll < best_oof:
            best_oof = avg_ll
            best_weights = {en: 1.0/n for en in combo}
            best_combo = list(combo)

log.info(f"\nBest ensemble: {best_combo}")
log.info(f"Best OOF: {best_oof:.5f}")
for en in exp_names:
    if en in best_weights:
        log.info(f"  {en}: weight={best_weights[en]:.3f}")

# Per-target best
log.info("\nPer-target ensemble OOF:")
for j, t in enumerate(TARGETS):
    ll = log_loss(y_train[t], combined_oof_cal[j], labels=[0,1])
    log.info(f"  {t}: {ll:.5f}")

# ============================================================
# Save
# ============================================================
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

oof_df = pd.DataFrame({t: combined_oof_cal[j] for j, t in enumerate(TARGETS)})
oof_df.insert(0, 'subject_id', feat['subject_id'].values)
oof_df.insert(1, 'sleep_date', feat['sleep_date'].values)
oof_df.insert(2, 'lifelog_date', feat['lifelog_date'].values)
oof_path = DATA / f'oof_v110_{ts}.csv'
oof_df.to_csv(oof_path, index=False)

sub_df = pd.DataFrame({t: combined_test_cal[j] for j, t in enumerate(TARGETS)})
sub_df.insert(0, 'subject_id', feat_test['subject_id'].values)
sub_path = SUBMIT / f'submission_v110_{ts}.csv'
sub_df.to_csv(sub_path, index=False)

exp_log = {
    'version': 'V110',
    'timestamp': ts,
    'experiments': {en: {
        'oof': round(results[en]['avg_oof'], 5),
        'configs': results[en]['configs'],
        'n_seeds': len(results[en]['seeds']),
        'interactions': results[en]['interactions'],
        'time_s': round(results[en]['time'], 0),
    } for en in exp_names},
    'ensemble': {
        'best_combo': best_combo,
        'best_weights': {k: round(v, 4) for k, v in best_weights.items()},
        'best_oof': round(best_oof, 5),
        'per_target_oof': {t: round(log_loss(y_train[t], combined_oof_cal[j], labels=[0,1]), 5) for j, t in enumerate(TARGETS)},
    },
    'test_submission': str(sub_path.name),
    'test_means': {t: round(sub_df[t].mean(), 4) for t in TARGETS},
    'test_stds': {t: round(sub_df[t].std(), 4) for t in TARGETS},
    'n_interaction_features': len(inter_cols),
    'total_features': len(all_features),
    'total_time_s': round(time.time() - t_start, 0),
}
with open(EXPERIMENTS / f'v110_{ts}.json', 'w') as f:
    json.dump(exp_log, f, indent=2, default=str)

log.info(f"\nSaved OOF: {oof_path}")
log.info(f"Saved submission: {sub_path}")
log.info(f"Log: {EXPERIMENTS / f'v110_{ts}.json'}")
log.info(f"Done in {time.time()-t_start:.0f}s")
