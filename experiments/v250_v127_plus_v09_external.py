"""
V250: V127 Baseline + V09 External Feature Engineering

Combines:
1. V127 baseline pipeline (GroupKFold 5-fold, per-target config/n_feat/seeds)
2. V09 external proxy features (9 features derived from internal sensors)
3. Target-specific external selection (n_ext=1 optimal per V08/V09)

Key difference from V09: Uses features_clean_v60.parquet (already has z-scores)
and adds external features as additional columns before feature selection.

Hypothesis: V09 external features can improve V127 baseline OOF by ~0.02
→ Expected LB: ~0.62-0.63 (from ~0.648)
"""

import os, sys, gc, re, json, warnings, time, hashlib
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)  # line-buffered
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
warnings.filterwarnings('ignore')

ROOT = Path('/home/mwoo423/projects/dacon2')
DATA = ROOT / 'data_processed'
EXTERNAL = ROOT / 'external_data'
EXPERIMENTS = ROOT / 'experiments'
SUBMIT = ROOT / 'submissions'
for d in [EXPERIMENTS, SUBMIT]:
    d.mkdir(exist_ok=True)

TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']
META_COLS = {'subject_id','lifelog_date','sleep_date','date'}
SEEDS = [42, 7, 999, 777]

# --- V127 configs ---
V53_SWEEP = {
    'Q1': {'cfg': 'deep', 'n_feat': 19},
    'Q2': {'cfg': 'deep', 'n_feat': 14},
    'Q3': {'cfg': 'v48', 'n_feat': 11},
    'S1': {'cfg': 'wide', 'n_feat': 21},
    'S2': {'cfg': 'deep', 'n_feat': 19},
    'S3': {'cfg': 'safety','n_feat': 23},
    'S4': {'cfg': 'wide', 'n_feat': 20},
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

# --- V10 multi-config ensemble weights ---
V10_ENSEMBLE = {
    'Q1':  {'wide': 0.5, 'deep': 0.5},
    'Q2':  {'deep': 1.0},
    'Q3':  {'wide': 0.45, 'deep': 0.45, 'safety': 0.1},
    'S1':  {'v48': 0.6, 'wide': 0.2, 'deep': 0.2},
    'S2':  {'deep': 1.0},
    'S3':  {'v48': 0.75, 'safety': 0.25},
    'S4':  {'v48': 0.8, 'wide': 0.2},
}

# Leakage columns (same as V127)
LEAK_S = {'wlight_w_light_mean','wlight_w_light_std','wlight_w_light_min','wlight_w_light_max','wlight_w_light_count',
          'whr_hr_mean','whr_hr_std','whr_hr_min','whr_hr_max','whr_hr_median','whr_hr_count',
          'wpedo_pedo_step_mean','wpedo_pedo_step_sum','wpedo_pedo_step_frequency_mean','wpedo_pedo_step_frequency_sum',
          'wpedo_pedo_running_step_mean','wpedo_pedo_running_step_sum','wpedo_pedo_walking_step_mean','wpedo_pedo_walking_step_sum',
          'wpedo_pedo_distance_mean','wpedo_pedo_distance_sum','wpedo_pedo_speed_mean','wpedo_pedo_speed_sum',
          'wpedo_pedo_burned_calories_mean','wpedo_pedo_burned_calories_sum'}
LEAK_Q = {'whr_hr_mean','whr_hr_std','whr_hr_min','whr_hr_max','whr_hr_median','whr_hr_count'}

def sanitize_col(n):
    return re.sub(r'[^a-zA-Z0-9_]', '_', n)

def mean_match(pred, tm):
    return np.clip(pred + (tm - pred.mean()), 0.0001, 0.9999)

def remove_leak(cols, t):
    if t.startswith('S'): return [c for c in cols if c not in LEAK_S]
    elif t.startswith('Q'): return [c for c in cols if c not in LEAK_Q]
    return cols

def get_feature_cols(df):
    ex = META_COLS | set(TARGETS)
    return [c for c in df.columns if c not in ex and df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]

def rank_features(feat, fcols, target, seed=42):
    """Rank features by LightGBM importance (50 trees)."""
    y = feat[target].values.astype(np.float64)
    X = feat[fcols].fillna(0).values.astype(np.float64)
    spw = max(((y==0).sum()) / max((y==1).sum(), 1), 0.1)
    p = {
        'objective':'binary','metric':'binary_logloss','verbose':-1,
        'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':50,
        'subsample':0.7,'colsample_bytree':0.7,'reg_alpha':1.0,'reg_lambda':3.0,
        'scale_pos_weight':spw,'random_state':seed,'min_child_samples':10,
        'force_row_wise':True,'n_jobs':1,
    }
    sn = [sanitize_col(c) for c in fcols]
    ds = lgb.Dataset(X, label=y, feature_name=sn)
    m = lgb.train(p, ds, num_boost_round=50)
    imp = m.feature_importance(importance_type='gain')
    ranked = sorted(zip(fcols, imp), key=lambda x: -x[1])
    del m, ds; gc.collect()
    return [r[0] for r in ranked]

def cfg_to_params(cfg_s, seed, spw):
    params = dict(cfg_s)
    params['scale_pos_weight'] = spw
    params['random_state'] = seed
    params['force_row_wise'] = True
    params['n_jobs'] = 1
    return params

def train_cv(feat, ftst, cols, y, seeds, cfg):
    """Train with GroupKFold 5-fold CV. Returns (oof, test_preds)."""
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros((len(y), len(seeds)))
    tp = np.zeros((len(ftst), len(seeds))) if ftst is not None else None
    sn = [sanitize_col(c) for c in cols]
    Xf = feat[cols].fillna(0).values.astype(np.float64)
    Xt = ftst[cols].fillna(0).values.astype(np.float64) if ftst is not None else None
    spw = max(((y==0).sum()) / max((y==1).sum(), 1), 0.1)

    for si, seed in enumerate(seeds):
        p = cfg_to_params(cfg, seed, spw)
        for tri, vai in gkf.split(feat, y, feat['subject_id']):
            ds = lgb.Dataset(Xf[tri], label=y[tri], feature_name=sn)
            if Xt is not None:
                vd = lgb.Dataset(Xf[vai], label=y[vai], feature_name=sn, reference=ds)
                m = lgb.train(p, ds, num_boost_round=cfg['n_estimators'],
                             valid_sets=[vd],
                             callbacks=[lgb.early_stopping(50, verbose=False),
                                       lgb.log_evaluation(0)])
                oof[vai, si] = m.predict(Xf[vai])
                tp[:, si] = m.predict(Xt)
                del vd
            else:
                m = lgb.train(p, ds, num_boost_round=cfg['n_estimators'],
                             callbacks=[lgb.log_evaluation(0)])
                oof[vai, si] = m.predict(Xf[vai])
            del ds, m; gc.collect()

    if tp is not None:
        tp = np.clip(tp, 0.0001, 0.9999)
    return oof, tp

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 60)
print("V250: V127 + V09 External Feature Engineering")
print("=" * 60)

print("\nLoading features_clean_v60...")
feat = pd.read_parquet(DATA / 'features_clean_v60.parquet')
ftst = pd.read_parquet(DATA / 'test_features_clean_v60.parquet')

# Sanitize column names (for LightGBM)
feat.columns = [sanitize_col(c) for c in feat.columns]
ftst.columns = [sanitize_col(c) for c in ftst.columns]

# Date columns to string
for df in [feat, ftst]:
    for c in ['sleep_date','lifelog_date','date']:
        if c in df.columns:
            df[c] = df[c].astype(str)

print(f"Train: {feat.shape}, Test: {ftst.shape}")

# ============================================================
# ADD EXTERNAL PROXY FEATURES (from V09)
# ============================================================
print("\nAdding external proxy features...")

f = feat.copy()
ft = ftst.copy()
fcols_all = get_feature_cols(f)
print(f"Base features: {len(fcols_all)}")

added_features = {}

# 1. ext_activity_z — step mean z-score
if 'wpedo_pedo_step_mean' in fcols_all:
    s = f['wpedo_pedo_step_mean'].fillna(0); s_t = ft['wpedo_pedo_step_mean'].fillna(0)
    f['ext_activity_z'] = (s - s.mean()) / max(s.std(), 1e-8)
    ft['ext_activity_z'] = (s_t - s.mean()) / max(s_t.std(), 1e-8)
    added_features['ext_activity_z'] = True

# 2. ext_charging_z — charging mean z-score
if 'macstatus_m_charging_mean' in fcols_all:
    ch = f['macstatus_m_charging_mean'].fillna(0); ch_t = ft['macstatus_m_charging_mean'].fillna(0)
    f['ext_charging_z'] = (ch - ch.mean()) / max(ch.std(), 1e-8)
    ft['ext_charging_z'] = (ch_t - ch.mean()) / max(ch_t.std(), 1e-8)
    added_features['ext_charging_z'] = True

# 3. ext_health_composite — step(+) - charging(-) + screen(+) * 0.3 + hr(*) * 0.1
req3 = ['wpedo_pedo_step_mean','macstatus_m_charging_mean','screenstatus_m_screen_use_mean','whr_hr_mean']
if all(c in fcols_all for c in req3):
    sa = f['wpedo_pedo_step_mean'].fillna(0); sc_h = f['macstatus_m_charging_mean'].fillna(0)
    ss = f['screenstatus_m_screen_use_mean'].fillna(0); hr = f['whr_hr_mean'].fillna(0)
    sa_t = ft['wpedo_pedo_step_mean'].fillna(0); sc_t = ft['macstatus_m_charging_mean'].fillna(0)
    ss_t = ft['screenstatus_m_screen_use_mean'].fillna(0); hr_t = ft['whr_hr_mean'].fillna(0)
    f['ext_health_composite'] = (
        (sa - sa.mean()) / max(sa.std(), 1e-8)
        - (sc_h - sc_h.mean()) / max(sc_h.std(), 1e-8)
        + (ss - ss.mean()) / max(ss.std(), 1e-8) * 0.3
        + (hr - hr.mean()) / max(hr.std(), 1e-8) * 0.1
    )
    ft['ext_health_composite'] = (
        (sa_t - sa.mean()) / max(sa.std(), 1e-8)
        - (sc_t - sc_t.mean()) / max(sc_t.std(), 1e-8)
        + (ss_t - ss_t.mean()) / max(ss_t.std(), 1e-8) * 0.3
        + (hr_t - hr_t.mean()) / max(hr_t.std(), 1e-8) * 0.1
    )
    added_features['ext_health_composite'] = True

# 4. ext_night_light — light_mean / hour_night ratio
if 'wlight_w_light_mean' in fcols_all and 'macstatus_hour_night' in fcols_all:
    f['ext_night_light'] = f['wlight_w_light_mean'].fillna(0) / (f['macstatus_hour_night'].fillna(0) + 1e-8)
    ft['ext_night_light'] = ft['wlight_w_light_mean'].fillna(0) / (ft['macstatus_hour_night'].fillna(0) + 1e-8)
    added_features['ext_night_light'] = True

# 5. ext_total_ambience — sum of ambience features
amb_cols = [c for c in fcols_all if 'ambience' in c.lower() and c.endswith('_sum')]
if amb_cols:
    f['ext_total_ambience'] = f[amb_cols].fillna(0).sum(axis=1)
    ft['ext_total_ambience'] = ft[amb_cols].fillna(0).sum(axis=1)
    added_features['ext_total_ambience'] = True

# 6. ext_hr_step — heart_rate * step_mean
if 'whr_hr_mean' in fcols_all and 'wpedo_pedo_step_mean' in fcols_all:
    f['ext_hr_step'] = f['whr_hr_mean'].fillna(0) * f['wpedo_pedo_step_mean'].fillna(0)
    ft['ext_hr_step'] = ft['whr_hr_mean'].fillna(0) * ft['wpedo_pedo_step_mean'].fillna(0)
    added_features['ext_hr_step'] = True

# 7. ext_screen_ratio — screen_use / (screen_use + epsilon)
if 'screenstatus_m_screen_use_mean' in fcols_all:
    sm = f['screenstatus_m_screen_use_mean'].fillna(0); sm_t = ft['screenstatus_m_screen_use_mean'].fillna(0)
    f['ext_screen_ratio'] = sm / (sm + 1e-8)
    ft['ext_screen_ratio'] = sm_t / (sm_t + 1e-8)
    added_features['ext_screen_ratio'] = True

# 8. ext_wifi_ble — wifi_sum / (ble_sum + epsilon)
wifi_cols = [c for c in fcols_all if 'wifi' in c.lower() and c.endswith('_sum')]
ble_cols = [c for c in fcols_all if 'ble' in c.lower() and c.endswith('_sum')]
if wifi_cols and ble_cols:
    w = f[wifi_cols].fillna(0).sum(axis=1); b = f[ble_cols].fillna(0).sum(axis=1)
    w_t = ft[wifi_cols].fillna(0).sum(axis=1); b_t = ft[ble_cols].fillna(0).sum(axis=1)
    f['ext_wifi_ble'] = w / (b + 1e-8)
    ft['ext_wifi_ble'] = w_t / (b_t + 1e-8)
    added_features['ext_wifi_ble'] = True

# 9. ext_activity_ambience — activity_z * total_ambience
if 'ext_activity_z' in f.columns and 'ext_total_ambience' in f.columns:
    f['ext_activity_ambience'] = f['ext_activity_z'] * f['ext_total_ambience']
    ft['ext_activity_ambience'] = ft['ext_activity_z'] * ft['ext_total_ambience']
    added_features['ext_activity_ambience'] = True

# 10. ext_step_consistency — step_std / step_mean
if 'wpedo_pedo_step_std' in fcols_all:
    f['ext_step_consistency'] = f['wpedo_pedo_step_std'].fillna(0) / (f['wpedo_pedo_step_mean'].fillna(0) + 1e-8)
    ft['ext_step_consistency'] = ft['wpedo_pedo_step_std'].fillna(0) / (ft['wpedo_pedo_step_mean'].fillna(0) + 1e-8)
    added_features['ext_step_consistency'] = True

print(f"Added {len(added_features)} external proxy features: {list(added_features.keys())}")

# Get all feature columns after adding external
all_cols = get_feature_cols(f)
non_const = [c for c in all_cols if f[c].std() > 0.001]  # minimal variance filter

ext_in = [c for c in non_const if c.startswith('ext_')]
non_ext_in = [c for c in non_const if not c.startswith('ext_')]
print(f"Total features: {len(non_const)} (ext: {len(ext_in)}, non-ext: {len(non_ext_in)})")

# Feature name mapping for targets
y_dict = {t: f[t].values.astype(np.float64) for t in TARGETS}

# ============================================================
# EXPERIMENT A: V127 Baseline (no external) — REFERENCE
# ============================================================
print("\n" + "=" * 60)
print("EXPERIMENT A: V127 Baseline (no external) — REFERENCE")
print("=" * 60)

baseline_results = {}
baseline_configs = {}

for target in TARGETS:
    sw = V53_SWEEP[target]
    cfg = CFGS[sw['cfg']]
    n_feat = sw['n_feat']
    y = y_dict[target]
    
    # Rank features (excluding external)
    leak_cols = remove_leak(non_ext_in, target)
    ranked = rank_features(f, leak_cols, target)
    
    # Top features
    top_cols = ranked[:max(n_feat, 25)]
    
    oof, _ = train_cv(f, None, top_cols, y, SEEDS, cfg)
    oof_avg = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
    cal = mean_match(oof_avg, y.mean())
    ll = log_loss(y, cal, labels=[0, 1])
    
    baseline_results[target] = ll
    baseline_configs[target] = {'cols': top_cols, 'cfg': sw['cfg'], 'n_feat': n_feat}
    
    print(f"  {target}: LL={ll:.5f} (cfg={sw['cfg']}, n_feat={n_feat})")

avg_baseline = np.mean(list(baseline_results.values()))
print(f"\n  AVG BASELINE OOF: {avg_baseline:.5f}")

# ============================================================
# EXPERIMENT B: V250 — V127 + V09 external features
# ============================================================
print("\n" + "=" * 60)
print("EXPERIMENT B: V250 (V127 + V09 External)")
print("=" * 60)

v250_results = {}
v250_configs = {}

for target in TARGETS:
    sw = V53_SWEEP[target]
    cfg = CFGS[sw['cfg']]
    n_feat = sw['n_feat']
    y = y_dict[target]
    
    # Rank ALL features (including external)
    all_cols_for_rank = non_ext_in + ext_in
    leak_cols = remove_leak(all_cols_for_rank, target)
    ranked = rank_features(f, leak_cols, target)
    
    ext_target = [c for c in ranked if c.startswith('ext_')]
    nonext_target = [c for c in ranked if not c.startswith('ext_')]
    
    # Strategy: try n_ext=0..4 with various n_total
    best_ll = float('inf')
    best_config = {}
    
    for n_ext in range(0, min(5, len(ext_target)) + 1):
        for n_total in range(max(n_ext, 8), 26):
            n_non = n_total - n_ext
            if n_non <= 0 or n_non > len(nonext_target):
                continue
            model_cols = ext_target[:n_ext] + nonext_target[:n_non]
            oof, _ = train_cv(f, None, model_cols, y, SEEDS, cfg)
            oof_avg = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
            ll = log_loss(y, mean_match(oof_avg, y.mean()), labels=[0, 1])
            
            if ll < best_ll:
                best_ll = ll
                best_config = {
                    'n_ext': n_ext,
                    'n_total': n_total,
                    'features': model_cols,
                    'cfg': sw['cfg'],
                }
    
    delta = best_ll - baseline_results[target]
    v250_results[target] = best_ll
    v250_configs[target] = best_config
    
    print(f"  {target}: LL={best_ll:.5f} base={baseline_results[target]:.5f} "
          f"Δ={delta:+.5f} (n_ext={best_config['n_ext']}, n_total={best_config['n_total']})")
    if best_config['n_ext'] > 0:
        print(f"    ext features: {best_config['features'][:best_config['n_ext']]}")

avg_v250 = np.mean(list(v250_results.values()))
avg_delta = avg_v250 - avg_baseline
print(f"\n  AVG V250 OOF: {avg_v250:.5f}")
print(f"  AVG DELTA:    {avg_delta:+.5f}")

# ============================================================
# EXPERIMENT C: V250 + V10 Multi-config Ensemble
# ============================================================
print("\n" + "=" * 60)
print("EXPERIMENT C: V250 + V10 Multi-config Ensemble")
print("=" * 60)

ens_results = {}

for target in TARGETS:
    y = y_dict[target]
    best = v250_configs[target]
    best_cols = best['features']
    
    v10_cfgs = V10_ENSEMBLE[target]
    
    # Train best config
    best_cfg = CFGS[best['cfg']]
    oof_best, _ = train_cv(f, None, best_cols, y, SEEDS, best_cfg)
    cal_best = mean_match(np.clip(oof_best.mean(axis=1), 0.0001, 0.9999), y.mean())
    ll_best = log_loss(y, cal_best, labels=[0, 1])
    
    # Train alternative configs
    alt_oofs = {}
    for cfg_name, weight in v10_cfgs.items():
        if cfg_name == best['cfg']:
            continue
        cfg = CFGS[cfg_name]
        oof, _ = train_cv(f, None, best_cols, y, SEEDS, cfg)
        cal = mean_match(np.clip(oof.mean(axis=1), 0.0001, 0.9999), y.mean())
        ll = log_loss(y, cal, labels=[0, 1])
        alt_oofs[cfg_name] = cal
        print(f"    {target}+{cfg_name}: LL={ll:.5f}")
    
    # Optimize ensemble weights
    cfg_names = list(alt_oofs.keys())
    best_ens_ll = ll_best
    
    if cfg_names:
        # Try all weight combinations
        for w_base in np.arange(0.3, 1.01, 0.1):
            n_alt = len(cfg_names)
            w_alt_each = (1 - w_base) / n_alt if n_alt > 0 else 0
            ens_cal = w_base * cal_best
            for cn in cfg_names:
                ens_cal = ens_cal + w_alt_each * alt_oofs[cn]
            ens_cal = mean_match(ens_cal, y.mean())
            ens_ll = log_loss(y, ens_cal, labels=[0, 1])
            if ens_ll < best_ens_ll:
                best_ens_ll = ens_ll
    
    delta_ens = best_ens_ll - baseline_results[target]
    ens_results[target] = best_ens_ll
    print(f"  {target}: ENS LL={best_ens_ll:.5f} Δ={delta_ens:+.5f}")

avg_ens = np.mean(list(ens_results.values()))
avg_ens_delta = avg_ens - avg_baseline
print(f"\n  AVG ENS OOF: {avg_ens:.5f}")
print(f"  AVG DELTA:   {avg_ens_delta:+.5f}")

# ============================================================
# LB ESTIMATION
# ============================================================
print("\n" + "=" * 60)
print("LB ESTIMATION")
print("=" * 60)

print(f"V127 baseline: OOF={avg_baseline:.5f}, est LB≈0.648")
print(f"V250 (ext):    OOF={avg_v250:.5f}, Δ={avg_delta:+.5f}, est LB≈{0.648 + avg_delta * 2.5:.5f}")
print(f"V250+ENS:      OOF={avg_ens:.5f}, Δ={avg_ens_delta:+.5f}, est LB≈{0.648 + avg_ens_delta * 2.5:.5f}")

# Shift analysis
print("\nShift analysis (V250 best config):")
shift_data = {}
for target in TARGETS:
    best = v250_configs[target]
    cfg = CFGS[best['cfg']]
    y = y_dict[target]
    oof, _ = train_cv(f, None, best['features'], y, SEEDS, cfg)
    oof_avg = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
    cal = mean_match(oof_avg, y.mean())
    
    train_mean = y.mean()
    oof_mean = cal.mean()
    shift = oof_mean - train_mean
    shift_data[target] = {
        'train_mean': float(train_mean),
        'oof_mean': float(oof_mean),
        'shift': float(shift),
    }
    print(f"  {target}: train={train_mean:.3f}, oof={oof_mean:.3f}, shift={shift:+.3f}")

# ============================================================
# GENERATE SUBMISSION (V250 best config)
# ============================================================
print("\n" + "=" * 60)
print("Generating V250 submission...")
print("=" * 60)

test_preds = {}
for target in TARGETS:
    best = v250_configs[target]
    cfg = CFGS[best['cfg']]
    y = y_dict[target]
    
    # Train on full training data
    spw = max(((y==0).sum()) / max((y==1).sum(), 1), 0.1)
    p = {**cfg, 'scale_pos_weight': spw, 'random_state': 42, 'force_row_wise': True, 'n_jobs': 1}
    
    train_X = f[best['features']].fillna(0).values.astype(np.float64)
    test_X = ft[best['features']].fillna(0).values.astype(np.float64)
    sn = [sanitize_col(c) for c in best['features']]
    
    full_ds = lgb.Dataset(train_X, label=y, feature_name=sn)
    full_model = lgb.train(p, full_ds, num_boost_round=cfg['n_estimators'])
    
    test_pred = np.clip(full_model.predict(test_X), 0.0001, 0.9999)
    test_pred = mean_match(test_pred, y.mean())
    
    test_preds[target] = test_pred
    print(f"  {target}: test_pred_mean={test_pred.mean():.3f}, train_mean={y.mean():.3f}")

# Save submission
submit_df = pd.DataFrame()
submit_df['subject_id'] = ft['subject_id'].values
submit_df['sleep_date'] = ft['sleep_date'].values
submit_df['lifelog_date'] = ft['lifelog_date'].values
for t in TARGETS:
    submit_df[sanitize_col(t)] = test_preds[t]

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
submit_path = SUBMIT / f'submission_v250_{ts}.csv'
submit_df.to_csv(submit_path, index=False)
print(f"\nSubmission saved: {submit_path}")

# ============================================================
# SAVE EXPERIMENT LOG
# ============================================================
experiment_log = {
    'name': 'V250_v127_plus_v09_external',
    'description': 'V127 baseline + V09 external proxy features + per-target selection + V10 ensemble',
    'timestamp': ts,
    'avg_baseline_oof': float(avg_baseline),
    'avg_v250_oof': float(avg_v250),
    'avg_v250_delta': float(avg_delta),
    'avg_ens_oof': float(avg_ens),
    'avg_ens_delta': float(avg_ens_delta),
    'baseline_per_target': {t: float(v) for t, v in baseline_results.items()},
    'v250_per_target': {t: float(v) for t, v in v250_results.items()},
    'v250_configs': {
        t: {k: v for k, v in cfg.items() if k != 'features'}
        for t, cfg in v250_configs.items()
    },
    'v250_feature_sets': {
        t: cfg['features'] for t, cfg in v250_configs.items()
    },
    'ens_per_target': {t: float(v) for t, v in ens_results.items()},
    'shift_analysis': shift_data,
    'added_external_features': list(added_features.keys()),
    'n_external_features': len(added_features),
    'n_total_features': len(non_const),
    'n_ext_features': len(ext_in),
    'n_non_ext_features': len(non_ext_in),
}

log_path = EXPERIMENTS / f'v250_{ts}.json'
with open(log_path, 'w') as fout:
    json.dump(experiment_log, fout, indent=2, default=str)
print(f"\nExperiment log saved: {log_path}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("=== V250 COMPLETE ===")
print("=" * 60)
print(f"V127 baseline OOF:  {avg_baseline:.5f}")
print(f"V250 OOF:           {avg_v250:.5f}")
print(f"V250 Δ:             {avg_delta:+.5f}")
print(f"V250+ENS OOF:       {avg_ens:.5f}")
print(f"V250+ENS Δ:         {avg_ens_delta:+.5f}")
print(f"Est. LB (baseline): ~0.648")
print(f"Est. LB (V250):     ~{0.648 + avg_delta * 2.5:.5f}")
print(f"Est. LB (V250+ENS): ~{0.648 + avg_ens_delta * 2.5:.5f}")
print(f"Submission: {submit_path}")
