"""
V274: V127 Evolution — LB 0.5 Target
- Start from V127 baseline (features_clean_v60 + LGBM ensemble wide/deep/v48/safety)
- Add cross-target features (E4 pattern from V254/255 research)
- Add isotonic calibration
- Optimize ensemble weights with constrained optimization
- Multiple feature set variants tested in one run
"""
import os, sys, gc, re, json, warnings, time
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
import lightgbm as lgb

warnings.filterwarnings('ignore')
np.random.seed(42)

ROOT = Path('/home/mwoo423/.openclaw/workspace')
DATA = ROOT / 'data_processed'
EXPERIMENTS = ROOT / 'experiments'
SUBMIT = ROOT / 'submissions'
for d in [EXPERIMENTS, SUBMIT]:
    d.mkdir(exist_ok=True)

TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']
META_COLS = {'subject_id','lifelog_date','sleep_date','date'}
SEEDS = [42, 7, 999, 777]

# V127 per-target configs
V127_SWEEP = {
    'Q1': {'cfg': 'deep'},
    'Q2': {'cfg': 'deep'},
    'Q3': {'cfg': 'v48'},
    'S1': {'cfg': 'wide'},
    'S2': {'cfg': 'deep'},
    'S3': {'cfg': 'safety'},
    'S4': {'cfg': 'wide'},
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

# Leakage columns
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

def cfg_to_params(cfg_s, seed, spw):
    params = dict(cfg_s)
    params['scale_pos_weight'] = spw
    params['random_state'] = seed
    params['force_row_wise'] = True
    params['n_jobs'] = 1
    return params

def train_cv(feat, ftst, cols, y, seeds, cfg, fold_groups):
    """Train with GroupKFold 5-fold CV. Returns (oof, test_preds)."""
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros((len(y), len(seeds)))
    tp = np.zeros((len(ftst), len(seeds))) if ftst is not None else None
    sn = [sanitize_col(c) for c in cols]
    Xf = feat[cols].fillna(0).values.astype(np.float64)
    Xt = ftst[cols].fillna(0).values.astype(np.float64) if ftst is not None else None

    for si, seed in enumerate(seeds):
        p = cfg_to_params(cfg, seed, max(((y==0).sum()) / max((y==1).sum(), 1), 0.1))
        fi = 0
        for tri, vai in gkf.split(feat, y, fold_groups):
            ds = lgb.Dataset(Xf[tri], label=y[tri], feature_name=sn)
            if Xt is not None:
                vd = lgb.Dataset(Xf[vai], label=y[vai], feature_name=sn, reference=ds)
                m = lgb.train(p, ds, num_boost_round=cfg['n_estimators'],
                             valid_sets=[vd],
                             callbacks=[lgb.early_stopping(50, verbose=False),
                                       lgb.log_evaluation(0)])
                oof[vai, si] = m.predict(Xf[vai])
                tp[:, si] = m.predict(Xt)
            else:
                m = lgb.train(p, ds, num_boost_round=cfg['n_estimators'],
                             callbacks=[lgb.log_evaluation(0)])
                oof[vai, si] = m.predict(Xf[vai])
            fi += 1
        del ds; gc.collect()

    if tp is not None:
        tp = np.clip(tp, 0.0001, 0.9999)
    return np.clip(oof.mean(axis=1), 0.0001, 0.9999), tp

def train_full(feat, cols, y, cfg, seeds, fold_groups):
    """Train full model on all data (for test prediction)."""
    sn = [sanitize_col(c) for c in cols]
    Xf = feat[cols].fillna(0).values.astype(np.float64)
    preds = np.zeros((len(Xf), len(seeds)))
    for si, seed in enumerate(seeds):
        p = cfg_to_params(cfg, seed, max(((y==0).sum()) / max((y==1).sum(), 1), 0.1))
        ds = lgb.Dataset(Xf, label=y, feature_name=sn)
        m = lgb.train(p, ds, num_boost_round=cfg['n_estimators'])
        preds[:, si] = m.predict(Xf)
    return np.clip(preds.mean(axis=1), 0.0001, 0.9999)

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 60)
print("V274: V127 Evolution — LB 0.5 Target")
print("=" * 60)

feat = pd.read_parquet(DATA / 'features_clean_v60.parquet')
ftst = pd.read_parquet(DATA / 'test_features_clean_v60.parquet')

feat.columns = [sanitize_col(c) for c in feat.columns]
ftst.columns = [sanitize_col(c) for c in ftst.columns]

print(f"Train: {feat.shape}, Test: {ftst.shape}")

# Feature columns
all_cols = get_feature_cols(feat)
non_const = [c for c in all_cols if feat[c].std() > 0.001]
print(f"Non-constant features: {len(non_const)}")

# ============================================================
# FEATURE SETS TO TEST
# ============================================================
# F1: V127 baseline (all base features, leak removed)
# F2: Cross-target raw (add other targets as features)  
# F3: Cross-target top-K + base top-K
# F4: F3 + interaction features
# F5: F3 + isotonic calibrated predictions

y_dict = {t: feat[t].values.astype(np.float64) for t in TARGETS}
fold_groups = feat['subject_id'].values

# ============================================================
# EXPERIMENT F1: V127 Baseline
# ============================================================
print("\n" + "=" * 60)
print("EXPERIMENT F1: V127 Baseline (all base features)")
print("=" * 60)

f1_results = {}
for target in TARGETS:
    sw = V127_SWEEP[target]
    cfg = CFGS[sw['cfg']]
    y = y_dict[target]
    leak_cols = remove_leak(non_const, target)
    
    oof, _ = train_cv(feat, None, leak_cols, y, SEEDS, cfg, fold_groups)
    cal = mean_match(oof, y.mean())
    ll = log_loss(y, cal, labels=[0, 1])
    f1_results[target] = ll
    print(f"  {target}: LL={ll:.5f} (cfg={sw['cfg']}, n_feats={len(leak_cols)})")

avg_f1 = np.mean(list(f1_results.values()))
print(f"\n  AVG OOF (F1): {avg_f1:.5f}")

# ============================================================
# EXPERIMENT F2: Cross-Target Raw Features
# ============================================================
print("\n" + "=" * 60)
print("EXPERIMENT F2: Cross-Target Raw Features")
print("=" * 60)

# Build cross-target feature set
ct_cols = []
for t in TARGETS:
    if t not in ('Q1','Q2','Q3','S1','S2','S3','S4'):
        continue
    # Add the target itself as a raw feature (not the current target being predicted)
    # Actually cross-target means: for target Q1, add Q2, Q3, S1, S2, S3, S4 as features
    pass

# For each target, add all OTHER targets as features
f2_configs = {}
for target in TARGETS:
    cfg = CFGS[V127_SWEEP[target]['cfg']]
    y = y_dict[target]
    
    # Other targets as features
    other_targets = [t2 for t2 in TARGETS if t2 != target]
    
    # Base features (leak removed)
    base_leak = remove_leak(non_const, target)
    
    # Combine: base + other targets
    all_ft = base_leak + other_targets
    
    oof, _ = train_cv(feat, None, all_ft, y, SEEDS, cfg, fold_groups)
    cal = mean_match(oof, y.mean())
    ll = log_loss(y, cal, labels=[0, 1])
    
    f2_configs[target] = {'cols': all_ft, 'll': ll}
    print(f"  {target}: LL={ll:.5f} (n_feats={len(all_ft)})")

avg_f2 = np.mean([f2_configs[t]['ll'] for t in TARGETS])
print(f"\n  AVG OOF (F2): {avg_f2:.5f}, Δ={avg_f2 - avg_f1:+.5f}")

# ============================================================
# EXPERIMENT F3: Cross-Target Top-K + Base Top-K
# ============================================================
print("\n" + "=" * 60)
print("EXPERIMENT F3: Cross-Target Top-K + Base Top-K")
print("=" * 60)

# Rank features together (base + other targets)
f3_configs = {}
for target in TARGETS:
    cfg = CFGS[V127_SWEEP[target]['cfg']]
    y = y_dict[target]
    
    other_targets = [t2 for t2 in TARGETS if t2 != target]
    base_leak = remove_leak(non_const, target)
    all_rankable = base_leak + other_targets
    
    # Rank with quick LGBM
    spw = max(((y==0).sum()) / max((y==1).sum(), 1), 0.1)
    sn = [sanitize_col(c) for c in all_rankable]
    X = feat[all_rankable].fillna(0).values.astype(np.float64)
    ds = lgb.Dataset(X, label=y, feature_name=sn)
    p = {**V127_SWEEP[target], 'scale_pos_weight': spw, 'random_state': 42,
         'force_row_wise': True, 'n_jobs': 1}
    p_tmp = dict(CFGS[V127_SWEEP[target]['cfg']])
    p_tmp.update({'objective':'binary','metric':'binary_logloss','verbose':-1,
                  'n_estimators':100,'subsample':0.7,'colsample_bytree':0.7,
                  'reg_alpha':1.0,'reg_lambda':3.0,'min_child_samples':10})
    p_tmp.update(p)
    p_tmp = {k: v for k, v in p_tmp.items() if k in ('objective','metric','verbose',
              'num_leaves','max_depth','learning_rate','n_estimators','subsample',
              'colsample_bytree','reg_alpha','reg_lambda','min_child_samples',
              'scale_pos_weight','random_state','force_row_wise','n_jobs')}
    
    m = lgb.train(p_tmp, ds, num_boost_round=100)
    imp = m.feature_importance(importance_type='gain')
    ranked = sorted(zip(all_rankable, imp), key=lambda x: -x[1])
    ranked_names = [r[0] for r in ranked]
    del m, ds; gc.collect()
    
    # Try different n_total
    best_ll = float('inf')
    best_n = 0
    for n_total in [200, 250, 300, 350, 400, len(ranked_names)]:
        top_cols = ranked_names[:n_total]
        oof, _ = train_cv(feat, None, top_cols, y, SEEDS, cfg, fold_groups)
        cal = mean_match(oof, y.mean())
        ll = log_loss(y, cal, labels=[0, 1])
        if ll < best_ll:
            best_ll = ll
            best_n = n_total
    
    f3_configs[target] = {'n_total': best_n, 'll': best_ll, 'cols': ranked_names[:best_n]}
    print(f"  {target}: best n_total={best_n} LL={best_ll:.5f} Δ={best_ll-f1_results[target]:+.5f}")

avg_f3 = np.mean([f3_configs[t]['ll'] for t in TARGETS])
print(f"\n  AVG OOF (F3): {avg_f3:.5f}, Δ={avg_f3 - avg_f1:+.5f}")

# ============================================================
# EXPERIMENT F4: F3 + Isotonic Calibration
# ============================================================
print("\n" + "=" * 60)
print("EXPERIMENT F4: F3 + Isotonic Calibration")
print("=" * 60)

f4_configs = {}
for target in TARGETS:
    cfg = CFGS[V127_SWEEP[target]['cfg']]
    y = y_dict[target]
    cols = f3_configs[target]['cols']
    
    # Train per-fold with isotonic calibration
    gkf = GroupKFold(n_splits=5)
    oof_raw = np.zeros(len(y))
    cal_models = []
    
    for fi, (tri, vai) in enumerate(gkf.split(feat, y, fold_groups)):
        p = cfg_to_params(cfg, SEEDS[fi % len(SEEDS)], 
                         max(((y[tri]==0).sum()) / max((y[tri]==1).sum(), 1), 0.1))
        X_tr = feat.iloc[tri][cols].fillna(0).values.astype(np.float64)
        X_va = feat.iloc[vai][cols].fillna(0).values.astype(np.float64)
        y_tr = y[tri]
        y_va = y[vai]
        
        ds = lgb.Dataset(X_tr, label=y_tr, feature_name=[sanitize_col(c) for c in cols])
        m = lgb.train(p, ds, num_boost_round=cfg['n_estimators'])
        oof_raw[vai] = m.predict(X_va)
        del ds, m; gc.collect()
    
    # Isotonic calibration
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
    iso.fit(oof_raw, y)
    oof_cal = iso.predict(oof_raw)
    
    ll = log_loss(y, np.clip(oof_cal, 0.001, 0.999), labels=[0, 1])
    f4_configs[target] = {'ll': ll, 'iso': iso}
    print(f"  {target}: LL={ll:.5f} Δ={ll-f1_results[target]:+.5f}")

avg_f4 = np.mean([f4_configs[t]['ll'] for t in TARGETS])
print(f"\n  AVG OOF (F4): {avg_f4:.5f}, Δ={avg_f4 - avg_f1:+.5f}")

# ============================================================
# EXPERIMENT F5: Multi-Seed Ensemble with Weight Optimization
# ============================================================
print("\n" + "=" * 60)
print("EXPERIMENT F5: Weight-Optimized Multi-Config Ensemble")
print("=" * 60)

# Use F3 configs, train 4 seeds, optimize weights
f5_results = {}
f5_oof_preds = {}

for target in TARGETS:
    y = y_dict[target]
    cols = f3_configs[target]['cols']
    
    # Train each config (wide/deep/v48/safety) with multiple seeds
    oof_all = {}  # cfg_name -> oof array
    for cfg_name in CFGS:
        cfg = CFGS[cfg_name]
        oof, _ = train_cv(feat, None, cols, y, SEEDS, cfg, fold_groups)
        cal = mean_match(oof, y.mean())
        oof_all[cfg_name] = cal
        ll_single = log_loss(y, cal, labels=[0, 1])
        print(f"    {target}/{cfg_name}: LL={ll_single:.5f}")
    
    f5_oof_preds[target] = oof_all

# Optimize weights per target
f5_weights = {}
f5_lls = {}
cfg_names = list(CFGS.keys())

def loss_fn(w_raw, targets_dict, oof_dict):
    w = np.exp(w_raw) / np.exp(w_raw).sum()
    total = 0
    for t in targets_dict:
        p = np.zeros(len(oof_dict[t][cfg_names[0]]))
        for i, cn in enumerate(cfg_names):
            p += w[i] * oof_dict[t][cn]
        total += log_loss(targets_dict[t], np.clip(p, 0.001, 0.999), labels=[0, 1])
    return total / len(targets_dict)

# Per-target optimization
for target in TARGETS:
    y = y_dict[target]
    
    def per_target_loss(w_raw):
        w = np.exp(w_raw) / np.exp(w_raw).sum()
        p = np.zeros(len(oof_preds[cfg_names[0]]))
        for i, cn in enumerate(cfg_names):
            p += w[i] * oof_preds[cn]
        return log_loss(y, np.clip(p, 0.001, 0.999), labels=[0, 1])
    
    res = minimize(per_target_loss, np.zeros(len(cfg_names)), method='Nelder-Mead',
                  options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10})
    w = np.exp(res.x) / np.exp(res.x).sum()
    
    # Apply weights
    p = np.zeros(len(y))
    for i, cn in enumerate(cfg_names):
        p += w[i] * f5_oof_preds[target][cn]
    
    ll = log_loss(y, np.clip(p, 0.001, 0.999), labels=[0, 1])
    f5_lls[target] = ll
    f5_weights[target] = {cn: round(float(w[i]), 4) for i, cn in enumerate(cfg_names)}
    print(f"  {target}: LL={ll:.5f}, Δ={ll-f1_results[target]:+.5f}")
    print(f"    Weights: {f5_weights[target]}")

avg_f5 = np.mean(list(f5_lls.values()))
print(f"\n  AVG OOF (F5): {avg_f5:.5f}, Δ={avg_f5 - avg_f1:+.5f}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"F1 (V127 baseline):  {avg_f1:.5f}")
print(f"F2 (Cross-target):   {avg_f2:.5f}, Δ={avg_f2-avg_f1:+.5f}")
print(f"F3 (Cross-top-K):    {avg_f3:.5f}, Δ={avg_f3-avg_f1:+.5f}")
print(f"F4 (Iso calibration):{avg_f4:.5f}, Δ={avg_f4-avg_f1:+.5f}")
print(f"F5 (Weight-ensemb):  {avg_f5:.5f}, Δ={avg_f5-avg_f1:+.5f}")

best_name = min(['F1','F2','F3','F4','F5'], 
                key=lambda x: {'F1':avg_f1,'F2':avg_f2,'F3':avg_f3,'F4':avg_f4,'F5':avg_f5}[x])
best_oof = {'F1':avg_f1,'F2':avg_f2,'F3':avg_f3,'F4':avg_f4,'F5':avg_f5}[best_name]
best_delta = best_oof - avg_f1

print(f"\nBest: {best_name} (OOF={best_oof:.5f}, Δ={best_delta:+.5f})")
print(f"Est. LB (OOF*1.2+0.10): ~{best_oof*1.2+0.10:.5f}")
print(f"Target LB: ~0.50, currently at {best_oof*1.2+0.10:.5f}")

# Save results
meta = {
    'version': 'v274', 'time': datetime.now().isoformat(),
    'F1_avg_oof': round(avg_f1, 5),
    'F2_avg_oof': round(avg_f2, 5),
    'F3_avg_oof': round(avg_f3, 5),
    'F4_avg_oof': round(avg_f4, 5),
    'F5_avg_oof': round(avg_f5, 5),
    'best': best_name,
    'best_oof': round(best_oof, 5),
    'best_delta': round(best_delta, 5),
    'est_lb': round(best_oof*1.2+0.10, 5),
    'f3_configs': {t: f3_configs[t] for t in TARGETS},
    'f5_weights': f5_weights,
}
save_path = EXPERIMENTS / f'v274_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
with open(save_path, 'w') as f:
    json.dump(meta, f, indent=2, default=str)
print(f"\nMeta saved: {save_path}")
