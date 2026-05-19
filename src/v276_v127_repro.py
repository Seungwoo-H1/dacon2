"""
V276: V127 Reproduction from features.parquet + Evolution
- Load features.parquet (153 cols, 450 rows) — same pipeline as V127
- Add z-scores (like V127)
- Test: Baseline, Cross-target, Top-K, Calibration, Multi-config ensemble
- Target: LB ~0.50
"""
import os, sys, gc, re, json, warnings, time
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
import lightgbm as lgb

warnings.filterwarnings('ignore')
np.random.seed(42)

ROOT = Path('/root/.openclaw/workspace')
DATA = ROOT / 'data_processed'
SUBMIT = ROOT / 'submissions'
EXP = ROOT / 'experiments'
for d in [SUBMIT, EXP]:
    d.mkdir(exist_ok=True)

TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']
SEEDS = [42, 7, 999, 777]

V127_SWEEP = {
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

def cfg_to_params(cfg_s, seed, spw):
    params = dict(cfg_s)
    params['scale_pos_weight'] = spw
    params['random_state'] = seed
    params['force_row_wise'] = True
    params['n_jobs'] = 1
    return params

# ============================================================
# LOAD
# ============================================================
t0 = time.time()
print("=" * 60)
print("V276: V127 Reproduction + Evolution from features.parquet")
print("=" * 60)

feat = pd.read_parquet(DATA / 'features.parquet')
ftst = None  # Will generate from raw data if needed
print(f"Train: {feat.shape}")

# Build z-scores (per-person)
exclude = {'subject_id','lifelog_date','sleep_date'} | set(TARGETS)
feat_cols = [c for c in feat.columns if c not in exclude and feat[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]

zscore_cols = []
for fcol in feat_cols:
    feat[fcol] = feat[fcol].astype(float)
    zcol = f'{fcol}_zscore'
    zscore_cols.append(zcol)
    for sid in feat['subject_id'].unique():
        mask = feat['subject_id'] == sid
        mn = feat.loc[mask, fcol].mean()
        sd = feat.loc[mask, fcol].std()
        if sd > 1e-8 and not np.isnan(sd):
            feat.loc[mask, zcol] = (feat.loc[mask, fcol] - mn) / sd
        else:
            feat.loc[mask, zcol] = 0.0

all_feat = feat_cols + zscore_cols
print(f"Features: {len(feat_cols)} base + {len(zscore_cols)} zscore = {len(all_feat)}")

# Non-constant features only
non_const = [c for c in all_feat if feat[c].std() > 0.001]
print(f"Non-constant: {len(non_const)}")

gkf = GroupKFold(n_splits=5)
groups = feat['subject_id'].values
y_dict = {t: feat[t].values.astype(np.float64) for t in TARGETS}

# Save processed
feat.to_parquet(DATA / 'features_v276.parquet', index=False)

# ============================================================
# HELPER: Train CV
# ============================================================
def train_cv(cols, target, seed_list=None):
    """Train per-target CV. Returns oof predictions (clipped, mean-matched)."""
    if seed_list is None:
        seed_list = SEEDS
    cfg = CFGS[V127_SWEEP[target]['cfg']]
    y = y_dict[target]
    oof = np.zeros(len(y))
    spw = max(((y==0).sum()) / max((y==1).sum(), 1), 0.1)
    sn = [sanitize_col(c) for c in cols]
    Xf = feat[cols].fillna(0).values.astype(np.float64)
    
    for fi, (tri, vai) in enumerate(gkf.split(Xf, y, groups)):
        p = cfg_to_params(cfg, SEEDS[fi % len(SEEDS)], spw)
        ds_tr = lgb.Dataset(Xf[tri], label=y[tri], feature_name=sn)
        ds_va = lgb.Dataset(Xf[vai], label=y[vai], feature_name=sn, reference=ds_tr)
        m = lgb.train(p, ds_tr, num_boost_round=cfg['n_estimators'],
                     valid_sets=[ds_va],
                     callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        oof[vai] = m.predict(Xf[vai])
        del ds_tr, ds_va, m; gc.collect()
    return np.clip(mean_match(oof, y.mean()), 0.001, 0.999)

def log_loss_oof(oof, target):
    return log_loss(y_dict[target], oof, labels=[0, 1])

# ============================================================
# PHASE 2: Baseline — V127
# ============================================================
print("\n[PHASE 2] V127 Baseline")
b1_results = {}
for t in TARGETS:
    leak = remove_leak(non_const, t)
    oof = train_cv(leak, t)
    ll = log_loss_oof(oof, t)
    b1_results[t] = ll
    print(f"  {t}: LL={ll:.5f} (n_feats={len(leak)})")
b1_avg = np.mean(list(b1_results.values()))
print(f"  AVG: {b1_avg:.5f}")

# ============================================================
# PHASE 3: Cross-Target Raw
# ============================================================
print("\n[PHASE 3] Cross-Target Raw")
ct_extra = {t: [t2 for t2 in TARGETS if t2 != t] for t in TARGETS}
b2_results = {}
for t in TARGETS:
    cols = non_const + ct_extra[t]
    cols = [c for c in cols if c in all_feat]
    leak = remove_leak(cols, t)
    oof = train_cv(leak, t)
    ll = log_loss_oof(oof, t)
    b2_results[t] = ll
    print(f"  {t}: LL={ll:.5f} (n_feats={len(leak)})")
b2_avg = np.mean(list(b2_results.values()))
print(f"  AVG: {b2_avg:.5f}, Δ={b2_avg-b1_avg:+.5f}")

# ============================================================
# PHASE 4: Top-K + Cross-Target (ranked)
# ============================================================
print("\n[PHASE 4] Top-K Feature Selection + Cross-Target")

b3_results = {}
b3_oof = {}
for t in TARGETS:
    y = y_dict[t]
    other = [t2 for t2 in TARGETS if t2 != t]
    rankable = non_const + other
    rankable = [c for c in rankable if c in all_feat]
    
    # Rank features
    spw = max(((y==0).sum()) / max((y==1).sum(), 1), 0.1)
    imp_list = []
    for fi in range(3):
        tr_idx = np.random.RandomState(fi).choice(len(y), size=min(200, len(y)), replace=False)
        ds = lgb.Dataset(feat.iloc[tr_idx][rankable].fillna(0).values.astype(np.float64), label=y[tr_idx])
        p_tmp = {**CFGS[V127_SWEEP[t]['cfg']], 'objective':'binary','metric':'binary_logloss','verbose':-1,
                 'n_estimators':100,'num_leaves':15,'max_depth':4,'learning_rate':0.03,
                 'subsample':0.7,'colsample_bytree':0.7,'reg_alpha':1.0,'reg_lambda':3.0,
                 'min_child_samples':10,'scale_pos_weight':spw,'random_state':42+fi,
                 'force_row_wise':True,'n_jobs':1}
        m = lgb.train(p_tmp, ds, num_boost_round=100)
        imp_list.append(m.feature_importance(importance_type='gain'))
        del m, ds; gc.collect()
    
    mean_imp = np.mean(imp_list, axis=0)
    ranked = sorted(zip(rankable, mean_imp), key=lambda x: -x[1])
    ranked_names = [r[0] for r in ranked]
    
    # Try k values
    best_ll = float('inf')
    best_k = len(ranked_names)
    for k in [50, 80, 100, 120, 150, 200, 250, 300, len(ranked_names)]:
        cols = ranked_names[:k]
        cols = remove_leak(cols, t)
        oof = train_cv(cols, t)
        ll = log_loss_oof(oof, t)
        if ll < best_ll:
            best_ll = ll
            best_k = k
    
    # Final
    cols = ranked_names[:best_k]
    cols = remove_leak(cols, t)
    oof = train_cv(cols, t)
    ll = log_loss_oof(oof, t)
    
    b3_results[t] = {'ll': ll, 'k': best_k, 'cols': cols}
    b3_oof[t] = oof
    print(f"  {t}: k={best_k}, LL={ll:.5f}, Δ={ll-b1_results[t]:+.5f}")

b3_avg = np.mean([v['ll'] for v in b3_results.values()])
print(f"  AVG: {b3_avg:.5f}, Δ={b3_avg-b1_avg:+.5f}")

# ============================================================
# PHASE 5: Isotonic Calibration on F4
# ============================================================
print("\n[PHASE 5] Isotonic Calibration on F4")
b4_results = {}
for t in TARGETS:
    oof = b3_oof[t]
    iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
    iso.fit(oof, y_dict[t])
    oof_cal = iso.predict(oof)
    ll = log_loss_oof(oof_cal, t)
    b4_results[t] = ll
    print(f"  {t}: LL={ll:.5f}, Δ={ll-b1_results[t]:+.5f}")
b4_avg = np.mean(list(b4_results.values()))
print(f"  AVG: {b4_avg:.5f}, Δ={b4_avg-b1_avg:+.5f}")

# ============================================================
# PHASE 6: Multi-Config Ensemble + Weight Optimization
# ============================================================
print("\n[PHASE 6] Multi-Config Ensemble (wide/deep/v48/safety) + Weight Opt")
b5_results = {}
b5_oof = {}
b5_weights = {}

for t in TARGETS:
    y = y_dict[t]
    cols = b3_results[t]['cols']
    
    # Train all 4 configs
    oof_configs = {}
    for cn, cfg in CFGS.items():
        oof_c = np.zeros(len(y))
        spw = max(((y==0).sum()) / max((y==1).sum(), 1), 0.1)
        sn = [sanitize_col(c) for c in cols]
        Xf = feat[cols].fillna(0).values.astype(np.float64)
        for fi, (tri, vai) in enumerate(gkf.split(Xf, y, groups)):
            p = cfg_to_params(cfg, SEEDS[fi % len(SEEDS)], spw)
            ds_tr = lgb.Dataset(Xf[tri], label=y[tri], feature_name=sn)
            ds_va = lgb.Dataset(Xf[vai], label=y[vai], feature_name=sn, reference=ds_tr)
            m = lgb.train(p, ds_tr, num_boost_round=cfg['n_estimators'],
                         valid_sets=[ds_va],
                         callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
            oof_c[vai] = m.predict(Xf[vai])
            del ds_tr, ds_va, m; gc.collect()
        oof_configs[cn] = np.clip(mean_match(oof_c, y.mean()), 0.001, 0.999)
        print(f"    {t}/{cn}: LL={log_loss_oof(oof_configs[cn], t):.5f}")
    
    # Optimize weights
    cfg_names = list(CFGS.keys())
    def loss_fn(w_raw):
        w = np.exp(w_raw) / np.exp(w_raw).sum()
        p = np.zeros(len(y))
        for i, cn in enumerate(cfg_names):
            p += w[i] * oof_configs[cn]
        return log_loss(y, np.clip(p, 0.001, 0.999), labels=[0, 1])
    
    res = minimize(loss_fn, np.zeros(len(cfg_names)), method='Nelder-Mead',
                  options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12})
    w = np.exp(res.x) / np.exp(res.x).sum()
    
    p_final = np.zeros(len(y))
    for i, cn in enumerate(cfg_names):
        p_final += w[i] * oof_configs[cn]
    p_final = np.clip(p_final, 0.001, 0.999)
    
    ll = log_loss(y, p_final, labels=[0, 1])
    b5_results[t] = ll
    b5_oof[t] = p_final
    b5_weights[t] = {cn: round(float(w[i]), 4) for i, cn in enumerate(cfg_names)}
    print(f"  {t}: LL={ll:.5f}, Δ={ll-b1_results[t]:+.5f}")
    print(f"    W: {b5_weights[t]}")

b5_avg = np.mean(list(b5_results.values()))
print(f"  AVG: {b5_avg:.5f}, Δ={b5_avg-b1_avg:+.5f}")

# ============================================================
# PHASE 7: Iso + Multi-Config (F5 + Iso)
# ============================================================
print("\n[PHASE 7] Iso Calibration on F6")
b6_results = {}
for t in TARGETS:
    iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
    iso.fit(b5_oof[t], y_dict[t])
    oof_cal = iso.predict(b5_oof[t])
    ll = log_loss_oof(oof_cal, t)
    b6_results[t] = ll
    print(f"  {t}: LL={ll:.5f}, Δ={ll-b1_results[t]:+.5f}")
b6_avg = np.mean(list(b6_results.values()))
print(f"  AVG: {b6_avg:.5f}, Δ={b6_avg-b1_avg:+.5f}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"B1 (V127 baseline):      {b1_avg:.5f}")
print(f"B2 (Cross-target raw):   {b2_avg:.5f},  Δ={b2_avg-b1_avg:+.5f}")
print(f"B3 (Cross-top-K):        {b3_avg:.5f},  Δ={b3_avg-b1_avg:+.5f}")
print(f"B4 (Iso on B3):          {b4_avg:.5f},  Δ={b4_avg-b1_avg:+.5f}")
print(f"B5 (Multi-config+w):     {b5_avg:.5f},  Δ={b5_avg-b1_avg:+.5f}")
print(f"B6 (Iso on B5):          {b6_avg:.5f},  Δ={b6_avg-b1_avg:+.5f}")

best_name = min(range(1,7), key=lambda i: [b1_avg,b2_avg,b3_avg,b4_avg,b5_avg,b6_avg][i-1])
best_oof = [b1_avg,b2_avg,b3_avg,b4_avg,b5_avg,b6_avg][best_name-1]
best_delta = best_oof - b1_avg
est_lb = best_oof * 2.0 + 0.05

print(f"\nBest: B{best_name} (OOF={best_oof:.5f}, Δ={best_delta:+.5f})")
print(f"Est. LB: ~{est_lb:.5f}")
print(f"Target LB: 0.50, current: {est_lb:.5f}, gap: {est_lb-0.50:+.5f}")

# Save meta
meta = {
    'version': 'v276', 'time': datetime.now().isoformat(),
    'b1_avg': round(b1_avg, 5), 'b2_avg': round(b2_avg, 5),
    'b3_avg': round(b3_avg, 5), 'b4_avg': round(b4_avg, 5),
    'b5_avg': round(b5_avg, 5), 'b6_avg': round(b6_avg, 5),
    'best': f'B{best_name}', 'best_oof': round(best_oof, 5),
    'est_lb': round(est_lb, 5),
    'n_features': len(non_const), 'n_zscore': len(zscore_cols),
    'b3_k': {t: b3_results[t]['k'] for t in TARGETS},
    'b5_weights': b5_weights,
}
save_path = EXP / f'v276_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
with open(save_path, 'w') as f:
    json.dump(meta, f, indent=2, default=str)
print(f"\nMeta: {save_path}")
print(f"Total time: {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f}min)")
print("V276 COMPLETE")
