"""
V275: V127 Evolution — Full Pipeline from Raw Data
- Rebuild features from raw data (like V273)
- Apply V127 config (wide/deep/v48/safety ensemble, 4 seeds)
- Test cross-target features, calibration, weight optimization
- Aim: LB ~0.50
"""
import os, sys, gc, re, json, warnings, time
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb

warnings.filterwarnings('ignore')
np.random.seed(42)

ROOT = Path('/root/.openclaw/workspace')
DATA_RAW = ROOT / 'data_raw'
DATA_PROC = ROOT / 'data_processed'
SUBMIT = ROOT / 'submissions'
EXP = ROOT / 'experiments'
for d in [DATA_PROC, SUBMIT, EXP]:
    d.mkdir(exist_ok=True)

TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']
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
# PHASE 1: Build features from raw data
# ============================================================
t0 = time.time()
print("=" * 60)
print("V275: V127 Evolution — Full Pipeline from Raw Data")
print("=" * 60)

# Load labels
labels = pd.read_csv(DATA_RAW / 'ch2026_metrics_train.csv')
labels['sleep_date_parsed'] = pd.to_datetime(labels['sleep_date']).dt.date
print(f"Labels: {labels.shape}")

label_pairs = set(zip(labels['subject_id'], labels['sleep_date_parsed']))
print(f"Subject-date pairs: {len(label_pairs)}")

# Process sensors
sensors = {}
DATA_DIR = DATA_RAW / 'ch2025_data_items'

def extract_floats(arr):
    vals = []
    if isinstance(arr, (list, np.ndarray)):
        for item in arr:
            if isinstance(item, (list, np.ndarray)):
                for sub in item:
                    try: vals.append(float(sub))
                    except: pass
            else:
                try: vals.append(float(item))
                except: pass
    return vals

all_feature_rows = {}

for fname in sorted(os.listdir(DATA_DIR)):
    if not fname.endswith('.parquet'):
        continue
    df = pd.read_parquet(DATA_DIR / fname)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    
    sname = fname.replace('ch2025_', '').replace('.parquet', '')
    sensors[sname] = df
    
    df = df[df.apply(lambda r: (r['subject_id'], r['date']) in label_pairs, axis=1)]
    if len(df) == 0:
        continue
    
    array_cols = []
    scalar_cols = []
    for col in df.columns:
        if col in ('subject_id', 'timestamp', 'date'):
            continue
        v = df[col].iloc[0]
        if isinstance(v, (list, np.ndarray)):
            array_cols.append(col)
        else:
            scalar_cols.append(col)
    
    # Array columns
    for col in array_cols:
        df[f'{col}_vals'] = df[col].apply(extract_floats)
        for func, suffix in [(np.nanmean, 'mean'), (np.nanstd, 'std'), (np.nanmin, 'min'),
                             (np.nanmax, 'max'), (np.nanmedian, 'median'), (np.nansum, 'sum')]:
            cn = f'{sname}_{col}_{suffix}'
            df[cn] = df[f'{col}_vals'].apply(lambda vs: func(vs) if len(vs) > 0 else np.nan)
        df[f'{sname}_{col}_cnt'] = df[f'{col}_vals'].apply(len)
        df = df.drop(columns=[f'{col}_vals'])
    
    # Scalar columns
    for col in scalar_cols:
        df[f'{sname}_{col}'] = pd.to_numeric(df[col], errors='coerce')
    
    # m_activity special handling
    mfeat_cols = []
    for col in scalar_cols + (['m_activity'] if 'm_activity' in df.columns else []):
        if col in ('subject_id', 'date'):
            continue
        if col == 'm_activity':
            col_key = f'{sname}_m_activity_mean'
        else:
            col_key = f'{sname}_{col}'
        mfeat_cols.append(col_key)
    
    # Group by subject + date
    feat_dict_cols = [c for c in df.columns if c not in ('subject_id', 'timestamp', 'date', 'm_activity')]
    feat_dict_cols = [c for c in feat_dict_cols if pd.api.types.is_numeric_dtype(df[c]) or c in mfeat_cols]
    
    if feat_dict_cols:
        grouped = df.groupby(['subject_id', 'date']).agg({
            c: 'mean' for c in feat_dict_cols if c != 'm_activity'
        }).reset_index()
        
        if 'm_activity' in df.columns:
            mact = df.groupby(['subject_id', 'date'])['m_activity'].mean().reset_index()
            mact = mact.rename(columns={'m_activity': 'mActivity_m_activity_mean'})
            grouped = grouped.merge(mact, on=['subject_id', 'date'], how='outer')
        
        for _, row in grouped.iterrows():
            key = (row['subject_id'], row['date'])
            if key not in all_feature_rows:
                all_feature_rows[key] = {}
            for fc in feat_dict_cols:
                if fc in ('subject_id', 'date'):
                    continue
                v = row.get(fc)
                if v is not None and not pd.isna(v):
                    all_feature_rows[key][fc] = float(v)
    
    gc.collect()

print(f"\n  Feature rows: {len(all_feature_rows)}")

# Build feature DataFrame
feat_list = []
for (sid, date), feats in all_feature_rows.items():
    row = {'subject_id': sid, 'sleep_date': date}
    row.update(feats)
    feat_list.append(row)

feat_df = pd.DataFrame(feat_list)
merged = labels.merge(feat_df, on=['subject_id', 'sleep_date'], how='left')
print(f"  Merged: {merged.shape}, missing: {merged.isna().sum().sum()}")

# Numeric feature columns (exclude targets and meta)
exclude_meta = {'subject_id','lifelog_date','sleep_date','sleep_date_parsed','date'} | set(TARGETS)
feat_cols = [c for c in merged.columns
             if c not in exclude_meta and
             merged[c].dtype in [np.float64, np.int64, float, int, np.float32, np.int32, bool, np.bool_]]

# Always generate z-scores (per-person)
zscore_cols = []
for fcol in feat_cols:
    merged[fcol] = merged[fcol].astype(float)
    zcol = f'{fcol}_zscore'
    zscore_cols.append(zcol)
    for sid in merged['subject_id'].unique():
        mask = merged['subject_id'] == sid
        mn = merged.loc[mask, fcol].mean()
        sd = merged.loc[mask, fcol].std()
        if sd > 1e-8 and not np.isnan(sd):
            merged.loc[mask, zcol] = (merged.loc[mask, fcol] - mn) / sd
        else:
            merged.loc[mask, zcol] = 0.0

all_feat = feat_cols + zscore_cols
print(f"  Total features (base+z): {len(all_feat)} ({len(feat_cols)}+{len(zscore_cols)})")

# Build X aligned with all_feat
for c in all_feat:
    if c not in merged.columns:
        merged[c] = 0.0
X = merged[all_feat].fillna(0).values.astype(np.float64)
X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
print(f"  X shape: {X.shape}")

# Save processed data
merged.to_parquet(DATA_PROC / 'features_v275.parquet', index=False)

gkf = GroupKFold(n_splits=5)
groups = merged['subject_id'].values
y_dict = {t: merged[t].values.astype(np.float64) for t in TARGETS}

# ============================================================
# PHASE 2: Feature Importance Ranking
# ============================================================
print("\n[PHASE 2] Feature Importance Ranking")
fold_imps = {}
for t in TARGETS:
    y = y_dict[t]
    imp_list = []
    for fi in range(3):
        spw = max(((y==0).sum()) / max((y==1).sum(), 1), 0.1)
        p = {'objective':'binary','metric':'binary_logloss','verbose':-1,
             'num_leaves':15,'max_depth':4,'learning_rate':0.05,
             'n_estimators':50,'subsample':0.8,'colsample_bytree':0.8,
             'reg_alpha':1.0,'reg_lambda':3.0,'scale_pos_weight':spw,
             'min_child_samples':15,'random_state':42+fi,'force_row_wise':True,'n_jobs':1}
        tr_idx = np.random.RandomState(fi).choice(len(y), size=min(200, len(y)), replace=False)
        ds = lgb.Dataset(X[tr_idx], label=y[tr_idx])
        m = lgb.train(p, ds, num_boost_round=50)
        imp_list.append(m.feature_importance(importance_type='gain'))
    fold_imps[t] = np.mean(imp_list, axis=0)
    top5 = np.argsort(fold_imps[t])[-5:][::-1]
    print(f"  {t}: {[all_feat[i] for i in top5]}")

# ============================================================
# PHASE 3: EXPERIMENT A — V127 Baseline (all features, leak removed)
# ============================================================
print("\n[PHASE 3A] V127 Baseline")

def run_experiment(feat_list, name, use_leak=True, extra_features=None):
    """Run CV experiments for all targets with given feature set."""
    results = {}
    oof_all = {}  # target -> oof predictions (used for weight opt)
    
    for target in TARGETS:
        cfg = CFGS[V127_SWEEP[target]['cfg']]
        y = y_dict[target]
        
        # Determine features
        cols = list(feat_list)
        
        # Add extra features (e.g., cross-target)
        if extra_features and target in extra_features:
            cols = cols + extra_features[target]
        
        # Remove leak
        if use_leak:
            cols = remove_leak(cols, target)
        
        # Ensure all columns exist in merged AND in X (all_feat)
        cols = [c for c in cols if c in all_feat]
        
        # Build X for this feature set
        X_sub = merged[cols].fillna(0).values.astype(np.float64)
        
        # Train CV
        oof = np.zeros(len(y))
        spw = max(((y==0).sum()) / max((y==1).sum(), 1), 0.1)
        
        sn = [sanitize_col(c) for c in cols]
        for fi, (tri, vai) in enumerate(gkf.split(X_sub, y, groups)):
            p = cfg_to_params(cfg, SEEDS[fi % len(SEEDS)], spw)
            ds_tr = lgb.Dataset(X_sub[tri], label=y[tri], feature_name=sn)
            ds_va = lgb.Dataset(X_sub[vai], label=y[vai], feature_name=sn, reference=ds_tr)
            m = lgb.train(p, ds_tr, num_boost_round=cfg['n_estimators'],
                         valid_sets=[ds_va],
                         callbacks=[lgb.early_stopping(100, verbose=False),
                                   lgb.log_evaluation(0)])
            oof[vai] = m.predict(X_sub[vai])
            del ds_tr, ds_va, m; gc.collect()
        
        cal = mean_match(oof, y.mean())
        ll = log_loss(y, np.clip(cal, 0.001, 0.999), labels=[0, 1])
        results[target] = ll
        oof_all[target] = cal
        print(f"  {target}: LL={ll:.5f} (n_feats={len(cols)})")
    
    avg = np.mean(list(results.values()))
    print(f"  AVG OOF: {avg:.5f}")
    return results, avg, oof_all

# A: All base features + z-scores, leak removed
a_results, a_avg, a_oof = run_experiment(all_feat, "A", use_leak=True)

# ============================================================
# PHASE 4: EXPERIMENT B — Cross-Target Raw Features
# ============================================================
print("\n[PHASE 4B] Cross-Target Raw Features")

ct_extra = {}
for target in TARGETS:
    others = [t for t in TARGETS if t != target]
    ct_extra[target] = others

b_results, b_avg, b_oof = run_experiment(all_feat, "B", use_leak=True, extra_features=ct_extra)
print(f"  Δ = {b_avg - a_avg:+.5f}")

# ============================================================
# PHASE 5: EXPERIMENT C — Top-K Feature Selection + Cross-Target
# ============================================================
print("\n[PHASE 5C] Top-K Feature Selection + Cross-Target")

c_results = {}
c_avg = 0
c_oof = {}

for target in TARGETS:
    cfg = CFGS[V127_SWEEP[target]['cfg']]
    y = y_dict[target]
    
    # Rank all features + other targets together
    other_targets = [t for t in TARGETS if t != target]
    rankable = all_feat + other_targets
    
    spw = max(((y==0).sum()) / max((y==1).sum(), 1), 0.1)
    p_tmp = {**CFGS[V127_SWEEP[target]['cfg']], 'scale_pos_weight': spw, 'random_state': 42,
             'force_row_wise': True, 'n_jobs': 1}
    p_tmp = {k: v for k, v in p_tmp.items() if k in CFGS[V127_SWEEP[target]['cfg']] or k in ('scale_pos_weight','random_state','force_row_wise','n_jobs')}
    
    imp_list = []
    for fi in range(3):
        tr_idx = np.random.RandomState(fi).choice(len(y), size=min(200, len(y)), replace=False)
        ds = lgb.Dataset(X[tr_idx], label=y[tr_idx])
        m = lgb.train({**p_tmp, 'n_estimators': 100, 'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03,
                       'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 1.0, 'reg_lambda': 3.0,
                       'min_child_samples': 10}, ds, num_boost_round=100)
        imp_list.append(m.feature_importance(importance_type='gain'))
    
    mean_imp = np.mean(imp_list, axis=0)
    ranked = sorted(zip(rankable, mean_imp), key=lambda x: -x[1])
    ranked_names = [r[0] for r in ranked]
    del m, ds; gc.collect()
    
    # Try different k values
    best_ll = float('inf')
    best_k = 0
    for k in [100, 150, 200, 250, 300, 350, len(ranked_names)]:
        top_cols = ranked_names[:k]
        
        oof = np.zeros(len(y))
        for fi, (tri, vai) in enumerate(gkf.split(X, y, groups)):
            p = cfg_to_params(cfg, SEEDS[fi % len(SEEDS)], spw)
            ds = lgb.Dataset(X[tri], label=y[tri])
            m = lgb.train(p, ds, num_boost_round=cfg['n_estimators'],
                         callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
            oof[vai] = m.predict(X[vai])
            del ds, m; gc.collect()
        
        cal = mean_match(oof, y.mean())
        ll = log_loss(y, np.clip(cal, 0.001, 0.999), labels=[0, 1])
        if ll < best_ll:
            best_ll = ll
            best_k = k
    
    # Final model with best k
    top_cols = ranked_names[:best_k]
    leak_cols = remove_leak(top_cols, target)
    
    oof = np.zeros(len(y))
    for fi, (tri, vai) in enumerate(gkf.split(X, y, groups)):
        p = cfg_to_params(cfg, SEEDS[fi % len(SEEDS)], spw)
        ds = lgb.Dataset(X[tri], label=y[tri])
        m = lgb.train(p, ds, num_boost_round=cfg['n_estimators'],
                     callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        oof[vai] = m.predict(X[vai])
        del ds, m; gc.collect()
    
    cal = mean_match(oof, y.mean())
    ll = log_loss(y, np.clip(cal, 0.001, 0.999), labels=[0, 1])
    
    c_results[target] = {'ll': ll, 'k': best_k, 'cols': leak_cols}
    c_avg += ll
    c_oof[target] = cal
    print(f"  {target}: best_k={best_k}, LL={ll:.5f}, Δ={ll-a_results[target]:+.5f}")

c_avg /= len(TARGETS)
print(f"\n  AVG OOF (C): {c_avg:.5f}, Δ={c_avg-a_avg:+.5f}")

# ============================================================
# PHASE 6: EXPERIMENT D — Isotonic Calibration on F3
# ============================================================
print("\n[PHASE 6D] Isotonic Calibration on F3")

d_results = {}
d_avg = 0
d_oof = {}

for target in TARGETS:
    y = y_dict[target]
    oof_raw = c_oof[target]
    
    iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
    iso.fit(oof_raw, y)
    oof_cal = iso.predict(oof_raw)
    
    ll = log_loss(y, np.clip(oof_cal, 0.001, 0.999), labels=[0, 1])
    d_results[target] = ll
    d_avg += ll
    d_oof[target] = oof_cal
    print(f"  {target}: LL={ll:.5f}, Δ={ll-a_results[target]:+.5f}")

d_avg /= len(TARGETS)
print(f"\n  AVG OOF (D): {d_avg:.5f}, Δ={d_avg-a_avg:+.5f}")

# ============================================================
# PHASE 7: EXPERIMENT E — Multi-Config Ensemble with Weight Opt
# ============================================================
print("\n[PHASE 7E] Multi-Config Ensemble (wide/deep/v48/safety) + Weight Opt")

e_results = {}
e_avg = 0
e_oof = {}
e_weights = {}

for target in TARGETS:
    y = y_dict[target]
    
    # Train each config with F3 best features
    cfg_configs = CFGS
    oof_configs = {}
    
    for cn, cfg in cfg_configs.items():
        cols = c_results[target]['cols']
        spw = max(((y==0).sum()) / max((y==1).sum(), 1), 0.1)
        
        oof = np.zeros(len(y))
        for fi, (tri, vai) in enumerate(gkf.split(X, y, groups)):
            p = cfg_to_params(cfg, SEEDS[fi % len(SEEDS)], spw)
            ds = lgb.Dataset(X[tri], label=y[tri])
            m = lgb.train(p, ds, num_boost_round=cfg['n_estimators'],
                         callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
            oof[vai] = m.predict(X[vai])
            del ds, m; gc.collect()
        
        cal = mean_match(oof, y.mean())
        oof_configs[cn] = cal
        ll_single = log_loss(y, np.clip(cal, 0.001, 0.999), labels=[0, 1])
        print(f"    {target}/{cn}: LL={ll_single:.5f}")
    
    # Optimize weights
    cfg_names = list(cfg_configs.keys())
    
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
    
    ll = log_loss(y, np.clip(p_final, 0.001, 0.999), labels=[0, 1])
    
    e_results[target] = ll
    e_avg += ll
    e_oof[target] = p_final
    e_weights[target] = {cn: round(float(w[i]), 4) for i, cn in enumerate(cfg_names)}
    print(f"  {target}: LL={ll:.5f}, Δ={ll-a_results[target]:+.5f}")
    print(f"    Weights: {e_weights[target]}")

e_avg /= len(TARGETS)
print(f"\n  AVG OOF (E): {e_avg:.5f}, Δ={e_avg-a_avg:+.5f}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"A (V127 baseline):       {a_avg:.5f}")
print(f"B (Cross-target raw):    {b_avg:.5f},  Δ={b_avg-a_avg:+.5f}")
print(f"C (Cross-top-K):         {c_avg:.5f},  Δ={c_avg-a_avg:+.5f}")
print(f"D (Iso calibration C):   {d_avg:.5f},  Δ={d_avg-a_avg:+.5f}")
print(f"E (Multi-config+weight): {e_avg:.5f},  Δ={e_avg-a_avg:+.5f}")

best_name = min(['A','B','C','D','E'], key=lambda x: {'A':a_avg,'B':b_avg,'C':c_avg,'D':d_avg,'E':e_avg}[x])
best_oof = {'A':a_avg,'B':b_avg,'C':c_avg,'D':d_avg,'E':e_avg}[best_name]
best_delta = best_oof - a_avg

print(f"\nBest: {best_name} (OOF={best_oof:.5f})")
print(f"Est. LB: ~{best_oof * 2.0 + 0.05:.5f}")
print(f"Gap to V127 LB (0.648): {best_oof * 2.0 + 0.05 - 0.648:+.5f}")
print(f"Gap to target (0.50):    {best_oof * 2.0 + 0.05 - 0.50:+.5f}")

meta = {
    'version': 'v275', 'time': datetime.now().isoformat(),
    'a_avg_oof': round(a_avg, 5),
    'b_avg_oof': round(b_avg, 5),
    'c_avg_oof': round(c_avg, 5),
    'd_avg_oof': round(d_avg, 5),
    'e_avg_oof': round(e_avg, 5),
    'best': best_name,
    'best_oof': round(best_oof, 5),
    'est_lb': round(best_oof * 2.0 + 0.05, 5),
    'c_best_k': {t: c_results[t]['k'] for t in TARGETS},
    'e_weights': e_weights,
}
save_path = EXP / f'v275_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
with open(save_path, 'w') as f:
    json.dump(meta, f, indent=2, default=str)
print(f"\nMeta: {save_path}")
print(f"Total time: {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f}min)")
print("V275 COMPLETE")
