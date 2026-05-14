"""
V251: V127 + external_data.parquet (real external date features)

Hypothesis: Merge external_data.parquet (holiday, month, daylight, season)
into features_clean_v60 and run V127 pipeline with additional external features.

external_data.parquet: 183 days × 8 features
- is_holiday, month, is_school_term, is_exam_period, is_lunar_holiday
- daylight_hours, daylight_ratio, season_index
100% coverage for all train (450) and test (250) dates.

Pipeline:
A: V127 baseline (base features only, same config as V127)
B: V251 (base + external, rank ALL together, pick top n_feat)
C: Ablation (try n_feat ±3, ±5)

KEY INSIGHT: external_data.parquet dates are string format, must map
to datetime then normalize before joining.
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
warnings.filterwarnings('ignore')

ROOT = Path('/home/mwoo423/projects/dacon2')
DATA = ROOT / 'data_processed'
EXPERIMENTS = ROOT / 'experiments'
SUBMIT = ROOT / 'submissions'
for d in [EXPERIMENTS, SUBMIT]:
    d.mkdir(exist_ok=True)

TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']
META_COLS = {'subject_id','lifelog_date','sleep_date','date'}
SEEDS = [42, 7, 999, 777]

# --- V127 configs (per target from V53 sweep) ---
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

# Leakage columns (sanitized)
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
    """Rank features by LightGBM gain importance (50 trees)."""
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
# LOAD DATA + MERGE EXTERNAL
# ============================================================
print("=" * 60)
print("V251: V127 + external_data.parquet Merge")
print("=" * 60)

print("\nLoading features_clean_v60...")
feat = pd.read_parquet(DATA / 'features_clean_v60.parquet')
ftst = pd.read_parquet(DATA / 'test_features_clean_v60.parquet')

# Sanitize column names for LightGBM
feat.columns = [sanitize_col(c) for c in feat.columns]
ftst.columns = [sanitize_col(c) for c in ftst.columns]

# Load external data and merge by date
print("Loading + merging external_data.parquet...")
ext = pd.read_parquet(DATA / 'external_data.parquet')
ext['date'] = pd.to_datetime(ext['lifelog_date']).dt.normalize()
date_map = {}
for _, row in ext.iterrows():
    date_map[row['date']] = {c: row[c] for c in ext.columns if c != 'lifelog_date'}

ext_feat_names = [c for c in date_map[next(iter(date_map))].keys()]
print(f"External features: {ext_feat_names}")

# Map to train
feat_dates = pd.to_datetime(feat['lifelog_date']).dt.normalize()
for col in ext_feat_names:
    feat[col] = feat_dates.map(lambda d: date_map.get(d, {}).get(col, np.nan))

# Map to test
ftst_dates = pd.to_datetime(ftst['lifelog_date']).dt.normalize()
for col in ext_feat_names:
    ftst[col] = ftst_dates.map(lambda d: date_map.get(d, {}).get(col, np.nan))

# Check coverage
train_cov = all(feat[col].notna().all() for col in ext_feat_names)
test_cov = all(ftst[col].notna().all() for col in ext_feat_names)
print(f"Coverage - Train: {train_cov}, Test: {test_cov}")
print(f"Train: {feat.shape}, Test: {ftst.shape}")

# ============================================================
# FEATURE COLUMNS
# ============================================================
all_cols = get_feature_cols(feat)
non_const = [c for c in all_cols if feat[c].std() > 0.001]

ext_feat_cols = [c for c in non_const if c in ext_feat_names]
base_feat_cols = [c for c in non_const if c not in ext_feat_names]

print(f"\nTotal features: {len(non_const)}")
print(f"  External ({len(ext_feat_cols)}): {ext_feat_cols}")
print(f"  Base ({len(base_feat_cols)}): ... (see below)")

y_dict = {t: feat[t].values.astype(np.float64) for t in TARGETS}

# ============================================================
# EXPERIMENT A: V127 Baseline (base features only)
# ============================================================
print("\n" + "=" * 60)
print("EXPERIMENT A: V127 Baseline (base features only)")
print("=" * 60)

baseline_results = {}
baseline_configs = {}

for target in TARGETS:
    sw = V53_SWEEP[target]
    cfg = CFGS[sw['cfg']]
    n_feat = sw['n_feat']
    y = y_dict[target]
    
    leak_cols = remove_leak(base_feat_cols, target)
    ranked = rank_features(feat, leak_cols, target)
    top_cols = ranked[:n_feat]
    
    oof, _ = train_cv(feat, None, top_cols, y, SEEDS, cfg)
    oof_avg = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
    cal = mean_match(oof_avg, y.mean())
    ll = log_loss(y, cal, labels=[0, 1])
    
    baseline_results[target] = ll
    baseline_configs[target] = {'cols': top_cols, 'cfg': sw['cfg'], 'n_feat': n_feat}
    
    print(f"  {target}: LL={ll:.5f} (cfg={sw['cfg']}, n_feat={n_feat})")

avg_baseline = np.mean(list(baseline_results.values()))
print(f"\n  AVG BASELINE OOF: {avg_baseline:.5f}")

# ============================================================
# EXPERIMENT B: V251 (all features ranked together)
# ============================================================
print("\n" + "=" * 60)
print("EXPERIMENT B: V251 (base + external, rank all together)")
print("=" * 60)

v251_results = {}
v251_configs = {}

for target in TARGETS:
    sw = V53_SWEEP[target]
    cfg = CFGS[sw['cfg']]
    n_feat = sw['n_feat']
    y = y_dict[target]
    
    all_rankable = base_feat_cols + ext_feat_cols
    leak_cols = remove_leak(all_rankable, target)
    ranked = rank_features(feat, leak_cols, target)
    
    top_cols = ranked[:n_feat]
    n_ext = sum(1 for c in top_cols if c in ext_feat_cols)
    
    oof, _ = train_cv(feat, None, top_cols, y, SEEDS, cfg)
    oof_avg = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
    cal = mean_match(oof_avg, y.mean())
    ll = log_loss(y, cal, labels=[0, 1])
    
    delta = ll - baseline_results[target]
    v251_results[target] = ll
    v251_configs[target] = {'cols': top_cols, 'cfg': sw['cfg'], 'n_feat': n_feat, 'n_ext': n_ext}
    
    print(f"  {target}: LL={ll:.5f} base={baseline_results[target]:.5f} "
          f"Δ={delta:+.5f} (ext={n_ext}/{n_feat})")
    if n_ext > 0:
        ext_used = [c for c in top_cols if c in ext_feat_cols]
        print(f"    ext selected: {ext_used}")

avg_v251 = np.mean(list(v251_results.values()))
avg_delta = avg_v251 - avg_baseline
print(f"\n  AVG V251 OOF: {avg_v251:.5f}")
print(f"  AVG DELTA:    {avg_delta:+.5f}")

# ============================================================
# EXPERIMENT C: Ablation — different n_feat
# ============================================================
print("\n" + "=" * 60)
print("EXPERIMENT C: Ablation (n_feat adjustments)")
print("=" * 60)

for n_adj in [-5, -3, -1, 0, +1, +3, +5]:
    target_results = []
    for target in TARGETS:
        sw = V53_SWEEP[target]
        cfg = CFGS[sw['cfg']]
        n_feat = sw['n_feat'] + n_adj
        if n_feat < 5:
            continue
        y = y_dict[target]
        
        all_rankable = base_feat_cols + ext_feat_cols
        leak_cols = remove_leak(all_rankable, target)
        ranked = rank_features(feat, leak_cols, target)
        top_cols = ranked[:min(n_feat, len(ranked))]
        
        oof, _ = train_cv(feat, None, top_cols, y, SEEDS, cfg)
        oof_avg = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
        cal = mean_match(oof_avg, y.mean())
        ll = log_loss(y, cal, labels=[0, 1])
        target_results.append(ll)
    
    if target_results:
        avg = np.mean(target_results)
        base_avg = np.mean([baseline_results[t] for t in TARGETS])
        print(f"  n_feat ±{n_adj:+d} (total={sw['n_feat']+n_adj}): avg LL={avg:.5f} vs base={base_avg:.5f} Δ={avg-base_avg:+.5f}")

# ============================================================
# GENERATE SUBMISSION (V251 best)
# ============================================================
print("\n" + "=" * 60)
print("Generating V251 submission...")
print("=" * 60)

test_preds = {}
for target in TARGETS:
    best = v251_configs[target]
    cfg = CFGS[best['cfg']]
    y = y_dict[target]
    
    spw = max(((y==0).sum()) / max((y==1).sum(), 1), 0.1)
    p = {**cfg, 'scale_pos_weight': spw, 'random_state': 42, 'force_row_wise': True, 'n_jobs': 1}
    
    train_X = feat[best['cols']].fillna(0).values.astype(np.float64)
    test_X = ftst[best['cols']].fillna(0).values.astype(np.float64)
    sn = [sanitize_col(c) for c in best['cols']]
    
    full_ds = lgb.Dataset(train_X, label=y, feature_name=sn)
    full_model = lgb.train(p, full_ds, num_boost_round=cfg['n_estimators'])
    
    test_pred = np.clip(full_model.predict(test_X), 0.0001, 0.9999)
    test_pred = mean_match(test_pred, y.mean())
    test_preds[target] = test_pred
    print(f"  {target}: test_pred_mean={test_pred.mean():.3f}, train_mean={y.mean():.3f}")

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
submit_df = pd.DataFrame()
submit_df['subject_id'] = ftst['subject_id'].values
submit_df['sleep_date'] = ftst['sleep_date'].values
submit_df['lifelog_date'] = ftst['lifelog_date'].values
for t in TARGETS:
    submit_df[sanitize_col(t)] = test_preds[t]

submit_path = SUBMIT / f'submission_v251_{ts}.csv'
submit_df.to_csv(submit_path, index=False)
print(f"\nSubmission saved: {submit_path}")

# ============================================================
# SAVE LOG
# ============================================================
experiment_log = {
    'name': 'V251_external_merge',
    'description': 'V127 + external_data.parquet (holiday, month, daylight, season) merge',
    'timestamp': ts,
    'avg_baseline_oof': float(avg_baseline),
    'avg_v251_oof': float(avg_v251),
    'avg_v251_delta': float(avg_delta),
    'baseline_per_target': {t: float(v) for t, v in baseline_results.items()},
    'v251_per_target': {t: float(v) for t, v in v251_results.items()},
    'v251_configs': {
        t: {k: v for k, v in cfg.items()}
        for t, cfg in v251_configs.items()
    },
    'n_external_features': len(ext_feat_cols),
    'n_base_features': len(base_feat_cols),
    'n_total_features': len(non_const),
    'ext_features': ext_feat_cols,
}

log_path = EXPERIMENTS / f'v251_{ts}.json'
with open(log_path, 'w') as fout:
    json.dump(experiment_log, fout, indent=2, default=str)
print(f"Experiment log saved: {log_path}")

print("\n" + "=" * 60)
print("=== V251 COMPLETE ===")
print("=" * 60)
print(f"V127 baseline OOF:  {avg_baseline:.5f}")
print(f"V251 OOF:           {avg_v251:.5f}")
print(f"V251 Δ:             {avg_delta:+.5f}")
print(f"Est. LB shift:      {avg_delta * 2.5:+.5f}")
print(f"Est. LB (V251):     ~{0.648 + avg_delta * 2.5:.5f}")
print(f"Submission: {submit_path}")
