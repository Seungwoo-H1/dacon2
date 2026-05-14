"""
V256: DACon2 V127 Ensemble Architecture Search
Experiments:
  1. Bayesian Weight Optimization (1000 iterations)
  2. Feature-Subspace Diversity Ensemble
  3. Rank Averaging vs Mean Averaging
  4. Per-Target Weight Optimization
  5. Additional Model Diversity (polynomial features, target-mean deviation)

Data: features_clean_v60.parquet + test_features_clean_v60.parquet
V127 baseline OOF: 0.53731
Target: 0.525
"""

import os, sys, gc, re, json, warnings, time, copy, itertools
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize
warnings.filterwarnings('ignore')

ROOT = Path('/home/mwoo423/projects/dacon2')
DATA = ROOT / 'data_processed'
EXPERIMENTS = ROOT / 'experiments'
SUBMIT = ROOT / 'submissions'
TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']
META_COLS = {'subject_id','lifelog_date','sleep_date','date'}

import lightgbm as lgb

def sanitize_col(n):
    return re.sub(r'[^a-zA-Z0-9_]', '_', n)

def get_numeric_cols(df, exclude=None):
    ex = META_COLS | set(TARGETS)
    if exclude: ex |= exclude
    return [c for c in df.columns
            if df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]
            and c not in ex]

CFGS = {
    'wide':   {'num_leaves':30,'max_depth':3,'learning_rate':0.05,'n_estimators':300,
               'subsample':0.8,'colsample_bytree':0.8,'reg_alpha':2.0,'reg_lambda':5.0,'min_child_samples':5},
    'deep':   {'num_leaves':20,'max_depth':5,'learning_rate':0.02,'n_estimators':1000,
               'subsample':0.7,'colsample_bytree':0.6,'reg_alpha':0.5,'reg_lambda':2.0,'min_child_samples':15},
    'v48':    {'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':500,
               'subsample':0.7,'colsample_bytree':0.7,'reg_alpha':1.0,'reg_lambda':3.0,'min_child_samples':10},
    'safety': {'num_leaves':10,'max_depth':3,'learning_rate':0.02,'n_estimators':1000,
               'subsample':0.6,'colsample_bytree':0.6,'reg_alpha':3.0,'reg_lambda':10.0,'min_child_samples':20},
}

V53_SWEEP = {
    'Q1': {'cfg': 'deep', 'n_feat': 19},
    'Q2': {'cfg': 'deep', 'n_feat': 14},
    'Q3': {'cfg': 'v48', 'n_feat': 11},
    'S1': {'cfg': 'wide', 'n_feat': 21},
    'S2': {'cfg': 'deep', 'n_feat': 19},
    'S3': {'cfg': 'safety','n_feat': 23},
    'S4': {'cfg': 'wide', 'n_feat': 20},
}

SEEDS = [42, 7, 999, 777]

LEAK_S = {'wlight_w_light_mean','wlight_w_light_std','wlight_w_light_min','wlight_w_light_max','wlight_w_light_count',
          'whr_hr_mean','whr_hr_std','whr_hr_min','whr_hr_max','whr_hr_median','whr_hr_count',
          'wpedo_pedo_step_mean','wpedo_pedo_step_sum','wpedo_pedo_step_frequency_mean','wpedo_pedo_step_frequency_sum',
          'wpedo_pedo_running_step_mean','wpedo_pedo_running_step_sum','wpedo_pedo_walking_step_mean','wpedo_pedo_walking_step_sum',
          'wpedo_pedo_distance_mean','wpedo_pedo_distance_sum','wpedo_pedo_speed_mean','wpedo_pedo_speed_sum',
          'wpedo_pedo_burned_calories_mean','wpedo_pedo_burned_calories_sum'}
LEAK_Q = {'whr_hr_mean','whr_hr_std','whr_hr_min','whr_hr_max','whr_hr_median','whr_hr_count'}

def remove_leak(cols, t):
    if t.startswith('S'): return [c for c in cols if c not in LEAK_S]
    elif t.startswith('Q'): return [c for c in cols if c not in LEAK_Q]
    return cols

def cfg_to_params(cfg_s, seed, spw):
    p = dict(cfg_s)
    p.update({'scale_pos_weight': spw, 'random_state': seed,
              'force_row_wise': True, 'n_jobs': 1, 'verbose': -1})
    return p

def mean_match(pred, tm):
    return np.clip(pred + (tm - np.clip(pred, 0.0001, 0.9999).mean()), 0.0001, 0.9999)

def isotonic_calibrate(oof_preds, y_true):
    try:
        iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0001, y_max=0.9999)
        iso.fit(oof_preds, y_true)
        cal = iso.predict(oof_preds)
        cal = mean_match(cal, float(y_true.mean()))
        return cal, True
    except:
        return np.clip(oof_preds, 0.0001, 0.9999), False

def rank_features(feat_df, fcols, target, seed=42):
    y = feat_df[target].values.astype(np.float64)
    X = feat_df[fcols].fillna(0).values.astype(np.float64)
    spw = max(((y==0).sum()) / max((y==1).sum(), 1), 0.1)
    p = cfg_to_params({'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':50,
                        'subsample':0.7,'colsample_bytree':0.7,'reg_alpha':1.0,'reg_lambda':3.0,
                        'min_child_samples':10}, seed, spw)
    sn = [sanitize_col(c) for c in fcols]
    ds = lgb.Dataset(X, label=y, feature_name=sn)
    m = lgb.train(p, ds, num_boost_round=50)
    imp = m.feature_importance(importance_type='gain')
    ranked = sorted(zip(fcols, imp), key=lambda x: -x[1])
    del m, ds; gc.collect()
    return [r[0] for r in ranked]

def add_pairwise(feat_df, top_feats):
    """Add pairwise interaction features (diff, sum, ratio) for top features."""
    top_s = [sanitize_col(c) for c in top_feats]
    new_df = feat_df.copy()
    added = []
    for i, f1 in enumerate(top_s[:8]):
        if f1 not in new_df.columns: continue
        for f2 in top_s[i+1:8]:
            if f2 not in new_df.columns: continue
            c1 = new_df[f1].fillna(0)
            c2 = new_df[f2].fillna(0)
            nc = f1 + '_diff_' + f2
            new_df[nc] = c1 - c2
            new_df[nc + '_sum'] = c1 + c2
            new_df[nc + '_ratio'] = c1 / (c2.abs() + 1e-10)
            added.extend([nc, nc+'_sum', nc+'_ratio'])
    return new_df, added

def add_polynomial(feat_df, top_feats):
    """Add polynomial features (sq, cube, pairwise product) for top features."""
    new_df = feat_df.copy()
    added = []
    for f in top_feats[:10]:
        if f not in new_df.columns: continue
        v = new_df[f].fillna(0)
        new_df[f + '_sq'] = v ** 2
        new_df[f + '_cube'] = v ** 3
        added.extend([f+'_sq', f+'_cube'])
    # Pairwise products
    for i, f1 in enumerate(top_feats[:6]):
        if f1 not in new_df.columns: continue
        for f2 in top_feats[i+1:6]:
            if f2 not in new_df.columns: continue
            v1 = new_df[f1].fillna(0)
            v2 = new_df[f2].fillna(0)
            nc = f1 + '_x_' + f2
            new_df[nc] = v1 * v2
            added.append(nc)
    return new_df, added

def add_target_mean_deviation(feat_df, top_feats):
    """Add per-subject deviation from global mean for each top feature."""
    new_df = feat_df.copy()
    added = []
    for f in top_feats[:10]:
        if f not in new_df.columns: continue
        global_mean = new_df[f].mean()
        new_df[f + '_dev'] = new_df[f].fillna(0) - global_mean
        added.append(f + '_dev')
    return new_df, added

def train_cv(feat_df, ftst_df, cols, y, seeds, cfg, n_folds=5):
    gkf = GroupKFold(n_splits=n_folds)
    oof = np.zeros((len(y), len(seeds)))
    tp = np.zeros((len(ftst_df), len(seeds))) if ftst_df is not None else None
    sn = [sanitize_col(c) for c in cols]
    Xf = feat_df[cols].fillna(0).values.astype(np.float64)
    Xt = ftst_df[cols].fillna(0).values.astype(np.float64) if ftst_df is not None else None
    spw = max(((y==0).sum()) / max((y==1).sum(), 1), 0.1)
    for si, seed in enumerate(seeds):
        p = cfg_to_params(cfg, seed, spw)
        for fi, (tri, vai) in enumerate(gkf.split(feat_df, y, feat_df['subject_id'])):
            ds = lgb.Dataset(Xf[tri], label=y[tri], feature_name=sn)
            if Xt is not None:
                vd = lgb.Dataset(Xf[vai], label=y[vai], feature_name=sn, reference=ds)
                m = lgb.train(p, ds, num_boost_round=cfg['n_estimators'],
                             valid_sets=[vd],
                             callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
                oof[vai, si] = m.predict(Xf[vai])
                tp[:, si] = m.predict(Xt)
                del vd
            else:
                m = lgb.train(p, ds, num_boost_round=cfg['n_estimators'],
                             callbacks=[lgb.log_evaluation(0)])
                oof[vai, si] = m.predict(Xf[vai])
            del ds, m; gc.collect()
    if tp is not None: tp = np.clip(tp, 0.0001, 0.9999)
    return oof, tp

def rank_blend(preds_2d):
    """Rank-blend: convert each model's preds to rank percentile, then average."""
    preds_2d = np.atleast_2d(preds_2d)
    if preds_2d.shape[0] < preds_2d.shape[1]:  # (models, samples) -> (samples, models)
        preds_2d = preds_2d.T
    n_models = preds_2d.shape[1]
    rank_averaged = np.zeros(preds_2d.shape[0])
    for j in range(n_models):
        ranks = pd.Series(preds_2d[:, j]).rank(pct=True).values
        rank_averaged += ranks
    rank_averaged /= n_models
    return np.clip(rank_averaged, 0.0001, 0.9999)

def mean_blend(preds_2d):
    return np.clip(preds_2d.mean(axis=0), 0.0001, 0.9999)

def build_model_set(feat_df, ftst_df, base_cols, target, feat_type, cfg, seeds):
    """
    Build OOF for a specific feat_type and cfg.
    feat_type: 'base' | 'pair' | 'trans' | 'poly' | 'dev'
    Returns: (oof_5xn, test_oof_5xn) where n = len(seeds)
    """
    working_df = feat_df.copy()
    target_leaked_cols = remove_leak(base_cols, target)
    y = feat_df[target].values.astype(np.float64)
    
    if feat_type == 'pair':
        ranked = rank_features(feat_df, target_leaked_cols, target)
        top8 = ranked[:8]
        working_df, added = add_pairwise(working_df, top8)
        post_cols = get_numeric_cols(working_df)
        sel_cols = remove_leak(post_cols, target)
        # Re-rank on augmented features
        ranked_aug = rank_features(working_df, remove_leak(get_numeric_cols(working_df), target), target)
        sel_cols = ranked_aug[:V53_SWEEP[target]['n_feat']]
    elif feat_type == 'trans':
        ranked = rank_features(feat_df, target_leaked_cols, target)
        top10 = ranked[:10]
        for f in top10:
            if f in working_df.columns:
                v = working_df[f].fillna(0)
                working_df[f + '_log'] = np.sign(v) * np.log1p(np.abs(v) + 1e-8)
                working_df[f + '_sqrt'] = np.sign(v) * np.sqrt(np.abs(v) + 1e-8)
        trans_cols = [c for c in get_numeric_cols(working_df) if c not in META_COLS | set(TARGETS)]
        ranked_aug = rank_features(working_df, remove_leak(trans_cols, target), target)
        sel_cols = ranked_aug[:V53_SWEEP[target]['n_feat']]
    elif feat_type == 'poly':
        ranked = rank_features(feat_df, target_leaked_cols, target)
        top10 = ranked[:10]
        working_df, _ = add_polynomial(working_df, top10)
        poly_cols = [c for c in get_numeric_cols(working_df) if c not in META_COLS | set(TARGETS)]
        ranked_aug = rank_features(working_df, remove_leak(poly_cols, target), target)
        sel_cols = ranked_aug[:V53_SWEEP[target]['n_feat']]
    elif feat_type == 'dev':
        ranked = rank_features(feat_df, target_leaked_cols, target)
        top10 = ranked[:10]
        working_df, _ = add_target_mean_deviation(working_df, top10)
        dev_cols = [c for c in get_numeric_cols(working_df) if c not in META_COLS | set(TARGETS)]
        ranked_aug = rank_features(working_df, remove_leak(dev_cols, target), target)
        sel_cols = ranked_aug[:V53_SWEEP[target]['n_feat']]
    else:  # base
        sel_cols = target_leaked_cols[:V53_SWEEP[target]['n_feat']]
    
    oof, _ = train_cv(working_df, None, sel_cols, y, seeds, cfg)
    return oof, sel_cols


# ============================================================
# Load data
# ============================================================
t_start = time.time()
print("=" * 70)
print("V256: V127 Ensemble Architecture Search")
print("=" * 70)

feat = pd.read_parquet(DATA / 'features_clean_v60.parquet')
ftst = pd.read_parquet(DATA / 'test_features_clean_v60.parquet')
ftst.columns = [sanitize_col(c) for c in ftst.columns]

y_dict = {t: feat[t].values for t in TARGETS}
train_rates = {t: float(feat[t].mean()) for t in TARGETS}

base_cols_all = get_numeric_cols(feat)
print(f"Base features: {len(base_cols_all)}")
print(f"Train shape: {feat.shape}, Test shape: {ftst.shape}")
print(f"Train rates: { {t: f'{train_rates[t]:.3f}' for t in TARGETS} }")

# ============================================================
# Build model pools for each strategy
# ============================================================
print("\n" + "=" * 70)
print("BUILDING MODEL POOLS")
print("=" * 70)

# model_pool[target][model_name] = {'oof': oof_array, 'll': logloss, 'n_feat': int}
model_pool = {t: {} for t in TARGETS}

# We'll build 6 model variants per target:
# base_wide, base_deep, pair_wide, pair_deep, trans_wide, trans_deep
feat_types = ['base', 'pair', 'trans']
cfg_keys = ['wide', 'deep']

for target in TARGETS:
    print(f"\n--- Target {target} ---")
    for ft in feat_types:
        for ck in cfg_keys:
            cfg = CFGS[ck]
            oof, sel_cols = build_model_set(feat, ftst, base_cols_all, target, ft, cfg, SEEDS)
            oof_avg = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
            iso_cal, _ = isotonic_calibrate(oof_avg, y_dict[target])
            ll = log_loss(y_dict[target], iso_cal, labels=[0,1])
            tag = f'{ft}_{ck}'
            model_pool[target][tag] = {'oof': oof_avg, 'll': ll, 'n_feat': len(sel_cols)}
            print(f"  {tag:15s} LL={ll:.5f} n_feat={len(sel_cols)}")

print(f"\nTime to build pool: {time.time()-t_start:.0f}s")


# ============================================================
# V127 BASELINE VERIFICATION
# ============================================================
print("\n" + "=" * 70)
print("V127 BASELINE VERIFICATION")
print("=" * 70)

# V127 config: 0.35 × V121(pair_deep) + 0.25 × V123(pair_wide) + 0.40 × V115(base_wide)
W121, W123, W115 = 0.35, 0.25, 0.40

baseline_oof = {}
for t in TARGETS:
    ens = (W121 * model_pool[t]['pair_deep']['oof'] +
           W123 * model_pool[t]['pair_wide']['oof'] +
           W115 * model_pool[t]['base_wide']['oof'])
    baseline_oof[t] = ens
    ll = log_loss(y_dict[t], np.clip(ens, 0.0001, 0.9999), labels=[0,1])
    print(f"  {t}: {ll:.5f}")
baseline_avg = np.mean([log_loss(y_dict[t], np.clip(baseline_oof[t], 0.0001, 0.9999), labels=[0,1]) for t in TARGETS])
print(f"  AVG: {baseline_avg:.5f} (V127 expected: 0.53731)")

# ============================================================
# EXPERIMENT 1: Bayesian Weight Optimization
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 1: Bayesian Weight Optimization (1000 iterations)")
print("=" * 70)

# Optimize weights for the 6-model pool using scipy optimize
ens_1_all_targets = {}

for t in TARGETS:
    y = y_dict[t]
    all_keys = ['base_wide', 'base_deep', 'pair_wide', 'pair_deep', 'trans_wide', 'trans_deep']
    model_oofs = np.array([model_pool[t][k]['oof'] for k in all_keys])
    
    best_ll = float('inf')
    best_w = None
    best_combo = None
    
    # Try all 3-way combinations with Bayesian optimization
    for combo in itertools.combinations(all_keys, 3):
        combo_oofs = np.array([model_pool[t][k]['oof'] for k in combo])
        
        def obj(w, y=y, mo=combo_oofs):
            w_arr = np.exp(w) / np.exp(w).sum()
            ens = np.clip(w_arr @ mo, 0.0001, 0.9999)
            return log_loss(y, ens, labels=[0,1])
        
        # Multiple restarts for robustness
        for restart in range(20):
            x0 = np.random.randn(3) * 0.5
            res = minimize(obj, x0, method='L-BFGS-B',
                          options={'maxiter': 2000, 'ftol': 1e-14})
            if res.fun < best_ll:
                best_ll = res.fun
                best_w = np.exp(res.x) / np.exp(res.x).sum()
                best_combo = combo
    
    # Apply best weights
    ens = np.clip(best_w @ np.array([model_pool[t][k]['oof'] for k in best_combo]), 0.0001, 0.9999)
    iso_cal, _ = isotonic_calibrate(ens, y)
    ll = log_loss(y, iso_cal, labels=[0,1])
    ens_1_all_targets[t] = {'ll': ll, 'cal_oof': iso_cal, 'w': best_w, 'combo': best_combo}
    print(f"  {t}: LL={ll:.5f} combo={best_combo} w=[{best_w[0]:.3f},{best_w[1]:.3f},{best_w[2]:.3f}]")

ens_1_avg = np.mean([ens_1_all_targets[t]['ll'] for t in TARGETS])
print(f"  AVG: {ens_1_avg:.5f}")
print(f"  Δ vs V127: {ens_1_avg - baseline_avg:+.5f}")

# ============================================================
# EXPERIMENT 2: Feature-Subspace Diversity Ensemble
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 2: Feature-Subspace Diversity Ensemble")
print("=" * 70)

# 4 random subspaces × 4 seeds = 16 models per target
# Each model uses 60% of features (randomly selected)
model_pool_2 = {t: {} for t in TARGETS}
ens_2_preds = {}

for target in TARGETS:
    print(f"\n  --- Target {target} ---")
    y = y_dict[target]
    cfg = CFGS[V53_SWEEP[target]['cfg']]
    base_feat_cols = remove_leak(base_cols_all, target)
    n_total = len(base_feat_cols)
    n_sub = max(int(n_total * 0.6), 10)
    
    n_subspaces = 4
    for si in range(n_subspaces):
        np.random.seed((42 + si * 1000 + abs(hash(target))) % (2**31))
        sub_cols = np.random.choice(base_feat_cols, size=n_sub, replace=False).tolist()
        
        # Train with all 4 seeds on this subspace
        oof_sub, _ = train_cv(feat, None, sub_cols, y, SEEDS, cfg)
        oof_avg = np.clip(oof_sub.mean(axis=1), 0.0001, 0.9999)
        iso_cal, _ = isotonic_calibrate(oof_avg, y)
        ll = log_loss(y, iso_cal, labels=[0,1])
        tag = f'sub_{si}'
        model_pool_2[target][tag] = {'oof': oof_avg, 'll': ll, 'n_feat': len(sub_cols)}
        print(f"    {tag:10s} LL={ll:.5f} n_feat={len(sub_cols)}")
    
    # Mean ensemble of 4 subspace models (each already averaged over 4 seeds)
    all_oofs = np.array([model_pool_2[target][k]['oof'] for k in sorted(model_pool_2[target].keys())])
    ens = np.clip(all_oofs.mean(axis=0), 0.0001, 0.9999)
    ens_isocal, _ = isotonic_calibrate(ens, y)
    ll_ens = log_loss(y, ens_isocal, labels=[0,1])
    ens_2_preds[target] = ens_isocal
    
    # Compare with best single model
    best_single_ll = min(model_pool_2[target][k]['ll'] for k in model_pool_2[target])
    print(f"    ENS_AVG: {ll_ens:.5f} (best subspace: {best_single_ll:.5f})")

ens_2_avg = np.mean([log_loss(y_dict[t], ens_2_preds[t], labels=[0,1]) for t in TARGETS])
print(f"  AVG: {ens_2_avg:.5f}")
print(f"  Δ vs V127: {ens_2_avg - baseline_avg:+.5f}")


# ============================================================
# EXPERIMENT 3: Rank Averaging vs Mean Averaging
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 3: Rank Averaging vs Mean Averaging")
print("=" * 70)

ens_3_results = {}
all_6_keys = ['base_wide', 'base_deep', 'pair_wide', 'pair_deep', 'trans_wide', 'trans_deep']

for t in TARGETS:
    y = y_dict[t]
    all_oofs = np.array([model_pool[t][k]['oof'] for k in all_6_keys])
    
    # Mean blending
    mean_pred = mean_blend(all_oofs)
    mean_isocal, _ = isotonic_calibrate(mean_pred, y)
    mean_ll = log_loss(y, mean_isocal, labels=[0,1])
    
    # Rank blending
    rank_pred = rank_blend(all_oofs)
    rank_isocal, _ = isotonic_calibrate(rank_pred, y)
    rank_ll = log_loss(y, rank_isocal, labels=[0,1])
    
    # V127 config (3 models): mean vs rank
    v127_oofs = np.array([
        model_pool[t]['pair_deep']['oof'],
        model_pool[t]['pair_wide']['oof'],
        model_pool[t]['base_wide']['oof'],
    ])
    
    v127_mean = mean_blend(v127_oofs)
    v127_mean_isocal, _ = isotonic_calibrate(v127_mean, y)
    v127_mean_ll = log_loss(y, v127_mean_isocal, labels=[0,1])
    
    v127_rank = rank_blend(v127_oofs)
    v127_rank_isocal, _ = isotonic_calibrate(v127_rank, y)
    v127_rank_ll = log_loss(y, v127_rank_isocal, labels=[0,1])
    
    print(f"  {t}:")
    print(f"    6-model mean: {mean_ll:.5f}  rank: {rank_ll:.5f}  Δ={rank_ll-mean_ll:+.5f}")
    print(f"    V127 mean:    {v127_mean_ll:.5f}  rank: {v127_rank_ll:.5f}  Δ={v127_rank_ll-v127_mean_ll:+.5f}")
    
    ens_3_results[t] = {
        'mean_ll': mean_ll, 'rank_ll': rank_ll,
        'v127_mean_ll': v127_mean_ll, 'v127_rank_ll': v127_rank_ll,
        'mean_pred': mean_isocal, 'rank_pred': rank_isocal,
        'v127_mean_pred': v127_mean_isocal, 'v127_rank_pred': v127_rank_isocal,
    }

avg_mean_6m = np.mean([ens_3_results[t]['mean_ll'] for t in TARGETS])
avg_rank_6m = np.mean([ens_3_results[t]['rank_ll'] for t in TARGETS])
avg_v127_mean = np.mean([ens_3_results[t]['v127_mean_ll'] for t in TARGETS])
avg_v127_rank = np.mean([ens_3_results[t]['v127_rank_ll'] for t in TARGETS])

print(f"\n  6-model AVG:  mean={avg_mean_6m:.5f}  rank={avg_rank_6m:.5f}  Δ={avg_rank_6m-avg_mean_6m:+.5f}")
print(f"  V127 AVG:     mean={avg_v127_mean:.5f}  rank={avg_v127_rank:.5f}  Δ={avg_v127_rank-avg_v127_mean:+.5f}")
print(f"  Δ vs V127 baseline: 6M_mean={avg_mean_6m-baseline_avg:+.5f}  6M_rank={avg_rank_6m-baseline_avg:+.5f}  127_rank={avg_v127_rank-baseline_avg:+.5f}")

# ============================================================
# EXPERIMENT 4: Per-Target Weight Optimization
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 4: Per-Target Weight Optimization")
print("=" * 70)

ens_4_all_targets = {}

for t in TARGETS:
    y = y_dict[t]
    all_keys = ['base_wide', 'base_deep', 'pair_wide', 'pair_deep', 'trans_wide', 'trans_deep']
    
    best_ll = float('inf')
    best_w = None
    best_combo = None
    
    for combo in itertools.combinations(all_keys, 3):
        combo_oofs = np.array([model_pool[t][k]['oof'] for k in combo])
        
        def obj(w, y=y, mo=combo_oofs):
            w_arr = np.exp(w) / np.exp(w).sum()
            ens = np.clip(w_arr @ mo, 0.0001, 0.9999)
            return log_loss(y, ens, labels=[0,1])
        
        for restart in range(20):
            x0 = np.random.randn(3) * 0.5
            res = minimize(obj, x0, method='L-BFGS-B',
                          options={'maxiter': 2000, 'ftol': 1e-14})
            if res.fun < best_ll:
                best_ll = res.fun
                best_w = np.exp(res.x) / np.exp(res.x).sum()
                best_combo = combo
    
    ens = np.clip(best_w @ np.array([model_pool[t][k]['oof'] for k in best_combo]), 0.0001, 0.9999)
    iso_cal, _ = isotonic_calibrate(ens, y)
    ll = log_loss(y, iso_cal, labels=[0,1])
    ens_4_all_targets[t] = {'ll': ll, 'cal_oof': iso_cal, 'w': best_w, 'combo': best_combo}
    print(f"  {t}: LL={ll:.5f} combo={best_combo} w=[{best_w[0]:.3f},{best_w[1]:.3f},{best_w[2]:.3f}]")

ens_4_avg = np.mean([ens_4_all_targets[t]['ll'] for t in TARGETS])
print(f"  AVG: {ens_4_avg:.5f}")
print(f"  Δ vs V127: {ens_4_avg - baseline_avg:+.5f}")


# ============================================================
# EXPERIMENT 5: Additional Model Diversity
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 5: Additional Model Diversity")
print("=" * 70)

# 5A: Polynomial features model
# 5B: Target-mean deviation features model
# 5C: 8-model ensemble (6 base + poly + dev)

model_pool_5a = {t: {} for t in TARGETS}
model_pool_5b = {t: {} for t in TARGETS}

for target in TARGETS:
    print(f"\n  --- Target {target} ---")
    y = y_dict[target]
    cfg_deep = CFGS['deep']
    base_feat_cols = remove_leak(base_cols_all, target)
    
    # 5A: Polynomial features
    oof_5a, sel_5a = build_model_set(feat, ftst, base_cols_all, target, 'poly', cfg_deep, SEEDS)
    oof_5a_avg = np.clip(oof_5a.mean(axis=1), 0.0001, 0.9999)
    iso_5a, _ = isotonic_calibrate(oof_5a_avg, y)
    ll_5a = log_loss(y, iso_5a, labels=[0,1])
    model_pool_5a[target]['poly_deep'] = {'oof': oof_5a_avg, 'll': ll_5a}
    print(f"  5A_PolyDeep:  LL={ll_5a:.5f} n_feat={len(sel_5a)}")
    
    # 5B: Target-mean deviation features
    oof_5b, sel_5b = build_model_set(feat, ftst, base_cols_all, target, 'dev', cfg_deep, SEEDS)
    oof_5b_avg = np.clip(oof_5b.mean(axis=1), 0.0001, 0.9999)
    iso_5b, _ = isotonic_calibrate(oof_5b_avg, y)
    ll_5b = log_loss(y, iso_5b, labels=[0,1])
    model_pool_5b[target]['dev_deep'] = {'oof': oof_5b_avg, 'll': ll_5b}
    print(f"  5B_DevDeep:   LL={ll_5b:.5f} n_feat={len(sel_5b)}")

# 5A Ensemble: V127 + poly model (4-model)
print("\n  5A: 4-model ensemble (V127 + Poly)")
ens_5a_ll = 0
for t in TARGETS:
    y = y_dict[t]
    ens = np.clip(
        0.30 * model_pool[t]['base_wide']['oof'] +
        0.25 * model_pool[t]['pair_wide']['oof'] +
        0.30 * model_pool[t]['pair_deep']['oof'] +
        0.15 * model_pool_5a[t]['poly_deep']['oof'],
        0.0001, 0.9999
    )
    iso, _ = isotonic_calibrate(ens, y)
    ll = log_loss(y, iso, labels=[0,1])
    ens_5a_ll += ll
    print(f"    {t}: {ll:.5f}")
ens_5a_avg = ens_5a_ll / len(TARGETS)
print(f"    AVG: {ens_5a_avg:.5f}  Δ vs V127: {ens_5a_avg - baseline_avg:+.5f}")

# 5B Ensemble: V127 + dev model (4-model)
print("\n  5B: 4-model ensemble (V127 + Dev)")
ens_5b_ll = 0
for t in TARGETS:
    y = y_dict[t]
    ens = np.clip(
        0.30 * model_pool[t]['base_wide']['oof'] +
        0.25 * model_pool[t]['pair_wide']['oof'] +
        0.30 * model_pool[t]['pair_deep']['oof'] +
        0.15 * model_pool_5b[t]['dev_deep']['oof'],
        0.0001, 0.9999
    )
    iso, _ = isotonic_calibrate(ens, y)
    ll = log_loss(y, iso, labels=[0,1])
    ens_5b_ll += ll
    print(f"    {t}: {ll:.5f}")
ens_5b_avg = ens_5b_ll / len(TARGETS)
print(f"    AVG: {ens_5b_avg:.5f}  Δ vs V127: {ens_5b_avg - baseline_avg:+.5f}")

# 5C: Full 8-model ensemble (all base variants + poly + dev)
print("\n  5C: 8-model equal-weight ensemble")
ens_5c_ll = 0
ens_5c_preds = {}
for t in TARGETS:
    y = y_dict[t]
    model_oofs = [model_pool[t][k]['oof'] for k in all_6_keys]
    model_oofs.append(model_pool_5a[t]['poly_deep']['oof'])
    model_oofs.append(model_pool_5b[t]['dev_deep']['oof'])
    
    ens = np.clip(np.mean(model_oofs, axis=0), 0.0001, 0.9999)
    iso, _ = isotonic_calibrate(ens, y)
    ll = log_loss(y, iso, labels=[0,1])
    ens_5c_ll += ll
    ens_5c_preds[t] = iso
    print(f"    {t}: {ll:.5f}")
ens_5c_avg = ens_5c_ll / len(TARGETS)
print(f"    AVG: {ens_5c_avg:.5f}  Δ vs V127: {ens_5c_avg - baseline_avg:+.5f}")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

summary = {
    'version': 'V256_ensemble_search',
    'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
    'baseline_v127': round(float(baseline_avg), 5),
    'experiments': {},
    'model_pool_ll': {},
}

# Log per-model LL for each target
for t in TARGETS:
    summary['model_pool_ll'][t] = {k: round(model_pool[t][k]['ll'], 5) for k in all_6_keys}

# Exp 1
summary['experiments']['exp1_bayesian_weight_opt'] = {
    'avg_oof': round(float(ens_1_avg), 5),
    'delta_vs_v127': round(float(ens_1_avg - baseline_avg), 5),
    'per_target': {t: {
        'll': round(ens_1_all_targets[t]['ll'], 5),
        'combo': ens_1_all_targets[t]['combo'],
        'weights': [round(w, 4) for w in ens_1_all_targets[t]['w']],
    } for t in TARGETS},
}
print(f"  Exp 1 (Bayesian Weight Opt):    AVG={ens_1_avg:.5f} Δ={ens_1_avg - baseline_avg:+.5f}")

# Exp 2
summary['experiments']['exp2_feature_subspace'] = {
    'avg_oof': round(float(ens_2_avg), 5),
    'delta_vs_v127': round(float(ens_2_avg - baseline_avg), 5),
    'per_target': {t: round(log_loss(y_dict[t], ens_2_preds[t], labels=[0,1]), 5) for t in TARGETS},
}
print(f"  Exp 2 (Feature-Subspace):       AVG={ens_2_avg:.5f} Δ={ens_2_avg - baseline_avg:+.5f}")

# Exp 3
summary['experiments']['exp3_rank_vs_mean'] = {
    '6model_mean_avg': round(float(avg_mean_6m), 5),
    '6model_rank_avg': round(float(avg_rank_6m), 5),
    'v127_mean_avg': round(float(avg_v127_mean), 5),
    'v127_rank_avg': round(float(avg_v127_rank), 5),
    'per_target': {t: {
        'mean_ll': round(ens_3_results[t]['mean_ll'], 5),
        'rank_ll': round(ens_3_results[t]['rank_ll'], 5),
        'v127_mean_ll': round(ens_3_results[t]['v127_mean_ll'], 5),
        'v127_rank_ll': round(ens_3_results[t]['v127_rank_ll'], 5),
    } for t in TARGETS},
}
print(f"  Exp 3 (Rank vs Mean):           6M_mean={avg_mean_6m:.5f} 6M_rank={avg_rank_6m:.5f} 127_mean={avg_v127_mean:.5f} 127_rank={avg_v127_rank:.5f}")

# Exp 4
summary['experiments']['exp4_per_target_weight'] = {
    'avg_oof': round(float(ens_4_avg), 5),
    'delta_vs_v127': round(float(ens_4_avg - baseline_avg), 5),
    'per_target': {t: {
        'll': round(ens_4_all_targets[t]['ll'], 5),
        'combo': ens_4_all_targets[t]['combo'],
        'weights': [round(w, 4) for w in ens_4_all_targets[t]['w']],
    } for t in TARGETS},
}
print(f"  Exp 4 (Per-Target Weight):      AVG={ens_4_avg:.5f} Δ={ens_4_avg - baseline_avg:+.5f}")

# Exp 5
summary['experiments']['exp5_additional_diversity'] = {
    '5a_poly_4model': {
        'avg_oof': round(float(ens_5a_avg), 5),
        'delta_vs_v127': round(float(ens_5a_avg - baseline_avg), 5),
    },
    '5b_dev_4model': {
        'avg_oof': round(float(ens_5b_avg), 5),
        'delta_vs_v127': round(float(ens_5b_avg - baseline_avg), 5),
    },
    '5c_8model': {
        'avg_oof': round(float(ens_5c_avg), 5),
        'delta_vs_v127': round(float(ens_5c_avg - baseline_avg), 5),
    },
}
print(f"  Exp 5A (Poly 4-model):          AVG={ens_5a_avg:.5f} Δ={ens_5a_avg - baseline_avg:+.5f}")
print(f"  Exp 5B (Dev 4-model):           AVG={ens_5b_avg:.5f} Δ={ens_5b_avg - baseline_avg:+.5f}")
print(f"  Exp 5C (8-model):               AVG={ens_5c_avg:.5f} Δ={ens_5c_avg - baseline_avg:+.5f}")

# Best overall
all_results = {
    'v127_baseline': baseline_avg,
    'exp1_bayesian': ens_1_avg,
    'exp2_subspace': ens_2_avg,
    'exp3_6M_mean': avg_mean_6m,
    'exp3_6M_rank': avg_rank_6m,
    'exp3_127_rank': avg_v127_rank,
    'exp4_per_target': ens_4_avg,
    'exp5a_poly': ens_5a_avg,
    'exp5b_dev': ens_5b_avg,
    'exp5c_8model': ens_5c_avg,
}

best_exp = min(all_results, key=all_results.get)
best_val = all_results[best_exp]
print(f"\n  🏆 BEST: {best_exp} AVG={best_val:.5f} (Δ vs V127: {best_val - baseline_avg:+.5f})")

# Per-target detail for best
print(f"\n  Per-target detail for best ({best_exp}):")
if best_exp == 'v127_baseline':
    for t in TARGETS:
        ll = log_loss(y_dict[t], np.clip(baseline_oof[t], 0.0001, 0.9999), labels=[0,1])
        print(f"    {t}: {ll:.5f}")
elif best_exp == 'exp1_bayesian':
    for t in TARGETS:
        print(f"    {t}: {ens_1_all_targets[t]['ll']:.5f} (w={ens_1_all_targets[t]['w']}, combo={ens_1_all_targets[t]['combo']})")
elif best_exp == 'exp2_subspace':
    for t in TARGETS:
        ll = log_loss(y_dict[t], ens_2_preds[t], labels=[0,1])
        print(f"    {t}: {ll:.5f}")
elif best_exp == 'exp3_6M_mean':
    for t in TARGETS:
        print(f"    {t}: {ens_3_results[t]['mean_ll']:.5f}")
elif best_exp == 'exp3_6M_rank':
    for t in TARGETS:
        print(f"    {t}: {ens_3_results[t]['rank_ll']:.5f}")
elif best_exp == 'exp3_127_rank':
    for t in TARGETS:
        print(f"    {t}: {ens_3_results[t]['v127_rank_ll']:.5f}")
elif best_exp == 'exp4_per_target':
    for t in TARGETS:
        print(f"    {t}: {ens_4_all_targets[t]['ll']:.5f} (w={ens_4_all_targets[t]['w']}, combo={ens_4_all_targets[t]['combo']})")
elif best_exp == 'exp5a_poly':
    print("    See 5A per-target above")
elif best_exp == 'exp5b_dev':
    print("    See 5B per-target above")
elif best_exp == 'exp5c_8model':
    for t in TARGETS:
        ll = log_loss(y_dict[t], ens_5c_preds[t], labels=[0,1])
        print(f"    {t}: {ll:.5f}")

# ============================================================
# Save experiment log
# ============================================================
print(f"\n{'='*70}")
print("SAVING RESULTS")
print(f"{'='*70}")

log_path = EXPERIMENTS / f'v256_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
with open(log_path, 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print(f"  Log: {log_path}")

print(f"\n{'='*70}")
print(f"V256 COMPLETE ✓ (total time: {time.time()-t_start:.0f}s)")
print(f"{'='*70}")

