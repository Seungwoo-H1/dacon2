"""
V256 Submission — Generate actual submission file from Exp 1 (Bayesian Weight Optimization) results.

Uses the best per-target combo + weights from V256 analysis:
  - Data: features_clean_v60.parquet + test_features_clean_v60.parquet (same as V256)
  - 6 model pool: base_wide, base_deep, pair_wide, pair_deep, trans_wide, trans_deep
  - Per-target: 3-model Bayesian optimized weights
  - 4 seeds per model
  - Isotonic calibration

This is the actual code that produced the OOF 0.58229 results.
We generate a submission file for manual upload to DaCon.
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
SUBMIT = ROOT / 'submissions'
SUBMIT.mkdir(exist_ok=True)
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

def build_model_set(feat_df, ftst_df, base_cols, target, feat_type, cfg, seeds):
    working_df = feat_df.copy()
    working_ftst = ftst_df.copy() if ftst_df is not None else None
    target_leaked_cols = remove_leak(base_cols, target)
    y = feat_df[target].values.astype(np.float64)
    
    if feat_type == 'pair':
        ranked = rank_features(feat_df, target_leaked_cols, target)
        top8 = ranked[:8]
        working_df, added = add_pairwise(working_df, top8)
        if working_ftst is not None:
            working_ftst, _ = add_pairwise(working_ftst, top8)
        post_cols = get_numeric_cols(working_df)
        sel_cols = remove_leak(post_cols, target)
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
            if working_ftst is not None and f in working_ftst.columns:
                v = working_ftst[f].fillna(0)
                working_ftst[f + '_log'] = np.sign(v) * np.log1p(np.abs(v) + 1e-8)
                working_ftst[f + '_sqrt'] = np.sign(v) * np.sqrt(np.abs(v) + 1e-8)
        trans_cols = [c for c in get_numeric_cols(working_df) if c not in META_COLS | set(TARGETS)]
        ranked_aug = rank_features(working_df, remove_leak(trans_cols, target), target)
        sel_cols = ranked_aug[:V53_SWEEP[target]['n_feat']]
    else:  # base
        sel_cols = target_leaked_cols[:V53_SWEEP[target]['n_feat']]
    
    oof, test_p = train_cv(working_df, working_ftst, sel_cols, y, seeds, cfg)
    return oof, test_p, sel_cols

def mean_blend(preds_2d):
    return np.clip(preds_2d.mean(axis=0), 0.0001, 0.9999)


# ============================================================
# Load data
# ============================================================
t_start = time.time()
print("=" * 70)
print("V256 Submission: Bayesian Weight + 6-model Ensemble")
print("=" * 70)

feat = pd.read_parquet(DATA / 'features_clean_v60.parquet')
ftst = pd.read_parquet(DATA / 'test_features_clean_v60.parquet')
ftst.columns = [sanitize_col(c) for c in ftst.columns]

y_dict = {t: feat[t].values for t in TARGETS}
train_rates = {t: float(feat[t].mean()) for t in TARGETS}

base_cols_all = get_numeric_cols(feat)
print(f"Base features: {len(base_cols_all)}")
print(f"Train: {feat.shape}, Test: {ftst.shape}")

# ============================================================
# Build model pools + test predictions
# ============================================================
print("\nBUILDING 6 MODELS per TARGET...")

# model_pool[target][name] = {'oof': cal_oof, 'test': test_preds, 'n_feat': int}
model_pool = {t: {} for t in TARGETS}

feat_types_cfgs = [
    ('base', 'wide'), ('base', 'deep'),
    ('pair', 'wide'), ('pair', 'deep'),
    ('trans', 'wide'), ('trans', 'deep'),
]

for target in TARGETS:
    print(f"\n--- {target} ---")
    for ft, ck in feat_types_cfgs:
        cfg = CFGS[ck]
        oof, test_p, sel_cols = build_model_set(feat, ftst, base_cols_all, target, ft, cfg, SEEDS)
        oof_avg = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
        iso_cal, _ = isotonic_calibrate(oof_avg, y_dict[target])
        ll = log_loss(y_dict[target], iso_cal, labels=[0,1])
        tag = f'{ft}_{ck}'
        model_pool[target][tag] = {'oof': oof_avg, 'll': ll, 'test': test_p, 'n_feat': len(sel_cols)}
        print(f"  {tag:12s} LL={ll:.5f} n_feat={len(sel_cols)}")

# ============================================================
# Exp 1: Bayesian Weight Optimization (same as V256 analysis)
# ============================================================
print("\n" + "=" * 70)
print("BAYESIAN WEIGHT OPTIMIZATION (per-target)")
print("=" * 70)

ens_1_all_targets = {}

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
    
    # Use best combo + weights for test predictions
    ens_test = np.clip(
        best_w @ np.array([model_pool[t][k]['test'] for k in best_combo]).mean(axis=0),
        0.0001, 0.9999
    )
    iso_cal_test, _ = isotonic_calibrate(ens_test, y)
    
    # Also calibrate OOF
    ens_oof = np.clip(best_w @ np.array([model_pool[t][k]['oof'] for k in best_combo]), 0.0001, 0.9999)
    iso_cal_oof, _ = isotonic_calibrate(ens_oof, y)
    ll = log_loss(y, iso_cal_oof, labels=[0,1])
    
    ens_1_all_targets[t] = {
        'll': ll,
        'oof_cal': iso_cal_oof,
        'test_cal': iso_cal_test,
        'w': best_w,
        'combo': best_combo,
    }
    print(f"  {t}: LL={ll:.5f} combo={best_combo}")
    print(f"       w=[{best_w[0]:.3f},{best_w[1]:.3f},{best_w[2]:.3f}]")

ens_1_avg = np.mean([ens_1_all_targets[t]['ll'] for t in TARGETS])
print(f"\n  AVG OOF: {ens_1_avg:.5f}")

# ============================================================
# Also compute V127 baseline for comparison
# ============================================================
print("\nV127 BASELINE (0.35×pair_deep + 0.25×pair_wide + 0.40×base_wide):")
v127_test = {}
for t in TARGETS:
    ens = np.clip(
        0.35 * model_pool[t]['pair_deep']['oof'] +
        0.25 * model_pool[t]['pair_wide']['oof'] +
        0.40 * model_pool[t]['base_wide']['oof'],
        0.0001, 0.9999
    )
    ll = log_loss(y_dict[t], ens, labels=[0,1])
    print(f"  {t}: {ll:.5f}")

# ============================================================
# Build submission
# ============================================================
print("\n" + "=" * 70)
print("BUILDING SUBMISSION")
print("=" * 70)

sub = pd.DataFrame()
sub['subject_id'] = ftst['subject_id'].values
sub['sleep_date'] = ftst['sleep_date'].values
sub['lifelog_date'] = ftst['lifelog_date'].values

for t in TARGETS:
    sub[t] = ens_1_all_targets[t]['test_cal']
    print(f"  {t}: min={sub[t].min():.4f} max={sub[t].max():.4f} mean={sub[t].mean():.4f}")

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
sub_path = SUBMIT / f"submission_v256_bayesian_{ts}.csv"
sub.to_csv(sub_path, index=False)
print(f"\n  Saved: {sub_path}")

# ============================================================
# Save metadata
# ============================================================
meta = {
    'version': 'V256_Bayesian',
    'name': 'V256: Bayesian Weight + 6-model Ensemble',
    'description': 'Per-target 3-model Bayesian weight optimization on 6-model pool',
    'models': ['base_wide', 'base_deep', 'pair_wide', 'pair_deep', 'trans_wide', 'trans_deep'],
    'seeds': 4,
    'avg_oof': round(float(ens_1_avg), 5),
    'per_target': {t: {
        'll': round(ens_1_all_targets[t]['ll'], 5),
        'combo': list(ens_1_all_targets[t]['combo']),
        'weights': [round(w, 4) for w in ens_1_all_targets[t]['w']],
        'test_mean': round(float(sub[t].mean()), 4),
    } for t in TARGETS},
    'submission_file': str(sub_path),
    'timestamp': ts,
    'total_time_s': round(time.time() - t_start, 0),
}

meta_path = SUBMIT / f'meta_v256_bayesian_{ts}.json'
with open(meta_path, 'w') as f:
    json.dump(meta, f, indent=2)
print(f"  Meta: {meta_path}")
print(f"\nTotal time: {time.time()-t_start:.0f}s")
