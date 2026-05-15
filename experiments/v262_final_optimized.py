#!/usr/bin/env python3
"""
V262: Final Optimized Model
Combines: V127 structure + Quantile Norm + PSI Filter + Clustering + Isotonic Cal

Strategy:
1. Reproduce V127 baseline (no quantile, no PSI filter, no clustering, no isotonic)
2. Quantile normalization (fit on train fold only)
3. Isotonic calibration (per-fold)
4. Clustering features (KMeans on stable features)
5. PSI-based feature filtering

Design: 2x2x2 factorial (Quantile x Isotonic x Clustering)
Then: PSI threshold sweep on best combo

4 seeds → 2 seeds (42, 7) for speed
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
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import QuantileTransformer
from sklearn.cluster import KMeans
warnings.filterwarnings('ignore')

ROOT = Path('/home/mwoo423/projects/dacon2')
DATA = ROOT / 'data_processed'
EXPERIMENTS = ROOT / 'experiments'
SUBMIT = ROOT / 'submissions'
for d in [EXPERIMENTS, SUBMIT]:
    d.mkdir(exist_ok=True)

TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']
META_COLS = {'subject_id','lifelog_date','sleep_date','date'}
SEEDS = [42, 7]  # Reduced from 4 → 2 for speed

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

LEAK_S = {'wlight_w_light_mean','wlight_w_light_std','wlight_w_light_min','wlight_w_light_max','wlight_w_light_count',
          'whr_hr_mean','whr_hr_std','whr_hr_min','whr_hr_max','whr_hr_median','whr_hr_count',
          'wpedo_pedo_step_mean','wpedo_pedo_step_sum','wpedo_pedo_step_frequency_mean','wpedo_pedo_step_frequency_sum',
          'wpedo_pedo_running_step_mean','wpedo_pedo_running_step_sum','wpedo_pedo_walking_step_mean','wpedo_pedo_walking_step_sum',
          'wpedo_pedo_distance_mean','wpedo_pedo_distance_sum','wpedo_pedo_speed_mean','wpedo_pedo_speed_sum',
          'wpedo_pedo_burned_calories_mean','wpedo_pedo_burned_calories_sum'}
LEAK_Q = {'whr_hr_mean','whr_hr_std','whr_hr_min','whr_hr_max','whr_hr_median','whr_hr_count'}

def sanitize_col(n): return re.sub(r'[^a-zA-Z0-9_]', '_', n)
def mean_match(pred, tm): return np.clip(pred + (tm - pred.mean()), 0.0001, 0.9999)

def cfg_to_params(cfg_s, seed, spw):
    params = dict(cfg_s)
    params['scale_pos_weight'] = spw
    params['random_state'] = seed
    params['force_row_wise'] = True
    params['n_jobs'] = 1
    return params

def train_cv(feat, ftst, cols, y, seeds, cfg, transform=None, transform_test=False):
    """Train with GroupKFold 5-fold CV.
    transform: 'quantile' for QuantileTransformer, None otherwise.
    When transform='quantile': fit QNT on train fold only, apply to val + test.
    """
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros((len(y), len(seeds)))
    tp = np.zeros((len(ftst), len(seeds))) if ftst is not None else None
    sn = [sanitize_col(c) for c in cols]
    spw = max(((y==0).sum()) / max((y==1).sum(), 1), 0.1)

    for si, seed in enumerate(seeds):
        p = cfg_to_params(cfg, seed, spw)
        for tri, vai in gkf.split(feat, y, feat['subject_id']):
            X_tr = feat[cols].fillna(0).values.astype(np.float64)[tri]
            X_val = feat[cols].fillna(0).values.astype(np.float64)[vai]
            
            if transform == 'quantile' and transform_test:
                qnt = QuantileTransformer(output_distribution='normal', random_state=seed)
                X_tr = qnt.fit_transform(X_tr)
                X_val = qnt.transform(X_val)
                if ftst is not None:
                    Xt = ftst[cols].fillna(0).values.astype(np.float64)
                    Xt = qnt.transform(Xt)
                else:
                    Xt = None
            else:
                Xt = None
            
            ds = lgb.Dataset(X_tr, label=y[tri], feature_name=sn)
            if X_val.shape[0] > 0:
                vd = lgb.Dataset(X_val, label=y[vai], feature_name=sn, reference=ds)
                m = lgb.train(p, ds, num_boost_round=cfg['n_estimators'],
                             valid_sets=[vd],
                             callbacks=[lgb.early_stopping(50, verbose=False),
                                       lgb.log_evaluation(0)])
                oof[vai, si] = m.predict(X_val)
                if Xt is not None:
                    tp[:, si] = m.predict(Xt)
            else:
                m = lgb.train(p, ds, num_boost_round=cfg['n_estimators'],
                             callbacks=[lgb.log_evaluation(0)])
                oof[vai, si] = m.predict(X_val)
            del ds, m, vd; gc.collect()

    if tp is not None:
        tp = np.clip(tp, 0.0001, 0.9999)
    return oof, tp

def compute_psi(expected, observed, buckets=100):
    eps = 1e-10
    min_p, max_p = 1e-6, 1 - 1e-6
    if len(np.unique(expected)) < 10:
        min_v, max_v = expected.min(), expected.max()
        if max_v - min_v < eps:
            return np.zeros(len(expected))
        edges = np.linspace(min_v, max_v, buckets + 1)
    else:
        edges = np.percentile(expected, np.linspace(0, 100, buckets + 1))
        edges[0] = edges[0] - eps
        edges[-1] = edges[-1] + eps
    expected_pct = np.histogram(expected, bins=edges)[0] / (len(expected) + eps)
    observed_pct = np.histogram(observed, bins=edges)[0] / (len(observed) + eps)
    expected_pct = np.clip(expected_pct, min_p, max_p)
    observed_pct = np.clip(observed_pct, min_p, max_p)
    psi = (expected_pct - observed_pct) * np.log(expected_pct / (observed_pct + eps) + eps)
    return psi.sum()  # Use TOTAL PSI (sum across bins), not mean per bin

def compute_calibration_oof(oof_preds, y, use_isotonic=False, subject_ids=None):
    """Apply mean matching (baseline) or per-fold isotonic calibration."""
    oof_avg = np.clip(oof_preds.mean(axis=1), 0.0001, 0.9999)
    if use_isotonic:
        gkf = GroupKFold(n_splits=5)
        oof_cal = oof_avg.copy()
        groups = subject_ids if subject_ids is not None else np.arange(len(y))
        for tri, vai in gkf.split(oof_avg, y, groups):
            if vai.shape[0] > 10:
                ir = IsotonicRegression(out_of_bounds='clip')
                try:
                    ir.fit(oof_avg[vai], y[vai])
                    oof_cal[vai] = ir.predict(oof_avg[vai])
                except:
                    pass
        return mean_match(oof_cal, y.mean())
    else:
        return mean_match(oof_avg, y.mean())

# ============================================================
# LOAD DATA
# ============================================================
t0 = time.time()
print("=" * 60)
print("V262: Final Optimized Model")
print("=" * 60)

feat = pd.read_parquet(DATA / 'features_clean_v60.parquet')
ftst = pd.read_parquet(DATA / 'test_features_clean_v60.parquet')

feat.columns = [sanitize_col(c) for c in feat.columns]
ftst.columns = [sanitize_col(c) for c in ftst.columns]

base_feat_cols = [c for c in feat.columns if c not in META_COLS | set(TARGETS) 
                  and feat[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]
                  and feat[c].std() > 0.001]
print(f"Base features: {len(base_feat_cols)}")

y_dict = {t: feat[t].values.astype(np.float64) for t in TARGETS}

# ============================================================
# PSI FEATURE SELECTION (train vs test)
# ============================================================
print("\nComputing PSI...")
psi_scores = {}
for c in base_feat_cols:
    train_vals = feat[c].fillna(0).values.astype(np.float64)
    test_vals = ftst[c].fillna(0).values.astype(np.float64)
    psi = compute_psi(train_vals, test_vals)
    psi_scores[c] = psi.mean()

psi_sorted = sorted(psi_scores.items(), key=lambda x: -x[1])
psi_total = np.mean(list(psi_scores.values()))
print(f"  Mean PSI: {psi_total:.4f}")
print(f"  Max PSI: {psi_sorted[0][1]:.4f}")
print(f"  Top 5: {[f'{n[:40]}: {s:.4f}' for n, s in psi_sorted[:5]]}")

# PSI filter columns
psi_filtered_cols = {}
for thresh in [0.1, 0.15, 0.2, 0.25, 0.5, None]:
    if thresh is None:
        psi_filtered_cols[thresh] = base_feat_cols.copy()
    else:
        filtered = [c for c in base_feat_cols if psi_scores.get(c, 0) < thresh]
        psi_filtered_cols[thresh] = filtered
        print(f"  PSI < {thresh}: {len(filtered)}/{len(base_feat_cols)} kept")

# ============================================================
# CLUSTERING FEATURES
# ============================================================
print("\nComputing clustering features...")
stable_features = [c for c, _ in psi_sorted[-80:]]
X_stable = feat[stable_features].fillna(0).values.astype(np.float64)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
feat['cluster_id'] = kmeans.fit_predict(X_stable)

centroids = kmeans.cluster_centers_
dists = np.linalg.norm(X_stable[:, None, :] - centroids[None, :, :], axis=2)
clustering_cols = []
for i in range(5):
    feat[f'cluster_dist_{i}'] = dists[:, i]
    clustering_cols.append(f'cluster_dist_{i}')

print(f"  {len(clustering_cols)} clustering features added")

# ============================================================
# 2x2x2 FACTORIAL: Quantile x Isotonic x Clustering
# ============================================================
print("\n" + "=" * 60)
print("V262: 2x2x2 Factorial (Quantile x Isotonic x Clustering)")
print("  No PSI filter (PSI is all <0.1, so no effect)")
print("=" * 60)

factorial_results = {}

for use_quantile in [False, True]:
    for use_isotonic in [False, True]:
        for use_clustering in [False, True]:
            exp_name = f"Q{use_quantile}_ISO{use_isotonic}_CLUST{use_clustering}"
            print(f"\n--- {exp_name} ---")
            
            cols = base_feat_cols.copy()
            if use_clustering:
                cols = cols + clustering_cols
            
            per_target_results = {}
            for target in TARGETS:
                sw = V53_SWEEP[target]
                cfg = CFGS[sw['cfg']]
                y = y_dict[target]
                
                oof, _ = train_cv(feat, None, cols, y, SEEDS, cfg,
                                 transform='quantile' if use_quantile else None,
                                 transform_test=use_quantile)
                
                cal = compute_calibration_oof(oof, y, use_isotonic, subject_ids=feat['subject_id'])
                ll = log_loss(y, cal, labels=[0, 1])
                per_target_results[target] = ll
                print(f"  {target:4s}: LL={ll:.5f}")
            
            avg_oof = np.mean(list(per_target_results.values()))
            factorial_results[exp_name] = avg_oof
            print(f"  AVG: {avg_oof:.5f}")

print("\n--- Factorial Summary ---")
for k, v in sorted(factorial_results.items(), key=lambda x: x[1]):
    print(f"  {k}: {v:.5f}")

best_combo = min(factorial_results, key=factorial_results.get)
baseline = factorial_results.get('QFalse_ISOFalse_CLUSTFalse', None)
print(f"\n  Best: {best_combo} = {factorial_results[best_combo]:.5f}")
if baseline is not None:
    print(f"  Baseline: QFalse_ISOFalse_CLUSTFalse = {baseline:.5f}")
    print(f"  Delta: {factorial_results[best_combo] - baseline:+.5f}")

# ============================================================
# PSI THRESHOLD SWEEP ON BEST COMBO
# ============================================================
print("\n" + "=" * 60)
print("V262: PSI Threshold Sweep on Best Combo")
print("=" * 60)

use_q = 'QTrue' in best_combo
use_i = 'ISOTrue' in best_combo
use_c = 'CLUSTTrue' in best_combo

psi_threshold_search = {}
for thresh in [0.05, 0.1, 0.15, 0.2, 0.25, 0.5, None]:
    cols = psi_filtered_cols.get(thresh, base_feat_cols.copy())
    if use_c:
        cols = cols + clustering_cols
    
    per_target_results = {}
    for target in TARGETS:
        sw = V53_SWEEP[target]
        cfg = CFGS[sw['cfg']]
        y = y_dict[target]
        
        oof, _ = train_cv(feat, None, cols, y, SEEDS, cfg,
                         transform='quantile' if use_q else None,
                         transform_test=use_q)
        
        cal = compute_calibration_oof(oof, y, use_i, subject_ids=feat['subject_id'])
        ll = log_loss(y, cal, labels=[0, 1])
        per_target_results[target] = ll
    
    avg_oof = np.mean(list(per_target_results.values()))
    psi_threshold_search[str(thresh)] = avg_oof
    print(f"  PSI < {thresh}: {len(cols)} feats, OOF={avg_oof:.5f}")

best_thresh = min(psi_threshold_search, key=psi_threshold_search.get)
print(f"\n  Best PSI: {best_thresh} (OOF={psi_threshold_search[best_thresh]:.5f})")

# ============================================================
# GENERATE SUBMISSION (best model, full data)
# ============================================================
print("\n" + "=" * 60)
print("V262: Generating Submission")
print("=" * 60)

test_preds = {}
best_cols = psi_filtered_cols.get(None if best_thresh == 'None' else float(best_thresh), base_feat_cols.copy())
if use_c:
    best_cols = best_cols + clustering_cols

for target in TARGETS:
    sw = V53_SWEEP[target]
    cfg = CFGS[sw['cfg']]
    y = y_dict[target]
    
    spw = max(((y==0).sum()) / max((y==1).sum(), 1), 0.1)
    p = cfg_to_params(cfg, SEEDS[0], spw)
    
    train_X = feat[best_cols].fillna(0).values.astype(np.float64)
    test_X = ftst[best_cols].fillna(0).values.astype(np.float64)
    sn = [sanitize_col(c) for c in best_cols]
    
    if use_q:
        qnt = QuantileTransformer(output_distribution='normal', random_state=SEEDS[0])
        train_X = qnt.fit_transform(train_X)
        test_X = qnt.transform(test_X)
    
    full_ds = lgb.Dataset(train_X, label=y, feature_name=sn)
    full_model = lgb.train(p, full_ds, num_boost_round=cfg['n_estimators'])
    
    test_pred = np.clip(full_model.predict(test_X), 0.0001, 0.9999)
    y_mean = y.mean()
    test_pred = mean_match(test_pred, y_mean)
    
    test_preds[target] = test_pred
    print(f"  {target}: test_pred_mean={test_pred.mean():.3f}, train_mean={y.mean():.3f}")

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
submit_df = pd.DataFrame()
submit_df['subject_id'] = ftst['subject_id'].values
submit_df['sleep_date'] = ftst['sleep_date'].values
submit_df['lifelog_date'] = ftst['lifelog_date'].values
for t in TARGETS:
    submit_df[sanitize_col(t)] = test_preds[t]

submit_path = SUBMIT / f'submission_v262_{ts}.csv'
submit_df.to_csv(submit_path, index=False)
print(f"\nSubmission saved: {submit_path}")

# ============================================================
# SAVE RESULT LOG
# ============================================================
result = {
    "version": "v262",
    "name": "Final Optimized Model",
    "timestamp": ts,
    "factorial_results": {k: float(v) for k, v in factorial_results.items()},
    "best_combo": best_combo,
    "psi_threshold_search": {k: float(v) for k, v in psi_threshold_search.items()},
    "best_psi_threshold": best_thresh,
    "use_quantile": use_q,
    "use_isotonic": use_i,
    "use_clustering": use_c,
    "n_features": len(best_cols),
    "clustering_features": clustering_cols,
    "submit_path": str(submit_path),
    "elapsed_seconds": round(time.time() - t0),
}

log_path = EXPERIMENTS / f'v262_result_{ts}.json'
with open(log_path, 'w') as f:
    json.dump(result, f, indent=2, default=str)
print(f"\nResult log saved: {log_path}")

print("\n" + "=" * 60)
print("=== V262 COMPLETE ===")
print(f"  Best combo: {best_combo}")
print(f"  Best PSI: {best_thresh}")
print(f"  OOF: {factorial_results[best_combo]:.5f}")
if baseline is not None:
    print(f"  Baseline: {baseline:.5f}, Δ={factorial_results[best_combo] - baseline:+.5f}")
print(f"  Submit: {submit_path}")
print(f"  Elapsed: {result['elapsed_seconds']}s")
print("=" * 60)
