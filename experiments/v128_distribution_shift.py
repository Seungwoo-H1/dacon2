"""
V128: Distribution Shift Research + Advanced Calibration

6 Experiments on OOF vs LB gap:
1. PSI (Population Stability Index) Analysis
2. Adversarial Validation
3. Quantile Normalization
4. Rank Stabilization
5. Temperature Scaling + Per-Target Sharpening
6. Calibration Error Analysis (ECE)

Baseline: V127 pipeline (GroupKFold 5-fold, 4 seeds, per-target configs)
"""

import os, sys, gc, re, json, warnings, time, copy
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
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

# V127 per-target configs
V53_SWEEP = {
    'Q1': {'cfg': 'deep'}, 'Q2': {'cfg': 'deep'}, 'Q3': {'cfg': 'v48'},
    'S1': {'cfg': 'wide'}, 'S2': {'cfg': 'deep'},
    'S3': {'cfg': 'safety'}, 'S4': {'cfg': 'wide'},
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

def train_cv(feat, ftst, cols, y, seeds, cfg):
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

def train_full(feat, cols, y, cfg):
    p = {**cfg, 'scale_pos_weight': max(((y==0).sum())/max((y==1).sum(),1), 0.1),
         'random_state': 42, 'force_row_wise': True, 'n_jobs': 1}
    sn = [sanitize_col(c) for c in cols]
    X = feat[cols].fillna(0).values.astype(np.float64)
    ds = lgb.Dataset(X, label=y, feature_name=sn)
    m = lgb.train(p, ds, num_boost_round=cfg['n_estimators'])
    return m, sn


# ============================================================
# EXPERIMENT 1: PSI (Population Stability Index) Analysis
# ============================================================
def compute_psi(expected_counts, actual_counts, buckets=10):
    """PSI between two distributions. Returns float."""
    eps = 1e-4
    total_expected = expected_counts.sum()
    total_actual = actual_counts.sum()
    
    expected_pct = expected_counts / (total_expected + eps)
    actual_pct = actual_counts / (total_actual + eps)
    expected_pct = np.clip(expected_pct, eps, 1.0)
    actual_pct = np.clip(actual_pct, eps, 1.0)
    
    psi = ((expected_pct - actual_pct) * np.log(expected_pct / actual_pct)).sum()
    return psi

def psi_per_feature(train_df, test_df, target_cols, buckets=20):
    """Compute PSI per feature between train and test."""
    psi_results = {}
    for col in train_df.columns:
        if train_df[col].dtype not in [np.float64, np.int64, float, int, np.float32, np.int32]:
            continue
        train_vals = train_df[col].dropna()
        test_vals = test_df[col].dropna()
        
        # Create quantile-based buckets from training data
        if len(train_vals) < 10:
            continue
        try:
            percentiles = np.linspace(0, 100, buckets + 1)
            breakpoints = np.percentile(train_vals.values, percentiles)
            breakpoints[0] = breakpoints[0] - 1  # include minimum
            breakpoints[-1] = breakpoints[-1] + 1  # include maximum
            
            # Remove duplicate breakpoints
            breakpoints = np.unique(breakpoints)
            if len(breakpoints) < 3:
                continue
                
            train_hist, _ = np.histogram(train_vals.values, bins=breakpoints)
            test_hist, _ = np.histogram(test_vals.values, bins=breakpoints)
            
            psi = compute_psi(train_hist, test_hist)
            psi_results[col] = psi
        except Exception:
            continue
    
    # Also compute per-target PSI (train vs test conditional on target)
    for tc in target_cols:
        for col in train_df.columns:
            if train_df[col].dtype not in [np.float64, np.int64, float, int, np.float32, np.int32]:
                continue
            try:
                for tgt_val in [0, 1]:
                    train_mask = train_df[tc].values == tgt_val
                    train_vals = train_df.loc[train_mask, col].dropna()
                    if len(train_vals) < 5:
                        continue
                    percentiles = np.linspace(0, 100, buckets + 1)
                    breakpoints = np.percentile(train_vals.values, percentiles)
                    breakpoints[0] -= 1
                    breakpoints[-1] += 1
                    breakpoints = np.unique(breakpoints)
                    if len(breakpoints) < 3:
                        continue
                    
                    train_hist, _ = np.histogram(train_vals.values, bins=breakpoints)
                    test_hist, _ = np.histogram(test_df[col].dropna().values, bins=breakpoints)
                    psi = compute_psi(train_hist, test_hist)
                    key = f"{col}_cond_{tc}_{tgt_val}"
                    psi_results[key] = psi
            except Exception:
                continue
    
    return psi_results

def experiment_psi(feat, ftst, psi_results, top_n=30):
    """Run PSI analysis and return results dict + reduced feature set."""
    results = {}
    
    # Per-target PSI
    per_target_psi = {}
    for target in TARGETS:
        train_cond = feat[feat[target] >= feat[target].median()]
        test_cond = ftst
        feature_psi = psi_per_feature(train_cond, test_cond, TARGETS)
        
        # Top features by PSI
        sorted_psi = sorted(feature_psi.items(), key=lambda x: -x[1])
        top_feats = [(name, score) for name, score in sorted_psi[:top_n]]
        per_target_psi[target] = {
            'top_psi_features': top_feats,
            'avg_psi': np.mean([s for _, s in sorted_psi[:50]]) if sorted_psi else 0,
            'max_psi': max([s for _, s in sorted_psi]) if sorted_psi else 0,
        }
        results[f'psi_{target}'] = per_target_psi[target]
        
        print(f"\n  PSI {target} (Top 10 features):")
        for name, score in top_feats[:10]:
            print(f"    {name}: PSI={score:.4f}")
    
    # Global PSI
    all_feature_cols = get_feature_cols(feat)
    feature_psi = psi_per_feature(feat[all_feature_cols], ftst[all_feature_cols], TARGETS)
    sorted_global = sorted(feature_psi.items(), key=lambda x: -x[1])
    results['psi_global_top'] = sorted_global[:20]
    results['psi_global_avg'] = np.mean([s for _, s in sorted_global[:50]]) if sorted_global else 0
    
    print(f"\n  Global PSI (Top 10):")
    for name, score in sorted_global[:10]:
        print(f"    {name}: PSI={score:.4f}")
    
    # Identify drift-heavy features to exclude
    drift_features = set()
    for target in TARGETS:
        top_feats = per_target_psi[target]['top_psi_features']
        for name, score in top_feats[:15]:
            # Strip _cond_* suffix for root feature name
            root_name = re.sub(r'_cond_[QSM]\d_\d$', '', name)
            if score > 0.1:  # significant drift threshold
                drift_features.add(root_name)
    
    results['drift_features'] = list(drift_features)
    results['n_drift_features'] = len(drift_features)
    
    print(f"\n  Features with PSI > 0.1: {len(drift_features)}")
    
    return results, drift_features


# ============================================================
# EXPERIMENT 2: Adversarial Validation
# ============================================================
def experiment_adversarial(feat, ftst, target_cols, top_n=20):
    """
    Adversarial validation: train a classifier to distinguish train vs test.
    Features with high importance = distribution drift features.
    """
    print("\n  Adversarial Validation:")
    
    n_train = len(feat)
    n_test = len(ftst)
    
    all_feature_cols = get_feature_cols(feat)
    non_const = [c for c in all_feature_cols if feat[c].std() > 0.001]
    
    # Build adversarial dataset
    labels_adv = np.array([1]*n_train + [0]*n_test)
    adv_df = pd.concat([feat[non_const], ftst[non_const]], axis=0).fillna(0)
    X_adv = adv_df[non_const].values.astype(np.float64)
    
    # Train 5-fold adversarial classifier
    gkf = GroupKFold(n_splits=5)
    # For adversarial, use sample_idx as groups
    adv_indices = np.arange(len(labels_adv))
    
    adv_preds = np.zeros(len(labels_adv))
    feature_importances = np.zeros(len(non_const))
    
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_adv, labels_adv, adv_indices)):
        ds = lgb.Dataset(X_adv[tr_idx], label=labels_adv[tr_idx], feature_name=non_const)
        vd = lgb.Dataset(X_adv[va_idx], label=labels_adv[va_idx], feature_name=non_const, reference=ds)
        params = {
            'objective': 'binary', 'metric': 'binary_logloss',
            'num_leaves': 31, 'max_depth': 5, 'learning_rate': 0.05,
            'n_estimators': 200, 'subsample': 0.8, 'colsample_bytree': 0.8,
            'reg_alpha': 1.0, 'reg_lambda': 5.0, 'min_child_samples': 20,
            'random_state': 42, 'verbose': -1, 'force_row_wise': True, 'n_jobs': 1,
        }
        model = lgb.train(params, ds, valid_sets=[vd],
                         callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)])
        adv_preds[va_idx] = model.predict(X_adv[va_idx])
        feature_importances += model.feature_importance(importance_type='gain')
        del model, ds, vd; gc.collect()
    
    feature_importances /= 5
    
    # AUC of adversarial classifier
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(labels_adv, adv_preds)
        print(f"    Adversarial AUC: {auc:.4f}")
    except:
        auc = 0.5
        print(f"    Adversarial AUC: N/A")
    
    # Rank features by importance
    ranked = sorted(zip(non_const, feature_importances), key=lambda x: -x[1])
    top_feats = ranked[:top_n]
    
    results = {
        'adversarial_auc': auc,
        'top_adversarial_features': [(name, float(imp)) for name, imp in top_feats],
        'feature_importances': {name: float(imp) for name, imp in ranked[:50]},
    }
    
    print(f"    Top 10 adversarial features:")
    for name, imp in top_feats[:10]:
        print(f"      {name}: {imp:.0f}")
    
    # Alternative: use feature importance from train/test score difference
    # Train small model, get train_score vs test_score per feature
    from sklearn.inspection import permutation_importance
    
    # Quick model for permutation importance
    p_quick = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.05,
        'n_estimators': 100, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_samples': 10,
        'random_state': 42, 'force_row_wise': True, 'n_jobs': 1,
    }
    sn = [sanitize_col(c) for c in non_const]
    ds_quick = lgb.Dataset(X_adv[:n_train], label=labels_adv[:n_train], feature_name=sn)
    m_quick = lgb.train(p_quick, ds_quick, num_boost_round=50)
    
    # Get feature contributions difference
    feat_imp_diff = {}
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_adv, labels_adv, adv_indices)):
        ds = lgb.Dataset(X_adv[tr_idx], label=labels_adv[tr_idx], feature_name=sn)
        vd = lgb.Dataset(X_adv[va_idx], label=labels_adv[va_idx], feature_name=sn, reference=ds)
        params = {**p_quick, 'random_state': 42+fold, 'n_estimators': 100}
        m = lgb.train(params, ds, valid_sets=[vd],
                     callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)])
        # Get split count per feature
        splits = m.feature_importance(importance_type='split')
        feat_imp_diff[fold] = dict(zip(non_const, splits))
        del m, ds, vd; gc.collect()
    
    avg_splits = {}
    for fc in non_const:
        avg_splits[fc] = np.mean([feat_imp_diff[f].get(fc, 0) for f in range(5)])
    
    ranked_splits = sorted(avg_splits.items(), key=lambda x: -x[1])
    results['top_split_features'] = [(name, float(cnt)) for name, cnt in ranked_splits[:top_n]]
    
    print(f"    Top 5 split-based features:")
    for name, cnt in ranked_splits[:5]:
        print(f"      {name}: split_count={cnt:.0f}")
    
    return results, ranked[:top_n]


# ============================================================
# EXPERIMENT 3: Quantile Normalization (train-to-test mapping)
# ============================================================
def quantile_normalize(pred, train_ref, test_ref):
    """
    Quantile mapping: transform predictions so their CDF matches
    between train reference and test reference.
    """
    # Build CDFs from the predictions themselves (train = oof, test = predicted)
    train_sorted = np.sort(train_ref)
    test_sorted = np.sort(test_ref)
    
    n = len(test_ref)
    # For each test prediction, find its rank in test, then map to that rank in train
    ranks = np.searchsorted(test_sorted, test_ref, side='left') / n
    ranks = np.clip(ranks, 0.001, 0.999)
    mapped = np.interp(ranks, np.linspace(0, 1, len(train_sorted)), train_sorted)
    
    return np.clip(mapped, 0.0001, 0.9999)

def experiment_quantile_norm(oof_all, test_all, y_dict):
    """
    Apply quantile normalization per target.
    Map test predictions to match the training (OOF) distribution.
    """
    print("\n  Quantile Normalization:")
    results = {}
    
    for target in TARGETS:
        oof = oof_all[target]  # shape (n_train, n_seeds)
        test_pred = test_all[target]  # shape (n_test, n_seeds)
        
        # Mean over seeds
        oof_mean = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
        test_mean = np.clip(test_pred.mean(axis=1), 0.0001, 0.9999)
        
        # Original OOF LL
        y = y_dict[target]
        oof_ll_orig = log_loss(y, oof_mean, labels=[0, 1])
        
        # Quantile normalize test → train distribution
        test_qn = quantile_normalize(test_mean, oof_mean, test_mean)
        
        # Mean-match after quantile norm
        test_qn_mm = mean_match(test_qn, y.mean())
        
        # Compute estimated test LL using a proxy
        # We estimate by: use OOF predictions to compute train LL,
        # then apply same transformation to estimate test effect
        test_ll_est = estimate_test_ll(oof_mean, test_qn_mm, y, method='quantile')
        
        results[target] = {
            'oof_ll_orig': float(oof_ll_orig),
            'test_mean_orig': float(test_mean.mean()),
            'test_mean_qn': float(test_qn_mm.mean()),
            'test_ll_estimate': float(test_ll_est),
        }
        
        print(f"    {target}: OOF LL={oof_ll_orig:.5f}, test_mean={test_qn_mm.mean():.3f}, est_test_ll={test_ll_est:.5f}")
    
    return results

def estimate_test_ll(oof_preds, test_preds, y_true, method='quantile'):
    """
    Estimate test log-loss using train/test distribution matching.
    Uses the approach: calibrate test predictions based on train calibration quality.
    """
    # Simple proxy: if test distribution shifted significantly,
    # estimate degradation proportionally to shift
    train_mean = oof_preds.mean()
    test_mean = test_preds.mean()
    shift = abs(train_mean - test_mean)
    
    # Train LL proxy (we know train LL from OOF)
    train_ll = log_loss(y_true, oof_preds, labels=[0, 1])
    
    # Estimate test LL: train LL + shift penalty
    # Larger distribution shift → larger degradation
    est_test_ll = train_ll + shift * 0.5
    return max(est_test_ll, train_ll - 0.05)  # small improvement possible with better calibration

# ============================================================
# EXPERIMENT 4: Rank Stabilization
# ============================================================
def rank_stabilize(pred, ref):
    """
    Rank stabilization: preserve the rank ordering of predictions
    but map the values to match the reference distribution's percentiles.
    Similar to quantile normalization but preserves within-group ranking.
    """
    # Get ranks of pred within ref distribution
    ref_sorted = np.sort(ref)
    n = len(ref_sorted)
    
    # For each prediction, find its rank percentile in ref
    ranks = np.searchsorted(ref_sorted, pred, side='left') / n
    ranks = np.clip(ranks, 0.001, 0.999)
    
    # Map ranks to ref percentiles
    mapped = np.interp(ranks, np.linspace(0, 1, n), ref_sorted)
    return np.clip(mapped, 0.0001, 0.9999)

def experiment_rank_stabilization(oof_all, test_all, y_dict, n_configs=300):
    """
    Try rank stabilization with different n_feat configurations.
    For each target, try rank-stabilized predictions with various feature subsets.
    """
    print("\n  Rank Stabilization:")
    results = {}
    
    for target in TARGETS:
        oof = oof_all[target]
        test_pred = test_all[target]
        y = y_dict[target]
        
        oof_mean = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
        test_mean = np.clip(test_pred.mean(axis=1), 0.0001, 0.9999)
        
        orig_ll = log_loss(y, oof_mean, labels=[0, 1])
        
        # Rank stabilize test predictions using OOF as reference
        test_rs = rank_stabilize(test_mean, oof_mean)
        test_rs_mm = mean_match(test_rs, y.mean())
        
        est_ll = estimate_test_ll(oof_mean, test_rs_mm, y, method='rank')
        
        results[target] = {
            'oof_ll': float(orig_ll),
            'test_mean_orig': float(test_mean.mean()),
            'test_mean_rs': float(test_rs_mm.mean()),
            'est_test_ll': float(est_ll),
        }
        
        print(f"    {target}: OOF={orig_ll:.5f}, test_mean={test_rs_mm.mean():.3f}, est_ll={est_ll:.5f}")
    
    return results


# ============================================================
# EXPERIMENT 5: Temperature Scaling + Per-Target Sharpening
# ============================================================
def temperature_scale(pred, temperature):
    """Apply temperature scaling to predictions (sigmoid-based)."""
    # Convert prob to logit, divide by temperature, convert back
    pred = np.clip(pred, 1e-7, 1 - 1e-7)
    logit = np.log(pred / (1 - pred))
    scaled_logit = logit / temperature
    return 1 / (1 + np.exp(-scaled_logit))

def find_optimal_temperature(oof_preds, y_true):
    """Find temperature that minimizes log-loss on OOF predictions."""
    def loss_fn(T):
        scaled = temperature_scale(oof_preds, T)
        return log_loss(y_true, scaled, labels=[0, 1])
    
    result = minimize_scalar(loss_fn, bounds=(0.1, 5.0), method='bounded')
    return result.x, result.fun

def experiment_temperature_scaling(feat, ftst, y_dict, drift_features_set=None):
    """
    Per-target temperature scaling with fold-level optimization.
    Try different temperature per fold, then ensemble.
    """
    print("\n  Temperature Scaling + Per-Target Sharpening:")
    results = {}
    
    for target in TARGETS:
        y = y_dict[target]
        sw = V53_SWEEP[target]
        cfg = CFGS[sw['cfg']]
        all_feature_cols = get_feature_cols(feat)
        cols = remove_leak(all_feature_cols, target)
        
        # If drift_features_set is provided, remove drift-heavy features
        if drift_features_set:
            cols = [c for c in cols if c not in drift_features_set]
        
        # Train with GroupKFold, find per-fold optimal T
        gkf = GroupKFold(n_splits=5)
        fold_temps = []
        fold_losses = []
        oof_temp = np.zeros((len(y), len(SEEDS)))
        test_temp = np.zeros((len(ftst), len(SEEDS)))
        
        all_feature_cols_s = get_feature_cols(feat)
        cols_s = remove_leak(all_feature_cols_s, target)
        if drift_features_set:
            cols_s = [c for c in cols_s if c not in drift_features_set]
        
        sn = [sanitize_col(c) for c in cols_s]
        Xf = feat[cols_s].fillna(0).values.astype(np.float64)
        Xt = ftst[cols_s].fillna(0).values.astype(np.float64)
        spw = max(((y==0).sum()) / max((y==1).sum(), 1), 0.1)
        
        fold_temps_list = []
        for si, seed in enumerate(SEEDS):
            fold_temps_seed = []
            for tri, vai in gkf.split(feat, y, feat['subject_id']):
                ds = lgb.Dataset(Xf[tri], label=y[tri], feature_name=sn)
                vd = lgb.Dataset(Xf[vai], label=y[vai], feature_name=sn, reference=ds)
                params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed,
                          'force_row_wise': True, 'n_jobs': 1}
                m = lgb.train(params, ds, valid_sets=[vd],
                             num_boost_round=cfg['n_estimators'],
                             callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
                
                oof_fold = m.predict(Xf[vai])
                test_fold = m.predict(Xt)
                
                # Find optimal temperature for this fold
                opt_T, opt_loss = find_optimal_temperature(oof_fold, y[vai])
                fold_temps_seed.append(opt_T)
                fold_losses.append(opt_loss)
                
                # Apply temperature scaling
                scaled_oof = np.clip(temperature_scale(oof_fold, opt_T), 0.0001, 0.9999)
                scaled_test = np.clip(temperature_scale(test_fold, opt_T), 0.0001, 0.9999)
                
                # Mean-match to training mean
                scaled_oof = mean_match(scaled_oof, y[tri].mean())
                oof_temp[vai, si] += scaled_oof
                test_temp[:, si] += scaled_test
                
                del ds, vd, m; gc.collect()
            
            fold_temps_list.append(fold_temps_seed)
        
        oof_temp = oof_temp.mean(axis=1)
        test_temp = test_temp.mean(axis=1)
        
        # Compute OOF LL with temperature scaling
        oof_ll_temp = log_loss(y, oof_temp, labels=[0, 1])
        
        # Estimate test LL
        train_ll_orig = log_loss(y, np.clip(oof.mean(axis=1), 0.0001, 0.9999), labels=[0, 1])
        est_test_ll = estimate_test_ll(oof_temp, test_temp, y, method='temp')
        
        results[target] = {
            'oof_ll': float(oof_ll_temp),
            'avg_temperature': float(np.mean(fold_temps_list)),
            'est_test_ll': float(est_test_ll),
            'test_mean': float(test_temp.mean()),
        }
        
        print(f"    {target}: OOF LL={oof_ll_temp:.5f}, avg_T={np.mean(fold_temps_list):.3f}, est_ll={est_test_ll:.5f}")
    
    return results


# ============================================================
# EXPERIMENT 6: Calibration Error Analysis (ECE)
# ============================================================
def compute_ece(y_true, y_pred, n_bins=20):
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_details = []
    
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i+1]
        mask = (y_pred >= lo) & (y_pred < hi)
        
        if mask.sum() == 0:
            bin_details.append({'bin': f'[{lo:.2f},{hi:.2f})', 'count': 0, 'accuracy': 0, 'mean_pred': 0, 'ece_contribution': 0})
            continue
        
        accuracy = y_true[mask].mean()
        mean_pred = y_pred[mask].mean()
        contribution = len(y_true[mask]) / len(y_true) * abs(accuracy - mean_pred)
        ece += contribution
        
        bin_details.append({
            'bin': f'[{lo:.2f},{hi:.2f})',
            'count': int(mask.sum()),
            'accuracy': float(accuracy),
            'mean_pred': float(mean_pred),
            'ece_contribution': float(contribution),
        })
    
    return float(ece), bin_details

def mce(y_true, y_pred, n_bins=10):
    """Maximum Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    max_ce = 0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i+1]
        mask = (y_pred >= lo) & (y_pred < hi)
        if mask.sum() > 0:
            ce = abs(y_true[mask].mean() - y_pred[mask].mean())
            max_ce = max(max_ce, ce)
    return float(max_ce)

def experiment_calibration_error(oof_all, test_all, y_dict):
    """Compute ECE per target and identify miscalibrated targets."""
    print("\n  Calibration Error Analysis (ECE):")
    results = {}
    
    for target in TARGETS:
        oof = oof_all[target]
        oof_mean = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
        y = y_dict[target]
        
        ece, bin_details = compute_ece(y, oof_mean, n_bins=20)
        mce_val = mce(y, oof_mean, n_bins=10)
        
        # Identify worst bins
        worst_bins = sorted(bin_details, key=lambda x: -x['ece_contribution'])[:5]
        
        results[target] = {
            'ece': float(ece),
            'mce': float(mce_val),
            'oof_ll': float(log_loss(y, oof_mean, labels=[0, 1])),
            'worst_bins': worst_bins,
        }
        
        print(f"    {target}: ECE={ece:.4f}, MCE={mce_val:.4f}, LL={log_loss(y, oof_mean, labels=[0, 1]):.5f}")
        print(f"      Worst bins:")
        for wb in worst_bins:
            if wb['count'] > 0:
                print(f"        {wb['bin']}: acc={wb['accuracy']:.3f}, pred={wb['mean_pred']:.3f}, contrib={wb['ece_contribution']:.4f}")
    
    # Identify targets with highest ECE for focused calibration
    ranked_by_ece = sorted(results.items(), key=lambda x: -x[1]['ece'])
    results['ece_ranking'] = [(t, r['ece']) for t, r in ranked_by_ece]
    results['most_miscalibrated'] = ranked_by_ece[0][0] if ranked_by_ece else None
    
    return results


# ============================================================
# MAIN PIPELINE
# ============================================================
print("=" * 70)
print("V128: Distribution Shift Research + Advanced Calibration")
print("=" * 70)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load data
print("\nLoading data...")
feat = pd.read_parquet(DATA / 'features_clean_v60.parquet')
ftst = pd.read_parquet(DATA / 'test_features_clean_v60.parquet')

# Sanitize column names
feat.columns = [sanitize_col(c) for c in feat.columns]
ftst.columns = [sanitize_col(c) for c in ftst.columns]

print(f"  Train: {feat.shape}, Test: {ftst.shape}")

# Feature columns
all_feature_cols = get_feature_cols(feat)
non_const = [c for c in all_feature_cols if feat[c].std() > 0.001]
print(f"  Total features: {len(non_const)}")

y_dict = {t: feat[t].values.astype(np.float64) for t in TARGETS}

# ============================================================
# BASELINE: V127 (all features, no selection)
# ============================================================
print("\n" + "=" * 70)
print("BASELINE: V127 (all non-leak features, no selection)")
print("=" * 70)

baseline_oof = {}
baseline_test = {}
baseline_configs_used = {}

for target in TARGETS:
    sw = V53_SWEEP[target]
    cfg = CFGS[sw['cfg']]
    y = y_dict[target]
    
    cols = remove_leak(all_feature_cols, target)
    baseline_configs_used[target] = {'cfg': sw['cfg'], 'n_feats': len(cols)}
    
    print(f"\n  Training {target} (cfg={sw['cfg']}, feats={len(cols)})...")
    oof, test_pred = train_cv(feat, ftst, cols, y, SEEDS, cfg)
    baseline_oof[target] = oof
    baseline_test[target] = test_pred
    
    oof_mean = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
    test_mean = np.clip(test_pred.mean(axis=1), 0.0001, 0.9999)
    
    oof_ll = log_loss(y, oof_mean, labels=[0, 1])
    print(f"    OOF LL: {oof_ll:.5f}, train_mean={oof_mean.mean():.4f}, test_mean={test_mean.mean():.4f}")

avg_baseline_oof = np.mean([log_loss(y_dict[t], np.clip(baseline_oof[t].mean(axis=1), 0.0001, 0.9999), labels=[0,1]) for t in TARGETS])
print(f"\n  AVG BASELINE OOF: {avg_baseline_oof:.5f}")

# ============================================================
# EXPERIMENT 1: PSI
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 1: PSI (Population Stability Index) Analysis")
print("=" * 70)

psi_results, drift_features = experiment_psi(feat, ftst, y_dict)
print(f"\n  Drift features (>0.1 PSI): {len(drift_features)}")

# PSI-based model: remove drift features and retrain
psi_results['oof_after_drift_removal'] = {}
for target in TARGETS:
    sw = V53_SWEEP[target]
    cfg = CFGS[sw['cfg']]
    y = y_dict[target]
    
    cols_orig = remove_leak(all_feature_cols, target)
    cols_reduced = [c for c in cols_orig if c not in drift_features]
    
    print(f"    {target}: {len(cols_orig)}→{len(cols_reduced)} feats")
    
    if len(cols_reduced) > 10:
        oof, _ = train_cv(feat, None, cols_reduced, y, SEEDS, cfg)
        oof_mean = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
        ll = log_loss(y, oof_mean, labels=[0, 1])
        baseline_oof[target] = oof
        psi_results['oof_after_drift_removal'][target] = {'ll': ll, 'n_feats': len(cols_reduced)}
        print(f"      OOF LL after drift removal: {ll:.5f}")
    else:
        print(f"      Skipping: too few features left")

# ============================================================
# EXPERIMENT 2: Adversarial Validation
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 2: Adversarial Validation")
print("=" * 70)

adv_results, adv_top_feats = experiment_adversarial(feat, ftst, TARGETS)
print(f"  Top adversarial features to exclude: {[n for n, _ in adv_top_feats[:10]]}")

# Adversarial-based model: exclude top adversarial features
adv_results['oof_after_adversarial_removal'] = {}
for target in TARGETS:
    sw = V53_SWEEP[target]
    cfg = CFGS[sw['cfg']]
    y = y_dict[target]
    
    cols_orig = remove_leak(all_feature_cols, target)
    # Remove top 20 adversarial features
    exclude_feats = set([n for n, _ in adv_top_feats[:20]])
    cols_reduced = [c for c in cols_orig if c not in exclude_feats]
    
    oof, _ = train_cv(feat, None, cols_reduced, y, SEEDS, cfg)
    oof_mean = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
    ll = log_loss(y, oof_mean, labels=[0, 1])
    adv_results['oof_after_adversarial_removal'][target] = {'ll': ll, 'n_feats': len(cols_reduced)}
    print(f"    {target}: {len(cols_orig)}→{len(cols_reduced)} feats, OOF LL={ll:.5f}")

# ============================================================
# EXPERIMENT 3: Quantile Normalization
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 3: Quantile Normalization")
print("=" * 70)

qn_results = experiment_quantile_norm(baseline_oof, baseline_test, y_dict)

# ============================================================
# EXPERIMENT 4: Rank Stabilization
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 4: Rank Stabilization")
print("=" * 70)

rs_results = experiment_rank_stabilization(baseline_oof, baseline_test, y_dict)


# ============================================================
# EXPERIMENT 5: Temperature Scaling
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 5: Temperature Scaling + Per-Target Sharpening")
print("=" * 70)

temp_results = experiment_temperature_scaling(feat, ftst, y_dict, drift_features_set=drift_features)

# Also run temperature scaling without drift feature removal (comparison)
print("\n  Temperature Scaling (without drift removal):")
temp_results_no_removal = experiment_temperature_scaling(feat, ftst, y_dict, drift_features_set=None)

# ============================================================
# EXPERIMENT 6: Calibration Error Analysis
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 6: Calibration Error Analysis (ECE)")
print("=" * 70)

ece_results = experiment_calibration_error(baseline_oof, baseline_test, y_dict)

# ============================================================
# COMBINED: ECE-guided per-target calibration
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 6b: ECE-guided Per-Target Calibration")
print("=" * 70)

ece_cal_results = {}
most_miscalib = ece_results.get('most_miscalibrated', 'S2')

for target in TARGETS:
    oof = baseline_oof[target]
    oof_mean = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
    y = y_dict[target]
    
    # Find fold-level optimal temperature
    gkf = GroupKFold(n_splits=5)
    all_feature_cols_s = get_feature_cols(feat)
    cols_s = remove_leak(all_feature_cols_s, target)
    sn = [sanitize_col(c) for c in cols_s]
    Xf = feat[cols_s].fillna(0).values.astype(np.float64)
    spw = max(((y==0).sum()) / max((y==1).sum(), 1), 0.1)
    
    fold_temps = []
    fold_scaled_oof = np.zeros(len(y))
    
    for si, seed in enumerate(SEEDS):
        p = {**CFGS[V53_SWEEP[target]['cfg']], 'scale_pos_weight': spw,
             'random_state': seed, 'force_row_wise': True, 'n_jobs': 1}
        
        for tri, vai in gkf.split(feat, y, feat['subject_id']):
            ds = lgb.Dataset(Xf[tri], label=y[tri], feature_name=sn)
            vd = lgb.Dataset(Xf[vai], label=y[vai], feature_name=sn, reference=ds)
            m = lgb.train(p, ds, valid_sets=[vd],
                         num_boost_round=CFGS[V53_SWEEP[target]['cfg']]['n_estimators'],
                         callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            
            raw_oof = m.predict(Xf[vai])
            
            # Try multiple temperatures and pick best ECE
            best_T = 1.0
            best_ece = 999
            for T_try in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]:
                scaled = np.clip(temperature_scale(raw_oof, T_try), 0.0001, 0.9999)
                scaled_mm = mean_match(scaled, y[tri].mean())
                ece_try, _ = compute_ece(y[vai], scaled_mm)
                if ece_try < best_ece:
                    best_ece = ece_try
                    best_T = T_try
            
            fold_temps.append(best_T)
            scaled_final = np.clip(temperature_scale(raw_oof, best_T), 0.0001, 0.9999)
            scaled_final = mean_match(scaled_final, y[tri].mean())
            fold_scaled_oof[vai] += scaled_final
            
            del ds, vd, m; gc.collect()
    
    fold_scaled_oof /= len(SEEDS)
    
    ece_final, _ = compute_ece(y, fold_scaled_oof)
    ll_final = log_loss(y, fold_scaled_oof, labels=[0, 1])
    ece_cal_results[target] = {
        'oof_ll': float(ll_final),
        'ece': float(ece_final),
        'avg_T': float(np.mean(fold_temps)),
        'oof_preds': fold_scaled_oof,
    }
    
    orig_ll = log_loss(y, oof_mean, labels=[0, 1])
    print(f"    {target}: orig_LL={orig_ll:.5f}→cal_LL={ll_final:.5f}, ECE={ece_final:.4f}, avg_T={np.mean(fold_temps):.3f}")

avg_ece_cal = np.mean([v['oof_ll'] for v in ece_cal_results.values()])
print(f"\n  AVG ECE-calibrated OOF: {avg_ece_cal:.5f}")

# ============================================================
# COMPREHENSIVE COMPARISON
# ============================================================
print("\n" + "=" * 70)
print("COMPREHENSIVE COMPARISON")
print("=" * 70)

comparison = {
    'baseline': {'avg_oof': float(avg_baseline_oof)},
    'psi_drift_removal': None,
    'adversarial_removal': None,
    'ece_calibration': {'avg_oof': float(avg_ece_cal)},
    'temperature_scaling': None,
}

# PSI
psi_lls = [v['ll'] for v in psi_results['oof_after_drift_removal'].values()]
if psi_lls:
    comparison['psi_drift_removal'] = {'avg_oof': float(np.mean(psi_lls)), 'per_target': {t: v['ll'] for t, v in psi_results['oof_after_drift_removal'].items()}}

# Adversarial
adv_lls = [v['ll'] for v in adv_results['oof_after_adversarial_removal'].values()]
if adv_lls:
    comparison['adversarial_removal'] = {'avg_oof': float(np.mean(adv_lls)), 'per_target': {t: v['ll'] for t, v in adv_results['oof_after_adversarial_removal'].items()}}

# Temperature
temp_lls = [v['oof_ll'] for v in temp_results.values()]
if temp_lls:
    comparison['temperature_scaling'] = {'avg_oof': float(np.mean(temp_lls)), 'per_target': {t: v['oof_ll'] for t, v in temp_results.items()}}

# Print comparison table
print(f"\n{'Method':<30} {'AVG OOF LL':>12} {'Δ vs Baseline':>14}")
print("-" * 58)
for method, data in comparison.items():
    if data is None:
        continue
    avg = data['avg_oof']
    delta = avg - avg_baseline_oof
    print(f"  {method:<28} {avg:>12.5f} {delta:>+13.5f}")

# Print detailed per-target
print("\n\nPer-Target OOF LL Comparison:")
print(f"{'Target':<8}", end="")
for method in ['baseline', 'psi_drift', 'adversarial', 'ece_cal', 'temp_scale']:
    short = method[:8]
    print(f"  {short:>8}", end="")
print()
print("-" * 58)

for target in TARGETS:
    print(f"  {target:<8}", end="")
    
    # Baseline
    b_ll = log_loss(y_dict[target], np.clip(baseline_oof[target].mean(axis=1), 0.0001, 0.9999), labels=[0,1])
    print(f"  {b_ll:>8.5f}", end="")
    
    # PSI
    if target in psi_results.get('oof_after_drift_removal', {}):
        print(f"  {psi_results['oof_after_drift_removal'][target]['ll']:>8.5f}", end="")
    else:
        print(f"  {'N/A':>8}", end="")
    
    # Adversarial
    if target in adv_results.get('oof_after_adversarial_removal', {}):
        print(f"  {adv_results['oof_after_adversarial_removal'][target]['ll']:>8.5f}", end="")
    else:
        print(f"  {'N/A':>8}", end="")
    
    # ECE calibration
    if target in ece_cal_results:
        print(f"  {ece_cal_results[target]['oof_ll']:>8.5f}", end="")
    else:
        print(f"  {'N/A':>8}", end="")
    
    # Temperature
    if target in temp_results:
        print(f"  {temp_results[target]['oof_ll']:>8.5f}", end="")
    else:
        print(f"  {'N/A':>8}", end="")
    
    print()

# ============================================================
# BEST CONFIG → GENERATE PREDICTIONS
# ============================================================
print("\n" + "=" * 70)
print("GENERATING BEST PREDICTIONS")
print("=" * 70)

# Find best method per target and generate predictions
# Strategy: use the method that gives best OOF per target
best_preds = {}
best_method_per_target = {}

# Gather all method results
methods = {}

# Baseline test predictions
for t in TARGETS:
    methods[f'baseline_{t}'] = (baseline_test[t], 'baseline')

# PSI
for t, v in psi_results.get('oof_after_drift_removal', {}).items():
    cols = remove_leak(get_feature_cols(feat), t)
    cols = [c for c in cols if c not in drift_features]
    if len(cols) > 10:
        oof, tp = train_cv(feat, ftst, cols, y_dict[t], SEEDS, CFGS[V53_SWEEP[t]['cfg']])
        methods[f'psi_{t}'] = (tp, 'psi_drift_removal')

# Adversarial
for t, v in adv_results.get('oof_after_adversarial_removal', {}).items():
    cols = remove_leak(get_feature_cols(feat), t)
    exclude_feats = set([n for n, _ in adv_top_feats[:20]])
    cols = [c for c in cols if c not in exclude_feats]
    oof, tp = train_cv(feat, ftst, cols, y_dict[t], SEEDS, CFGS[V53_SWEEP[t]['cfg']])
    methods[f'adv_{t}'] = (tp, 'adversarial_removal')

# Temperature
for t, v in temp_results.items():
    cols = remove_leak(get_feature_cols(feat), t)
    cols = [c for c in cols if c not in drift_features]
    oof, tp = train_cv(feat, ftst, cols, y_dict[t], SEEDS, CFGS[V53_SWEEP[t]['cfg']])
    # Apply temperature scaling to test predictions
    test_mean = tp.mean(axis=1)
    # Use the average temperature found during training
    T = v['avg_temperature']
    scaled = np.clip(temperature_scale(test_mean, T), 0.0001, 0.9999)
    scaled = mean_match(scaled, y_dict[t].mean())
    methods[f'temp_{t}'] = (scaled, 'temperature_scaling')

# ECE calibration
for t, v in ece_cal_results.items():
    # For test predictions, apply similar temperature scaling
    # Retrain full model and apply
    cols = remove_leak(get_feature_cols(feat), t)
    sn = [sanitize_col(c) for c in cols]
    Xf = feat[cols].fillna(0).values.astype(np.float64)
    Xt = ftst[cols].fillna(0).values.astype(np.float64)
    spw = max(((y_dict[t]==0).sum()) / max((y_dict[t]==1).sum(), 1), 0.1)
    
    full_tp = np.zeros(len(ftst))
    for seed in SEEDS:
        p = {**CFGS[V53_SWEEP[t]['cfg']], 'scale_pos_weight': spw,
             'random_state': seed, 'force_row_wise': True, 'n_jobs': 1}
        ds = lgb.Dataset(Xf, label=y_dict[t], feature_name=sn)
        m = lgb.train(p, ds, num_boost_round=CFGS[V53_SWEEP[t]['cfg']]['n_estimators'])
        pred = m.predict(Xt)
        T = v['avg_T']
        pred = np.clip(temperature_scale(pred, T), 0.0001, 0.9999)
        pred = mean_match(pred, y_dict[t].mean())
        full_tp += pred
    full_tp /= len(SEEDS)
    methods[f'ece_{t}'] = (full_tp, 'ece_calibration')

# Select best per target by OOF
# Pre-compute OOF LLs per method per target
method_oof = {}
# Baseline
for t in TARGETS:
    method_oof[('baseline', t)] = log_loss(y_dict[t], np.clip(baseline_oof[t].mean(axis=1), 0.0001, 0.9999), labels=[0,1])
# PSI
for t, v in psi_results.get('oof_after_drift_removal', {}).items():
    method_oof[('psi_drift_removal', t)] = v['ll']
# Adversarial
for t, v in adv_results.get('oof_after_adversarial_removal', {}).items():
    method_oof[('adversarial_removal', t)] = v['ll']
# Temperature
for t, v in temp_results.items():
    method_oof[('temperature_scaling', t)] = v['oof_ll']
# ECE calibration
for t, v in ece_cal_results.items():
    method_oof[('ece_calibration', t)] = v['oof_ll']

# Now pick best per target
for target in TARGETS:
    best_ll = float('inf')
    best_method = 'baseline'
    best_test_pred = baseline_test[target].mean(axis=1)
    
    for method_name in ['baseline', 'psi_drift_removal', 'adversarial_removal', 'temperature_scaling', 'ece_calibration']:
        oof_ll = method_oof.get((method_name, target), float('inf'))
        if oof_ll < best_ll:
            best_ll = oof_ll
            best_method = method_name
            # Get test predictions
            key = f'{method_name}_{target}'
            if key in methods:
                test_arr = methods[key][0]
                if isinstance(test_arr, np.ndarray) and test_arr.ndim == 2:
                    test_arr = test_arr.mean(axis=1)
                best_test_pred = test_arr
    
    best_method_per_target[target] = best_method
    best_preds[target] = np.clip(best_test_pred, 0.0001, 0.9999)
    print(f"  {target}: best_method={best_method}, OOF LL={best_ll:.5f}")

# ============================================================
# LB ESTIMATION
# ============================================================
print("\n" + "=" * 70)
print("LB ESTIMATION")
print("=" * 70)

est_lbs = {}
for target in TARGETS:
    oof_mean = np.clip(baseline_oof[target].mean(axis=1), 0.0001, 0.9999)
    test_mean = best_preds[target]
    
    train_ll = log_loss(y_dict[target], oof_mean, labels=[0, 1])
    shift = abs(oof_mean.mean() - test_mean.mean())
    est_test = estimate_test_ll(oof_mean, test_mean, y_dict[target], method='combined')
    
    # Use multiple estimation methods
    ests = [train_ll + shift * 0.3, train_ll + shift * 0.5, train_ll + shift * 0.7]
    est_lbs[target] = {
        'train_ll': float(train_ll),
        'shift': float(shift),
        'est_ll_low': float(ests[0]),
        'est_ll_mid': float(ests[1]),
        'est_ll_high': float(ests[2]),
    }

avg_est_low = np.mean([v['est_ll_low'] for v in est_lbs.values()])
avg_est_mid = np.mean([v['est_ll_mid'] for v in est_lbs.values()])
avg_est_high = np.mean([v['est_ll_high'] for v in est_lbs.values()])

print(f"  AVG Estimated LB: {avg_est_mid:.5f} (range: {avg_est_low:.5f} - {avg_est_high:.5f})")
print(f"  Current V53 Swept LB: 0.65358")
print(f"  Improvement: {0.65358 - avg_est_mid:+.5f}")

# ============================================================
# SAVE SUBMISSION
# ============================================================
print("\n" + "=" * 70)
print("SAVING SUBMISSION")
print("=" * 70)

ts = datetime.now().strftime('%Y%m%d_%H%M%S')

# Generate all method submissions for reference
all_submissions = {}
for method_name in ['baseline', 'psi_drift_removal', 'adversarial_removal', 'ece_calibration', 'temperature_scaling']:
    submit_df = pd.DataFrame()
    submit_df['subject_id'] = ftst['subject_id'].values
    submit_df['sleep_date'] = ftst['sleep_date'].values
    submit_df['lifelog_date'] = ftst['lifelog_date'].values
    
    method_worked = False
    for target in TARGETS:
        method_key = f'{method_name}_{target}'
        if method_key in methods:
            test_arr = methods[method_key][0]
            if isinstance(test_arr, np.ndarray) and test_arr.ndim == 2:
                test_arr = test_arr.mean(axis=1)
            if len(test_arr) == len(ftst):
                submit_df[sanitize_col(target)] = np.clip(test_arr, 0.0001, 0.9999)
                method_worked = True
    
    if method_worked:
        sub_name = f"submission_v128_{method_name}_{ts}"
        sub_path = SUBMIT / f"{sub_name}.csv"
        submit_df.to_csv(sub_path, index=False)
        all_submissions[method_name] = str(sub_path)
        print(f"  {method_name}: {sub_path}")

# Best per-target submission
submit_df = pd.DataFrame()
submit_df['subject_id'] = ftst['subject_id'].values
submit_df['sleep_date'] = ftst['sleep_date'].values
submit_df['lifelog_date'] = ftst['lifelog_date'].values
for target in TARGETS:
    submit_df[sanitize_col(target)] = best_preds[target]

best_sub_path = SUBMIT / f"submission_v128_best_per_target_{ts}.csv"
submit_df.to_csv(best_sub_path, index=False)
print(f"  Best per-target: {best_sub_path}")

# ============================================================
# SAVE EXPERIMENT LOG
# ============================================================
print("\n" + "=" * 70)
print("SAVING EXPERIMENT LOG")
print("=" * 70)

experiment_log = {
    'name': 'V128 Distribution Shift Research',
    'description': '6 experiments: PSI, Adversarial Validation, Quantile Normalization, Rank Stabilization, Temperature Scaling, ECE',
    'timestamp': ts,
    'baseline': {
        'avg_oof': float(avg_baseline_oof),
        'per_target': {t: float(log_loss(y_dict[t], np.clip(baseline_oof[t].mean(axis=1), 0.0001, 0.9999), labels=[0,1])) for t in TARGETS},
    },
    'experiment1_psi': {
        'drift_features': list(drift_features),
        'n_drift_features': len(drift_features),
        'oof_after_drift_removal': {t: {k: float(v) if isinstance(v, (int, float)) else v for k, v in vv.items()} for t, vv in psi_results.get('oof_after_drift_removal', {}).items()},
        'avg_psi': psi_results.get('psi_global_avg', 0),
    },
    'experiment2_adversarial': {
        'auc': float(adv_results.get('adversarial_auc', 0)),
        'top_features': adv_results.get('top_adversarial_features', [])[:20],
        'oof_after_removal': {t: {k: float(v) if isinstance(v, (int, float)) else v for k, v in vv.items()} for t, vv in adv_results.get('oof_after_adversarial_removal', {}).items()},
    },
    'experiment3_quantile_norm': {t: {k: float(v) if isinstance(v, (int, float)) else v for k, v in vv.items()} for t, vv in qn_results.items()},
    'experiment4_rank_stabilization': {t: {k: float(v) if isinstance(v, (int, float)) else v for k, v in vv.items()} for t, vv in rs_results.items()},
    'experiment5_temperature_scaling': {t: {k: float(v) if isinstance(v, (int, float)) else v for k, v in vv.items()} for t, vv in temp_results.items()},
    'experiment6_ece': {t: {'ece': vv.get('ece', 0), 'mce': vv.get('mce', 0), 'n_bins': len(vv.get('bins', [])),
                             'bins_summary': [{'bin': b.get('bin',''), 'count': b.get('count',0),
                                               'accuracy': round(b.get('accuracy',0),4),
                                               'mean_pred': round(b.get('mean_pred',0),4)}
                                              for b in vv.get('bins', [])][:10]
                            } for t, vv in ece_results.items()},
    'experiment6b_ece_calibration': {t: {k: float(v) if isinstance(v, (int, float)) else v for k, v in vv.items()} for t, vv in ece_cal_results.items()},
    'best_per_target': best_method_per_target,
    'est_lbs': {t: {k: float(v) for k, v in vv.items()} for t, vv in est_lbs.items()},
    'avg_est_lb_low': float(avg_est_low),
    'avg_est_lb_mid': float(avg_est_mid),
    'avg_est_lb_high': float(avg_est_high),
    'submissions': all_submissions,
    'best_submission': str(best_sub_path),
}

log_path = EXPERIMENTS / f'v128_distribution_shift_{ts}.json'
with open(log_path, 'w') as fout:
    json.dump(experiment_log, fout, indent=2, default=str)
print(f"  Experiment log: {log_path}")

print("\n" + "=" * 70)
print("=== V128 COMPLETE ===")
print("=" * 70)
print(f"Time: {datetime.now().strftime('%H:%M:%S')}")

