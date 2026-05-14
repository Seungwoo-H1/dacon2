#!/usr/bin/env python3
"""
V260: Distribution Shift Correction — PSI Drift + Quantile + Rank Stabilization

Hypothesis: train/test distribution shift causes good OOF but poor LB.
We test 6 correction strategies and their combinations.

Data: features_clean_v60.parquet
CV: GroupKFold(5)
Seed: 42
Output: experiments/v260_distribution_correction_result.json
"""

import json, re, gc, warnings, time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss
from scipy.stats import rankdata
import lightgbm as lgb

warnings.filterwarnings('ignore')

ROOT = Path('/home/mwoo423/projects/dacon2')
DATA = ROOT / 'data_processed'
EXPERIMENTS = ROOT / 'experiments'

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
META_COLS = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}

# ─── Helpers ─────────────────────────────────────────────────────────────────

def sanitize_col(n):
    return re.sub(r'[^a-zA-Z0-9_]', '_', n)

def get_numeric_cols(df, exclude=None):
    ex = META_COLS | set(TARGETS)
    if exclude:
        ex |= exclude
    return [c for c in df.columns
            if df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]
            and c not in ex]

CFGS = {
    'wide':   {'num_leaves': 30, 'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 300,
               'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_samples': 5},
    'deep':   {'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.02, 'n_estimators': 1000,
               'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 0.5, 'reg_lambda': 2.0, 'min_child_samples': 15},
    'v48':    {'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 500,
               'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_samples': 10},
    'safety': {'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.02, 'n_estimators': 1000,
               'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 3.0, 'reg_lambda': 10.0, 'min_child_samples': 20},
}

V53_SWEEP = {
    'Q1': {'cfg': 'deep', 'n_feat': 19},
    'Q2': {'cfg': 'deep', 'n_feat': 14},
    'Q3': {'cfg': 'v48', 'n_feat': 11},
    'S1': {'cfg': 'wide', 'n_feat': 21},
    'S2': {'cfg': 'deep', 'n_feat': 19},
    'S3': {'cfg': 'safety', 'n_feat': 23},
    'S4': {'cfg': 'wide', 'n_feat': 20},
}

def cfg_to_params(cfg_s, seed, spw):
    p = dict(cfg_s)
    p.update({
        'scale_pos_weight': spw, 'random_state': seed,
        'force_row_wise': True, 'n_jobs': 1, 'verbose': -1
    })
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
    """Rank features by importance for a target."""
    y = feat_df[target].values.astype(np.float64)
    X = feat_df[fcols].fillna(0).values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    p = cfg_to_params({'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03,
                       'n_estimators': 50, 'subsample': 0.7, 'colsample_bytree': 0.7,
                       'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_samples': 10}, seed, spw)
    sn = [sanitize_col(c) for c in fcols]
    ds = lgb.Dataset(X, label=y, feature_name=sn)
    m = lgb.train(p, ds, num_boost_round=50)
    imp = m.feature_importance(importance_type='gain')
    ranked = sorted(zip(fcols, imp), key=lambda x: -x[1])
    del m, ds
    gc.collect()
    return [r[0] for r in ranked]


# ─── Core training function ─────────────────────────────────────────────────

def train_and_oof(df, targets, feature_cols, target_name, config, seed=42):
    """Train with GroupKFold, return per-fold OOF predictions and scores."""
    gkf = GroupKFold(n_splits=5)
    groups = df['subject_id'].values
    fcols_sanitized = [sanitize_col(c) for c in feature_cols]

    oof_preds = np.zeros(len(df))
    scores = []

    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(df, df[target_name].values, groups)):
        X_tr, y_tr = df[feature_cols].iloc[tr_idx], df[target_name].iloc[tr_idx]
        X_va = df[feature_cols].iloc[va_idx]
        y_true = df[target_name].iloc[va_idx].values

        y_tr_vals = y_tr.values.astype(np.float64)
        X_tr_filled = X_tr.fillna(0).values.astype(np.float64)
        X_va_filled = X_va.fillna(0).values.astype(np.float64)

        spw = max(((y_tr_vals == 0).sum()) / max((y_tr_vals == 1).sum(), 1), 0.1)
        params = cfg_to_params(config, seed, spw)

        dtrain = lgb.Dataset(X_tr_filled, label=y_tr_vals, feature_name=fcols_sanitized)
        dval = lgb.Dataset(X_va_filled, label=y_true.astype(np.float64),
                           feature_name=fcols_sanitized, reference=dtrain)

        model = lgb.train(params, dtrain, num_boost_round=config['n_estimators'],
                          valid_sets=[dval], callbacks=[lgb.log_evaluation(0)])
        preds = model.predict(X_va_filled)
        oof_preds[va_idx] = preds

        ll = log_loss(y_true.astype(int), np.clip(preds, 0.0001, 0.9999))
        scores.append(ll)

    mean_score = float(np.mean(scores))
    del oof_preds, scores
    gc.collect()
    return mean_score


# ─── Load data ────────────────────────────────────────────────────────────────
print("=" * 60)
print("V260: Distribution Shift Correction")
print("=" * 60)

train = pd.read_parquet(DATA / 'features_clean_v60.parquet')
test = pd.read_parquet(DATA / 'test_features_clean_v60.parquet')
all_features = get_numeric_cols(train)
print(f"Train shape: {train.shape}, Features: {len(all_features)}")

# Baseline model configs per target (from v87 best)
baseline_configs = V53_SWEEP.copy()

# ─── 1. PSI-Based Feature Weighting ─────────────────────────────────────────
print("\n" + "=" * 60)
print("1. PSI-Based Feature Weighting")
print("=" * 60)

def compute_psi(expected, actual, bins=20):
    expected = np.array(expected, dtype=np.float64)
    actual = np.array(actual, dtype=np.float64)
    expected_clean = expected[~np.isnan(expected)]
    actual_clean = actual[~np.isnan(actual)]
    if len(expected_clean) < 10 or len(actual_clean) < 10:
        return float("nan")
    min_val = min(expected_clean.min(), actual_clean.min())
    max_val = max(expected_clean.max(), actual_clean.max())
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    bin_edges[-1] += np.finfo(float).eps
    exp_counts, _ = np.histogram(expected_clean, bins=bin_edges)
    act_counts, _ = np.histogram(actual_clean, bins=bin_edges)
    exp_prop = (exp_counts + 1e-6) / (len(expected_clean) + bins * 1e-6)
    act_prop = (act_counts + 1e-6) / (len(actual_clean) + bins * 1e-6)
    return float(np.sum((act_prop - exp_prop) * np.log(act_prop / exp_prop)))

# Compute PSI for each feature
per_feature_psi = {}
for col in all_features:
    psi = compute_psi(train[col].values, test[col].values)
    per_feature_psi[col] = round(psi, 6)

high_psi_features = [c for c, p in per_feature_psi.items() if p > 0.25]
print(f"  Features with PSI > 0.25: {len(high_psi_features)}")

# PSI-filtered features (remove high-PSI features)
psi_filtered_features = [c for c in all_features if per_feature_psi.get(c, 0) <= 0.25]
print(f"  Features after PSI filter (>0.25 removed): {len(psi_filtered_features)}")

# Train with PSI-filtered features
psi_filter_results = {}
baseline_avg_oof = 0
for target in TARGETS:
    cfg = baseline_configs[target]
    model_cfg = CFGS[cfg['cfg']]
    # Use top-n feats (rank them)
    ranked = rank_features(train, all_features, target, seed=42)
    top_feats = ranked[:cfg['n_feat']]
    # Get PSI-filtered top feats
    top_feats_psi_filtered = [f for f in top_feats if f in psi_filtered_features]
    if len(top_feats_psi_filtered) < 3:
        top_feats_psi_filtered = top_feats[:max(len(top_feats) // 2, 3)]

    baseline_score = train_and_oof(train, all_features, top_feats, target, model_cfg, seed=42)
    psi_filtered_score = train_and_oof(train, all_features, top_feats_psi_filtered, target, model_cfg, seed=42)

    delta = psi_filtered_score - baseline_score
    psi_filter_results[target] = {
        'baseline_oof': round(baseline_score, 6),
        'psi_filtered_oof': round(psi_filtered_score, 6),
        'delta': round(delta, 6)
    }
    baseline_avg_oof += baseline_score
    print(f"  {target}: baseline={baseline_score:.6f}, psi_filtered={psi_filtered_score:.6f}, delta={delta:+.6f}")

baseline_avg_oof /= len(TARGETS)
psi_filter_avg = np.mean([r['psi_filtered_oof'] for r in psi_filter_results.values()])
psi_filter_delta = psi_filter_avg - baseline_avg_oof
print(f"  Avg baseline OOF: {baseline_avg_oof:.6f}")
print(f"  Avg PSI-filtered OOF: {psi_filter_avg:.6f}")
print(f"  Delta: {psi_filter_delta:+.6f}")

# ─── 2. Quantile Normalization (Train-Test Matching) ────────────────────────
print("\n" + "=" * 60)
print("2. Quantile Normalization")
print("=" * 60)

def quantile_normalize(train_df, test_df, feature_cols):
    """Map train distribution to test distribution quantile-by-quantile."""
    result = train_df.copy()
    for col in feature_cols:
        train_vals = train_df[col].values.astype(np.float64)
        test_vals = test_df[col].values.astype(np.float64)

        # Handle NaN
        train_nan_mask = np.isnan(train_vals)
        test_nan_mask = np.isnan(test_vals)
        train_clean = train_vals[~train_nan_mask]
        test_clean = test_vals[~test_nan_mask]

        if len(train_clean) < 10 or len(test_clean) < 10:
            continue

        # Use quantile mapping
        # For each quantile q in [0,1], map train value at q to test value at q
        q = np.linspace(0.001, 0.999, 500)
        train_q_vals = np.quantile(train_clean, q)
        test_q_vals = np.quantile(test_clean, q)

        # Create interpolation function
        try:
            interp_fn = np.interp
            mapped = interp_fn(np.linspace(0, 1, len(train_clean)),
                              train_q_vals, test_q_vals)
            result.loc[~train_nan_mask, col] = mapped
        except:
            pass
    return result

# Apply quantile normalization to training data only
quantile_normalized_train = quantile_normalize(train, test, all_features)

quantile_norm_results = {}
for target in TARGETS:
    cfg = baseline_configs[target]
    model_cfg = CFGS[cfg['cfg']]
    ranked = rank_features(train, all_features, target, seed=42)
    top_feats = ranked[:cfg['n_feat']]

    # Use quantile-normalized features for training
    quant_score = train_and_oof(quantile_normalized_train, all_features, top_feats, target, model_cfg, seed=42)
    delta = quant_score - baseline_avg_oof  # compare to overall baseline
    quantile_norm_results[target] = {
        'oof': round(quant_score, 6),
        'delta': round(delta, 6)
    }
    print(f"  {target}: oof={quant_score:.6f}, delta={delta:+.6f}")

quantile_norm_avg = np.mean([r['oof'] for r in quantile_norm_results.values()])
quantile_norm_delta = quantile_norm_avg - baseline_avg_oof
print(f"  Avg quantile norm OOF: {quantile_norm_avg:.6f}")
print(f"  Delta from baseline: {quantile_norm_delta:+.6f}")

# ─── 3. Rank Stabilization ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. Rank Stabilization")
print("=" * 60)

def rank_stabilize(df, feature_cols, group_col='subject_id'):
    """Replace feature values with within-subject ranks."""
    result = df.copy()
    for col in feature_cols:
        ranks = np.zeros(len(df))
        for _, grp_idx in df.groupby(group_col).groups.items():
            vals = df[col].iloc[grp_idx].values.astype(np.float64)
            ranks[grp_idx] = rankdata(vals, method='average')
        result[col] = ranks
    return result

rank_stabilized_train = rank_stabilize(train, all_features)

rank_stab_results = {}
for target in TARGETS:
    cfg = baseline_configs[target]
    model_cfg = CFGS[cfg['cfg']]
    ranked = rank_features(rank_stabilized_train, all_features, target, seed=42)
    top_feats = ranked[:cfg['n_feat']]

    rank_score = train_and_oof(rank_stabilized_train, all_features, top_feats, target, model_cfg, seed=42)
    delta = rank_score - baseline_avg_oof
    rank_stab_results[target] = {
        'oof': round(rank_score, 6),
        'delta': round(delta, 6)
    }
    print(f"  {target}: oof={rank_score:.6f}, delta={delta:+.6f}")

rank_stab_avg = np.mean([r['oof'] for r in rank_stab_results.values()])
rank_stab_delta = rank_stab_avg - baseline_avg_oof
print(f"  Avg rank stabilization OOF: {rank_stab_avg:.6f}")
print(f"  Delta from baseline: {rank_stab_delta:+.6f}")

# ─── 4. Calibration Transfer ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. Calibration Transfer")
print("=" * 60)

# We do calibration transfer differently: we train a model, compute calibration curves
# on train OOF vs true, then apply inverse mapping calibrated on the test distribution
cal_transfer_results = {}
for target in TARGETS:
    cfg = baseline_configs[target]
    model_cfg = CFGS[cfg['cfg']]
    gkf = GroupKFold(n_splits=5)
    groups = train['subject_id'].values
    ranked = rank_features(train, all_features, target, seed=42)
    top_feats = ranked[:cfg['n_feat']]

    oof_raw = np.zeros(len(train))
    fcols_sanitized = [sanitize_col(c) for c in top_feats]

    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(train, train[target].values, groups)):
        X_tr, y_tr = train[top_feats].iloc[tr_idx], train[target].iloc[tr_idx]
        X_va = train[top_feats].iloc[va_idx]
        y_true = train[target].iloc[va_idx].values

        y_tr_vals = y_tr.values.astype(np.float64)
        X_tr_filled = X_tr.fillna(0).values.astype(np.float64)
        X_va_filled = X_va.fillna(0).values.astype(np.float64)

        spw = max(((y_tr_vals == 0).sum()) / max((y_tr_vals == 1).sum(), 1), 0.1)
        params = cfg_to_params(model_cfg, 42, spw)

        dtrain = lgb.Dataset(X_tr_filled, label=y_tr_vals, feature_name=fcols_sanitized)
        dval = lgb.Dataset(X_va_filled, label=y_true.astype(np.float64),
                           feature_name=fcols_sanitized, reference=dtrain)
        model = lgb.train(params, dtrain, num_boost_round=model_cfg['n_estimators'],
                          valid_sets=[dval], callbacks=[lgb.log_evaluation(0)])
        oof_raw[va_idx] = model.predict(X_va_filled)

    # Standard calibration
    cal_preds, cal_ok = isotonic_calibrate(oof_raw, train[target].values)

    # Calibration transfer: train a mapping from train OOF distribution to test predictions
    # We approximate this by: compute mean_match with train rate, then apply isotonic on OOF
    # The "transfer" is: apply isotonic calibration to raw OOF, then match mean to train rate
    train_rate = float(train[target].mean())
    mm_pred = mean_match(oof_raw, train_rate)
    # Then apply isotonic calibration on the mean-matched predictions
    try:
        iso2 = IsotonicRegression(out_of_bounds='clip', y_min=0.0001, y_max=0.9999)
        iso2.fit(oof_raw, train[target].values)
        cal_pred = iso2.predict(mm_pred)
        cal_pred = mean_match(cal_pred, train_rate)
    except:
        cal_pred = mm_pred

    # Evaluate with log_loss
    ll_raw = log_loss(train[target].values.astype(int), np.clip(oof_raw, 0.0001, 0.9999))
    ll_cal = log_loss(train[target].values.astype(int), np.clip(cal_pred, 0.0001, 0.9999))

    cal_transfer_results[target] = {
        'raw_oof': round(ll_raw, 6),
        'cal_transfer_oof': round(ll_cal, 6),
        'delta': round(ll_cal - baseline_avg_oof, 6)
    }
    print(f"  {target}: raw={ll_raw:.6f}, cal_transfer={ll_cal:.6f}, delta={ll_cal - baseline_avg_oof:+.6f}")

cal_transfer_avg = np.mean([r['cal_transfer_oof'] for r in cal_transfer_results.values()])
cal_transfer_delta = cal_transfer_avg - baseline_avg_oof
print(f"  Avg calibration transfer OOF: {cal_transfer_avg:.6f}")
print(f"  Delta from baseline: {cal_transfer_delta:+.6f}")

# ─── 5. Shift-Sensitive Feature Analysis ─────────────────────────────────────
print("\n" + "=" * 60)
print("5. Shift-Sensitive Feature Analysis")
print("=" * 60)

# For each feature, measure its importance on train-split vs test-like splits
# Features important in train but with high PSI are "shift-sensitive"
shift_sensitive_features = []
feat_analysis = {}

for col in all_features:
    # Get feature importance on a quick model
    for target in TARGETS[:3]:  # Quick check on first 3 targets
        cfg = baseline_configs[target]
        model_cfg = CFGS[cfg['cfg']]
        gkf = GroupKFold(n_splits=5)
        groups = train['subject_id'].values

        importances = np.zeros(1)
        for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(train, train[target].values, groups)):
            X_tr = train[[col]].iloc[tr_idx].fillna(0).values.astype(np.float64)
            y_tr = train[target].iloc[tr_idx].values.astype(np.float64)
            X_va = train[[col]].iloc[va_idx].fillna(0).values.astype(np.float64)

            spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
            params = cfg_to_params(model_cfg, 42, spw)

            dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=[sanitize_col(col)])
            dval = lgb.Dataset(X_va, label=train[target].iloc[va_idx].values.astype(np.float64),
                               feature_name=[sanitize_col(col)], reference=dtrain)
            m = lgb.train(params, dtrain, num_boost_round=model_cfg['n_estimators'],
                          valid_sets=[dval], callbacks=[lgb.log_evaluation(0)])
            imp = m.feature_importance(importance_type='gain')
            importances += imp
            del m, dtrain, dval

        avg_imp = float(np.mean(importances))
        psi_val = per_feature_psi.get(col, 0)

        if avg_imp > 0 and psi_val > 0.25:
            shift_sensitive_features.append(col)

        feat_analysis[col] = {
            'train_importance': round(float(avg_imp), 2),
            'psi': psi_val,
            'shift_sensitive': avg_imp > 0 and psi_val > 0.25
        }

shift_sensitive_features = list(set(shift_sensitive_features))
print(f"  Shift-sensitive features (important + PSI>0.25): {len(shift_sensitive_features)}")

# Train without shift-sensitive features
non_shift_sens_features = [c for c in all_features if c not in shift_sensitive_features]
shift_sensitive_results = {}
for target in TARGETS:
    cfg = baseline_configs[target]
    model_cfg = CFGS[cfg['cfg']]
    ranked = rank_features(train, all_features, target, seed=42)
    top_feats = ranked[:cfg['n_feat']]
    top_feats_no_shift = [f for f in top_feats if f not in shift_sensitive_features]
    if len(top_feats_no_shift) < 3:
        top_feats_no_shift = top_feats[:max(len(top_feats) // 2, 3)]

    no_shift_score = train_and_oof(train, all_features, top_feats_no_shift, target, model_cfg, seed=42)
    delta = no_shift_score - baseline_avg_oof
    shift_sensitive_results[target] = {
        'oof': round(no_shift_score, 6),
        'delta': round(delta, 6)
    }
    print(f"  {target}: no_shift_sens={no_shift_score:.6f}, delta={delta:+.6f}")

shift_sensitive_avg = np.mean([r['oof'] for r in shift_sensitive_results.values()])
shift_sensitive_delta = shift_sensitive_avg - baseline_avg_oof
print(f"  Avg shift-sensitive exclusion OOF: {shift_sensitive_avg:.6f}")
print(f"  Delta from baseline: {shift_sensitive_delta:+.6f}")

# ─── 6. Best Combination ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. Testing Best Combinations")
print("=" * 60)

# Strategy: use quantile-normalized features, but with PSI-filtered subset
best_combined_avg = baseline_avg_oof
best_components = []

# Try: quantile normalization + PSI filtering
for target in TARGETS:
    cfg = baseline_configs[target]
    model_cfg = CFGS[cfg['cfg']]
    ranked = rank_features(quantile_normalized_train, all_features, target, seed=42)
    top_feats = ranked[:cfg['n_feat']]
    # PSI-filtered version
    top_feats_comb = [f for f in top_feats if per_feature_psi.get(f, 0) <= 0.25]
    if len(top_feats_comb) < 3:
        top_feats_comb = top_feats[:max(len(top_feats) // 2, 3)]

    comb_score = train_and_oof(quantile_normalized_train, all_features, top_feats_comb, target, model_cfg, seed=42)
    delta = comb_score - baseline_avg_oof
    print(f"  {target}: quantile+psi_filter={comb_score:.6f}, delta={delta:+.6f}")

    if delta < 0:  # Lower log_loss is better
        best_components.append(f"{target}+q+psi")

# Also try rank stabilization + PSI filtering
for target in TARGETS:
    cfg = baseline_configs[target]
    model_cfg = CFGS[cfg['cfg']]
    ranked = rank_features(rank_stabilized_train, all_features, target, seed=42)
    top_feats = ranked[:cfg['n_feat']]
    top_feats_comb = [f for f in top_feats if per_feature_psi.get(f, 0) <= 0.25]
    if len(top_feats_comb) < 3:
        top_feats_comb = top_feats[:max(len(top_feats) // 2, 3)]

    comb_score = train_and_oof(rank_stabilized_train, all_features, top_feats_comb, target, model_cfg, seed=42)
    delta = comb_score - baseline_avg_oof
    print(f"  {target}: rank+psi_filter={comb_score:.6f}, delta={delta:+.6f}")

    if delta < 0:
        best_components.append(f"{target}+rank+psi")

# Determine best combination by overall average
all_combination_scores = {}
for target in TARGETS:
    cfg = baseline_configs[target]
    model_cfg = CFGS[cfg['cfg']]
    ranked = rank_features(quantile_normalized_train, all_features, target, seed=42)
    top_feats = ranked[:cfg['n_feat']]
    top_feats_comb = [f for f in top_feats if per_feature_psi.get(f, 0) <= 0.25]
    if len(top_feats_comb) < 3:
        top_feats_comb = top_feats[:max(len(top_feats) // 2, 3)]
    score = train_and_oof(quantile_normalized_train, all_features, top_feats_comb, target, model_cfg, seed=42)
    all_combination_scores[target] = score

best_combined_avg = np.mean(list(all_combination_scores.values()))
best_combined_delta = best_combined_avg - baseline_avg_oof

# Also try: PSI filter + rank stabilization + quantile (all three)
print("\n  Testing all-three combination...")
for target in TARGETS:
    cfg = baseline_configs[target]
    model_cfg = CFGS[cfg['cfg']]
    ranked = rank_features(quantile_normalized_train, all_features, target, seed=42)
    top_feats = ranked[:cfg['n_feat']]
    top_feats_all = [f for f in top_feats if per_feature_psi.get(f, 0) <= 0.25]
    if len(top_feats_all) < 3:
        top_feats_all = top_feats[:max(len(top_feats) // 2, 3)]
    score = train_and_oof(quantile_normalized_train, all_features, top_feats_all, target, model_cfg, seed=42)
    delta = score - baseline_avg_oof
    print(f"  {target}: all_three={score:.6f}, delta={delta:+.6f}")

    if delta < best_combined_delta:
        best_combined_avg = score
        best_combined_delta = delta
        best_components = ['quantile_normalization', 'psi_filtering', 'rank_stabilization', 'calibration_transfer']

print(f"\n  Best combined OOF: {best_combined_avg:.6f}")
print(f"  Best combined delta: {best_combined_delta:+.6f}")
print(f"  Best components: {best_components}")

# ─── Build result ─────────────────────────────────────────────────────────────
# Find best combination by component
best_oof = best_combined_avg
best_delta = best_combined_delta
best_comp_list = ['quantile_normalization', 'psi_filtering', 'calibration_transfer']

# Check each component's delta; include if delta < 0
comp_deltas = {
    'psi_filtering': psi_filter_delta,
    'quantile_normalization': quantile_norm_delta,
    'rank_stabilization': rank_stab_delta,
    'calibration_transfer': cal_transfer_delta,
    'shift_sensitive_exclusion': shift_sensitive_delta,
}

for comp, d in comp_deltas.items():
    print(f"  Component {comp}: delta={d:+.6f}")

# The best combination is the one with lowest average OOF
all_results = {
    'psi_filtering': psi_filter_avg,
    'quantile_normalization': quantile_norm_avg,
    'rank_stabilization': rank_stab_avg,
    'calibration_transfer': cal_transfer_avg,
    'shift_sensitive_exclusion': shift_sensitive_avg,
    'combined_q_psi': best_combined_avg,
}

best_method = min(all_results, key=all_results.get)
best_method_oof = all_results[best_method]
best_method_delta = best_method_oof - baseline_avg_oof

# Determine which components are in the best method
if 'combined' in best_method:
    components = ['quantile_normalization', 'psi_filtering']
elif best_method == 'psi_filtering':
    components = ['psi_filtering']
elif best_method == 'quantile_normalization':
    components = ['quantile_normalization']
elif best_method == 'rank_stabilization':
    components = ['rank_stabilization']
elif best_method == 'calibration_transfer':
    components = ['calibration_transfer']
elif best_method == 'shift_sensitive_exclusion':
    components = ['shift_sensitive_exclusion']
else:
    components = []

result = {
    "version": "v260_distribution_correction",
    "baseline_avg_oof": round(baseline_avg_oof, 6),
    "psi_filtering": {
        "high_psi_features_removed": len(high_psi_features),
        "oof": round(psi_filter_avg, 6),
        "delta": round(psi_filter_delta, 6),
        "per_target": psi_filter_results
    },
    "quantile_normalization": {
        "oof": round(quantile_norm_avg, 6),
        "delta": round(quantile_norm_delta, 6),
        "per_target": quantile_norm_results
    },
    "rank_stabilization": {
        "oof": round(rank_stab_avg, 6),
        "delta": round(rank_stab_delta, 6),
        "per_target": rank_stab_results
    },
    "calibration_transfer": {
        "oof": round(cal_transfer_avg, 6),
        "delta": round(cal_transfer_delta, 6),
        "per_target": cal_transfer_results
    },
    "shift_sensitive_features": {
        "n_features": len(shift_sensitive_features),
        "exclusion_oof": round(shift_sensitive_avg, 6),
        "exclusion_delta": round(shift_sensitive_delta, 6),
        "features": shift_sensitive_features[:20]
    },
    "best_combination": {
        "oof": round(best_method_oof, 6),
        "delta": round(best_method_delta, 6),
        "method": best_method,
        "components": components
    },
    "feature_psi_summary": {
        "overall_psi": round(np.mean(list(per_feature_psi.values())), 6),
        "max_psi": round(max(per_feature_psi.values()), 6),
        "min_psi": round(min(per_feature_psi.values()), 6),
        "features_above_025": len(high_psi_features),
        "per_feature_psi": per_feature_psi
    },
    "notes": (
        f"Baseline avg OOF: {baseline_avg_oof:.6f}. "
        f"Best correction: {best_method} with OOF={best_method_oof:.6f} "
        f"(delta={best_method_delta:+.6f}). "
        f"PSI>0.25 features removed: {len(high_psi_features)}. "
        f"Shift-sensitive features: {len(shift_sensitive_features)}. "
        f"Seed=42, GroupKFold(5)."
    )
}

output_path = ROOT / 'experiments/v260_distribution_correction_result.json'
with open(output_path, 'w') as f:
    json.dump(result, f, indent=2)

print(f"\n{'=' * 60}")
print(f"Results written to {output_path}")
print(f"{'=' * 60}")
