"""
V259: Calibration Pipeline — Temperature Scaling, Quantile Correction, Rank Stabilization

Comprehensive calibration experiments on the V127 baseline pipeline:
1. Temperature Scaling (per-target)
2. Quantile Normalization
3. Rank Stabilization
4. Local Calibration (Isotonic Regression)
5. Combined: Temperature Scaling + Rank Stabilization

Uses GroupKFold(5), seed=42, features_clean_v60.parquet.
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
from scipy.optimize import minimize_scalar
from scipy.stats import rankdata
warnings.filterwarnings('ignore')

ROOT = Path('/home/mwoo423/projects/dacon2')
DATA = ROOT / 'data_processed'
EXPERIMENTS = ROOT / 'experiments'
for d in [EXPERIMENTS]:
    d.mkdir(exist_ok=True)

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
META_COLS = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}
SEED = 42

# V127 per-target configs
V53_SWEEP = {
    'Q1': {'cfg': 'deep'},
    'Q2': {'cfg': 'deep'},
    'Q3': {'cfg': 'v48'},
    'S1': {'cfg': 'wide'},
    'S2': {'cfg': 'deep'},
    'S3': {'cfg': 'safety'},
    'S4': {'cfg': 'wide'},
}

CFGS = {
    'wide':   {'num_leaves': 30, 'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 300,
               'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0,
               'min_child_samples': 5},
    'deep':   {'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.02, 'n_estimators': 1000,
               'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 0.5, 'reg_lambda': 2.0,
               'min_child_samples': 15},
    'v48':    {'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 500,
               'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 1.0, 'reg_lambda': 3.0,
               'min_child_samples': 10},
    'safety': {'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.02, 'n_estimators': 1000,
               'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 3.0, 'reg_lambda': 10.0,
               'min_child_samples': 20},
}

LEAK_S = {'wlight_w_light_mean', 'wlight_w_light_std', 'wlight_w_light_min', 'wlight_w_light_max', 'wlight_w_light_count',
          'whr_hr_mean', 'whr_hr_std', 'whr_hr_min', 'whr_hr_max', 'whr_hr_median', 'whr_hr_count',
          'wpedo_pedo_step_mean', 'wpedo_pedo_step_sum', 'wpedo_pedo_step_frequency_mean', 'wpedo_pedo_step_frequency_sum',
          'wpedo_pedo_running_step_mean', 'wpedo_pedo_running_step_sum', 'wpedo_pedo_walking_step_mean', 'wpedo_pedo_walking_step_sum',
          'wpedo_pedo_distance_mean', 'wpedo_pedo_distance_sum', 'wpedo_pedo_speed_mean', 'wpedo_pedo_speed_sum',
          'wpedo_pedo_burned_calories_mean', 'wpedo_pedo_burned_calories_sum'}
LEAK_Q = {'whr_hr_mean', 'whr_hr_std', 'whr_hr_min', 'whr_hr_max', 'whr_hr_median', 'whr_hr_count'}

def sanitize_col(n):
    return re.sub(r'[^a-zA-Z0-9_]', '_', n)

def mean_match(pred, tm):
    return np.clip(pred + (tm - np.clip(pred, 0.0001, 0.9999).mean()), 0.0001, 0.9999)

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
    """Train with GroupKFold 5-fold CV. Returns (oof, test_preds)."""
    gkf = GroupKFold(n_splits=5)
    n_seeds = len(seeds)
    oof = np.zeros((len(y), n_seeds))
    tp = np.zeros((len(ftst), n_seeds)) if ftst is not None else None
    sn = [sanitize_col(c) for c in cols]
    Xf = feat[cols].fillna(0).values.astype(np.float64)
    Xt = ftst[cols].fillna(0).values.astype(np.float64) if ftst is not None else None
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)

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
            del ds, m
            gc.collect()

    if tp is not None:
        tp = np.clip(tp, 0.0001, 0.9999)
    return oof, tp


# ============================================================
# Load data
# ============================================================
t_start = time.time()
print("=" * 70)
print("V259: Calibration Pipeline — Temperature Scaling & Friends")
print("=" * 70)

feat = pd.read_parquet(DATA / 'features_clean_v60.parquet')
ftst = pd.read_parquet(DATA / 'test_features_clean_v60.parquet')

feat.columns = [sanitize_col(c) for c in feat.columns]
ftst.columns = [sanitize_col(c) for c in ftst.columns]

y_dict = {t: feat[t].values.astype(np.float64) for t in TARGETS}
train_rates = {t: float(feat[t].mean()) for t in TARGETS}
base_cols = get_feature_cols(feat)

print(f"Train: {feat.shape}, Test: {ftst.shape}")
print(f"Train rates: { {t: f'{train_rates[t]:.3f}' for t in TARGETS} }")

# ============================================================
# STEP 0: Train base models (same as V127) to get OOF predictions
# ============================================================
print("\n" + "=" * 70)
print("STEP 0: Training base LGBM models (V127 configs, single seed=42)")
print("=" * 70)

# Use multiple seeds for robust OOF, then average per fold
SEEDS = [42, 7, 999, 777]

all_oof = {}       # target -> (N_train, n_seeds) raw predictions
all_test = {}      # target -> (N_test, n_seeds) raw predictions
per_target_ll = {} # target -> baseline log-loss

for target in TARGETS:
    sw = V53_SWEEP[target]
    cfg = CFGS[sw['cfg']]
    y = y_dict[target]
    cols = remove_leak(base_cols, target)
    
    print(f"\n  Training {target} (cfg={sw['cfg']}, n_feats={len(cols)})...", end=' ')
    oof, test = train_cv(feat, ftst, cols, y, SEEDS, cfg)
    all_oof[target] = oof
    all_test[target] = test
    
    # Average over seeds
    oof_avg = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
    cal = mean_match(oof_avg, train_rates[target])
    ll = log_loss(y, cal, labels=[0, 1])
    per_target_ll[target] = ll
    print(f"LL={ll:.5f}")

avg_baseline = np.mean(list(per_target_ll.values()))
print(f"\n  AVG BASELINE OOF: {avg_baseline:.5f}")

# ============================================================
# HELPER: Calibrate OOF predictions for a single target
# ============================================================
def calibrate_and_score(target, method, oof_raw, method_params=None, y=None):
    """
    Apply calibration method to OOF predictions, compute log-loss.
    Returns (calibrated_oof, log_loss, extra_info_dict)
    """
    if y is None:
        y = y_dict[target]
    cfg = CFGS[V53_SWEEP[target]['cfg']]

    # Average over seeds first
    oof_avg = np.clip(oof_raw.mean(axis=1), 0.0001, 0.9999)

    if method == 'raw':
        cal = mean_match(oof_avg, train_rates[target])
        return cal, log_loss(y, cal, labels=[0, 1]), {}

    elif method == 'temperature':
        # Temperature scaling: calibrated = sigmoid(logit(raw) / T)
        raw_clipped = np.clip(oof_avg, 0.0001, 0.9999)
        logit = np.log(raw_clipped / (1 - raw_clipped + 1e-10) + 1e-10)

        # Find optimal T on validation (OOF) predictions
        # Minimize log_loss with temperature-scaled predictions
        T_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0]
        best_T = 1.0
        best_ll = float('inf')

        for T in T_values:
            scaled = 1.0 / (1.0 + np.exp(-logit / T))
            scaled = np.clip(scaled, 0.0001, 0.9999)
            ll = log_loss(y, scaled, labels=[0, 1])
            if ll < best_ll:
                best_ll = ll
                best_T = T

        # Also try fine-grained around best
        if best_T not in [0.1, 5.0]:
            for T in np.arange(max(0.05, best_T - 0.3), best_T + 0.31, 0.05):
                T = round(T, 2)
                if T <= 0: continue
                scaled = 1.0 / (1.0 + np.exp(-logit / T))
                scaled = np.clip(scaled, 0.0001, 0.9999)
                ll = log_loss(y, scaled, labels=[0, 1])
                if ll < best_ll:
                    best_ll = ll
                    best_T = T

        cal = np.clip(scaled, 0.0001, 0.9999)
        cal = mean_match(cal, train_rates[target])
        final_ll = log_loss(y, cal, labels=[0, 1])
        return cal, final_ll, {'T': best_T}

    elif method == 'isotonic':
        raw_clipped = np.clip(oof_avg, 0.0001, 0.9999)
        try:
            iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0001, y_max=0.9999)
            iso.fit(raw_clipped, y)
            cal = iso.predict(raw_clipped)
        except:
            cal = raw_clipped
        cal = mean_match(cal, train_rates[target])
        return cal, log_loss(y, cal, labels=[0, 1]), {}

    elif method == 'quantile_norm':
        # Map predictions to standard normal quantiles
        # Step 1: compute empirical CDF of predictions
        # Step 2: map to normal quantile
        # Step 3: map back using target's predicted distribution

        # Use rank-based CDF estimation
        sorted_idx = np.argsort(oof_avg)
        ranks = rankdata(oof_avg, method='average')
        n = len(oof_avg)
        cdf_vals = (ranks - 0.5) / n  # mid-rank CDF
        cdf_vals = np.clip(cdf_vals, 1e-10, 1 - 1e-10)

        # Map to standard normal quantiles
        norm_quantiles = scipy_norm_ppf(cdf_vals)

        # Now we want to map these to the target's empirical distribution
        # But for binary classification, we map to the target rate
        # Strategy: use the norm quantiles to determine which predictions are "high" vs "low"
        # and set high=high_target_rate, low=low_target_rate, interpolate
        target_rate = train_rates[target]

        # Convert norm quantiles back to probability using logistic
        # This preserves the ranking but maps to a more normal shape
        scaled = 1.0 / (1.0 + np.exp(-norm_quantiles))
        scaled = np.clip(scaled, 0.0001, 0.9999)

        # Re-rank to match original prediction order
        # Actually, quantile norm changes the distribution but preserves ranking
        # We want to reorder so that high CDF -> high prediction
        cal = np.zeros_like(scaled)
        # Map: sorted(norm_quantiles) positions -> sorted(oof_avg) positions
        # Simply: the quantile-normalized value at rank i should replace the raw value at rank i
        orig_sorted_idx = np.argsort(oof_avg)
        # Actually, quantile normalization maps: rank_i -> quantile_i
        # We want the calibrated value to be quantile_i mapped to target scale
        cal = np.zeros(n)
        cal[orig_sorted_idx] = scaled

        cal = mean_match(cal, target_rate)
        return cal, log_loss(y, cal, labels=[0, 1]), {}

    elif method == 'rank_stabilization':
        # Within each fold, rank predictions, then map to calibrated probability
        # This reduces sensitivity to absolute prediction values
        n_seeds = oof_raw.shape[1]
        gkf = GroupKFold(n_splits=5)
        folds = list(gkf.split(feat, y, feat['subject_id']))

        rank_cal = np.zeros(len(y))
        seed_mean = np.zeros(len(y))

        for fi, (tri, vai) in enumerate(folds):
            for si in range(n_seeds):
                fold_oof = oof_raw[vai, si]
                # Rank within this fold
                ranks = rankdata(fold_oof, method='average')
                n_fold = len(vai)
                pct = (ranks - 0.5) / n_fold
                pct = np.clip(pct, 0.0001, 0.9999)
                rank_cal[vai] += pct
            seed_mean[vai] += rank_cal[vai]

        rank_cal /= (n_seeds * len(folds))
        rank_cal = np.clip(rank_cal, 0.0001, 0.9999)
        rank_cal = mean_match(rank_cal, train_rates[target])
        return rank_cal, log_loss(y, rank_cal, labels=[0, 1]), {}

    elif method == 'rank_stab_iso':
        # Rank stabilization followed by isotonic regression
        # Same rank process, then apply isotonic calibration
        n_seeds = oof_raw.shape[1]
        gkf = GroupKFold(n_splits=5)
        folds = list(gkf.split(feat, y, feat['subject_id']))

        rank_cal = np.zeros(len(y))
        for fi, (tri, vai) in enumerate(folds):
            for si in range(n_seeds):
                fold_oof = oof_raw[vai, si]
                ranks = rankdata(fold_oof, method='average')
                n_fold = len(vai)
                pct = (ranks - 0.5) / n_fold
                pct = np.clip(pct, 0.0001, 0.9999)
                rank_cal[vai] += pct

        rank_cal /= (n_seeds * len(folds))
        rank_cal = np.clip(rank_cal, 0.0001, 0.9999)

        try:
            iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0001, y_max=0.9999)
            iso.fit(rank_cal, y)
            rank_cal = iso.predict(rank_cal)
        except:
            pass
        rank_cal = mean_match(rank_cal, train_rates[target])
        return rank_cal, log_loss(y, rank_cal, labels=[0, 1]), {}

    elif method == 'combined':
        # Temperature scaling + rank stabilization
        # First do rank stabilization, then temperature scale
        n_seeds = oof_raw.shape[1]
        gkf = GroupKFold(n_splits=5)
        folds = list(gkf.split(feat, y, feat['subject_id']))

        rank_cal = np.zeros(len(y))
        for fi, (tri, vai) in enumerate(folds):
            for si in range(n_seeds):
                fold_oof = oof_raw[vai, si]
                ranks = rankdata(fold_oof, method='average')
                n_fold = len(vai)
                pct = (ranks - 0.5) / n_fold
                pct = np.clip(pct, 0.0001, 0.9999)
                rank_cal[vai] += pct

        rank_cal /= (n_seeds * len(folds))
        rank_cal = np.clip(rank_cal, 0.0001, 0.9999)

        # Now apply temperature scaling
        logit = np.log(rank_cal / (1 - rank_cal + 1e-10) + 1e-10)
        T_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0]
        best_T = 1.0
        best_ll = float('inf')

        for T in T_values:
            scaled = 1.0 / (1.0 + np.exp(-logit / T))
            scaled = np.clip(scaled, 0.0001, 0.9999)
            ll = log_loss(y, scaled, labels=[0, 1])
            if ll < best_ll:
                best_ll = ll
                best_T = T

        cal = np.clip(scaled, 0.0001, 0.9999)
        cal = mean_match(cal, train_rates[target])
        return cal, log_loss(y, cal, labels=[0, 1]), {'T': best_T}

    elif method == 'temp_per_seed':
        # Temperature scaling applied per seed, then averaged
        n_seeds = oof_raw.shape[1]
        per_seed_cal = np.zeros((len(y), n_seeds))
        best_Ts = []

        for si in range(n_seeds):
            raw_clipped = np.clip(oof_raw[:, si], 0.0001, 0.9999)
            logit = np.log(raw_clipped / (1 - raw_clipped + 1e-10) + 1e-10)
            T_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0]
            best_T = 1.0
            best_ll = float('inf')
            for T in T_values:
                scaled = 1.0 / (1.0 + np.exp(-logit / T))
                scaled = np.clip(scaled, 0.0001, 0.9999)
                ll = log_loss(y, scaled, labels=[0, 1])
                if ll < best_ll:
                    best_ll = ll
                    best_T = T
            per_seed_cal[:, si] = np.clip(scaled, 0.0001, 0.9999)
            best_Ts.append(best_T)

        avg_cal = np.clip(per_seed_cal.mean(axis=1), 0.0001, 0.9999)
        avg_cal = mean_match(avg_cal, train_rates[target])
        return avg_cal, log_loss(y, avg_cal, labels=[0, 1]), {'T_per_seed': best_Ts}

    else:
        raise ValueError(f"Unknown method: {method}")


# Need scipy for norm.ppf
from scipy.stats import norm as scipy_norm

def scipy_norm_ppf(p):
    return scipy_norm.ppf(p)


# ============================================================
# EXPERIMENT 1: Temperature Scaling (per-target)
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 1: Temperature Scaling (per-target)")
print("=" * 70)

temp_scaling_result = {}
temp_result = {}  # per-target LL for temperature
temp_avg_ll = 0

for target in TARGETS:
    cal_oof, ll, info = calibrate_and_score(target, 'temperature', all_oof[target])
    temp_scaling_result[target] = info
    temp_result[target] = ll
    temp_avg_ll += ll
    print(f"  {target}: LL={ll:.5f} (baseline: {per_target_ll[target]:.5f}) T={info.get('T', 'N/A')}")

temp_avg_ll /= len(TARGETS)
temp_delta = temp_avg_ll - avg_baseline
print(f"\n  AVG: {temp_avg_ll:.5f} (baseline: {avg_baseline:.5f}) Δ={temp_delta:+.5f}")


# ============================================================
# EXPERIMENT 2: Quantile Normalization
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 2: Quantile Normalization")
print("=" * 70)

quantile_result = {}
quantile_avg_ll = 0

for target in TARGETS:
    cal_oof, ll, info = calibrate_and_score(target, 'quantile_norm', all_oof[target])
    quantile_result[target] = ll
    quantile_avg_ll += ll
    print(f"  {target}: LL={ll:.5f} (baseline: {per_target_ll[target]:.5f}) Δ={ll - per_target_ll[target]:+.5f}")

quantile_avg_ll /= len(TARGETS)
quantile_delta = quantile_avg_ll - avg_baseline
print(f"\n  AVG: {quantile_avg_ll:.5f} Δ={quantile_delta:+.5f}")


# ============================================================
# EXPERIMENT 3: Rank Stabilization
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 3: Rank Stabilization")
print("=" * 70)

rank_result = {}
rank_avg_ll = 0

for target in TARGETS:
    cal_oof, ll, info = calibrate_and_score(target, 'rank_stabilization', all_oof[target])
    rank_result[target] = ll
    rank_avg_ll += ll
    print(f"  {target}: LL={ll:.5f} (baseline: {per_target_ll[target]:.5f}) Δ={ll - per_target_ll[target]:+.5f}")

rank_avg_ll /= len(TARGETS)
rank_delta = rank_avg_ll - avg_baseline
print(f"\n  AVG: {rank_avg_ll:.5f} Δ={rank_delta:+.5f}")


# ============================================================
# EXPERIMENT 4: Local Calibration (Isotonic Regression)
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 4: Local Calibration (Isotonic Regression)")
print("=" * 70)

iso_result = {}
iso_avg_ll = 0

for target in TARGETS:
    cal_oof, ll, info = calibrate_and_score(target, 'isotonic', all_oof[target])
    iso_result[target] = ll
    iso_avg_ll += ll
    print(f"  {target}: LL={ll:.5f} (baseline: {per_target_ll[target]:.5f}) Δ={ll - per_target_ll[target]:+.5f}")

iso_avg_ll /= len(TARGETS)
iso_delta = iso_avg_ll - avg_baseline
print(f"\n  AVG: {iso_avg_ll:.5f} Δ={iso_delta:+.5f}")


# ============================================================
# EXPERIMENT 5: Combined — Temperature Scaling + Rank Stabilization
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 5: Combined (Temperature + Rank Stabilization)")
print("=" * 70)

combined_result = {}
combined_result_ll = {}
combined_avg_ll = 0

for target in TARGETS:
    cal_oof, ll, info = calibrate_and_score(target, 'combined', all_oof[target])
    combined_result[target] = info
    combined_result_ll[target] = ll
    combined_avg_ll += ll
    print(f"  {target}: LL={ll:.5f} (baseline: {per_target_ll[target]:.5f}) T={info.get('T', 'N/A')}")

combined_avg_ll /= len(TARGETS)
combined_delta = combined_avg_ll - avg_baseline
print(f"\n  AVG: {combined_avg_ll:.5f} Δ={combined_delta:+.5f}")


# ============================================================
# EXTRA: Per-seed temperature scaling (applies temp per model before averaging)
# ============================================================
print("\n" + "=" * 70)
print("EXTRA: Temperature Scaling Per-Seed")
print("=" * 70)

temp_per_seed_result = {}
temp_per_seed_result_ll = {}
temp_per_seed_avg_ll = 0

for target in TARGETS:
    cal_oof, ll, info = calibrate_and_score(target, 'temp_per_seed', all_oof[target])
    temp_per_seed_result[target] = info
    temp_per_seed_result_ll[target] = ll
    temp_per_seed_avg_ll += ll
    print(f"  {target}: LL={ll:.5f} T_per_seed={info.get('T_per_seed', 'N/A')}")

temp_per_seed_avg_ll /= len(TARGETS)
temp_per_seed_delta = temp_per_seed_avg_ll - avg_baseline
print(f"\n  AVG: {temp_per_seed_avg_ll:.5f} Δ={temp_per_seed_delta:+.5f}")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

results = {
    'baseline': avg_baseline,
    'temperature': temp_avg_ll,
    'quantile_norm': quantile_avg_ll,
    'rank_stabilization': rank_avg_ll,
    'isotonic': iso_avg_ll,
    'combined': combined_avg_ll,
    'temp_per_seed': temp_per_seed_avg_ll,
}

print(f"\n  Method                     AVG OOF    Δ vs baseline")
print(f"  {'─' * 60}")
for method, val in results.items():
    delta = val - avg_baseline
    print(f"  {method:28s} {val:.5f}    {delta:+.5f}")

best_method = min(results, key=results.get)
best_val = results[best_method]
print(f"\n  🏆 Best: {best_method} (OOF={best_val:.5f}, Δ={best_val - avg_baseline:+.5f})")

# Per-target detail
print(f"\n  Per-target log-loss breakdown:")
print(f"  {'Target':10s} {'Baseline':>10s} {'Temp':>10s} {'Quantile':>10s} {'Rank':>10s} {'Iso':>10s} {'Combined':>10s}")
print(f"  {'─' * 70}")
for t in TARGETS:
    print(f"  {t:10s} {per_target_ll[t]:10.5f} {temp_result.get(t, 0):10.5f} "
          f"{quantile_result[t]:10.5f} {rank_result[t]:10.5f} {iso_result[t]:10.5f} "
          f"{combined_result_ll.get(t, 0):10.5f}")


# ============================================================
# SAVE RESULTS
# ============================================================
ts = datetime.now().strftime('%Y%m%d_%H%M%S')

# Build result for temp_scaling with per-target T
temp_per_target_T = {}
for t in TARGETS:
    temp_per_target_T[t] = temp_scaling_result[t].get('T', 1.0)

# Build final JSON
final_result = {
    "version": "v259_calibration",
    "baseline_avg_oof": round(float(avg_baseline), 7),
    "per_target_baseline": {t: round(per_target_ll[t], 7) for t in TARGETS},
    "temperature_scaling": {
        "per_target_T": {t: round(temp_per_target_T[t], 4) for t in TARGETS},
        "avg_oof": round(float(temp_avg_ll), 7),
        "delta": round(float(temp_delta), 7),
    },
    "quantile_norm": {
        "avg_oof": round(float(quantile_avg_ll), 7),
        "delta": round(float(quantile_delta), 7),
    },
    "rank_stabilization": {
        "avg_oof": round(float(rank_avg_ll), 7),
        "delta": round(float(rank_delta), 7),
    },
    "isotonic_regression": {
        "avg_oof": round(float(iso_avg_ll), 7),
        "delta": round(float(iso_delta), 7),
    },
    "combined": {
        "avg_oof": round(float(combined_avg_ll), 7),
        "delta": round(float(combined_delta), 7),
        "per_target_T": {t: round(combined_result[t].get('T', 1.0), 4) for t in TARGETS},
    },
    "temperature_per_seed": {
        "avg_oof": round(float(temp_per_seed_avg_ll), 7),
        "delta": round(float(temp_per_seed_delta), 7),
        "per_target_info": {t: {
            "ll": round(temp_per_seed_result_ll[t], 7),
            "T_per_seed": temp_per_seed_result[t].get('T_per_seed', []),
        } for t in TARGETS},
    },
    "per_target_details": {
        t: {
            "baseline": round(per_target_ll[t], 7),
            "temperature": round(temp_result.get(t, 0), 7),
            "quantile_norm": round(quantile_result[t], 7),
            "rank_stabilization": round(rank_result[t], 7),
            "isotonic": round(iso_result[t], 7),
            "combined": round(combined_result_ll.get(t, 0), 7),
        } for t in TARGETS
    },
    "best_method": best_method,
    "best_avg_oof": round(float(best_val), 7),
    "best_delta": round(float(best_val - avg_baseline), 7),
    "timestamp": ts,
    "notes": (
        f"Trained with GroupKFold(5), seed=42, features_clean_v60.parquet. "
        f"V127 baseline uses wide/deep/v48/safety per-target configs. "
        f"Temperature scaling searched T in [0.1..5.0] with fine-grained refinement. "
        f"Rank stabilization uses within-fold rank mapping averaged across folds and seeds. "
        f"Isotonic regression with out_of_bounds=clip, y_min/max=[0.0001, 0.9999]. "
        f"Combined applies rank stabilization first, then temperature scaling on rank-probabilities."
    ),
}

result_path = EXPERIMENTS / 'v259_calibration_result.json'
with open(result_path, 'w') as fout:
    json.dump(final_result, fout, indent=2)

print(f"\n  Result saved: {result_path}")
print(f"\n{'=' * 70}")
print(f"V259 COMPLETE ✓ (total time: {time.time() - t_start:.0f}s)")
print(f"{'=' * 70}")
