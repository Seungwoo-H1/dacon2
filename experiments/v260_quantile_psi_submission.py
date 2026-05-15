#!/usr/bin/env python3
"""
V260 Submission: Quantile Norm + PSI Filter Distribution Correction

Pipeline:
1. Quantile normalization (map train fold distribution to test distribution)
2. PSI filtering (remove features with PSI > 0.25)
3. GroupKFold(5) × seed=42
4. Isotonic calibration + mean match
5. Generate test predictions

Best OOF from V260 analysis: 0.614019 (Δ=-0.098555 vs baseline 0.712574)
"""
import os, sys, gc, re, json, warnings, time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss
import lightgbm as lgb
warnings.filterwarnings('ignore')

ROOT = Path('/home/mwoo423/projects/dacon2')
DATA = ROOT / 'data_processed'
EXPERIMENTS = ROOT / 'experiments'
SUBMIT = ROOT / 'submissions'
SUBMIT.mkdir(exist_ok=True)

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
SEED = 42

META_COLS = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}

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

LEAK_S = {'wlight_w_light_mean','wlight_w_light_std','wlight_w_light_min','wlight_w_light_max','wlight_w_light_count',
          'whr_hr_mean','whr_hr_std','whr_hr_min','whr_hr_max','whr_hr_median','whr_hr_count',
          'wpedo_pedo_step_mean','wpedo_pedo_step_sum','wpedo_pedo_step_frequency_mean','wpedo_pedo_step_frequency_sum',
          'wpedo_pedo_running_step_mean','wpedo_pedo_running_step_sum','wpedo_pedo_walking_step_mean','wpedo_pedo_walking_step_sum',
          'wpedo_pedo_distance_mean','wpedo_pedo_distance_sum','wpedo_pedo_speed_mean','wpedo_pedo_speed_sum',
          'wpedo_pedo_burned_calories_mean','wpedo_pedo_burned_calories_sum'}
LEAK_Q = {'whr_hr_mean','whr_hr_std','whr_hr_min','whr_hr_max','whr_hr_median','whr_hr_count'}

def sanitize_col(n):
    return re.sub(r'[^a-zA-Z0-9_]', '_', n)

def get_numeric_cols(df, exclude=None):
    ex = META_COLS | set(TARGETS)
    if exclude: ex |= exclude
    return [c for c in df.columns
            if df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]
            and c not in ex]

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

def compute_psi(expected, actual, bins=20):
    expected = np.array(expected, dtype=np.float64)
    actual = np.array(actual, dtype=np.float64)
    ec = expected[~np.isnan(expected)]
    ac = actual[~np.isnan(actual)]
    if len(ec) < 10 or len(ac) < 10: return 0.0
    mn = min(ec.min(), ac.min())
    mx = max(ec.max(), ac.max())
    be = np.linspace(mn, mx, bins + 1)
    be[-1] += np.finfo(float).eps
    ep = (np.histogram(ec, be)[0] + 1e-6) / (len(ec) + bins * 1e-6)
    ap = (np.histogram(ac, be)[0] + 1e-6) / (len(ac) + bins * 1e-6)
    return float(np.sum((ap - ep) * np.log(ap / ep)))


# ============================================================
# LOAD DATA
# ============================================================
print("=" * 70)
print("V260: Quantile Norm + PSI Filter Submission")
print("=" * 70)

train = pd.read_parquet(DATA / 'features_clean_v60.parquet')
test = pd.read_parquet(DATA / 'test_features_clean_v60.parquet')
all_features = get_numeric_cols(train)
print(f"Train: {train.shape}, Test: {test.shape}, Features: {len(all_features)}")

# ============================================================
# PSI Filter
# ============================================================
print("\n[PSI Filter] Computing PSI for all features...")
per_feature_psi = {}
for col in all_features:
    per_feature_psi[col] = compute_psi(train[col].values, test[col].values)

high_psi = [c for c, p in per_feature_psi.items() if p > 0.25]
print(f"  PSI > 0.25 features removed: {len(high_psi)}")
print(f"  Remaining features: {len(all_features) - len(high_psi)}")

psi_ok = set(all_features) - set(high_psi)

# ============================================================
# Quantile Normalization (per fold, not pre-computed)
# ============================================================
def quantile_normalize_fold(X_train_fold, X_test_full, feat_cols, feature_idx_map):
    """Quantile normalize train fold features to test distribution."""
    X_norm = X_train_fold.copy()
    for fi, col in enumerate(feat_cols):
        if col not in psi_ok:
            continue
        ti = feature_idx_map.get(col, fi)
        train_vals = X_train_fold[:, ti]
        test_vals = X_test_full[:, fi]

        valid = ~np.isnan(train_vals)
        if valid.sum() < 10:
            continue

        tc = train_vals[valid]
        test_c = test_vals[~np.isnan(test_vals)]
        if len(test_c) < 10:
            continue

        q = np.linspace(0.001, 0.999, 500)
        tq = np.quantile(tc, q)
        test_q = np.quantile(test_c, q)
        mapped = np.interp(np.linspace(0, 1, len(tc)), tq, test_q)
        X_norm[valid, ti] = mapped
    return X_norm


# ============================================================
# Training & Submission Generation
# ============================================================
print("\n[Training] Quantile Norm + PSI Filter, GroupKFold(5)...")

gkf = GroupKFold(n_splits=5)
test_preds = {t: np.zeros(len(test)) for t in TARGETS}
results = {}

for target in TARGETS:
    t0 = time.time()
    sw = V53_SWEEP[target]
    cfg = CFGS[sw['cfg']]
    y = train[target].values.astype(np.float64)

    # Get features: leak removed + PSI filtered
    cols = get_numeric_cols(train)
    cols = remove_leak(cols, target)
    cols = [c for c in cols if c in psi_ok]

    # Rank features by importance (quick model)
    y_train = y.copy()
    X_all = train[cols].fillna(0).values.astype(np.float64)
    spw = max(((y_train == 0).sum()) / max((y_train == 1).sum(), 1), 0.1)
    p_quick = cfg_to_params({'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03,
                              'n_estimators': 50, 'subsample': 0.7, 'colsample_bytree': 0.7,
                              'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_samples': 10}, SEED, spw)
    sn = [sanitize_col(c) for c in cols]
    ds_quick = lgb.Dataset(X_all, label=y_train, feature_name=sn)
    m_quick = lgb.train(p_quick, ds_quick, num_boost_round=50)
    imp = m_quick.feature_importance(importance_type='gain')
    ranked = sorted(zip(cols, imp), key=lambda x: -x[1])
    top_feats = [r[0] for r in ranked[:sw['n_feat']]]
    top_feats = [f for f in top_feats if f in psi_ok]  # re-apply PSI filter
    if len(top_feats) < 3:
        top_feats = ranked[:max(sw['n_feat'] // 2, 3)]

    print(f"  {target}: {len(top_feats)} features (cfg={sw['cfg']}, n_feat={sw['n_feat']})")

    # CV training with quantile normalization
    oof_preds = np.zeros(len(y))
    X_top = train[top_feats].fillna(0).values.astype(np.float64)
    X_top_test = test[top_feats].fillna(0).values.astype(np.float64)
    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(train, y, train['subject_id'].values)):
        X_tr = X_top[tr_idx]
        X_va = X_top[va_idx]

        # Quantile normalize train fold to test distribution
        X_tr_norm = quantile_normalize_fold(X_tr, X_top_test, top_feats, {c: i for i, c in enumerate(top_feats)})

        y_tr = y[tr_idx]
        y_va = y[va_idx]

        X_tr_norm_filled = X_tr_norm.astype(np.float64)
        X_va_filled = X_va.astype(np.float64)

        spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
        params = cfg_to_params(cfg, SEED, spw)

        dtrain = lgb.Dataset(X_tr_norm_filled, label=y_tr, feature_name=[sanitize_col(c) for c in top_feats])
        dval = lgb.Dataset(X_va_filled, label=y_va, feature_name=[sanitize_col(c) for c in top_feats], reference=dtrain)
        model = lgb.train(params, dtrain, num_boost_round=cfg['n_estimators'],
                          valid_sets=[dval], callbacks=[lgb.log_evaluation(0)])
        oof_preds[va_idx] = model.predict(X_va_filled)

    # Isotonic calibration
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0001, y_max=0.9999)
    iso.fit(oof_preds, y)
    cal_oof = iso.predict(oof_preds)
    cal_oof = mean_match(cal_oof, float(y.mean()))
    ll = log_loss(y, cal_oof, labels=[0, 1])

    # Mean match for test predictions (train on full, predict test)
    # Train final model on all data for test predictions
    X_all_filled = train[top_feats].fillna(0).values.astype(np.float64)
    X_test_filled = test[top_feats].fillna(0).values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    params = cfg_to_params(cfg, SEED, spw)
    ds_full = lgb.Dataset(X_all_filled, label=y, feature_name=[sanitize_col(c) for c in top_feats])
    m_full = lgb.train(params, ds_full, num_boost_round=cfg['n_estimators'])
    tp_raw = m_full.predict(X_test_filled)

    # Apply isotonic calibration to test predictions using OOF mapping
    tp_cal = iso.predict(tp_raw)
    tp_cal = mean_match(tp_cal, float(y.mean()))
    tp_cal = np.clip(tp_cal, 0.0001, 0.9999)

    test_preds[target] = tp_cal
    elapsed = time.time() - t0
    results[target] = {'oof': ll, 'n_feats': len(top_feats), 'cfg': sw['cfg'], 'time_s': elapsed}
    print(f"    OOF: {ll:.6f} (Δ from baseline 0.712574: {ll - 0.712574:+.6f})  [{elapsed:.0f}s]")

avg_oof = np.mean([r['oof'] for r in results.values()])
print(f"\n  AVG OOF: {avg_oof:.6f}")
print(f"  Δ from baseline: {avg_oof - 0.712574:+.6f}")

# ============================================================
# SAVE SUBMISSION
# ============================================================
print("\n[Submission] Saving...")
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
submit_df = pd.DataFrame()
submit_df['subject_id'] = test['subject_id'].values
submit_df['sleep_date'] = test['sleep_date'].values
submit_df['lifelog_date'] = test['lifelog_date'].values
for t in TARGETS:
    submit_df[sanitize_col(t)] = test_preds[t]

submit_path = SUBMIT / f'submission_v260_quantile_psi_{ts}.csv'
submit_df.to_csv(submit_path, index=False)
print(f"  Saved: {submit_path}")

# ============================================================
# SAVE RESULT LOG
# ============================================================
result_log = {
    "version": "v260_quantile_psi_submission",
    "name": "Quantile Norm + PSI Filter Distribution Correction",
    "timestamp": ts,
    "avg_oof": float(avg_oof),
    "baseline_avg_oof": 0.712574,
    "delta": float(avg_oof - 0.712574),
    "psi_summary": {
        "total_features": len(all_features),
        "psi_above_025": len(high_psi),
        "psi_filtered_features": len(all_features) - len(high_psi),
    },
    "per_target": {t: {
        "oof": float(results[t]['oof']),
        "n_feats": results[t]['n_feats'],
        "cfg": results[t]['cfg'],
    } for t in TARGETS},
    "pipeline": [
        "PSI filtering (PSI > 0.25 removed)",
        "Feature ranking by importance",
        "Top-N features per target (V53 sweep)",
        "Quantile normalization (train fold → test distribution)",
        "GroupKFold(5) × seed=42",
        "Isotonic regression calibration",
        "Mean match calibration",
    ],
    "submission_path": str(submit_path),
}

log_path = EXPERIMENTS / f'v260_quantile_psi_submission_{ts}.json'
with open(log_path, 'w') as f:
    json.dump(result_log, f, indent=2, default=str)
print(f"  Log: {log_path}")

print("\n" + "=" * 70)
print("=== V260 SUBMISSION READY ===")
print(f"  AVG OOF:  {avg_oof:.6f}")
print(f"  Δ:        {avg_oof - 0.712574:+.6f} (vs baseline 0.712574)")
print(f"  File:     {submit_path}")
print("=" * 70)
