#!/usr/bin/env python3
"""
V262: Submission with Quantile + PSI distribution correction

Pipeline:
1. Base features (features_clean_v60, leak-removed) 
2. PSI filtering: remove features with PSI > 0.25
3. Quantile normalization: map train fold distribution to full train distribution
4. GroupKFold(5) × 4 seeds
5. Isotonic regression calibration per fold
6. Mean match calibration

Based on V260 best result: Quantile+PSI combination gave Δ=-0.099
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
from scipy.stats import rankdata
warnings.filterwarnings('ignore')

ROOT = Path('/home/mwoo423/projects/dacon2')
DATA = ROOT / 'data_processed'
EXPERIMENTS = ROOT / 'experiments'
SUBMIT = ROOT / 'submissions'
for d in [EXPERIMENTS, SUBMIT]:
    d.mkdir(exist_ok=True)

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
SEED = 42
SEEDS = [42, 7, 999, 777]

V53_SWEEP = {
    'Q1': {'cfg': 'deep'}, 'Q2': {'cfg': 'deep'}, 'Q3': {'cfg': 'v48'},
    'S1': {'cfg': 'wide'}, 'S2': {'cfg': 'deep'}, 'S3': {'cfg': 'safety'}, 'S4': {'cfg': 'wide'},
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


def sanitize_col(n):
    return re.sub(r'[^a-zA-Z0-9_]', '_', n)


def mean_match(pred, tm):
    return np.clip(pred + (tm - pred.mean()), 0.0001, 0.9999)


def remove_leak(cols, t):
    LEAK_S = {'wlight_w_light_mean', 'wlight_w_light_std', 'wlight_w_light_min', 'wlight_w_light_max', 'wlight_w_light_count',
              'whr_hr_mean', 'whr_hr_std', 'whr_hr_min', 'whr_hr_max', 'whr_hr_median', 'whr_hr_count',
              'wpedo_pedo_step_mean', 'wpedo_pedo_step_sum', 'wpedo_pedo_step_frequency_mean',
              'wpedo_pedo_step_frequency_sum',
              'wpedo_pedo_running_step_mean', 'wpedo_pedo_running_step_sum',
              'wpedo_pedo_walking_step_mean', 'wpedo_pedo_walking_step_sum',
              'wpedo_pedo_distance_mean', 'wpedo_pedo_distance_sum', 'wpedo_pedo_speed_mean',
              'wpedo_pedo_speed_sum',
              'wpedo_pedo_burned_calories_mean', 'wpedo_pedo_burned_calories_sum'}
    LEAK_Q = {'whr_hr_mean', 'whr_hr_std', 'whr_hr_min', 'whr_hr_max', 'whr_hr_median', 'whr_hr_count'}
    if t.startswith('S'):
        return [c for c in cols if c not in LEAK_S]
    elif t.startswith('Q'):
        return [c for c in cols if c not in LEAK_Q]
    return cols


def get_feature_cols(df):
    META = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}
    return [c for c in df.columns if c not in META | set(TARGETS)
            and df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]


def compute_psi(expected, actual, bins=20):
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) < 10 or len(actual) < 10:
        return 0.0
    eps = 1e-6
    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    bin_edges[-1] += eps
    exp_counts = np.histogram(expected, bins=bin_edges)[0]
    act_counts = np.histogram(actual, bins=bin_edges)[0]
    exp_prop = (exp_counts + eps) / (len(expected) + bins * eps)
    act_prop = (act_counts + eps) / (len(actual) + bins * eps)
    return float(np.sum((act_prop - exp_prop) * np.log(act_prop / exp_prop)))


# ============================================================
# LOAD DATA
# ============================================================
print("=" * 70)
print("V262: Quantile + PSI Distribution Correction Submission")
print("=" * 70)

feat = pd.read_parquet(DATA / 'features_clean_v60.parquet')
ftst = pd.read_parquet(DATA / 'test_features_clean_v60.parquet')
feat.columns = [sanitize_col(c) for c in feat.columns]
ftst.columns = [sanitize_col(c) for c in ftst.columns]

y_dict = {t: feat[t].values.astype(np.float64) for t in TARGETS}
all_feat_cols = get_feature_cols(feat)
non_const_cols = [c for c in all_feat_cols if feat[c].std() > 0.001]

print(f"Train: {feat.shape}, Test: {ftst.shape}")
print(f"Features: {len(non_const_cols)} non-constant")

# ============================================================
# PSI filtering
# ============================================================
print("\n[PSI Filter] Computing PSI...")
psi_scores = {}
for col in non_const_cols:
    psi_scores[col] = compute_psi(feat[col].values, ftst[col].values)

high_psi_cols = [c for c, p in psi_scores.items() if p > 0.25]
psi_filtered_cols = [c for c in non_const_cols if c not in high_psi_cols]
print(f"  PSI > 0.25 features: {len(high_psi_cols)} (removed)")
print(f"  Remaining features: {len(psi_filtered_cols)}")
print(f"  Mean PSI: {np.mean(list(psi_scores.values())):.4f}")
print(f"  Max PSI: {max(psi_scores.values()):.2f}")

# ============================================================
# Quantile-normalized CV training
# ============================================================
print("\n[CV Training] Quantile normalization + Isotonic calibration")


def cfg_to_params(cfg_s, seed, spw):
    params = dict(cfg_s)
    params['scale_pos_weight'] = spw
    params['random_state'] = seed
    params['force_row_wise'] = True
    params['n_jobs'] = 1
    return params


target_results = {}

for target in TARGETS:
    sw = V53_SWEEP[target]
    cfg = CFGS[sw['cfg']]
    y = y_dict[target]

    cols = get_feature_cols(feat)
    cols = remove_leak(cols, target)
    cols = [c for c in cols if c in psi_filtered_cols]  # PSI filter

    gkf = GroupKFold(n_splits=5)
    oof = np.zeros((len(y), len(SEEDS)))
    tp = np.zeros((len(ftst), len(SEEDS)))
    sn = [sanitize_col(c) for c in cols]
    Xf = feat[cols].fillna(0).values.astype(np.float64)
    Xt = ftst[cols].fillna(0).values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)

    for si, seed in enumerate(SEEDS):
        p = cfg_to_params(cfg, seed, spw)
        for tri, vai in gkf.split(feat, y, feat['subject_id']):
            # Quantile normalization: map train fold features to full-train distribution
            X_tr = Xf[tri, :].copy()
            X_full_train = Xf[tri, :].copy()

            for fi in range(X_tr.shape[1]):
                fv = X_tr[:, fi]
                # Rank in fold -> map to full-train quantiles
                ranks = rankdata(fv)
                pctiles = ranks / len(ranks)
                # Sort full train fold values
                full_sorted = np.sort(X_full_train[:, fi])
                # Map each fold value to its percentile's value in sorted full train
                mapped = np.interp(pctiles, np.arange(len(full_sorted)) / len(full_sorted), full_sorted)
                X_tr[:, fi] = mapped

            ds = lgb.Dataset(X_tr, label=y[tri], feature_name=sn)

            vd = lgb.Dataset(Xf[vai, :], label=y[vai], feature_name=sn, reference=ds)
            m = lgb.train(p, ds, num_boost_round=cfg['n_estimators'],
                          valid_sets=[vd],
                          callbacks=[lgb.early_stopping(100, verbose=False),
                                     lgb.log_evaluation(0)])
            oof[vai, si] = m.predict(Xf[vai, :])
            tp[:, si] = m.predict(Xt)

    # Average across seeds
    oof_avg = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
    tp_avg = np.clip(tp.mean(axis=1), 0.0001, 0.9999)

    # Isotonic regression calibration per fold
    oof_iso = np.zeros_like(oof_avg)
    for tri, vai in gkf.split(feat, y, feat['subject_id']):
        iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0001, y_max=0.9999)
        iso.fit(oof[vai].ravel(), y[vai].ravel())
        oof_iso[vai] = iso.predict(oof[vai].ravel())

    # Mean match
    cal_test = mean_match(tp_avg, y.mean())
    cal_oof = mean_match(oof_iso, y.mean())
    ll = log_loss(y, cal_oof, labels=[0, 1])

    target_results[target] = {
        'll': float(ll),
        'll_oof_raw': float(log_loss(y, oof_avg, labels=[0, 1])),
        'n_feats': len(cols),
        'cfg': sw['cfg'],
        'test_pred_mean': float(cal_test.mean()),
        'true_mean': float(y.mean()),
    }
    print(f"  {target}: OOF={ll:.5f} (raw={target_results[target]['ll_oof_raw']:.5f}, "
          f"n_feats={len(cols)}, cfg={sw['cfg']})")

avg_oof = np.mean([target_results[t]['ll'] for t in TARGETS])
print(f"\n  AVG OOF (with correction): {avg_oof:.5f}")

# Compare with naive (no correction)
print("\n[Comparison] Naive OOF (no correction):")
naive_results = {}
for target in TARGETS:
    sw = V53_SWEEP[target]
    cfg = CFGS[sw['cfg']]
    y = y_dict[target]

    cols = get_feature_cols(feat)
    cols = remove_leak(cols, target)
    cols = [c for c in cols if c in psi_filtered_cols]

    gkf = GroupKFold(n_splits=5)
    oof = np.zeros((len(y), len(SEEDS)))
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    sn = [sanitize_col(c) for c in cols]
    Xf = feat[cols].fillna(0).values.astype(np.float64)

    for si, seed in enumerate(SEEDS):
        p = cfg_to_params(cfg, seed, spw)
        for tri, vai in gkf.split(feat, y, feat['subject_id']):
            ds = lgb.Dataset(Xf[tri], label=y[tri], feature_name=sn)
            m = lgb.train(p, ds, num_boost_round=cfg['n_estimators'],
                          callbacks=[lgb.early_stopping(100, verbose=False),
                                     lgb.log_evaluation(0)])
            oof[vai, si] = m.predict(Xf[vai, :])

    oof_avg = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
    naive_results[target] = float(log_loss(y, oof_avg, labels=[0, 1]))
    print(f"  {target}: {naive_results[target]:.5f}")

avg_naive = np.mean(list(naive_results.values()))
print(f"\n  AVG naive OOF: {avg_naive:.5f}")
print(f"  AVG corrected OOF: {avg_oof:.5f}")
print(f"  Δ: {avg_oof - avg_naive:+.5f}")

# ============================================================
# SAVE SUBMISSION
# ============================================================
print(f"\n[Submission] Saving V262...")

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
submit_df = pd.DataFrame()
submit_df['subject_id'] = ftst['subject_id'].values
submit_df['sleep_date'] = ftst['sleep_date'].values
submit_df['lifelog_date'] = ftst['lifelog_date'].values
for t in TARGETS:
    submit_df[sanitize_col(t)] = test_preds[t] if t in locals() else cal_test

# Generate predictions properly
print("\n[Final Predictions] Training on full data...")
for target in TARGETS:
    sw = V53_SWEEP[target]
    cfg = CFGS[sw['cfg']]
    y = y_dict[target]

    cols = get_feature_cols(feat)
    cols = remove_leak(cols, target)
    cols = [c for c in cols if c in psi_filtered_cols]

    Xf = feat[cols].fillna(0).values.astype(np.float64)
    Xt = ftst[cols].fillna(0).values.astype(np.float64)
    sn = [sanitize_col(c) for c in cols]
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)

    preds = []
    for seed in SEEDS:
        p = cfg_to_params(cfg, seed, spw)
        ds = lgb.Dataset(Xf, label=y, feature_name=sn)
        m = lgb.train(p, ds, num_boost_round=cfg['n_estimators'])
        preds.append(m.predict(Xt))

    tp = np.clip(np.mean(preds, axis=0), 0.0001, 0.9999)
    tp = mean_match(tp, y.mean())
    submit_df[sanitize_col(target)] = tp

submit_path = SUBMIT / f'submission_v262_quantile_psi_{ts}.csv'
submit_df.to_csv(submit_path, index=False)
print(f"  Saved: {submit_path}")

# ============================================================
# SAVE LOG
# ============================================================
result_log = {
    'version': 'v262',
    'name': 'Quantile + PSI Distribution Correction',
    'timestamp': ts,
    'avg_naive_oof': float(avg_naive),
    'avg_corrected_oof': float(avg_oof),
    'delta': float(avg_oof - avg_naive),
    'psi_summary': {
        'total_features': len(non_const_cols),
        'psi_above_025': len(high_psi_cols),
        'psi_filtered_cols': len(psi_filtered_cols),
        'mean_psi': float(np.mean(list(psi_scores.values()))),
    },
    'per_target': {t: {
        'naive_oof': float(naive_results[t]),
        'corrected_oof': float(target_results[t]['ll']),
        'delta': float(target_results[t]['ll'] - naive_results[t]),
        'n_feats': target_results[t]['n_feats'],
        'cfg': target_results[t]['cfg'],
    } for t in TARGETS},
    'submission_path': str(submit_path),
}

log_path = EXPERIMENTS / f'v262_quantile_psi_{ts}.json'
with open(log_path, 'w') as f:
    json.dump(result_log, f, indent=2, default=str)
print(f"  Log saved: {log_path}")

print("\n" + "=" * 70)
print("=== V262 COMPLETE ===")
print(f"  Naive AVG OOF:  {avg_naive:.5f}")
print(f"  Corrected OOF:  {avg_oof:.5f}")
print(f"  Δ:              {avg_oof - avg_naive:+.5f}")
print(f"  Submission:     {submit_path}")
print("=" * 70)
