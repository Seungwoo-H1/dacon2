#!/usr/bin/env python3
"""DACon2 v259: PSI Drift & Adversarial Analysis"""

import json
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
from scipy.stats import gaussian_kde
from pathlib import Path

# Sanitize feature names for LightGBM (no commas, parentheses etc.)
def sanitize_name(name):
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)

# ─── Load data ────────────────────────────────────────────────────────────────
ROOT = Path('/home/mwoo423/projects/dacon2')
train = pd.read_parquet(ROOT / 'data_processed/features_clean_v60.parquet')
test = pd.read_parquet(ROOT / 'data_processed/test_features_clean_v60.parquet')

# Exclude non-numeric columns (e.g. categorical strings)
non_numeric = {"mAmbience_ambience_max_cat"}
feature_cols = [
    c for c in train.columns
    if c not in ["subject_id", "lifelog_date", "sleep_date",
                  "Q1", "Q2", "Q3", "S1", "S2", "S3", "S4", "date"]
    and c not in non_numeric
]

# Create sanitized names for LightGBM
feature_names_sanitized = [sanitize_name(c) for c in feature_cols]

# ─── PSI helper ───────────────────────────────────────────────────────────────
def compute_psi(expected, actual, bins=20):
    """PSI between expected (train) and actual (test) distributions."""
    # Filter out NaN independently
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    
    if len(expected) < 10 or len(actual) < 10:
        return float("nan")
    
    # Create bins from expected range
    eps = np.finfo(float).eps
    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())
    
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    # Avoid zero-width bins
    bin_edges[-1] += eps  # extend upper edge slightly
    
    expected_counts = np.histogram(expected, bins=bin_edges)[0]
    actual_counts = np.histogram(actual, bins=bin_edges)[0]
    
    # Proportions with epsilon
    expected_prop = (expected_counts + 1e-6) / (len(expected) + bins * 1e-6)
    actual_prop = (actual_counts + 1e-6) / (len(actual) + bins * 1e-6)
    
    psi = np.sum((actual_prop - expected_prop) * np.log(actual_prop / expected_prop))
    return float(psi)


# ─── 1. PSI: Feature-level ───────────────────────────────────────────────────
print("Computing PSI per feature...")
per_feature_psi = {}
features_above_01 = []
features_above_025 = []

for col in feature_cols:
    psi = compute_psi(train[col].values, test[col].values)
    per_feature_psi[col] = round(psi, 6)
    if psi > 0.1:
        features_above_01.append(col)
    if psi > 0.25:
        features_above_025.append(col)

# Global PSI (average of per-feature PSIs)
overall_psi = round(np.mean(list(per_feature_psi.values())), 6)
print(f"  Overall PSI: {overall_psi}")
print(f"  Features PSI > 0.1: {len(features_above_01)}")
print(f"  Features PSI > 0.25: {len(features_above_025)}")
if features_above_025:
    print(f"  Top 10 high-drift features:")
    top_drift = sorted(per_feature_psi.items(), key=lambda x: x[1], reverse=True)[:10]
    for name, val in top_drift:
        print(f"    {name}: {val:.4f}")


# ─── 2. PSI: Per subject group ────────────────────────────────────────────────
print("\nComputing PSI per subject group...")
subjects = sorted(train["subject_id"].unique())
all_train_features = train[feature_cols].values
per_subject_psi = {}

for subj in subjects:
    test_mask = test["subject_id"] == subj
    test_subset = test[test_mask]
    if test_subset.shape[0] == 0:
        per_subject_psi[subj] = float("nan")
        continue
    test_features = test_subset[feature_cols].values
    
    # PSI: each subject's test vs overall train
    psi_vals = []
    for col_idx, col in enumerate(feature_cols):
        psi = compute_psi(train[col].values, test_features[:, col_idx])
        psi_vals.append(psi)
    subj_psi = np.mean(psi_vals)
    per_subject_psi[subj] = round(subj_psi, 6)

print(f"  Per-subject PSI: {per_subject_psi}")


# ─── 3. Adversarial Validation ───────────────────────────────────────────────
print("\nAdversarial validation (LGBM, 5-fold GroupKFold)...")

X_train = train[feature_cols].astype(float).values
X_test = test[feature_cols].astype(float).values
y = np.concatenate([np.zeros(len(train)), np.ones(len(test))])
X = np.vstack([X_train, X_test])

# Group is subject_id
groups_train = train["subject_id"].values
groups_test = test["subject_id"].values
groups = np.concatenate([groups_train, groups_test])

gkf = GroupKFold(n_splits=5)

lgbm_params = dict(
    n_estimators=100,
    num_leaves=15,
    max_depth=3,
    learning_rate=0.05,
    seed=42,
    verbose=2,
    n_jobs=1,
)

fold_aucs = []
train_importance = np.zeros(len(feature_cols))
fold_discriminative_features = set()

for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    
    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names_sanitized)
    dval = lgb.Dataset(X_va, label=y_va, feature_name=feature_names_sanitized, reference=dtrain)
    
    model = lgb.train(
        lgbm_params,
        dtrain,
        num_boost_round=100,
        valid_sets=[dval],
        callbacks=[lgb.log_evaluation(0)],
    )
    
    preds = model.predict(X_va)
    
    # Compute AUC
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_va, preds)
    fold_aucs.append(round(auc, 6))
    print(f"  Fold {fold_idx}: AUC = {auc:.4f}")
    
    # Feature importance
    imp = model.feature_importance(importance_type="gain")
    train_importance += imp
    
    # Top 20 features for this fold
    top_indices = np.argsort(imp)[::-1][:20]
    for i in top_indices:
        fold_discriminative_features.add(feature_cols[i])  # original name

mean_auc = np.mean(fold_aucs)
print(f"\n  Mean CV AUC: {mean_auc:.4f} (±{np.std(fold_aucs):.4f})")

# Top discriminative features (average importance)
avg_importance = train_importance / 5
top_discriminative = [feature_cols[i] for i in np.argsort(avg_importance)[::-1][:15]]
print(f"  Top discriminative features: {top_discriminative[:5]}")


# ─── 4. Fold-level Drift ─────────────────────────────────────────────────────
print("\nFold-level drift analysis...")
per_fold_psi = {}

# GroupKFold on train only (groups = subject_id)
gkf_train = GroupKFold(n_splits=5)
fold_drift_psis = []

for fold_idx, (tr_idx, va_idx) in enumerate(gkf_train.split(X_train, train["Q1"].values, train["subject_id"].values)):
    fold_psis = []
    for col in feature_cols:
        psi = compute_psi(train.iloc[tr_idx][col].values, train.iloc[va_idx][col].values)
        if not np.isnan(psi):
            fold_psis.append(psi)
    avg_psi = np.mean(fold_psis)
    per_fold_psi[f"fold_{fold_idx}"] = round(avg_psi, 6)
    fold_drift_psis.append(avg_psi)
    print(f"  Fold {fold_idx} drift (train split): PSI = {avg_psi:.4f}")

fold_variance = round(np.var(fold_drift_psis), 6)
print(f"  Fold drift variance: {fold_variance}")


# ─── 5. Target Distribution Analysis ─────────────────────────────────────────
print("\nTarget distribution analysis...")
target_cols = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]

# 5a. Per-target by subject
per_target_by_subject = {}
for tcol in target_cols:
    per_target_by_subject[tcol] = {}
    for subj in subjects:
        subj_data = train[train["subject_id"] == subj]
        per_target_by_subject[tcol][subj] = round(float(subj_data[tcol].mean()), 6)

print("  Per-target by subject (sample Q1):")
for subj in subjects:
    print(f"    {subj}: {per_target_by_subject['Q1'][subj]:.4f}")

# 5b. Temporal drift (split by median date)
train_copy = train.copy()
median_date = train_copy["lifelog_date"].median()
train_copy["period"] = train_copy["lifelog_date"].apply(
    lambda x: "first_half" if x <= median_date else "second_half"
)

temporal_drift = {}
for tcol in target_cols:
    first_mean = train_copy[train_copy["period"] == "first_half"][tcol].mean()
    second_mean = train_copy[train_copy["period"] == "second_half"][tcol].mean()
    temporal_drift[tcol] = {
        "first_half_mean": round(float(first_mean), 6),
        "second_half_mean": round(float(second_mean), 6),
        "drift": round(float(second_mean - first_mean), 6),
    }
    print(f"  {tcol}: first_half={first_mean:.4f}, second_half={second_mean:.4f}, drift={second_mean-first_mean:.4f}")

# ─── Build result ─────────────────────────────────────────────────────────────
result = {
    "version": "v259",
    "psi": {
        "overall": overall_psi,
        "features_above_01": features_above_01,
        "features_above_025": features_above_025,
        "per_feature_psi": per_feature_psi,
        "per_subject_psi": per_subject_psi,
    },
    "adversarial": {
        "cv_auc": round(mean_auc, 6),
        "top_discriminative_features": top_discriminative,
        "fold_aucs": fold_aucs,
    },
    "fold_drift": {
        "per_fold_psi": per_fold_psi,
        "fold_variance": fold_variance,
    },
    "target_distribution": {
        "per_target_by_subject": per_target_by_subject,
        "temporal_drift": temporal_drift,
    },
}

output_path = ROOT / 'experiments/v259_psi_adversarial_result.json'
with open(output_path, "w") as f:
    json.dump(result, f, indent=2)

print(f"\n✅ Results written to {output_path}")
