#!/usr/bin/env python3
"""DACon2 v259 EDA — Advanced Feature Discovery (fast version)"""
import sys, os, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
warnings.filterwarnings("ignore")

DATA_DIR = Path("/home/mwoo423/projects/dacon2/data_processed")
OUT_FILE = Path("/home/mwoo423/projects/dacon2/experiments/v259_eda_features_result.json")
TRAIN_PATH = DATA_DIR / "features_clean_v60.parquet"
TEST_PATH = DATA_DIR / "test_features_clean_v60.parquet"
TARGETS = ["Q1", "Q2", "Q3"]
ID_COLS = ["subject_id", "lifelog_date", "sleep_date", "date"]
S_COLS = ["S1", "S2", "S3", "S4"]
GROUP_COL = "subject_id"
N_SEEDS = 4
N_FOLDS = 5

lg = lambda x: print(x, flush=True)
lg("=" * 60)
lg("DACon2 v259 EDA — Advanced Feature Discovery")
lg("=" * 60)

# ── Load ──
lg("\n[1/7] Loading data...")
df_train = pd.read_parquet(TRAIN_PATH)
df_test = pd.read_parquet(TEST_PATH)
numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in numeric_cols if c not in ID_COLS + TARGETS + S_COLS]
lg(f"  Train: {df_train.shape}, Features: {len(feature_cols)}, Targets: {TARGETS}")

# ── 2. Baseline LGBM ──
lg("\n[2/7] Baseline LGBM OOF + Feature Importance...")
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

lgbm_params = dict(
    n_estimators=100, learning_rate=0.05, max_depth=4, num_leaves=15,
    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.5, reg_lambda=0.5, verbose=-1, n_jobs=4,
)

gkf = GroupKFold(n_splits=N_FOLDS)
baseline_oof = []
# Per target: list of (n_features,) arrays, one per (seed, fold) pair
feature_imp_stable = {t: [] for t in TARGETS}

for target in TARGETS:
    lg(f"\n  Target {target}:")
    y = df_train[target].values
    
    for seed in range(N_SEEDS):
        oof_proba = np.zeros(len(y))
        imp_accum = np.zeros(len(feature_cols))
        
        for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(df_train[feature_cols], y, df_train[GROUP_COL].values)):
            X_tr, X_va = df_train.iloc[tr_idx][feature_cols].values, df_train.iloc[va_idx][feature_cols].values
            y_tr, y_va = y[tr_idx], y[va_idx]
            
            model = LGBMClassifier(**lgbm_params, random_state=seed * 100 + fold_idx)
            model.fit(X_tr, y_tr)
            oof_proba[va_idx] = model.predict_proba(X_va)[:, 1]
            imp_accum += model.feature_importances_
        
        try:
            auc = roc_auc_score(y, oof_proba)
            lg(f"    seed={seed} OOF: {auc:.6f}")
            baseline_oof.append(auc)
        except Exception as e:
            lg(f"    seed={seed} OOF: N/A ({e})")
            baseline_oof.append(0.0)
        
        feature_imp_stable[target].append(imp_accum)

results = {
    "version": "v259_eda",
    "interactions_generated": 0,
    "top_interactions_by_target": {},
    "nonlinear_transforms_applied": 0,
    "stable_features": [],
    "top_30_features_per_target": {},
    "baseline_avg_oof": round(float(np.mean(baseline_oof)), 6),
    "new_feature_avg_oof": 0.0,
    "delta": 0.0,
}
lg(f"\n  Baseline avg OOF: {results['baseline_avg_oof']:.6f}")

# Stable features analysis
all_mean_imp = {f: 0.0 for f in feature_cols}
for ti, target in enumerate(TARGETS):
    imp_arr = np.stack(feature_imp_stable[target])  # (4, 275)
    fi_arr = imp_arr.mean(axis=0)
    for fi, f in enumerate(feature_cols):
        all_mean_imp[f] += fi_arr[fi]

avg_all_imp = {f: v / len(TARGETS) for f, v in all_mean_imp.items()}
stable_threshold = np.median(list(avg_all_imp.values()))
lg(f"  Stability threshold (median): {stable_threshold:.2f}")

stable_features = []
for fname in feature_cols:
    mean_val = avg_all_imp[fname]
    if mean_val <= stable_threshold:
        continue
    fi = feature_cols.index(fname)
    all_stable = True
    for target in TARGETS:
        imp_arr = np.stack(feature_imp_stable[target])
        vals = imp_arr[:, fi]
        if vals.mean() > 0:
            cv = vals.std() / vals.mean()
            if cv > 0.5:
                all_stable = False
                break
    if all_stable:
        stable_features.append(fname)

results["stable_features"] = stable_features
lg(f"  Stable features: {len(stable_features)} / {len(feature_cols)}")

# Top 30 per target from baseline
for target in TARGETS:
    imp_arr = np.stack(feature_imp_stable[target])
    mean_imp = imp_arr.mean(axis=0)
    top30 = np.argsort(mean_imp)[::-1][:30]
    results["top_30_features_per_target"][target] = [feature_cols[i] for i in top30]
    lg(f"    {target} top 5: {[feature_cols[i] for i in top30[:5]]}")

# ── 3. Nonlinear transforms ──
lg("\n[3/7] Nonlinear transforms...")
train_features = df_train[feature_cols]
skewed = []
for col in feature_cols:
    sk = train_features[col].skew()
    if not np.isnan(sk) and abs(sk) > 1.0:
        skewed.append((col, sk))
lg(f"  Skewed features (|skew|>1.0): {len(skewed)}")

nonlinear_new = {}
nonlinear_count = 0

for feat_name, sk in sorted(skewed, key=lambda x: abs(x[1]), reverse=True)[:10]:
    pos = train_features[feat_name].values.copy()
    if pos.min() <= 0:
        pos = pos - pos.min() + 1e-8
    
    nonlinear_new[f"log1p_{feat_name}"] = np.log1p(pos)
    nonlinear_count += 1
    nonlinear_new[f"sqrt_{feat_name}"] = np.sqrt(np.maximum(pos, 0))
    nonlinear_count += 1
    nonlinear_new[f"rank_{feat_name}"] = scipy_stats.rankdata(pos) / len(pos)
    nonlinear_count += 1

# Box-Cox on few
from sklearn.preprocessing import PowerTransformer
for feat_name in feature_cols[:10]:
    series = train_features[feat_name]
    if series.min() > 0:
        try:
            bc = PowerTransformer(method='box-cox', standardize=False)
            nonlinear_new[f"bc_{feat_name}"] = bc.fit_transform(series.values.reshape(-1,1)).flatten()
            nonlinear_count += 1
        except: pass

results["nonlinear_transforms_applied"] = nonlinear_count
lg(f"  Nonlinear transforms: {nonlinear_count}")

# ── 4. Feature interactions ──
lg("\n[4/7] Feature interactions (top 10 features)...")
top10 = sorted(avg_all_imp.items(), key=lambda x: x[1], reverse=True)[:10]
top10_names = [f[0] for f in top10]
lg(f"  Top 10: {top10_names}")

interactions = {}
interaction_count = 0

for i in range(len(top10_names)):
    for j in range(i+1, len(top10_names)):
        a, b = top10_names[i], top10_names[j]
        va, vb = train_features[a].values, train_features[b].values
        
        denom = np.maximum(np.abs(vb), 1e-8)
        interactions[f"r_{a}_by_{b}"] = va / denom
        interaction_count += 1
        
        denom2 = np.maximum(np.abs(va), 1e-8)
        interactions[f"r_{b}_by_{a}"] = vb / denom2
        interaction_count += 1
        
        interactions[f"d_{a}_minus_{b}"] = va - vb
        interaction_count += 1
        
        interactions[f"s_{a}_plus_{b}"] = va + vb
        interaction_count += 1
        
        mx = np.maximum(np.maximum(np.abs(va), np.abs(vb)), 1e-8)
        interactions[f"nd_{a}_{b}"] = np.abs(va - vb) / mx
        interaction_count += 1

results["interactions_generated"] = interaction_count
lg(f"  Interaction features: {interaction_count}")

# ── 5. Target-conditioned delta features ──
lg("\n[5/7] Target-conditioned delta features...")
deltas = {}
delta_count = 0

for target in TARGETS:
    y = df_train[target].values
    for col in feature_cols:
        vals = df_train[col].values
        m1 = np.mean(vals[y == 1]) if np.sum(y == 1) > 0 else 0
        s1 = np.std(vals[y == 1]) if np.sum(y == 1) > 1 else 0
        
        deltas[f"delta_{target}_m1_{col}"] = vals - m1
        delta_count += 1
        
        if s1 > 1e-8:
            deltas[f"z_{target}_{col}"] = (vals - m1) / s1
            delta_count += 1

lg(f"  Delta features: {delta_count}")

# ── 6. Combine & test improvement ──
lg("\n[6/7] Training with combined features...")

# Select best new features by correlation
new_candidates = {**interactions, **nonlinear_new, **deltas}
candidate_scores = {}
for fname, fvals in new_candidates.items():
    scores = []
    for target in TARGETS:
        y = df_train[target].values
        corr = np.corrcoef(y, fvals)[0, 1]
        if not np.isnan(corr):
            scores.append(abs(corr))
    candidate_scores[fname] = np.mean(scores) if scores else 0

# Top 30 new features to keep things fast
top_n_new = 30
top_new = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:top_n_new]
top_new_names = [f[0] for f in top_new]
lg(f"  Selected top {len(top_new_names)} new features (out of {len(new_candidates)} candidates)")
lg(f"  Top scores: {[f'{n}:{s:.4f}' for n,s in top_new[:5]]}")

# Build combined matrix
X_base = df_train[feature_cols].values
X_new = np.column_stack([
    np.array(new_candidates[c]) for c in top_new_names
])
X_combined = np.hstack([X_base, X_new])
all_names = feature_cols + top_new_names
lg(f"  Combined feature count: {len(all_names)}")

# OOF with combined — just 2 seeds for speed
combined_oof = []
for target in TARGETS:
    lg(f"\n  Combined training for {target}...")
    y = df_train[target].values
    oof_proba = np.zeros(len(y))
    
    for seed in range(N_SEEDS):
        for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X_combined, y, df_train[GROUP_COL].values)):
            X_tr, X_va = X_combined[tr_idx], X_combined[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]
            
            model = LGBMClassifier(**lgbm_params, random_state=seed * 100 + fold_idx)
            model.fit(X_tr, y_tr)
            oof_proba[va_idx] = model.predict_proba(X_va)[:, 1]
    
    auc = roc_auc_score(y, oof_proba)
    lg(f"    Combined OOF ({target}): {auc:.6f}")
    combined_oof.append(auc)

results["new_feature_avg_oof"] = round(float(np.mean(combined_oof)), 6)
results["delta"] = round(results["new_feature_avg_oof"] - results["baseline_avg_oof"], 6)
lg(f"\n  Baseline avg: {results['baseline_avg_oof']}")
lg(f"  New avg: {results['new_feature_avg_oof']}")
lg(f"  Delta: {results['delta']}")

# Top 30 from combined model
for target in TARGETS:
    y = df_train[target].values
    imp_acc = np.zeros(len(all_names))
    
    for seed in range(N_SEEDS):
        for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X_combined, y, df_train[GROUP_COL].values)):
            X_tr = X_combined[tr_idx]
            y_tr = y[tr_idx]
            
            model = LGBMClassifier(**lgbm_params, random_state=seed * 100 + fold_idx)
            model.fit(X_tr, y_tr)
            imp_acc += model.feature_importances_
    
    imp_acc /= (N_SEEDS * N_FOLDS)
    top30 = np.argsort(imp_acc)[::-1][:30]
    results["top_30_features_per_target"][target] = [all_names[i] for i in top30]
    
    # Top interactions (new features only)
    new_in_top = [n for n in results["top_30_features_per_target"][target] if n not in feature_cols][:10]
    results["top_interactions_by_target"][target] = new_in_top

lg(f"\n  Top interactions by target:")
for t in TARGETS:
    lg(f"    {t}: {results['top_interactions_by_target'][t]}")

# ── 7. SHAP-style analysis ──
lg("\n[7/7] SHAP-style feature importance analysis...")
for target in TARGETS:
    y = df_train[target].values
    seed, fold_idx = 0, 0
    tr_idx, va_idx = list(gkf.split(X_combined, y, df_train[GROUP_COL].values))[0]
    model = LGBMClassifier(**lgbm_params, random_state=42)
    model.fit(X_combined[tr_idx], y[tr_idx])
    imp = model.feature_importances_
    
    top5 = np.argsort(imp)[::-1][:10]
    lg(f"    {target} tree-importance top 10:")
    for idx in top5:
        lg(f"      {all_names[idx]}: {imp[idx]:.2f}")

# ── Save ──
lg(f"\n{'=' * 60}")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_FILE, "w") as f:
    json.dump(results, f, indent=2, default=str)
lg(f"Results saved: {OUT_FILE}")
lg(f"  interactions: {results['interactions_generated']}")
lg(f"  nonlinear: {results['nonlinear_transforms_applied']}")
lg(f"  stable_features: {len(results['stable_features'])}")
lg(f"  baseline_oof: {results['baseline_avg_oof']}")
lg(f"  new_oof: {results['new_feature_avg_oof']}")
lg(f"  delta: {results['delta']}")
lg("=" * 60)
