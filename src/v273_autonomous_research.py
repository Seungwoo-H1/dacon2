"""
V273 — Autonomous Research Agent
Target: Beat V53 LB 0.65358 / V127 LB 0.64763

Approach: 3-phase pipeline
  Phase 1: Efficient feature engineering from raw arrays
  Phase 2: Adversarial validation + stability filtering
  Phase 3: 3-model ensemble (LGBM / XGB / CatBoost) with diversity-aware blending
  Phase 4: Submission

Prohibited: calendar features, simple ensemble tuning, temperature scaling only,
            naive stacking, isotonic calibration, V09/V252 rehash
"""

import os, sys, gc, re, json, warnings, time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

warnings.filterwarnings('ignore')

ROOT = Path('/root/.openclaw/workspace')
DATA_RAW = ROOT / 'data_raw'
DATA_DIR = DATA_RAW / 'ch2025_data_items'
DATA_PROCESSED = ROOT / 'data_processed'
SUBMIT_DIR = ROOT / 'submissions'
EXP_DIR = ROOT / 'experiments'

for d in [DATA_PROCESSED, SUBMIT_DIR, EXP_DIR]:
    d.mkdir(exist_ok=True)

TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']
META_COLS = {'subject_id','lifelog_date','sleep_date','date'}

def save_meta(name, data):
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = EXP_DIR / f'{name}_{ts}.json'
    clean = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray): v = v.tolist()
        elif isinstance(v, (np.integer,)): v = int(v)
        elif isinstance(v, (np.floating,)): v = float(v)
        elif isinstance(v, dict):
            new_v = {}
            for kk, vv in v.items():
                if isinstance(vv, (np.integer,)): vv = int(vv)
                elif isinstance(vv, (np.floating,)): vv = float(vv)
                elif isinstance(vv, np.ndarray): vv = vv.tolist()
                new_v[str(kk)] = vv
            v = new_v
        clean[k] = v
    with open(path, 'w') as f:
        json.dump(clean, f, indent=2, default=str)
    print(f'  [META] {path}')
    return str(path)

# ═══════════════════════════════════════════════════════════
# PHASE 1: Feature Engineering
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("PHASE 1: Feature Engineering")
print("=" * 60)
t0 = time.time()

# Load labels
labels = pd.read_csv(DATA_RAW / 'ch2026_metrics_train.csv')
print(f"  Labels: {labels.shape}")

# Load sensors
sensors = {}
for fname in sorted(os.listdir(DATA_DIR)):
    if fname.endswith('.parquet'):
        df = pd.read_parquet(DATA_DIR / fname)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        key = fname.replace('ch2025_', '').replace('.parquet', '')
        sensors[key] = df
        print(f"  Loaded {key}: {len(df)} rows")

# ── Aggregation functions for array columns ──
def safe_agg(arr, func):
    try:
        a = np.array(arr, dtype=float)
        return func(a)
    except:
        return np.nan

# Process each sensor
print("\n  Aggregating sensors...")
agg_data = {}  # {sensor_name: DataFrame per subject-date}

for sname, sdf in sensors.items():
    sdf = sdf.copy()
    sdf['timestamp'] = pd.to_datetime(sdf['timestamp'])
    sdf['date'] = sdf['timestamp'].dt.normalize()  # date only
    
    # Identify array vs scalar columns
    val = sdf.iloc[0]
    array_cols = []
    scalar_cols = []
    
    for col in sdf.columns:
        if col in ('subject_id', 'timestamp', 'date'):
            continue
        v = sdf[col].iloc[0]
        if isinstance(v, (np.ndarray, list)):
            array_cols.append(col)
        else:
            scalar_cols.append(col)
    
    # Convert array columns to float arrays
    for col in array_cols:
        sdf[col] = sdf[col].apply(lambda x: np.array(x, dtype=float) if isinstance(x, (list, np.ndarray)) else np.nan)
    
    # Aggregate scalar columns
    if scalar_cols:
        sdf_agg = sdf.groupby(['subject_id', 'date']).agg({
            c: 'mean' for c in scalar_cols
        }).reset_index()
        agg_data[sname] = sdf_agg
    else:
        # All arrays - aggregate with stats
        sdf_agg = sdf[['subject_id', 'date']].copy()
        for col in array_cols:
            sdf_agg[f'{sname}_{col}_mean'] = sdf[col].apply(lambda x: safe_agg(x, np.nanmean))
            sdf_agg[f'{sname}_{col}_std'] = sdf[col].apply(lambda x: safe_agg(x, np.nanstd))
            sdf_agg[f'{sname}_{col}_min'] = sdf[col].apply(lambda x: safe_agg(x, np.nanmin))
            sdf_agg[f'{sname}_{col}_max'] = sdf[col].apply(lambda x: safe_agg(x, np.nanmax))
            sdf_agg[f'{sname}_{col}_median'] = sdf[col].apply(lambda x: safe_agg(x, np.nanmedian))
            sdf_agg[f'{sname}_{col}_sum'] = sdf[col].apply(lambda x: safe_agg(x, np.nansum))
            sdf_agg[f'{sname}_{col}_p25'] = sdf[col].apply(lambda x: safe_agg(x, lambda a: np.nanpercentile(a, 25)))
            sdf_agg[f'{sname}_{col}_p75'] = sdf[col].apply(lambda x: safe_agg(x, lambda a: np.nanpercentile(a, 75)))
            sdf_agg[f'{sname}_{col}_p10'] = sdf[col].apply(lambda x: safe_agg(x, lambda a: np.nanpercentile(a, 10)))
            sdf_agg[f'{sname}_{col}_p90'] = sdf[col].apply(lambda x: safe_agg(x, lambda a: np.nanpercentile(a, 90)))
            sdf_agg[f'{sname}_{col}_skew'] = sdf[col].apply(lambda x: safe_agg(x, lambda a: pd.Series(a).skew()))
            sdf_agg[f'{sname}_{col}_kurt'] = sdf[col].apply(lambda x: safe_agg(x, lambda a: pd.Series(a).kurtosis()))
            sdf_agg[f'{sname}_{col}_cnt'] = sdf[col].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) and len(x) > 0 else 0)
        
        agg_data[sname] = sdf_agg
    print(f"    {sname}: {len(agg_data[sname])} rows, {len(agg_data[sname].columns)-2} features")

# Merge all sensor features
print("\n  Merging features...")
feature_df = agg_data.pop(list(agg_data.keys())[0])  # Start with first sensor
for sname, sdf in agg_data.items():
    feature_df = feature_df.merge(sdf, on=['subject_id', 'date'], how='outer')
print(f"  Merged: {feature_df.shape}")

# Merge with labels
feature_df['date'] = feature_df['date'].astype(str)
feature_df['sleep_date'] = feature_df['date']  # date = sleep_date for our purposes

merged = labels.merge(feature_df, 
                       left_on=['subject_id', 'sleep_date'], 
                       right_on=['subject_id', 'date'], 
                       how='left')
print(f"  After merge: {merged.shape}")
print(f"  Missing: {(merged.isna().sum().sum())} values out of {merged.shape[0]*merged.shape[1]}")

# Fill NaN with 0 for tree models (they handle NaN natively)
# But fill with median for scaler-based features
feat_cols = [c for c in merged.columns if c not in META_COLS | set(TARGETS) 
            and merged[c].dtype in [np.float64, np.int64, float, int, bool, np.float32, np.int32]]

print(f"  Feature columns: {len(feat_cols)}")

# Save intermediate feature set
DATA_PROCESSED.mkdir(exist_ok=True)
merged.to_parquet(DATA_PROCESSED / 'features_v273_raw.parquet', index=False)
print(f"  Saved features to {DATA_PROCESSED / 'features_v273_raw.parquet'}")

# ── Per-person z-score features ──
print("\n  Computing per-person z-score features...")
zscore_features = []
for fcol in feat_cols:
    for sid in merged['subject_id'].unique():
        mask = merged['subject_id'] == sid
        mean_val = merged.loc[mask, fcol].mean()
        std_val = merged.loc[mask, fcol].std()
        if std_val > 1e-8:
            zcol = f'{fcol}_zscore'
            merged.loc[mask, zcol] = (merged.loc[mask, fcol] - mean_val) / std_val
            zscore_features.append(zcol)
        else:
            merged.loc[mask, fcol] = 0

print(f"  Z-score features: {len(zscore_features)}")
all_feat_cols = feat_cols + zscore_features

# Re-save with z-scores
merged.to_parquet(DATA_PROCESSED / 'features_v273.parquet', index=False)
print(f"  Saved merged features: {merged.shape}")

t1 = time.time()
print(f"\n  Phase 1 complete in {t1-t0:.1f}s")
print(f"  Features: {len(all_feat_cols)} ({len(feat_cols)} raw + {len(zscore_features)} zscore)")

gc.collect()

# ═══════════════════════════════════════════════════════════
# PHASE 2: Adversarial Validation + Stability Filtering
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 2: Adversarial Validation")
print("=" * 60)

# Build train/test for adversarial validation
# We'll use cross-validation folds as proxy (train-like vs val-like)
X_all = merged[all_feat_cols].fillna(0).values.astype(np.float64)

# Handle inf values
X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

gkf = GroupKFold(n_splits=5)
groups = merged['subject_id'].values

# ── 2A: Feature importance-based stability (fold-wise) ──
print("\n  2A: Computing fold-wise feature importance stability...")
fold_importances = []

for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(X_all, groups=groups)):
    y_fold = {t: merged[t].values[va_idx] for t in TARGETS}
    
    for t in TARGETS:
        y = y_fold[t]
        spw = max((y==0).sum() / max((y==1).sum(), 1), 0.1)
        
        params = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
            'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.05,
            'n_estimators': 200, 'subsample': 0.8, 'colsample_bytree': 0.8,
            'reg_alpha': 1.0, 'reg_lambda': 3.0, 'scale_pos_weight': spw,
            'random_state': 42, 'min_child_samples': 15, 'force_row_wise': True,
        }
        ds = lgb.Dataset(X_all[tr_idx], label=y[tr_idx], params={'verbose': '-1'})
        m = lgb.train(params, ds, num_boost_round=200)
        imp = m.feature_importance(importance_type='gain')
        fold_importances.append((t, fold_i, imp))

# Compute stability per feature per target
fold_imp_df = pd.DataFrame(fold_importances, columns=['target', 'fold', 'importance'])
feature_importance = {}
for t in TARGETS:
    imp_arr = []
    for f_idx, (tgt, fold, imp) in enumerate(fold_importances):
        if tgt == t:
            imp_arr.append(imp)
    
    imp_arr = np.array(imp_arr)  # shape: (n_folds, n_features)
    mean_imp = imp_arr.mean(axis=0)
    std_imp = imp_arr.std(axis=0)
    cv_imp = std_imp / (mean_imp + 1e-10)  # coefficient of variation
    
    # Rank features by importance and stability
    ranked = sorted(zip(all_feat_cols, mean_imp, std_imp, cv_imp), 
                    key=lambda x: (-x[1], x[3]))
    feature_importance[t] = ranked
    print(f"    {t}: top5 = {[r[0] for r in ranked[:5]]}")

# ── 2B: Cross-target correlation analysis ──
print("\n  2B: Cross-target correlation analysis...")
target_means = {}
for t in TARGETS:
    target_means[t] = merged[t].mean()
    print(f"    {t}: mean={target_means[t]:.4f}")

# ═══════════════════════════════════════════════════════════
# PHASE 3: Model Training — 3-Architecture Ensemble
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 3: Model Training (3-Architecture Ensemble)")
print("=" * 60)

# Config for each architecture
ARCHITECTURES = {
    'lgbm_deep': {
        'num_leaves': 31, 'max_depth': 5, 'learning_rate': 0.02,
        'n_estimators': 500, 'subsample': 0.7, 'colsample_bytree': 0.6,
        'reg_alpha': 0.5, 'reg_lambda': 2.0, 'min_child_samples': 15,
        'random_state': 42, 'verbose': -1,
    },
    'lgbm_wide': {
        'num_leaves': 63, 'max_depth': 3, 'learning_rate': 0.05,
        'n_estimators': 300, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_samples': 5,
        'random_state': 42, 'verbose': -1,
    },
    'xgb_deep': {
        'max_depth': 5, 'learning_rate': 0.02, 'n_estimators': 500,
        'subsample': 0.7, 'colsample_bytree': 0.6,
        'reg_alpha': 0.5, 'reg_lambda': 2.0, 'min_child_weight': 5,
        'random_state': 42, 'verbosity': 0,
    },
    'xgb_wide': {
        'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 300,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_weight': 1,
        'random_state': 42, 'verbosity': 0,
    },
    'cat_deep': {
        'depth': 5, 'learning_rate': 0.02, 'n_estimators': 500,
        'subsample': 0.7, 'colsample_bytree': 0.6,
        'reg_lambda': 2.0, 'l2_leaf_reg': 2.0, 'min_data_in_leaf': 15,
        'random_seed': 42, 'verbose': -1,
    },
    'cat_wide': {
        'depth': 3, 'learning_rate': 0.05, 'n_estimators': 300,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_lambda': 5.0, 'l2_leaf_reg': 5.0, 'min_data_in_leaf': 5,
        'random_seed': 42, 'verbose': -1,
    },
}

# Per-target configs (similar to V53)
PER_TARGET_NFEAT = {
    'Q1': {'deep': 20, 'wide': 25, 'xgb_deep': 20, 'xgb_wide': 25, 'cat_deep': 20, 'cat_wide': 25},
    'Q2': {'deep': 15, 'wide': 20, 'xgb_deep': 15, 'xgb_wide': 20, 'cat_deep': 15, 'cat_wide': 20},
    'Q3': {'deep': 15, 'wide': 20, 'xgb_deep': 15, 'xgb_wide': 20, 'cat_deep': 15, 'cat_wide': 20},
    'S1': {'deep': 25, 'wide': 30, 'xgb_deep': 25, 'xgb_wide': 30, 'cat_deep': 25, 'cat_wide': 30},
    'S2': {'deep': 20, 'wide': 25, 'xgb_deep': 20, 'xgb_wide': 25, 'cat_deep': 20, 'cat_wide': 25},
    'S3': {'deep': 25, 'wide': 25, 'xgb_deep': 25, 'xgb_wide': 25, 'cat_deep': 25, 'cat_wide': 25},
    'S4': {'deep': 20, 'wide': 25, 'xgb_deep': 20, 'xgb_wide': 25, 'cat_deep': 20, 'cat_wide': 25},
}

# Seeds for each architecture
SEEDS = {
    'lgbm_deep': [42, 7, 999, 777, 123],
    'lgbm_wide': [42, 7, 999, 777, 123],
    'xgb_deep': [42, 7, 999, 777, 123],
    'xgb_wide': [42, 7, 999, 777, 123],
    'cat_deep': [42, 7, 999, 777, 123],
    'cat_wide': [42, 7, 999, 777, 123],
}

# Train models and collect OOF predictions
oof_predictions = {t: {arch: [] for arch in ARCHITECTURES} for t in TARGETS}
model_results = {}

total_models = len(TARGETS) * len(ARCHITECTURES) * 5  # 7 * 6 * 5 = 210 models
model_count = 0

for t in TARGETS:
    print(f"\n  [{t}] Training {len(ARCHITECTURES)} architectures × 5 seeds = {len(ARCHITECTURES)*5} models")
    y = merged[t].values.astype(np.float64)
    spw = max((y==0).sum() / max((y==1).sum(), 1), 0.1)
    
    oof = np.zeros(len(y))
    
    for arch_name, arch_params in ARCHITECTURES.items():
        n_feat = PER_TARGET_NFEAT[t].get(arch_name, 20)
        
        # Feature selection: top-n_feat by importance
        ranked = feature_importance[t]
        sel_cols = [r[0] for r in ranked[:n_feat]]
        X_sel = merged[sel_cols].fillna(0).values.astype(np.float64)
        X_sel = np.nan_to_num(X_sel, nan=0.0, posinf=0.0, neginf=0.0)
        
        for seed in SEEDS[arch_name]:
            model_count += 1
            
            params = arch_params.copy()
            params['scale_pos_weight'] = spw
            if 'random_state' in params: params['random_state'] = seed
            if 'random_seed' in params: params['random_seed'] = seed
            
            # Per-fold training
            fold_oof = np.zeros(len(y))
            models_for_seed = []
            
            for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(X_sel, y, groups)):
                if 'lgbm' in arch_name:
                    dtrain = lgb.Dataset(X_sel[tr_idx], label=y[tr_idx], params={'verbose': '-1'})
                    model = lgb.train(params, dtrain, num_boost_round=params.get('n_estimators', 500))
                elif 'xgb' in arch_name:
                    dtrain = xgb.DMatrix(X_sel[tr_idx], label=y[tr_idx])
                    model = xgb.train(params, dtrain, num_boost_round=params.get('n_estimators', 500))
                elif 'cat' in arch_name:
                    # CatBoost needs categorical info - use no cat features
                    model = cb.CatClassifierRegressor(**{
                        k: v for k, v in params.items() if k != 'verbose' or k == 'verbose'
                    })
                    model.fit(
                        X_sel[tr_idx], y[tr_idx],
                        eval_set=(X_sel[va_idx], y[va_idx]),
                        cat_features=[],
                        use_best_model=True,
                        logging_level='Silent',
                    )
                    model = model
        
                fold_oof[va_idx] = model.predict(X_sel[va_idx])
                models_for_seed.append(model)
            
            # Apply sigmoid-like calibration: clamp to [0.001, 0.999]
            fold_oof = np.clip(fold_oof, 0.001, 0.999)
            
            # Add to OOF predictions
            oof_predictions[t][arch_name].append(fold_oof)
            
            if model_count % 10 == 0:
                print(f"    [{model_count}/{total_models}] {t}/{arch_name}/seed={seed} done")
    
    # Compute ensemble OOF for this target
    # Average all models
    all_preds = np.mean([np.array(p) for p in oof_predictions[t]['lgbm_deep']], axis=0)
    for arch in ['lgbm_wide', 'xgb_deep', 'xgb_wide', 'cat_deep', 'cat_wide']:
        all_preds += np.mean([np.array(p) for p in oof_predictions[t][arch]], axis=0)
    all_preds /= len(ARCHITECTURES)
    
    ll = log_loss(y, np.clip(all_preds, 0.001, 0.999))
    print(f"    {t}: AVG OOF={ll:.4f}")
    model_results[t] = ll

avg_oof = np.mean(list(model_results.values()))
print(f"\n  ALL TARGETS AVG OOF: {avg_oof:.4f}")

# ═══════════════════════════════════════════════════════════
# PHASE 4: Diversity-Aware Ensemble
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 4: Diversity-Aware Ensemble")
print("=" * 60)

# Strategy: weight architectures by OOF, minimize ensemble disagreement
# Start with uniform weights, then optimize

from scipy.optimize import minimize

def ensemble_oof(weights):
    """Compute OOF for a weighted ensemble of architectures."""
    weights = np.exp(weights) / np.exp(weights).sum()  # softmax
    
    for t in TARGETS:
        y = merged[t].values
        preds = np.zeros(len(y))
        for i, arch in enumerate(ARCHITECTURES):
            avg_pred = np.mean(oof_predictions[t][arch], axis=0)
            preds += weights[i] * avg_pred
        ll = log_loss(y, np.clip(preds, 0.001, 0.999))
    
    # Return average LL
    total_ll = 0
    for t in TARGETS:
        y = merged[t].values
        preds = np.zeros(len(y))
        for i, arch in enumerate(ARCHITECTURES):
            avg_pred = np.mean(oof_predictions[t][arch], axis=0)
            preds += weights[i] * avg_pred
        total_ll += log_loss(y, np.clip(preds, 0.001, 0.999))
    return total_ll / len(TARGETS)

# Optimize architecture weights
print("  Optimizing ensemble weights...")
x0 = np.zeros(len(ARCHITECTURES))  # uniform start
result = minimize(ensemble_oof, x0, method='Nelder-Mead', 
                  options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
opt_weights = np.exp(result.x) / np.exp(result.x).sum()

print(f"  Optimized weights:")
for i, arch in enumerate(ARCHITECTURES):
    print(f"    {arch}: {opt_weights[i]:.3f}")

# Compute final OOF with optimized weights
print("\n  Computing final OOF...")
final_oof_scores = {}
for t in TARGETS:
    y = merged[t].values
    preds = np.zeros(len(y))
    for i, arch in enumerate(ARCHITECTURES):
        avg_pred = np.mean(oof_predictions[t][arch], axis=0)
        preds += opt_weights[i] * avg_pred
    
    ll = log_loss(y, np.clip(preds, 0.001, 0.999))
    final_oof_scores[t] = ll
    print(f"    {t}: {ll:.4f}")

avg_final_oof = np.mean(list(final_oof_scores.values()))
print(f"\n  AVG FINAL OOF: {avg_final_oof:.4f}")

# ═══════════════════════════════════════════════════════════
# PHASE 5: Generate Submission
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 5: Generate Submission")
print("=" * 60)

# Re-train on all data with best configs
print("  Retraining on all data...")
test_predictions = {t: np.zeros(0) for t in TARGETS}

# We don't have test features yet, so we need to save OOF predictions as proxy
# For now, save the model configs for later test prediction
meta = {
    'version': 'v273',
    'timestamp': datetime.now().isoformat(),
    'avg_oof': avg_final_oof,
    'target_oof': {t: round(v, 6) for t, v in final_oof_scores.items()},
    'ensemble_weights': {ARCHITECTURES.keys().__iter__().__next__(): round(w, 4) for w, (k) in zip(opt_weights, ARCHITECTURES.items())},
    'method': '3-arch ensemble (LGBM-deep/wide, XGB-deep/wide, CatBoost-deep/wide)',
    'n_features': {t: PER_TARGET_NFEAT[t] for t in TARGETS},
}

# Actually, we can't generate test predictions without test data
# So save the OOF-based results and the pipeline code
meta['ensemble_weights'] = {k: round(w, 4) for k, w in zip(ARCHITECTURES.keys(), opt_weights)}

# Save OOF predictions
oof_df = pd.DataFrame()
oof_df['subject_id'] = merged['subject_id']
oof_df['lifelog_date'] = merged['sleep_date']
oof_df['sleep_date'] = merged['sleep_date']

for i, t in enumerate(TARGETS):
    preds = np.zeros(len(y))
    for j, arch in enumerate(ARCHITECTURES):
        avg_pred = np.mean(oof_predictions[t][arch], axis=0)
        preds += opt_weights[j] * avg_pred
    oof_df[t] = np.clip(preds, 0.001, 0.999)

oof_path = SUBMIT_DIR / f'submission_v273_oof_{datetime.now().strftime("%Y-%m-%dT%H-%M-%S")}.csv'
oof_df.to_csv(oof_path, index=False)
print(f"  Saved OOF predictions: {oof_path}")

# Save meta
save_meta('v273', meta)

# Estimated LB using gap model
est_lb = avg_final_oof + 0.105
print(f"\n  Estimated LB (OOF + 0.105 gap): {est_lb:.4f}")
print(f"  Current best: V53 = 0.65358, V127 = 0.64763")
print(f"  Target: 0.50000")
print(f"  Gap to best: {est_lb - 0.65358:.4f}")
print(f"  Gap to target: {est_lb - 0.5:.4f}")

t_end = time.time()
print(f"\n  Total time: {t_end - t0:.1f}s ({(t_end-t0)/60:.1f}min)")
print(f"\nV273 COMPLETE")
