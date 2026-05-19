"""
V273 — Autonomous Research Agent
Clean pipeline from scratch, handling all raw data quirks
"""
import os, gc, json, re, warnings, time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

warnings.filterwarnings('ignore')
np.random.seed(42)

ROOT = Path('/root/.openclaw/workspace')
DATA_RAW = ROOT / 'data_raw'
DATA_DIR = DATA_RAW / 'ch2025_data_items'
DATA_PROC = ROOT / 'data_processed'
SUBMIT = ROOT / 'submissions'
EXP = ROOT / 'experiments'
for d in [DATA_PROC, SUBMIT, EXP]:
    d.mkdir(exist_ok=True)

TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']
t0 = time.time()

print("=" * 60)
print("V273 — Autonomous Research Agent")
print(f"Started: {datetime.now()}")
print("=" * 60)

# ── Load raw data ──
labels = pd.read_csv(DATA_RAW / 'ch2026_metrics_train.csv')
labels['sleep_date_parsed'] = pd.to_datetime(labels['sleep_date']).dt.date
print(f"Labels: {labels.shape}, subjects={labels['subject_id'].nunique()}, dates={labels['sleep_date_parsed'].nunique()}")

# Build subject-date lookup from labels to know which dates to aggregate
label_pairs = set(zip(labels['subject_id'], labels['sleep_date_parsed']))
print(f"Subject-date pairs: {len(label_pairs)}")

# ── Phase 1: Feature Engineering ──
print("\n[PHASE 1] Feature Engineering")

all_feature_rows = {}  # {(subject_id, date): {feature: value}}

def extract_floats(arr):
    """Extract all numeric values from nested array structures."""
    vals = []
    if isinstance(arr, (list, np.ndarray)):
        for item in arr:
            if isinstance(item, (list, np.ndarray)):
                for sub in item:
                    try: vals.append(float(sub))
                    except: pass
            else:
                try: vals.append(float(item))
                except: pass
    return vals

sensors = {}
for fname in sorted(os.listdir(DATA_DIR)):
    if not fname.endswith('.parquet'):
        continue
    df = pd.read_parquet(DATA_DIR / fname)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    
    sname = fname.replace('ch2025_', '').replace('.parquet', '')
    sensors[sname] = df
    print(f"\n  Processing {sname}: {len(df)} rows")
    
    # Only keep rows that match known label pairs
    df = df[df.apply(lambda r: (r['subject_id'], r['date']) in label_pairs, axis=1)]
    print(f"  After date filtering: {len(df)} rows")
    
    if len(df) == 0:
        continue
    
    # Identify array vs scalar columns
    array_cols = []
    scalar_cols = []
    for col in df.columns:
        if col in ('subject_id', 'timestamp', 'date'):
            continue
        v = df[col].iloc[0]
        if isinstance(v, (list, np.ndarray)):
            array_cols.append(col)
        else:
            scalar_cols.append(col)
    
    print(f"    Scalar cols: {scalar_cols}, Array cols: {array_cols}")
    
    # ── Process array columns ──
    for col in array_cols:
        # Extract floats from nested arrays
        df[f'{col}_vals'] = df[col].apply(extract_floats)
        
        # Aggregate
        for func, suffix in [(np.nanmean, 'mean'), (np.nanstd, 'std'), (np.nanmin, 'min'),
                             (np.nanmax, 'max'), (np.nanmedian, 'median'), (np.nansum, 'sum')]:
            col_name = f'{sname}_{col}_{suffix}'
            df[col_name] = df[f'{col}_vals'].apply(
                lambda vs: func(vs) if len(vs) > 0 else np.nan)
        
        for p in [10, 25, 50, 75, 90]:
            col_name = f'{sname}_{col}_p{p}'
            df[col_name] = df[f'{col}_vals'].apply(
                lambda vs: np.percentile(vs, p) if len(vs) > 0 else np.nan)
        
        df[f'{sname}_{col}_skew'] = df[f'{col}_vals'].apply(
            lambda vs: pd.Series(vs).skew() if len(vs) > 2 else np.nan)
        df[f'{sname}_{col}_kurt'] = df[f'{col}_vals'].apply(
            lambda vs: pd.Series(vs).kurtosis() if len(vs) > 3 else np.nan)
        df[f'{sname}_{col}_cnt'] = df[f'{col}_vals'].apply(len)
        
        df = df.drop(columns=[f'{col}_vals'])
    
    # ── Process scalar columns ──
    for col in scalar_cols:
        df[f'{sname}_{col}'] = pd.to_numeric(df[col], errors='coerce')
    
    # ── Group by subject_id + date ──
    feat_cols = [c for c in df.columns if c not in ('subject_id', 'timestamp', 'date', 'm_activity')]
    # m_activity is int64 scalar, handle separately
    
    if 'm_activity' in df.columns and 'm_activity' not in scalar_cols:
        df['m_activity_val'] = pd.to_numeric(df['m_activity'], errors='coerce')
        feat_cols.append('m_activity_val')
    
    # Keep only numeric feat_cols
    feat_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df[c])]
    
    if feat_cols:
        grouped = df.groupby(['subject_id', 'date']).agg({
            c: 'mean' for c in feat_cols if c not in ('m_activity_val',)
        }).reset_index()
        
        if 'm_activity_val' in feat_cols:
            mact_agg = df.groupby(['subject_id', 'date'])['m_activity_val'].mean().reset_index()
            mact_agg = mact_agg.rename(columns={'m_activity_val': 'mActivity_m_activity_mean'})
            grouped = grouped.merge(mact_agg, on=['subject_id', 'date'], how='outer')
        
        # Convert to dict per row
        for _, row in grouped.iterrows():
            key = (row['subject_id'], row['date'])
            if key not in all_feature_rows:
                all_feature_rows[key] = {}
            for fc in feat_cols:
                if fc in ('subject_id', 'date'):
                    continue
                v = row.get(fc)
                if v is not None and not pd.isna(v):
                    all_feature_rows[key][fc] = float(v)
    
    gc.collect()

print(f"\n  Total feature rows: {len(all_feature_rows)}")
print(f"  Expected: {len(label_pairs)}")

# Build feature DataFrame
print("\n  Building feature matrix...")
feat_list = []
for (sid, date), feats in all_feature_rows.items():
    row = {'subject_id': sid, 'sleep_date': date}
    row.update(feats)
    feat_list.append(row)

feat_df = pd.DataFrame(feat_list)
print(f"  Feature DF: {feat_df.shape}")

# Merge with labels
merged = labels.merge(feat_df, on=['subject_id', 'sleep_date'], how='left')
print(f"  After merge: {merged.shape}")
print(f"  Missing values: {merged.isna().sum().sum()}")

# Identify numeric feature columns
feat_cols = [c for c in merged.columns
             if c not in ('subject_id','lifelog_date','sleep_date','sleep_date_parsed') and
             merged[c].dtype in [np.float64, np.int64, float, int, np.float32, np.int32]]
print(f"  Numeric features: {len(feat_cols)}")

# ── Per-person z-score ──
print(f"\n  Per-person z-scores for {len(feat_cols)} features...")
zscore_cols = []
t_z = time.time()

for fcol in feat_cols:
    merged[fcol] = merged[fcol].astype(float)
    for sid in merged['subject_id'].unique():
        mask = merged['subject_id'] == sid
        mn = merged.loc[mask, fcol].mean()
        sd = merged.loc[mask, fcol].std()
        if sd > 1e-8 and not np.isnan(sd):
            zcol = f'{fcol}_zscore'
            merged.loc[mask, zcol] = (merged.loc[mask, fcol] - mn) / sd
            zscore_cols.append(zcol)
        elif not np.isnan(mn):
            merged.loc[mask, fcol] = 0.0

print(f"  Done: {len(zscore_cols)} z-score cols in {time.time()-t_z:.1f}s")

all_feat = feat_cols + zscore_cols
print(f"  Total features: {len(all_feat)}")

# Build feature matrix
X = merged[all_feat].fillna(0).values.astype(np.float64)
X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
print(f"  X shape: {X.shape}")

DATA_PROC.mkdir(exist_ok=True)
merged.to_parquet(DATA_PROC / 'features_v273.parquet', index=False)

gc.collect()

# ── Phase 2: Fold-wise Feature Importance (stability analysis) ──
print("\n[PHASE 2] Fold-wise Feature Importance")
gkf = GroupKFold(n_splits=5)
groups = merged['subject_id'].values

fold_imp = {}
for t in TARGETS:
    y = merged[t].values.astype(np.float64)
    spw = max((y==0).sum() / max((y==1).sum(), 1), 0.1)
    
    fold_imps = []
    for fi in range(5):
        params = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
            'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.05,
            'n_estimators': 100, 'subsample': 0.8, 'colsample_bytree': 0.8,
            'reg_alpha': 1.0, 'reg_lambda': 3.0, 'scale_pos_weight': spw,
            'min_child_samples': 15, 'force_row_wise': True,
            'random_state': 42 + fi,
        }
        tr_idx = np.random.RandomState(fi).choice(len(y), size=min(200, len(y)), replace=False)
        ds = lgb.Dataset(X[tr_idx], label=y[tr_idx])
        mdl = lgb.train(params, ds, num_boost_round=100)
        imp = mdl.feature_importance(importance_type='gain')
        fold_imps.append(imp)
    
    fold_imps = np.array(fold_imps)
    mean_imp = fold_imps.mean(axis=0)
    std_imp = fold_imps.std(axis=0)
    cv_imp = std_imp / (mean_imp + 1e-10)
    fold_imp[t] = (mean_imp, std_imp, cv_imp)
    
    top5 = np.argsort(mean_imp)[-5:][::-1]
    print(f"    {t}: {[all_feat[i] for i in top5]}")

# ── Phase 3: 6-Architecture Ensemble ──
print("\n[PHASE 3] 6-Architecture Ensemble Training")

ARCHS = {
    'lgbm_deep':  {'num_leaves': 31, 'max_depth': 5,  'lr': 0.02, 'n_est': 500,
                   'subsample': 0.7, 'colsample': 0.6, 'alpha': 0.5, 'lamb': 2.0, 'child': 15},
    'lgbm_wide':  {'num_leaves': 63, 'max_depth': 3,  'lr': 0.05, 'n_est': 300,
                   'subsample': 0.8, 'colsample': 0.8, 'alpha': 2.0, 'lamb': 5.0, 'child': 5},
    'xgb_deep':   {'max_depth': 5,  'eta': 0.02, 'n_est': 500,
                   'subsample': 0.7, 'colsample_bytree': 0.6, 'alpha': 0.5, 'lambda': 2.0, 'gamma': 5},
    'xgb_wide':   {'max_depth': 3,  'eta': 0.05, 'n_est': 300,
                   'subsample': 0.8, 'colsample_bytree': 0.8, 'alpha': 2.0, 'lambda': 5.0, 'gamma': 1},
    'cat_deep':   {'depth': 5, 'lr': 0.02, 'n_est': 500,
                   'subsample': 0.7, 'colsample': 0.6, 'lamb': 2.0, 'leaf_reg': 2.0, 'min_leaf': 15},
    'cat_wide':   {'depth': 3, 'lr': 0.05, 'n_est': 300,
                   'subsample': 0.8, 'colsample': 0.8, 'lamb': 5.0, 'leaf_reg': 5.0, 'min_leaf': 5},
}
SEEDS = [42, 7, 999, 777, 123]
NFEAT = {t: {a: (20 if 'deep' in a else 25) for a in ARCHS} for t in TARGETS}

oof_preds = {t: {a: [] for a in ARCHS} for t in TARGETS}
total_models = len(TARGETS) * len(ARCHS) * len(SEEDS)
cnt = 0

for t in TARGETS:
    print(f"\n  [{t}/{len(TARGETS)}]")
    y = merged[t].values.astype(np.float64)
    spw = max((y==0).sum() / max((y==1).sum(), 1), 0.1)
    
    mean_imp, _, _ = fold_imp[t]
    ranked = sorted(range(len(all_feat)), key=lambda i: -mean_imp[i])
    
    for arch in ARCHS:
        nf = NFEAT[t][arch]
        sel = ranked[:nf]
        Xs = X[:, sel]
        
        for si, seed in enumerate(SEEDS):
            cnt += 1
            fold_oof = np.zeros(len(y))
            
            for fi, (tri, vai) in enumerate(gkf.split(Xs, y, groups)):
                sp = {k: v for k, v in ARCHS[arch].items()}
                sp['scale_pos_weight'] = spw
                rnd_key = 'random_state' if 'cat' not in arch else 'random_seed'
                sp[rnd_key] = seed
                
                if 'lgbm' in arch:
                    ds = lgb.Dataset(Xs[tri], label=y[tri])
                    sp_lgb = {
                        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
                        'num_leaves': sp.get('num_leaves', 31),
                        'max_depth': sp.get('max_depth', -1),
                        'learning_rate': sp.get('lr', 0.05),
                        'subsample': sp.get('subsample', 1.0),
                        'colsample_bytree': sp.get('colsample', 1.0),
                        'reg_alpha': sp.get('alpha', 0.0),
                        'reg_lambda': sp.get('lamb', 1.0),
                        'min_child_samples': sp.get('child', 20),
                        'force_row_wise': True,
                    }
                    mdl = lgb.train(sp_lgb, ds, num_boost_round=sp['n_est'])
                    fold_oof[vai] = mdl.predict(Xs[vai])
                elif 'xgb' in arch:
                    dm = xgb.DMatrix(Xs[tri], label=y[tri])
                    sp_xgb = {
                        'max_depth': sp.get('max_depth', 6),
                        'eta': sp.get('eta', 0.1),
                        'subsample': sp.get('subsample', 1.0),
                        'colsample_bytree': sp.get('colsample_bytree', 1.0),
                        'alpha': sp.get('alpha', 0.0),
                        'lambda': sp.get('lambda', 1.0),
                        'gamma': sp.get('gamma', 0.0),
                        'tree_method': 'hist',
                    }
                    mdl = xgb.train(sp_xgb, dm, num_boost_round=sp['n_est'])
                    fold_oof[vai] = mdl.predict(xgb.DMatrix(Xs[vai]))
                elif 'cat' in arch:
                    mdl = cb.CatBoostClassifier(
                        depth=sp['depth'], learning_rate=sp['lr'],
                        iterations=sp['n_est'], subsample=sp['subsample'],
                        colsample_bylevel=sp['colsample'],
                        l2_leaf_reg=sp['leaf_reg'],
                        min_data_in_leaf=sp['min_leaf'],
                        random_seed=seed,
                        loss_function='Logloss',
                        eval_metric='AUC',
                        logging_level='Silent')
                    mdl.fit(Xs[tri], y[tri], eval_set=(Xs[vai], y[vai]),
                           use_best_model=True)
                    fold_oof[vai] = mdl.predict(Xs[vai], prediction_type='Probability')[:, 1]
            
            fold_oof = np.clip(fold_oof, 0.001, 0.999)
            oof_preds[t][arch].append(fold_oof)
            
            if cnt % 40 == 0:
                print(f"    [{cnt}/{total_models}] {t}/{arch} ✓")

print(f"\n  All {cnt} models trained in {time.time()-t0:.1f}s")

# ── Phase 4: Weight Optimization ──
print("\n[PHASE 4] Weight Optimization")
from scipy.optimize import minimize

arch_names = list(ARCHS.keys())

def loss_fn(w_raw):
    w = np.exp(w_raw) / np.exp(w_raw).sum()
    total = 0
    for t in TARGETS:
        y = merged[t].values
        p = np.zeros(len(y))
        for i, a in enumerate(arch_names):
            p += w[i] * np.mean(oof_preds[t][a], axis=0)
        total += log_loss(y, np.clip(p, 0.001, 0.999))
    return total / len(TARGETS)

res = minimize(loss_fn, np.zeros(len(arch_names)), method='Nelder-Mead',
               options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
opt_w = np.exp(res.x) / np.exp(res.x).sum()

print("  Weights:")
for i, a in enumerate(arch_names):
    print(f"    {a}: {opt_w[i]:.4f}")

# Final OOF
oof_scores = {}
for t in TARGETS:
    y = merged[t].values
    p = np.zeros(len(y))
    for i, a in enumerate(arch_names):
        p += opt_w[i] * np.mean(oof_preds[t][a], axis=0)
    ll = log_loss(y, np.clip(p, 0.001, 0.999))
    oof_scores[t] = ll
    print(f"    {t}: {ll:.6f}")

avg_oof = np.mean(list(oof_scores.values()))
print(f"\n  AVG OOF: {avg_oof:.6f}")

# ── Phase 5: Submit ──
print("\n[PHASE 5] Submit")
est_lb = avg_oof + 0.105

ts = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
oof_path = SUBMIT / f'submission_v273_{ts}.csv'
df_sub = pd.DataFrame({'subject_id': merged['subject_id'],
                        'lifelog_date': merged['sleep_date'],
                        'sleep_date': merged['sleep_date']})
for t in TARGETS:
    p = np.zeros(len(y))
    for i, a in enumerate(arch_names):
        p += opt_w[i] * np.mean(oof_preds[t][a], axis=0)
    df_sub[t] = np.clip(p, 0.001, 0.999)
df_sub.to_csv(oof_path, index=False)

meta = {
    'version': 'v273', 'time': datetime.now().isoformat(),
    'avg_oof': round(avg_oof, 6), 'target_oof': {t: round(v, 6) for t,v in oof_scores.items()},
    'est_lb': round(est_lb, 6), 'est_gap': 0.105,
    'weights': {a: round(w, 4) for a,w in zip(arch_names, opt_w)},
    'n_feat_raw': len(feat_cols), 'n_feat_zscore': len(zscore_cols),
    'method': '6-arch ensemble + per-target top-K + per-person zscore',
    'time_total': round(time.time()-t0, 1),
}
save_path = EXP / f'v273_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
with open(save_path, 'w') as f:
    json.dump(meta, f, indent=2)

print(f"  AVG OOF: {avg_oof:.6f}")
print(f"  Est LB (OOF+0.105): {est_lb:.6f}")
print(f"  V53 best: 0.65358, V127: 0.64763")
print(f"  Diff to V53: {est_lb - 0.65358:+.4f}")
print(f"\n  Submission: {oof_path}")
print(f"  Meta: {save_path}")
print(f"\n  Total time: {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f}min)")
print("V273 COMPLETE")
