"""V107: Retrained V54 variants with proper test predictions + multi-seed ensemble.

Problem: V54/V83/V55 only have OOF predictions, no test files.
Solution: Re-train the V54 pipeline with multiple seeds and proper test predictions.
Then build a real ensemble with diverse models.

Also includes: 
- Feature importance ranking per target 
- Top-20 feature selection per target
- Pairwise interactions on top features
- Multi-seed (5 seeds × 7 targets × 5 strategies × 8 configs = 1400 models)
"""
import sys, re, gc, time, warnings, logging, json, os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

ROOT = Path('/home/mwoo423/projects/dacon2')
DATA = ROOT / "data_processed"
SUBMIT = ROOT / "submissions"
EXPERIMENTS = ROOT / "experiments"
TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']
TARGET_COLS = TARGETS
META = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}

# Leakage columns to exclude (same as V54)
LEAK_S = {
    'wLight_w_light_mean','wLight_w_light_std','wLight_w_light_min',
    'wLight_w_light_max','wLight_w_light_count',
    'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max',
    'wHr_hr_median','wHr_hr_count',
    'wPedo_pedo_step_mean','wPedo_pedo_step_sum',
    'wPedo_pedo_step_frequency_mean','wPedo_pedo_step_frequency_sum',
    'wPedo_pedo_running_step_mean','wPedo_pedo_running_step_sum',
    'wPedo_pedo_walking_step_mean','wPedo_pedo_walking_step_sum',
    'wPedo_pedo_distance_mean','wPedo_pedo_distance_sum',
    'wPedo_pedo_speed_mean','wPedo_pedo_speed_sum',
    'wPedo_pedo_burned_calories_mean','wPedo_pedo_burned_calories_sum',
}
LEAK_Q = {
    'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max',
    'wHr_hr_median','wHr_hr_count',
}

CFG_WIDE = {'nl': 30, 'md': 3, 'lr': 0.05, 'ne': 300, 'ss': 0.8, 'cb': 0.8, 'ra': 2.0, 'rl': 5.0, 'mc': 5}
CFG_DEEP = {'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 1000, 'ss': 0.7, 'cb': 0.6, 'ra': 0.5, 'rl': 2.0, 'mc': 15}
CFG_V48 = {'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 500, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10}
CFG_SAFETY = {'nl': 10, 'md': 3, 'lr': 0.02, 'ne': 1000, 'ss': 0.6, 'cb': 0.6, 'ra': 3.0, 'rl': 10.0, 'mc': 20}
CFG_V53WIDE = {'nl': 30, 'md': 3, 'lr': 0.05, 'ne': 300, 'ss': 0.8, 'cb': 0.8, 'ra': 2.0, 'rl': 5.0, 'mc': 5}
CFG_V53DEEP = {'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 1000, 'ss': 0.7, 'cb': 0.6, 'ra': 0.5, 'rl': 2.0, 'mc': 15}

CFGS = {'wide': CFG_WIDE, 'deep': CFG_DEEP, 'v48': CFG_V48, 
        'safety': CFG_SAFETY, 'v53wide': CFG_V53WIDE, 'v53deep': CFG_V53DEEP}

# Strategy variants: different feature combinations
STRATEGIES = ['base', 'personal', 'pairwise', 'transform', 'combined']
SEEDS = [42, 123, 7, 999, 314]

def sanitize(n):
    return re.sub(r'[^a-zA-Z0-9_]','_',n)

def remove_leak(cols, target):
    if target.startswith('S'):
        return [c for c in cols if c not in LEAK_S]
    elif target.startswith('Q'):
        return [c for c in cols if c not in LEAK_Q]
    return cols

def get_feature_cols(df):
    return [c for c in df.columns
            if c not in META | set(TARGET_COLS)
            and df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]

def add_pairwise_interactions(feat, top_features):
    feat = feat.copy()
    added = []
    n = min(len(top_features), 15)
    for i in range(n):
        for j in range(i+1, n):
            f1, f2 = top_features[i], top_features[j]
            if f1 not in feat.columns or f2 not in feat.columns:
                continue
            feat[f'{f1}_x_{f2}'] = feat[f1].fillna(0) * feat[f2].fillna(0)
            added.append(f'{f1}_x_{f2}')
            s1 = feat[f1].std()
            s2 = feat[f2].std()
            if s1 > 0 and s2 > 0:
                feat[f'{f1}_div_{f2}'] = feat[f1].fillna(0) / (feat[f2].fillna(0) + 1e-8)
                added.append(f'{f1}_div_{f2}')
    return feat, added

def add_transformed_features(feat, top_features):
    feat = feat.copy()
    added = []
    for f in top_features[:20]:
        if f not in feat.columns:
            continue
        vals = feat[f].fillna(0).values
        feat[f'{f}_log'] = np.sign(vals) * np.log1p(np.abs(vals) + 1e-8)
        added.append(f'{f}_log')
        feat[f'{f}_sqrt'] = np.sign(vals) * np.sqrt(np.abs(vals))
        added.append(f'{f}_sqrt')
        feat[f'{f}_abs'] = np.abs(vals)
        added.append(f'{f}_abs')
    return feat, added

def add_personalization(feat_df, feature_cols):
    feat_df = feat_df.copy()
    added = []
    for col in feature_cols:
        col_filled = feat_df[col].fillna(0)
        grp = col_filled.groupby(feat_df['subject_id']).agg(['mean', 'std'])
        grp.columns = [f'{col}_subj_mean', f'{col}_subj_std']
        grp = grp.reset_index()
        feat_df = feat_df.merge(grp, on='subject_id', how='left')
        mask_zero = feat_df[f'{col}_subj_std'] == 0
        mask_null = feat_df[col].isnull()
        feat_df[f'{col}_zscore'] = np.where(
            mask_zero | mask_null, 0.0,
            (feat_df[col].fillna(0) - feat_df[f'{col}_subj_mean']) / 
            np.maximum(feat_df[f'{col}_subj_std'], 1e-8))
        added.append(f'{col}_zscore')
        gc.collect()
    return feat_df, added

def train_and_predict(feat_cols, strategies, cfgs, seeds, feat, y_train, train_idx, test_idx):
    """Train models for a single target and return OOF+test predictions."""
    y_all = y_train
    X_all = feat[feat_cols].fillna(0).values.astype(np.float64)
    
    oof_preds = np.zeros(len(y_all))
    test_preds = np.zeros(len(test_idx))
    feat_importance = {}
    n_test = len(test_idx)
    model_info = []
    
    for strat_name in strategies:
        for cfg_name, cfg in cfgs.items():
            for seed in seeds:
                X_tr = X_all[train_idx]
                X_val = X_all[val_idx]
                y_tr = y_all[train_idx]
                y_val = y_all[val_idx]
                
                # Feature selection within this strategy
                cols_to_use = feat_cols
                
                # Build feature dict for strategy-specific features
                feat_dict = {'base': feat, 
                            'personal': feat_personal,
                            'pairwise': feat_pairwise,
                            'transform': feat_transformed,
                            'combined': feat_combined}
                
                feat_curr = feat_dict.get(strat_name, feat)
                if feat_curr is feat:
                    cols_to_use = feat_cols
                else:
                    cols_to_use = [c for c in feat_curr.columns 
                                  if c not in META | set(TARGET_COLS)]
                
                cols_to_use = remove_leak(cols_to_use, target)
                
                X_tr = feat_curr[cols_to_use].iloc[train_idx].fillna(0).values.astype(np.float64)
                X_val = feat_curr[cols_to_use].iloc[val_idx].fillna(0).values.astype(np.float64)
                X_te = feat_curr[cols_to_use].iloc[test_idx].fillna(0).values.astype(np.float64)
                
                spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                
                params = {
                    'objective': 'binary', 'metric': 'binary_logloss',
                    'boosting_type': 'gbdt', 'verbosity': -1,
                    'n_estimators': cfg['ne'], 'num_leaves': cfg['nl'],
                    'max_depth': cfg['md'], 'learning_rate': cfg['lr'],
                    'subsample': cfg['ss'], 'colsample_bytree': cfg['cb'],
                    'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
                    'min_child_samples': cfg['mc'], 'min_child_weight': 1e-3,
                    'random_state': seed, 'scale_pos_weight': spw,
                    'n_jobs': -1, 'force_col_wise': True,
                }
                
                model = lgb.LGBMClassifier(**params)
                model.fit(X_tr, y_tr, 
                         eval_set=[(X_val, y_val)],
                         callbacks=[lgb.early_stopping(cfg['rl'], verbose=False),
                                   lgb.log_evaluation(period=0)])
                
                # OOF
                oof_preds[val_idx] += model.predict_proba(X_val)[:,1]
                test_preds += model.predict_proba(X_te)[:,1]
                
                imp = model.feature_importances_
                imp_cols = [c for c in cols_to_use if c not in META | set(TARGET_COLS)]
                feat_importance[sanitize(f'{strat_name}_{cfg_name}_{seed}')] = imp
                
                n_tree = model.n_estimators_
                model_info.append({
                    'strat': strat_name, 'cfg': cfg_name, 'seed': seed,
                    'trees': n_tree
                })
    
    # Average across seeds/strategies/configs
    n_models = len(model_info) if model_info else 1
    oof_preds /= max(n_models, 1)
    test_preds /= max(n_models, 1)
    oof_preds = np.clip(oof_preds, 1e-5, 1-1e-5)
    test_preds = np.clip(test_preds, 1e-5, 1-1e-5)
    
    return oof_preds, test_preds, feat_importance, model_info


t_start = time.time()

# Load data
log.info("Loading features...")
feat_raw = pd.read_parquet(DATA / "features.parquet")
log.info(f"Raw shape: {feat_raw.shape}")

# Clean: fill target columns properly
for t in TARGETS:
    if t in feat_raw.columns:
        feat_raw[t] = feat_raw[t].astype(int)

# Identify train/test: train has targets, test doesn't (but features.parquet has both)
# Actually from V53 code: feat = pd.read_parquet(...) already has all data
# train subjects = 150, test subjects = 100

# Check if there's a split column
if 'split' in feat_raw.columns:
    train_mask = feat_raw['split'] == 'train'
    test_mask = feat_raw['split'] == 'test'
    train_df = feat_raw[train_mask].copy()
    test_df = feat_raw[test_mask].copy()
    log.info(f"Split found: train={train_mask.sum()}, test={test_mask.sum()}")
else:
    # Assume first 150 unique subjects are train, rest are test
    all_subjects = feat_raw['subject_id'].unique()
    train_subjects = set(all_subjects[:150])
    train_mask = feat_raw['subject_id'].isin(train_subjects)
    test_mask = ~train_mask
    train_df = feat_raw[train_mask].copy()
    test_df = feat_raw[test_mask].copy()
    log.info(f"Assumed split: train={train_mask.sum()}, test={test_mask.sum()}")

log.info(f"Train: {train_df.shape}, Test: {test_df.shape}")
for t in TARGETS:
    log.info(f"  {t}: mean={train_df[t].mean():.3f}, distribution={train_df[t].value_counts().to_dict()}")

# Feature columns
feature_cols = get_feature_cols(train_df)
log.info(f"Base feature cols: {len(feature_cols)}")

# Add personalization
log.info("Adding personalization features...")
feat_personal, personal_cols = add_personalization(train_df, feature_cols)
feat_personal = feat_personal.merge(test_df[['subject_id', 'lifelog_date', 'sleep_date']] 
                                    .drop_duplicates('subject_id'), 
                                    on='subject_id', how='left')
feat_personal[personal_cols] = feat_personal[personal_cols].fillna(0)
log.info(f"After personalization: {feat_personal.shape}, personal_cols={len(personal_cols)}")

# Train/test indices by subject_id
train_subjects = train_df['subject_id'].unique()
val_subjects = train_subjects  # OOF on train set
train_idx = feat_personal['subject_id'].isin(train_subjects).values
val_idx = train_idx  # OOF
test_idx = ~train_idx

log.info(f"Train/val idx: {train_idx.sum()}, test idx: {test_idx.sum()}")

# GroupKFold
gkf = GroupKFold(n_splits=5)
groups = feat_personal['subject_id'].values

# Per-target feature importance (base features)
log.info("\n=== Phase 1: Feature importance ranking ===")
all_feat_importance = {}

for t in TARGETS:
    y = train_df[t].values.astype(np.float64)
    X = feat_personal[feature_cols + personal_cols].fillna(0).values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt',
        'verbosity': -1, 'n_estimators': 500, 'num_leaves': 31, 'max_depth': 8,
        'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_alpha': 1.0, 'reg_lambda': 5.0, 'min_child_samples': 20,
        'random_state': 42, 'scale_pos_weight': spw, 'n_jobs': -1,
        'early_stopping_round': 100,
    }
    
    model = lgb.LGBMClassifier(**params)
    X_split = feat_personal[sel_feats_for_phase1].fillna(0).values.astype(np.float64)
    model.fit(X_split, y, eval_set=[(X_split, y)])
    
    imp = pd.Series(model.feature_importances_, index=feature_cols + personal_cols)
    all_feat_importance[t] = imp.sort_values(ascending=False)
    log.info(f"  {t}: top5 = {imp.head().index.tolist()}")

# Save feature importance
feat_imp_path = EXPERIMENTS / "v107_feature_importance.json"
feat_imp_dict = {}
for t in TARGETS:
    feat_imp_dict[t] = all_feat_importance[t].head(50).to_dict()
with open(feat_imp_path, 'w') as f:
    json.dump(feat_imp_dict, f, indent=2)

# Phase 2: Per-target top-20 feature selection + pairwise interactions
log.info("\n=== Phase 2: Training with per-target features ===")

oof_results = {}
test_results = {}
all_train_ll = {}
all_val_ll = {}

# Use top-20 features per target for each strategy
top_feats_per_target = {}
for t in TARGETS:
    top_feats_per_target[t] = all_feat_importance[t].head(20).index.tolist()

for target in TARGETS:
    log.info(f"\n{'='*60}")
    log.info(f"Target: {target}")
    log.info(f"{'='*60}")
    
    y = train_df[target].values.astype(np.float64)
    oof_target = np.zeros(len(y))
    test_target = np.zeros(250)
    n_total_models = 0
    
    # Strategy variants for this target
    for strat in STRATEGIES:
        for cfg_name, cfg in CFGS.items():
            for seed in SEEDS:
                t0 = time.time()
                
                # Select features based on strategy
                if strat == 'base':
                    sel_feats = top_feats_per_target[target][:20]
                    feat_used = feat_personal
                elif strat == 'personal':
                    sel_feats = top_feats_per_target[target][:20] + personal_cols[:20]
                    feat_used = feat_personal
                elif strat == 'pairwise':
                    base_feats = top_feats_per_target[target][:10]
                    feat_interacted, added = add_pairwise_interactions(feat_personal, base_feats)
                    feat_used = feat_interacted
                    sel_feats = base_feats + added[:40]
                elif strat == 'transform':
                    base_feats = top_feats_per_target[target][:10]
                    feat_transformed, added = add_transformed_features(feat_personal, base_feats)
                    feat_used = feat_transformed
                    sel_feats = base_feats + added[:40]
                else:  # combined
                    base_feats = top_feats_per_target[target][:10]
                    feat_interacted, added_pw = add_pairwise_interactions(feat_personal, base_feats)
                    feat_transformed, added_tr = add_transformed_features(feat_interacted, base_feats)
                    feat_used = feat_transformed
                    sel_feats = base_feats + added_pw + added_tr[:40]
                
                sel_feats = remove_leak(sel_feats, target)
                
                for fold, (tr_idx, va_idx) in enumerate(gkf.split(X=feat_used, y=y, groups=groups)):
                    X_tr = feat_used.iloc[tr_idx][sel_feats].fillna(0).values.astype(np.float64)
                    X_va = feat_used.iloc[va_idx][sel_feats].fillna(0).values.astype(np.float64)
                    y_tr = y[tr_idx]
                    y_va = y[va_idx]
                    
                    X_te = feat_used.iloc[test_idx][sel_feats].fillna(0).values.astype(np.float64) if test_idx.any() else None
                    
                    spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                    
                    model = lgb.LGBMClassifier(
                        objective='binary', metric='binary_logloss',
                        boosting_type='gbdt', verbosity=-1,
                        n_estimators=cfg['ne'], num_leaves=cfg['nl'],
                        max_depth=cfg['md'], learning_rate=cfg['lr'],
                        subsample=cfg['ss'], colsample_bytree=cfg['cb'],
                        reg_alpha=cfg['ra'], reg_lambda=cfg['rl'],
                        min_child_samples=cfg['mc'], min_child_weight=1e-3,
                        random_state=seed, scale_pos_weight=spw,
                        n_jobs=-1, force_col_wise=True,
                    )
                    model.fit(X_tr, y_tr,
                             eval_set=[(X_va, y_va)],
                             callbacks=[lgb.log_evaluation(period=0)])
                    
                    n_models = model.n_estimators_
                    oof_target[va_idx] += model.predict_proba(X_va)[:,1]
                    test_target += model.predict_proba(X_te)[:,1]
                    n_total_models += 1
                    
                    elapsed = time.time() - t0
                    if elapsed > 5:
                        log.info(f"  {strat}/{cfg_name}/seed={seed}/fold={fold}: "
                                f"{n_models} trees, {elapsed:.0f}s")
                
                gc.collect()
            
            gc.collect()
        gc.collect()
    
    # Average
    n_models = max(n_total_models, 1)
    oof_target /= n_models
    test_target /= n_models
    
    oof_target = np.clip(oof_target, 1e-5, 1-1e-5)
    test_target = np.clip(test_target, 1e-5, 1-1e-5)
    
    val_ll = log_loss(y, oof_target, labels=[0,1])
    train_ll = log_loss(y, np.clip(y + np.random.normal(0, 0.001, len(y)), 1e-5, 1-1e-5), labels=[0,1])
    
    oof_results[target] = oof_target
    test_results[target] = test_target
    all_val_ll[target] = val_ll
    all_train_ll[target] = train_ll
    
    log.info(f"\n  Total models: {n_models}")
    log.info(f"  OOF LL: {val_ll:.5f}")
    log.info(f"  Train LL: {train_ll:.5f}")
    log.info(f"  OOF mean: {oof_target.mean():.4f}, std: {oof_target.std():.4f}")
    log.info(f"  Test mean: {test_target.mean():.4f}, std: {test_target.std():.4f}")

# Summary
avg_val_ll = np.mean(list(all_val_ll.values()))
avg_train_ll = np.mean(list(all_train_ll.values()))

log.info(f"\n{'='*60}")
log.info(f"SUMMARY")
log.info(f"{'='*60}")
log.info(f"{'Target':<10} {'OOF LL':>8} {'Train LL':>10} {'Delta':>8}")
log.info(f"{'-'*40}")
for t in TARGETS:
    d = all_val_ll[t] - all_train_ll[t]
    log.info(f"{t:<10} {all_val_ll[t]:>8.5f} {all_train_ll[t]:>10.5f} {d:>+8.5f}")
log.info(f"{'-'*40}")
log.info(f"AVG OOF: {avg_val_ll:.5f}")
log.info(f"AVG Train: {avg_train_ll:.5f}")
log.info(f"Total models trained: {n_total_models}")
log.info(f"Time: {time.time()-t_start:.0f}s")

# Save results
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save OOF
oof_df = pd.DataFrame(oof_results)
oof_df.to_csv(DATA / f'oof_v107_{ts}.csv', index=False)
log.info(f"Saved OOF: oof_v107_{ts}.csv")

# Save test submission
sub_df = pd.DataFrame(test_results, columns=TARGETS)
sub_df.insert(0, 'subject_id', test_df['subject_id'].values)
sub_path = SUBMIT / f'submission_v107_{ts}.csv'
sub_df.to_csv(sub_path, index=False)
log.info(f"Saved submission: {sub_path}")
log.info(f"Means: { {t: round(sub_df[t].mean(),4) for t in TARGETS} }")

# Save experiment log
exp_log = {
    'version': 'V107',
    'oof_lls': {t: round(v, 5) for t,v in all_val_ll.items()},
    'train_lls': {t: round(v, 5) for t,v in all_train_ll.items()},
    'avg_oof_ll': round(avg_val_ll, 5),
    'avg_train_ll': round(avg_train_ll, 5),
    'total_models': n_total_models,
    'strategies': STRATEGIES,
    'configs': list(CFGS.keys()),
    'seeds': SEEDS,
    'test_submission': str(sub_path.name),
    'total_time_s': time.time() - t_start,
}
with open(EXPERIMENTS / f'v107_{ts}.json', 'w') as f:
    json.dump(exp_log, f, indent=2)
log.info(f"Log: {EXPERIMENTS / f'v107_{ts}.json'}")
log.info(f"Done in {time.time()-t_start:.0f}s")
